import sys, os, argparse, time, json
from logging import basicConfig, getLogger, INFO, DEBUG; logger = getLogger()
from collections import deque
import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework import nest
from ..components.inference import Inference
from ..components.decoding import length_penalty, beam_search_decode_V2
from ..components.model import align_to_right, remove_offsets, Decoder
from ..components import dataprocessing as dp
from ..language_model import language_model
from ..language_model.inference import Inference as LMInference
from .tm_lm_fusion_beam_search import fusion_beam_search, fusion_beam_search_delayed_pmi


class PMIFusionDecoder(Inference):
    def __init__(self, lm_dir, *args, lm_checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tm = self.model

        # Creating language model
        self.lmi = LMInference(lm_dir, graph=self.graph, n_gpus=self.n_gpus, n_cpu_cores=self.n_cpu_cores, batch_capacity=self.batch_capacity//2, checkpoint=lm_checkpoint)
        self.lm = self.lmi.model

        with self.graph.as_default():
            self.ph_init_y_last_ctok = tf.placeholder(tf.bool, [])
            self.ph_lambda = tf.placeholder(tf.float32, [])
            self.ph_delay = tf.placeholder(tf.int32, [])
            # Triple batch placeholder
            ph_x_y_c = tuple((tf.placeholder(tf.int32, [None,None]), tf.placeholder(tf.int32, [None])) for i in range(3))


            self.op_fusion_decode = self.make_op(self.fn_fusion_decode)
            self.op_fusion_decode2 = self.make_op(
                self.fn_fusion_decode, 1, {'beam_size':self.ph_beam_size})
            self.op_fusion_decode3 = self.make_op(
                self.fn_fusion_decode, 2, {'beam_size': self.ph_beam_size})
            self.op_fusion_decode4 = self.make_op(
                self.fn_fusion_decode, 3, {'beam_size': self.ph_beam_size, 'lambda': self.ph_lambda})
            self.op_fusion_decode10 = self.make_op(self.fn_fusion_decode, 10)
            self.op_fusion_decode11 = self.make_op(self.fn_fusion_decode, 11)
            self.op_cumu_pmi = self.make_op(self.fn_cumulative_pmi)

            self.op_cumu_score = self.make_op(self.fn_cumulative_score, input_phs=ph_x_y_c)
            self.op_cumu_score2 = self.make_op(self.fn_cumulative_score, 1, {'beam_size':self.ph_beam_size}, input_phs=ph_x_y_c)

            self.op_top_pmi = self.make_op(self.fn_top_pmi, input_phs=ph_x_y_c)


    def make_session(self, *args, **kwargs):
        super().make_session(*args, **kwargs)
        self.lmi.make_session(self.session, load_checkpoint=True)


    def fn_fusion_decode(self, inputs, option=0, config=None):
        (x, x_len), (y, y_len) = inputs
        hypos, scores = self.decode(x, x_len, y, y_len, self.ph_beam_size, option, config)
        return hypos, scores

    
    def fusion_decode(self, x, ctx, beam_size=8, ret_search_detail=False, option=0, length_penalty=None, init_y_last_ctok=False, pmi_delay=3):
        batch_capacity = self.batch_capacity // beam_size
        batches = self.make_batches(x, ctx, batch_capacity)
        #x_IDs = dp.gen_line2IDs(x, self.src_vocab)
        #y_IDs = dp.gen_line2IDs(ctx, self.vocab)
        #batches = [(([x_ID], [len(x_ID)]), ([y_ID], [len(y_ID)])) for x_ID, y_ID in zip(x_IDs, y_IDs)]

        if length_penalty is None: length_penalty = self.params['test']['decode_config']['length_penalty_a']

        if option == 0:
            op = self.op_fusion_decode
        elif option == 1:
            op = self.op_fusion_decode2
        elif option == 2:
            op = self.op_fusion_decode3
        elif option == 3:
            op = self.op_fusion_decode4
        elif option == 10:
            op = self.op_fusion_decode10
        elif option == 11:
            op = self.op_fusion_decode11
        else:
            raise ValueError

        candidates, scores = self.execute_op(op, batches,
            {
                self.ph_beam_size: beam_size,
                self.ph_length_penalty: length_penalty,
                self.ph_init_y_last_ctok: init_y_last_ctok,
                self.ph_delay: pmi_delay})

        if ret_search_detail:
            # flatten 
            candidates = sum(candidates, []) # [nsamples*beam_size, length(variable)]
            # convert to string
            candidates = self.IDs2text(candidates) #[nsamples*beam_size]
            # restore shape
            candidates = [candidates[i:i + beam_size] for i in range(0, len(candidates), beam_size)]

            return candidates, scores
        else:
            # take top 1
            candidates = [beam[0] for beam in candidates] # [nsamples, length(variable)]
            # convert to string
            candidates = self.IDs2text(candidates) #[nsamples]

            return candidates


    def decode(self, x, x_len, ctx, ctx_len, beam_size, option=0, config=None):
        """
        Notes:
            Sequences in `y` must have EOS at the end and must not have SOS.
            """
        cache = {}

        batch_size = tf.shape(x)[0]
        init_seq = tf.fill([batch_size, 1], self.tm.params['vocab']['SOS_ID'])

        #### TM initialization
        cache['TM'] = self.tm.make_cache(x, x_len, training=False, layer_cache=True)

        #### contextual LM initialization
        # Add sos and remove EOS. [batch, length] -> [batch, length]
        init_ctx = tf.concat([init_seq, ctx], axis=1)[:, :-1]
        init_ctx, offsets = align_to_right(init_ctx, ctx_len)
        
        cache['ctxLM'] = self.lm.make_cache(batch_size, layer_cache=True, offsets=offsets)

        # Cache the context sequence
        cache['init_ctx'] = init_ctx

        #### LM initialization
        cache['init_y'] = tf.cond(
            self.ph_init_y_last_ctok,
            lambda: tf.concat([init_seq, init_ctx[:, -1:]], axis=1),
            lambda: tf.tile(init_seq, [1, 2]))
        init_y_offsets = tf.where(
            tf.equal(cache['init_y'][:, -1], init_seq[:, -1]),
            tf.ones([batch_size], dtype=tf.int32),
            tf.zeros([batch_size], dtype=tf.int32)) 
        cache['LM'] = self.lm.make_cache(batch_size, layer_cache=True, offsets=init_y_offsets)
        

        # Maximum target length
        maxlens = tf.minimum(
            tf.minimum(
                self.params['network']['max_length'] - 10,
                x_len * 3 + 10),
            self.lm.params['network']['max_length'] - tf.shape(init_ctx)[1] - 10)
        

        __get_logits_fn = lambda decI, cache: self.__get_logits_fn(decI, cache, option, config, option >= 10)

        # Execute
        if option < 10:
            hypos, scores = beam_search_decode_V2(
                __get_logits_fn,
                cache,
                init_seq,
                beam_size,
                maxlens,
                self.vocab.EOS_ID,
                self.vocab.PAD_ID,
                params={'length_penalty_a': self.ph_length_penalty},
                normalize_logits=False)
        elif option == 10:
            hypos, scores = fusion_beam_search(__get_logits_fn,
                cache,
                init_seq,
                beam_size,
                maxlens,
                self.vocab.EOS_ID,
                self.vocab.PAD_ID,
                params={'length_penalty_a': self.ph_length_penalty})
        elif option == 11:
            hypos, scores = fusion_beam_search_delayed_pmi(__get_logits_fn,
                cache,
                init_seq,
                beam_size,
                self.ph_delay,
                maxlens,
                self.vocab.EOS_ID,
                self.vocab.PAD_ID,
                params={'length_penalty_a': self.ph_length_penalty})
        else:
            raise
        
        return hypos, scores


    def __get_logits_fn(self, dec_inputs, cache, option=0, config=None, ret_triple=False):
        first_pos = tf.equal(0, self.tm.decoder.get_layer_cache_length(cache['TM']))
        ctx_inputs, y_inputs = tf.cond(
            first_pos,
            lambda: (cache['init_ctx'], cache['init_y']),
            lambda: (dec_inputs, dec_inputs))
        # Shape of dec_inputs: [batch * beam, 1]
        # Logits [batch * beam, 1, vocab]
        pTM = tf.math.log_softmax(self.tm.get_logits_w_cache(dec_inputs, cache['TM']), axis=-1)
        pLM = tf.math.log_softmax(self.lm.get_logits_w_cache(y_inputs, cache['LM'])[:, -1:], axis=-1)
        pCTXLM = tf.math.log_softmax(self.lm.get_logits_w_cache(ctx_inputs, cache['ctxLM'])[:, -1:], axis=-1)

        if ret_triple:
            return pTM, pLM, pCTXLM

        # Fusion
        if option == 0:
            # Normal fusion by adding logits
            p_fusion = pTM + pCTXLM - pLM
        elif option == 1:
            # Only keep top k=beam_size which are suggested by translation model. PMI for EOS is always 0.
            PMI = pCTXLM - pLM
            p_fusion = pTM + PMI
            max_mt = tf.math.reduce_max(pTM, axis=-1, keepdims=True)
            threshold = max_mt - np.log(100)
            bias = 1e9 * tf.minimum(tf.sign(pTM - threshold), 0)
            p_fusion += bias
        elif option == 2:
            # PMI as a score. Only keep ones suggested by TM
            TM_top_logits, TM_top_inds = tf.math.top_k(pTM, config['beam_size'], True)
            bias = 1e9 * tf.minimum(tf.sign(pTM - TM_top_logits[:, :, config['beam_size'] - 1:config['beam_size']]), 0)
            p_fusion = pCTXLM - pLM + bias
        elif option == 3:
            p_fusion = pTM + (pCTXLM - pLM) * config['lambda']
            TM_top_logits, TM_top_inds = tf.math.top_k(pTM, config['beam_size'], True)
            bias = 1e9 * tf.minimum(tf.sign(pTM - TM_top_logits[:, :, config['beam_size'] - 1:config['beam_size']]), 0)
            p_fusion += bias


        return p_fusion

    
    def conditional_y_logits(self, y, y_len, c, c_len):
        batch_size = tf.shape(y)[0]
        # Shift c
        c = tf.concat([tf.fill([batch_size, 1], self.tm.params['vocab']['SOS_ID']), c[:, :-1]], axis=1)
        c, offsets = align_to_right(c, c_len)
        # Join c and y and remove the last token
        cy = tf.concat([c, y], axis=1)[:, :-1]
        cy_len = c_len + y_len - 1

        return self.lm.get_logits(cy, shift_dec_inputs=False, offsets=offsets
            )[:, tf.shape(c)[1] - 1:]

    def pmi_logits(self, y, y_len, c, c_len):
        # top pmis [batch, y_len, vocab]
        py_logits = self.lm.get_logits(y)
        cond_y_logits = self.conditional_y_logits(y, y_len, c, c_len)
        pmi_logits = tf.math.log_softmax(cond_y_logits) - tf.math.log_softmax(py_logits)

        return pmi_logits


    def fn_cumulative_score(self, inputs, mode=0, config=None):
        (x, x_len), (y, y_len), (c, c_len) = inputs
        pmi_logits = self.pmi_logits(y, y_len, c, c_len)
        mt_logits = tf.math.log_softmax(self.tm.get_logits(x, y, x_len, y_len))
        score_logits = pmi_logits + mt_logits

        if mode == 1:
            mt_top_logits, top_inds = tf.math.top_k(mt_logits, config['beam_size'], True)
            bias = 1e9 * tf.minimum(tf.sign(mt_logits - mt_top_logits[:, :, config['beam_size'] - 1:config['beam_size']]), 0)
            score_logits += bias

        score = tf.batch_gather(score_logits, y[:, :, None])[:, :, 0]
        cum_score = tf.math.cumsum(score, axis=1)

        return cum_score, score


    def fn_cumulative_pmi(self, inputs):
        (y, y_len), (c, c_len) = inputs
        y_logits = self.lm.get_logits(y)
        cond_y_logits = self.conditional_y_logits(y, y_len, c, c_len)

        norm_y = tf.batch_gather(tf.math.log_softmax(y_logits), y[:,:,None])[:,:,0]
        norm_cond_y = tf.batch_gather(tf.math.log_softmax(cond_y_logits), y[:,:,None])[:,:,0]

        cum_y = tf.math.cumsum(norm_y, axis=1)
        cum_cond_y = tf.math.cumsum(norm_cond_y, axis=1)
        pmi = norm_cond_y - norm_y
        cum_pmi = cum_cond_y - cum_y
        return cum_pmi, pmi, cum_y, norm_y, cum_cond_y, norm_cond_y


    def fn_top_pmi(self, inputs):
        (x, x_len), (y, y_len), (c, c_len) = inputs

        pmi_logits = self.pmi_logits(y, y_len, c, c_len)
        pmi, pmi_inds = tf.math.top_k(pmi_logits, self.ph_beam_size)

        # top log p(y|x)
        pxy_logits = tf.math.log_softmax(self.tm.get_logits(x, y, x_len, y_len))
        logpyx, pyx_inds = tf.math.top_k(pxy_logits, self.ph_beam_size, True)

        # Score
        fusion_logits = pmi_logits + pxy_logits
        fusion_score, fusion_inds = tf.math.top_k(fusion_logits, self.ph_beam_size)

        # trans pmi
        trans_pmi_logits = pxy_logits - tf.math.log_softmax(self.lm.get_logits(y))
        trans_pmi, trans_pmi_inds = tf.math.top_k(trans_pmi_logits, self.ph_beam_size)

        # pmi for top logp(y|x)
        top_pyx_pmi = tf.batch_gather(pmi_logits, pyx_inds)

        # score for top logp(y|x)
        top_pyx_score = tf.batch_gather(fusion_logits, pyx_inds)

        # trans pmi for top logp(y|x)
        top_pyx_trans_pmi = tf.batch_gather(trans_pmi_logits, pyx_inds)


        return pmi, pmi_inds, logpyx, pyx_inds, fusion_score, fusion_inds, trans_pmi, trans_pmi_inds, top_pyx_pmi, top_pyx_score, top_pyx_trans_pmi


    def top_pmi_analysis(self, x, y, c, k=8):
        x = dp.gen_line2IDs(x)
        y,c = (dp.gen_line2IDs(_, self.vocab) for _ in (y, c))
        batches = list(dp.gen_multi_padded_batch(
            zip(x, y, c), self.batch_capacity // 128, self.vocab.PAD_ID))
        return self.execute_op(self.op_top_pmi, batches, {self.ph_beam_size: k})


    def cumulative_score(self, x, y, c, option=0, beam_size=1):
        x = dp.gen_line2IDs(x, self.src_vocab)
        y,c = (dp.gen_line2IDs(_, self.vocab) for _ in (y, c))
        batches = list(dp.gen_multi_padded_batch(
            zip(x, y, c), self.batch_capacity // 128, self.vocab.PAD_ID))

        if option == 0:
            op = self.op_cumu_score
        elif option == 1:
            op = self.op_cumu_score2
        return self.execute_op(op, batches, {self.ph_beam_size: beam_size})

    
    def cumulative_pmi(self, y, c):
        c = dp.gen_line2IDs(c, self.vocab)
        y = dp.gen_line2IDs(y, self.vocab)
        batches = list(dp.gen_dual_const_capacity_batch(zip(y,c), self.batch_capacity, self.vocab.PAD_ID))
        return self.execute_op(self.op_cumu_pmi, batches)

    
    def calculate_trans_score_w_context(self, x, y, c):
        pmi = self.lmi.calculate_pmi(c, y)
        logp = self.calculate_translation_score(x, y, length_penalty_a=0)
        return np.array(pmi) + np.array(logp)


def print_score(args):
    decode_config = json.loads(args.decode_config) if args.decode_config is not None else None
    fdec = PMIFusionDecoder(
        args.lm_dir, args.tm_dir,
        lm_checkpoint=args.lm_checkpoint,
        n_gpus=args.n_gpus, batch_capacity=args.batch_capacity,
        decode_config=decode_config)
    fdec.make_session()

    if args.type == 'fusion':
        x, y, c = (list(_) for _ in zip(*(line.split(args.delimiter) for line in sys.stdin)))
        for s in fdec.calculate_trans_score_w_context(x, y, c):
            print(s)
    else:
        raise


def gen_docs(line_iter):
    doc = []
    for line in line_iter:
        line = line.strip()
        if len(line) == 0:
            assert len(doc) > 0
            yield doc
            doc = []
        else:
            doc.append(line)

    if len(doc) > 0:
        yield doc


def translate_documents_rerank(args):
    decode_config = json.loads(args.decode_config) if args.decode_config is not None else None
    if args.max_context_tokens is None and args.max_context_sents is None:
        args.max_context_sents = 1
    assert (args.max_context_sents is None) or (args.max_context_tokens is None)
    
    input_sents = sys.stdin.readlines()
    docs = list(gen_docs(input_sents))
    logger.info('#docs: {}, #sents: {}'.format(len(docs), len(input_sents) - len(docs)))

    # Load translator
    fdec = PMIFusionDecoder(
        args.lm_dir, args.tm_dir,
        lm_checkpoint=args.lm_checkpoint,
        n_gpus=args.n_gpus, batch_capacity=args.batch_capacity,
        decode_config=decode_config)
    fdec.make_session()

    # Create N-best lists
    logger.debug('Creating n-best lists.')
    hypo_docs = []
    for doc in docs:
        hypos, scores = fdec.translate_sentences(doc,
            beam_size=args.beam_size,
            return_search_results=True)
        hypo_docs.append(hypos)

    # Rerank
    logger.debug('Reranking.')
    trans_docs = []
    for doc, hypos in zip(docs, hypo_docs):
        ctx_q = deque()
        trans = []
        for x, nbest in zip(doc, hypos):
            ctx_tiled = [' '.join(ctx_q)] * args.beam_size
            x_tiled = [x] * args.beam_size

            if args.method == 'pmi_rerank':
                scores = fdec.lmi.calculate_pmi(ctx_tiled, nbest)
            elif args.method == 'fusion_rerank':
                scores = fdec.calculate_trans_score_w_context(x_tiled, nbest, ctx_tiled)
            else:
                raise

            out = nbest[np.argmax(scores)]
            trans.append(out)

            # Update context queue
            if args.max_context_sents is not None:
                ctx_q.append(out)
                if len(ctx_q) > args.max_context_sents:
                    ctx_q.popleft()
            else:
                for tok in out.split():
                    ctx_q.append(tok)
                while len(ctx_q) > args.max_context_tokens:
                    ctx_q.popleft()

        trans_docs.append(trans)

    # Output
    for doc in trans_docs:
        for line in doc:
            print(line)
        print('')


def translate_documents(args):
    decode_config = json.loads(args.decode_config) if args.decode_config is not None else None

    if args.max_context_tokens is None and args.max_context_sents is None:
        args.max_context_sents = 1
    assert (args.max_context_sents is None) or (args.max_context_tokens is None)
    
    input_sents = sys.stdin.readlines()
    docs = list(gen_docs(input_sents))
    logger.info('#docs: {}, #sents: {}'.format(len(docs), len(input_sents) - len(docs)))

    # time_aligned [max_doc_len , ndocs(var), 2] (conteporaries)
    max_doc_len = max(len(doc) for doc in docs)
    time_aligned = [[(doc[i], doci) for doci, doc in enumerate(docs) if len(doc) > i] for i in range(max_doc_len)]
    
    # [ndocs, n_past_sents(at most max_context_sents)]
    contexts = [deque() for i in range(len(docs))]

    translations = [[] for i in range(len(docs))]

    # Load translator
    fdec = PMIFusionDecoder(
        args.lm_dir, args.tm_dir,
        lm_checkpoint=args.lm_checkpoint,
        n_gpus=args.n_gpus, batch_capacity=args.batch_capacity,
        decode_config=decode_config)
    fdec.make_session()

    # Translate
    _count = 0
    _start_t = time.time()
    for a in time_aligned:
        lines, doc_inds = zip(*a)
        ctx = [' '.join(contexts[i]) for i in doc_inds]

        outs = fdec.fusion_decode(
            lines, ctx, beam_size=args.beam_size, option=args.mode,
            init_y_last_ctok=not args.init_y_sos,
            pmi_delay=args.pmi_delay)


        _count += len(outs)
        logger.debug('Translated {}/{}. {} s/sentence'.format(
            _count, len(input_sents), (time.time() - _start_t)/_count))

        for i, out in zip(doc_inds, outs):
            # Store translations
            translations[i].append(out)

            # Update context queue
            if args.max_context_sents is not None:
                contexts[i].append(out)
                if len(contexts[i]) > args.max_context_sents:
                    contexts[i].popleft()
            else:
                for tok in out.split():
                    contexts[i].append(tok)
                while len(contexts[i]) > args.max_context_tokens:
                    contexts[i].popleft()


    # Output
    for doc in translations:
        for line in doc:
            print(line)
        print('')


def translate_oracle_context(args):
    decode_config = json.loads(args.decode_config) if args.decode_config is not None else None

    x, c = zip(*(line.split(args.delimiter) for line in sys.stdin))
    logger.info('Number of sents.: {}'.format(len(x)))
    logger.debug('[2nd src] {}\n[2nd ctx] {}'.format(x[1], c[1]))

    fdec = PMIFusionDecoder(
        args.lm_dir, args.tm_dir,
        lm_checkpoint=args.lm_checkpoint,
        n_gpus=args.n_gpus,
        batch_capacity=args.batch_capacity, decode_config=decode_config)
    fdec.make_session()
    
    if args.method == 'beam_search':
        trans = fdec.fusion_decode(x, c, args.beam_size, option=args.mode, init_y_last_ctok=not args.init_y_sos, pmi_delay=args.pmi_delay)
    elif args.method == 'pmi_rerank' or args.method == 'fusion_rerank' or args.method == 'normal_rerank':
        hypos, scores = fdec.translate_sentences(x, args.beam_size, return_search_results=True)
        hypos = sum(hypos, [])

        # Tile c and x
        c = list(sum(zip(*([c] * args.beam_size)), tuple()))
        x = list(sum(zip(*([x] * args.beam_size)), tuple()))

        if args.method == 'pmi_rerank':
            scores = fdec.lmi.calculate_pmi(c, hypos)
        elif args.method == 'fusion_rerank':
            scores = fdec.calculate_trans_score_w_context(x, hypos, c)
        elif args.method == 'normal_rerank':
            scores = fdec.calculate_translation_score(x, hypos)
        trans = []
        for i in range(0, len(hypos), args.beam_size):
            _hypos = hypos[i: i+args.beam_size]
            _scores = scores[i: i+args.beam_size]
            trans.append(_hypos[np.argmax(_scores)])
    else:
        raise

    for line in trans:
        print(line)


def translate(args):
    logger.info('Arguments: {}'.format(str(args)))
    if args.oracle:
        translate_oracle_context(args)
        
    else:
        if args.method == 'beam_search':
            translate_documents(args)
        elif args.method == 'fusion_rerank' or args.method == 'pmi_rerank':
            translate_documents_rerank(args)


def main():
    parser = argparse.ArgumentParser()
    logLvDic = {'info': INFO, 'debug':DEBUG}
    parser.add_argument('--log_level', choices=list(logLvDic.keys()), default="info")
    parser.add_argument('--tm_dir', '-tm', '-t', type=str, required=True)
    parser.add_argument('--lm_dir', '-lm', '-l', type=str, required=True)
    parser.add_argument('--lm_checkpoint', default=None, type=str)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--batch_capacity', '--capacity', type=int, default=None)

    subparsers = parser.add_subparsers()

    # Translation parser
    trans = subparsers.add_parser('trans')
    # Translation with the oracle context or not
    trans.add_argument('--oracle', action='store_true')
    trans.add_argument('--delimiter', '--delim', '-delim', type=str, default='\t')
    trans.add_argument('--max_context_sents', '-sents', type=int, default=None, help='Num. of ctx sents.')
    trans.add_argument('--max_context_tokens', '-tokens', type=int, default=None, help='Num. of ctx sents.')
    methods = ['beam_search', 'pmi_rerank', 'fusion_rerank', 'normal_rerank']
    trans.add_argument('--method', type=str, default='beam_search', choices=methods)
    trans.add_argument('--beam_size', type=int, default=8)
    trans.add_argument('--mode', type=int, default=1)
    trans.add_argument('--init_y_sos', action='store_true')
    trans.add_argument('--pmi_delay', type=int, default=4)
    trans.add_argument('--decode_config', type=str, default=None)
    trans.set_defaults(handler=translate)

    # Score calculation parser
    calc = subparsers.add_parser('calc')
    calc.set_defaults(handler=print_score)
    calc.add_argument('--delimiter', '--delim', '-delim', type=str, default='\t')
    calc.add_argument('--type', default='fusion')
    calc.add_argument('--decode_config', type=str, default=None)


    args = parser.parse_args()

    basicConfig(level=logLvDic[args.log_level])

    args.handler(args)



if __name__ == '__main__':
    main()
