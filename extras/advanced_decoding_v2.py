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
from .tm_lm_fusion_beam_search_v2 import fusion_beam_search_delayed_pmi_v2


class PMIFusionDecoder(Inference):
    def __init__(self, lm_dir, *args, lm_checkpoint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tm = self.model

        # Creating language model
        self.lmi = LMInference(lm_dir, graph=self.graph, n_gpus=self.n_gpus, n_cpu_cores=self.n_cpu_cores, batch_capacity=self.batch_capacity//2, checkpoint=lm_checkpoint)
        self.lm = self.lmi.model


    def make_session(self, *args, **kwargs):
        super().make_session(*args, **kwargs)
        self.lmi.make_session(self.session, load_checkpoint=True)


    def fn_fusion_decode(self, inputs, null_ctx, tm_eos, lm_eos, beam_size, delay):
        (x, x_len), (c, c_len) = inputs
        hypos, scores = self.decode_fn(x, x_len, c, c_len, null_ctx, tm_eos, lm_eos, beam_size, delay)
        return hypos, scores

    
    def fusion_decode(self, x, ctx, null_ctx, tm_eos, lm_eos, beam_size, ret_search_detail=False, delay=0):

        if not hasattr(self, 'op_fusion_beam_search'):
            with self.graph.as_default():
                param_phs = {
                    'null_ctx': tf.placeholder(tf.int32, [None]),
                    'tm_eos': tf.placeholder(tf.int32, []),
                    'lm_eos': tf.placeholder(tf.int32, []),
                    'beam_size': tf.placeholder(tf.int32, []),
                    'delay': tf.placeholder(tf.int32, [])
                }
                self.op_fusion_beam_search = self.make_op(
                    self.fn_fusion_decode,
                    (
                        (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None])),
                        (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None]))
                    ),
                    **param_phs,
                    param_phs = param_phs)

        batch_capacity = self.batch_capacity // beam_size
        batches = self.make_batches(x, ctx, batch_capacity)
        candidates, scores = self.execute_op(
            self.op_fusion_beam_search,
            batches,
            null_ctx = self.vocab.line2IDs(null_ctx),
            tm_eos = self.vocab.tok2ID[tm_eos] if type(tm_eos) == str else tm_eos,
            lm_eos = self.vocab.tok2ID[lm_eos] if type(lm_eos) == str else lm_eos,
            beam_size = beam_size,
            delay = delay)

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


    def decode_fn(self, x, x_len, ctx, ctx_len, null_ctx, tm_stop_token, lm_stop_token, beam_size, delay):
        """
            x: '<s> source sentence </s>' * batch_size
            ctx: '<s> context sentence <sep>' * batch_size
            null_ctx:  '<s> <sep>' (shape [None])
            """
        cache = {}
        static_cache = {}

        batch_size = tf.shape(x)[0]

        # Start token
        header = self.tm.params['inference']['header']
        if type(header) == str:
            header = self.vocab.tok2ID[header]
        init_seq = tf.fill([batch_size, 1], header)

        #### TM initialization
        cache['TM'] = self.tm.make_cache(x, x_len, training=False, layer_cache=True)

        #### contextual LM initialization
        init_ctx, offsets = align_to_right(ctx, ctx_len)
        
        cache['ctxLM'] = self.lm.make_cache(batch_size, layer_cache=True, offsets=offsets)

        # Cache the context sequence
        static_cache['init_ctx'] = init_ctx

        #### LM initialization
        static_cache['null_ctx'] = tf.tile(null_ctx[None], [batch_size, 1])
        cache['LM'] = self.lm.make_cache(batch_size, layer_cache=True)
        
        # Maximum target length
        maxlens = tf.minimum(
            tf.minimum(
                self.params['network']['max_length'] - 10,
                x_len * 3 + 10),
            self.lm.params['network']['max_length'] - tf.shape(init_ctx)[1] - 10)
        
        # PMI queue for delayed fusion beam search
        # [batch, delay, vocab].(replicated along axis=0 in the beam search function)
        # Initial value is 0 (flat pmi distribution)
        cache['pmi_q'] = tf.zeros(
            [batch_size, delay, self.params['vocab']['vocab_size']],
            dtype = tf.float32)

        def __get_logits_fn(*args, **kwargs):
            return self.__get_logits_fn(*args, **kwargs, tm_eos=tm_stop_token, lm_eos=lm_stop_token)

        # Execute
        hypos, scores = fusion_beam_search_delayed_pmi_v2(
            __get_logits_fn,
            cache,
            static_cache,
            init_seq,
            beam_size,
            maxlens,
            eos_id = tm_stop_token,
            pad_id = self.vocab.PAD_ID,
            normalize_logits = False)
        
        return hypos, scores


    def __get_logits_fn(self, dec_inputs, cache, static_cache, _i, tm_eos, lm_eos):
        NEGINF = -1e10
        first_pos = tf.equal(0, _i)
        ctx_inputs, y_inputs = tf.cond(
            first_pos,
            lambda: (static_cache['init_ctx'], static_cache['null_ctx']),
            lambda: (dec_inputs, dec_inputs))

        # Shape of dec_inputs: [batch * beam, 1]
        # Logits [batch * beam, 1, vocab]
        pTM = tf.math.log_softmax(self.tm.get_logits_w_cache(dec_inputs, cache['TM']), axis=-1)
        pLM = tf.math.log_softmax(self.lm.get_logits_w_cache(y_inputs, cache['LM'])[:, -1:], axis=-1)
        pCTXLM = tf.math.log_softmax(self.lm.get_logits_w_cache(ctx_inputs, cache['ctxLM'])[:, -1:], axis=-1)

        # PMI
        pmi = pCTXLM - pLM

        # Move lm_eos to tm_eos and set NEGINF to lm_eos
        batch_size = tf.shape(pmi)[0]
        eos_diff = pmi[:, 0, lm_eos] - pmi[:, 0, tm_eos]
        # [batch_size, 2]
        updates = tf.concat([eos_diff[:, None], tf.fill([batch_size, 1], NEGINF)], axis=-1)
        indices = tf.concat([
            tf.tile(tf.range(batch_size)[:, None], [1, 2])[:, :, None],
            tf.zeros([batch_size, 2, 1], tf.int32),
            tf.tile([[[tm_eos], [lm_eos]]], [batch_size, 1, 1])
        ], axis=-1)
        pmi = pmi + tf.scatter_nd(indices, updates, tf.shape(pmi))

        # push and pop
        #pmi_q = tf.concat([cache['pmi_q'], pmi], axis=1)
        #pmi, cache['pmi_q'] = pmi_q[:, :1], pmi_q[:, 1:]

        p_fusion = pTM + pmi
        return p_fusion


    def conditional_y_logits(self, y, y_len, c, c_len):
        c, offsets = align_to_right(c, c_len)
        cy = tf.concat([c, y], axis=1)
        cy_len = c_len + y_len

        return self.lm.get_logits(cy, offsets=offsets)[:, tf.shape(c)[1] - 1:]


    def fn_cumulative_scores(self, inputs, null_c, sos, tm_eos, lm_eos):
        """
        Usual values:
            null_c: [(id of sos), (id of sep)]
            sos: (id of sos)
            tm_eos: (id of eos)
            lm_eos: (id of sep)
            """
        # x: wrapped, y: naked
        (x, x_len), (y, y_len), (c, c_len) = inputs
        b = tf.shape(x)[0]

        out_y_len = y_len + 1
        out_mask = tf.sequence_mask(out_y_len, tf.shape(y)[1] + 1, dtype=tf.float32)

        tm_y = tf.concat([tf.fill([b, 1], sos), y, tf.zeros([b, 1], tf.int32)], axis=1)
        tm_y += tf.scatter_nd(
            tf.concat([tf.range(b)[:,None], (y_len + 1)[:, None]], axis=1),
            tf.fill([b], tm_eos),
            tf.shape(tm_y)
        )

        null_c = tf.tile(null_c[None], [b, 1])
        null_c_len = tf.tile(tf.shape(null_c)[1][None], [b])
        lm_y = tf.concat([y, tf.zeros([b,1],tf.int32)], axis=1)
        lm_y += tf.scatter_nd(
            tf.concat([tf.range(b)[:,None], y_len[:,None]], axis=1),
            tf.fill([b], lm_eos),
            tf.shape(lm_y)
        )

        tm_in, tm_out = tm_y[:, :-1], tm_y[:, 1:]
        tm_logp_y = tf.math.log_softmax(self.tm.get_logits(x, tm_in, x_len, out_y_len))
        tm_logp_y = tf.batch_gather(tm_logp_y, tm_out[:,:,None])[:,:,0] * out_mask
        
        lm_in, lm_out = lm_y[:, :-1], lm_y
        lm_logp = tf.math.log_softmax(self.conditional_y_logits(
            lm_in, out_y_len - 1, null_c, null_c_len))
        lm_cond_logp = tf.math.log_softmax(self.conditional_y_logits(
            lm_in, out_y_len - 1, c, c_len))
        pmi = tf.batch_gather(lm_cond_logp - lm_logp, lm_out[:,:,None])[:,:,0] * out_mask

        fusion = tm_logp_y + pmi

        return (
            tf.cumsum(fusion, axis=1),
            fusion,
            tf.cumsum(tm_logp_y, axis=1),
            tm_logp_y,
            tf.cumsum(pmi, axis=1),
            pmi,
            tm_in, tm_out, lm_in, lm_out
        )


    def cumulative_scores(self, x, y, c, null_c, sos, tm_eos, lm_eos):
        if not hasattr(self, 'op_cumu_scores'):
            param_phs = {
                'null_c': tf.placeholder(tf.int32, [None]),
                'sos': tf.placeholder(tf.int32, []),
                'tm_eos': tf.placeholder(tf.int32, []),
                'lm_eos': tf.placeholder(tf.int32, [])
            }
            self.op_cumu_scores = self.make_op(
                self.fn_cumulative_scores,
                (
                     (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None])),
                     (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None])),
                     (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None]))
                ), **param_phs,
                param_phs=param_phs)
        
        sos, tm_eos, lm_eos = (self.vocab.tok2ID[x] if type(x)==str else x
            for x in [sos, tm_eos, lm_eos])
        null_c = self.vocab.line2IDs(null_c)

        x_data = dp.gen_line2IDs(x, self.src_vocab)
        y_data = dp.gen_line2IDs(y, self.vocab)
        c_data = dp.gen_line2IDs(c, self.vocab)
        data = zip(x_data, y_data, c_data)
        data = dp.gen_const_capacity_batch_multi_seq(data, self.batch_capacity)
        return self.execute_op(self.op_cumu_scores,
            list(data),
            null_c = null_c,
            sos = sos,
            tm_eos = tm_eos,
            lm_eos = lm_eos)
    
    def cumulative_pmi(self, y, c):
        c = dp.gen_line2IDs(c, self.vocab)
        y = dp.gen_line2IDs(y, self.vocab)
        batches = list(dp.gen_dual_const_capacity_batch(zip(y,c), self.batch_capacity, self.vocab.PAD_ID))
        return self.execute_op(self.op_cumu_pmi, batches)

    
    def calculate_trans_score_w_context(self, x, y, c, null_c):
        pmi = self.lmi.calculate_pmi_v2(c, null_c, y)
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
