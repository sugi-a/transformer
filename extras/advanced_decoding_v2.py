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


    def fn_fusion_decode(self, inputs, null_ctx, tm_eos, lm_eos, beam_size, delay, length_penalty_a):
        (x, x_len), (c, c_len) = inputs
        hypos, scores = self.decode_fn(x, x_len, c, c_len, null_ctx, tm_eos, lm_eos, beam_size, delay, length_penalty_a)
        return hypos, scores

    
    def fusion_decode(self, x, ctx, null_ctx, tm_eos, lm_eos, beam_size, ret_search_detail=False, delay=0, length_penalty_a=0):

        if not hasattr(self, 'op_fusion_beam_search'):
            with self.graph.as_default():
                param_phs = {
                    'null_ctx': tf.placeholder(tf.int32, [None]),
                    'tm_eos': tf.placeholder(tf.int32, []),
                    'lm_eos': tf.placeholder(tf.int32, []),
                    'beam_size': tf.placeholder(tf.int32, []),
                    'delay': tf.placeholder(tf.int32, []),
                    'length_penalty_a': tf.placeholder(tf.float64, [])
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
            delay = delay,
            length_penalty_a = length_penalty_a)

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


    def decode_fn(self, x, x_len, ctx, ctx_len, null_ctx, tm_stop_token, lm_stop_token, beam_size, delay, length_penalty_a):
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
            [batch_size, tf.maximum(delay - 1, 0)],
            dtype = tf.float32)

        def __get_logits_fn(*args, **kwargs):
            return self.__get_logits_fn_delay(*args, **kwargs, delay=delay, tm_eos=tm_stop_token, lm_eos=lm_stop_token)

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
            length_penalty_a=length_penalty_a,
            normalize_logits = False)
        
        return hypos, scores


    def __get_logits_fn_delay(self, dec_inputs, cache, static_cache, _i, prev_pmi_chosen, delay, tm_eos, lm_eos):
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

        # prev_chosen_pmi: [batch*beam]
        # push [batch*beam, delay]
        pmi_q = tf.concat([cache['pmi_q'], prev_pmi_chosen[:, None]], axis=1)
        # pop [batch*beam], [batch*beam, delay - 1]
        popped, cache['pmi_q'] = pmi_q[:, 0], pmi_q[:, 1:]

        # [batch*beam, 1, 1]
        pmi_to_add = tf.cond(tf.equal(delay, 0), lambda: pmi, lambda: popped[:, None, None])

        p_fusion = pTM + pmi_to_add

        return p_fusion, pmi[:, 0]


    def replace_logits(logits, index, values):
        """
        logits: [batch, length, vocab]
        index: 0 <= index < vocab
        values: [1 or batch, 1 or length]"""
        
        values = tf.broadcast_to(values, tf.shape(logits))
        return tf.concat([logits[:, :, :index], values, logits[:, :, index+1:]], axis=2)



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
            pmi, x, y, c, null_c
        )


    def cumulative_scores(self, x, y, c, null_c, sos, tm_eos, lm_eos, ret_input_info=False):
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
        ret = self.execute_op(self.op_cumu_scores,
            list(data),
            null_c = null_c,
            sos = sos,
            tm_eos = tm_eos,
            lm_eos = lm_eos)
        
        if ret_input_info:
            return ret
        else:
            return ret[:-4]
    
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


def fusion_rerank(docs, fdec, tm_sos, tm_eos, lm_sos, lm_sep, beam_size=1, length_penalty_a=0, context_limit=1, method='fusion_rerank'):
    nsents = len(sum(docs, []))
    logger.info('#docs: {}, #sents: {}'.format(len(docs), nsents))

    # Create N-best lists
    logger.debug('Creating n-best lists.')

    doc_offset = [0] + list(np.cumsum([len(v) for v in docs]))
    flatten = sum(docs, [])
    hypos, scores = fdec.translate_sentences(
        flatten,
        beam_size=beam_size,
        length_penalty_a=length_penalty_a,
        return_search_results=True)

    # [doc, n_sents, hypos]
    hypo_docs = [hypos[doc_offset[i]: doc_offset[i+1]] for i in range(len(docs))]

    # transpose [sentId, docId, hypos]
    maxsents = max(map(len, hypo_docs))
    transposed = [[doc[i] for doc in hypo_docs if len(doc) > i] for i in range(maxsents)]
    docIds = [[j for j, doc in enumerate(hypo_docs) if len(doc) > i] for i in range(maxsents)]

    # Rerank
    logger.debug('Reranking.')

    # Context queue
    ctx_qs = [deque() for i in range(len(hypo_docs))]

    # translations
    trans_docs = [[] for i in range(len(docs))]

    for sentId, (sents, ids) in enumerate(zip(transposed, docIds)):
        # sents: [docIndex, hypos]
        if sentId == 0:
            out = [v[0] for v in sents]
        else:
            # [docIndex, hypos]
            ctx_tiled = [[' {} '.format(lm_sep).join(ctx_qs[i])] * beam_size for i in ids]
            x_tiled = [[docs[i][sentId]] * beam_size for i in ids]
            n_best_wrapped_lm = [['{} {}'.format(line, lm_sep) for line in nbest] for nbest in sents]
            n_best_wrapped_tm = [['{} {} {}'.format(tm_sos, line, tm_eos) for line in nbest] for nbest in sents]

            if method == 'fusion_rerank':
                scores = np.array(fdec.lmi.calculate_pmi(
                    sum(ctx_tiled, []),
                    sum(n_best_wrapped_lm, []),
                    sep=lm_sep,
                    head=lm_sos))
                scores += np.array(fdec.calculate_translation_score(
                    sum(x_tiled, []),
                    sum(n_best_wrapped_tm, []),
                    length_penalty_a=length_penalty_a))
                scores = scores.reshape([-1, beam_size])
                top1Indices = np.argmax(scores, axis=1)
                out = [v[i] for v,i in zip(sents, top1Indices)]
        for i, o in zip(ids, out):
            trans_docs[i].append(o)

            ctx_qs[i].append(o)
            if len(ctx_qs[i]) > context_limit:
                ctx_qs[i].popleft()

    return trans_docs


def successively_translate(sents, fdec, tm_sos, tm_eos, lm_sos, lm_sep, beam_size=1, delay=0, length_penalty_a=0, context_limit_type='sentence', context_limit=1, trans_log=None):
    docs = list(gen_docs(sents))
    logger.info('#docs: {}, #sents: {}'.format(len(docs), len(sents) - len(docs)))

    # time_aligned [max_doc_len , ndocs(var), 2] (conteporaries)
    max_doc_len = max(len(doc) for doc in docs)
    time_aligned = [[(doc[i], doci) for doci, doc in enumerate(docs) if len(doc) > i] for i in range(max_doc_len)]
    
    # [ndocs, n_past_sents(at most max_context_sents)]
    contexts = [deque() for i in range(len(docs))]
    translations = [[] for i in range(len(docs))]

    null_ctx = lm_sos + ' ' + lm_sep
    # Translate
    _count = 0
    _start_t = time.time()
    for a in time_aligned:
        lines, doc_inds = zip(*a)
        ctx = [lm_sos + ' ' + ' '.join(contexts[i])  + ' ' + lm_sep for i in doc_inds]

        outs = fdec.fusion_decode(
            lines,
            ctx,
            null_ctx = null_ctx,
            tm_eos = tm_eos,
            lm_eos = lm_sep,
            beam_size = beam_size,
            delay = delay,
            length_penalty_a = length_penalty_a)

        if trans_log is not None:
            trans_log.extend(zip(lines, ctx, outs))

        _count += len(outs)
        logger.debug('Translated {}/{}. {} s/sentence'.format(
            _count, len(sents) - len(docs), (time.time() - _start_t)/_count))

        for i, out in zip(doc_inds, outs):
            # Store translations
            translations[i].append(out)

            # Update context queue
            if context_limit_type == 'sentence':
                contexts[i].append(lm_sep + ' ' + out)
                if len(contexts[i]) > context_limit:
                    contexts[i].popleft()
            elif context_limit_type == 'token':
                contexts[i].append(lm_sep)
                contexts[i].extend(out.split())
                while len(contexts[i]) > context_limit:
                    contexts[i].popleft()
            else:
                raise ValueError

    return translations

def translate_with_context(x_docs, y_docs, fdec, tm_sos, tm_eos, lm_sos, lm_sep, beam_size=1, delay=0, length_penalty_a=0, context_limit=1, return_x_c=False):

    x, c = [], []
    for x_doc, y_doc in zip(x_docs, y_docs):
        x.extend(x_doc)
        c.extend([
            '{} {} {}'.format(
                lm_sos,
                ' '.join([lm_sep + ' ' + line for line in y_doc[max(0, i-context_limit): i]]),
                lm_sep)
            for i in range(len(y_doc))])
    logger.info('Number of sents.: {}'.format(len(x)))
    logger.debug('{}\n{}'.format(x[:5], c[:5]))
    
    null_ctx = '{} {}'.format(lm_sos, lm_sep)
    trans = fdec.fusion_decode(
        x, c,
        null_ctx = null_ctx,
        tm_eos = tm_eos,
        lm_eos = lm_sep,
        beam_size = beam_size,
        delay = delay,
        length_penalty_a = length_penalty_a)
    
    if return_x_c:
        return trans, x, c
    else:
        return trans

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
