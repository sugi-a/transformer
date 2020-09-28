import sys, os, argparse, time, json, itertools
from logging import basicConfig, getLogger, INFO, DEBUG; logger = getLogger()
from collections import deque
import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework import nest
from ..components.inference import Inference
from ..components.decoding import length_penalty, beam_search_decode_V2, force_decoding
from ..components.model import align_to_right, remove_offsets, Decoder
from ..components import dataprocessing as dp
from ..language_model import language_model
from ..language_model.inference import Inference as LMInference
from .tm_lm_fusion_beam_search_v2 import fusion_beam_search_delayed_pmi_v2

def replace_logits(logits, index, values):
    """
    logits: [batch, length, vocab]
    index: 0 <= index < vocab
    values: [1 or batch, 1 or length]"""
    
    values = tf.broadcast_to(values, tf.shape(logits[:,:,0]))
    return tf.concat([logits[:, :, :index], values[:,:,None], logits[:, :, index+1:]], axis=2)


def pmi_mask(allow_list, top1):
    '''
    Args:
        allow_list: [V, L]
        top1: [B, T]
    Returns:
        tf.bool [B, T, V]

    V: vocabulary size
    L: max list length
    B: batch size
    T: max sequence length
        '''

    B = tf.shape(top1)[0]
    T = tf.shape(top1)[1]
    V = tf.shape(allow_list)[0]
    L = tf.shape(allow_list)[1]

    # [B, T, L]
    V_inds = tf.gather(allow_list, top1)
    B_inds = tf.tile(tf.range(B)[:, None, None], [1, T, L])
    T_inds = tf.tile(tf.range(T)[None, :, None], [B, 1, L])

    # [B, T, L, 3]
    indices = tf.stack([B_inds, T_inds, V_inds], -1)
    
    # [B, T, L]
    updates = tf.ones(tf.shape(indices)[:-1], dtype=tf.bool)

    # [B, T, V]
    ret = tf.scatter_nd(indices, updates, [B, T, V])
    
    return ret


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


    def fn_fusion_score(self, inputs, nullc, pmi_smoothing_a, pmi_smoothing_T, pmi_constraints):
        (x, x_len), (tmy, tmy_len), (c, c_len), (lmy, lmy_len) = inputs
        
        tm_score, = self.fn_translation_score(inputs[:2], 0)

        ret = self.fn_pmi_detail(((c, c_len), (lmy, lmy_len)), nullc, pmi_smoothing_a, pmi_smoothing_T)
        pmi = ret[3] - ret[4]

        # make pmi constrant mask
        tm_logits = self.model.get_logits(x, tmy[:,:-1], x_len, tmy_len - 1, False)
        top1 = tf.math.argmax(tm_logits, axis=-1)
        V = tf.shape(tm_logits)[-1]

        # user friendly
        pmi_constraints = tf.cond(
            tf.equal(tf.size(pmi_constraints), 0),
            lambda: tf.zeros([V, 0], tf.int32),
            lambda: pmi_constraints)
        
        # pmi_constraints [V, L], top1 [B, T], pmi [B, T]
        # [B, T, L]
        mask = tf.gather(pmi_constraints, top1)
        # [B, T, L]
        mask = tf.equal(mask, lmy[:,:,None])
        # [B, T]
        mask = tf.cast(tf.math.reduce_any(mask, axis=-1), tf.float32)
        pmi *= mask

        return tm_score + tf.reduce_sum(pmi, axis=-1), pmi, tm_score


    def fn_pmi_detail(self, inputs, null_ctx, pmi_smoothing_a, pmi_smoothing_T):
        # null_ctx: [L]
        (c, c_len), (x, x_len) = inputs

        is_target = tf.sequence_mask(x_len, tf.shape(x)[1], dtype=tf.float32)
        batch_size = tf.shape(c)[0]
        nullc = tf.tile(null_ctx[None], [batch_size, 1])
        nullc_len = tf.ones([batch_size], dtype=tf.int32) * tf.shape(null_ctx)[0]

        # [batch, len, vocab]
        condp, = self.lmi.fn_cond_tok_logp_dist(((x, x_len), (c, c_len)))
        nullp, = self.lmi.fn_cond_tok_logp_dist(((x, x_len), (nullc, nullc_len)))

        # add-a smoothing
        condp = self.smoothing_methodA_logp(condp, pmi_smoothing_a)
        nullp = self.smoothing_methodA_logp(nullp, pmi_smoothing_a)

        # Temperature scaling smoothing
        condp = tf.math.log_softmax(condp * pmi_smoothing_T)
        nullp = tf.math.log_softmax(nullp * pmi_smoothing_T)

        condp, nullp = [
            tf.batch_gather(v, x[:,:,None])[:,:,0] * is_target for v in [condp, nullp]]

        seq_condp = tf.reduce_sum(condp, axis=1)
        seq_nullp = tf.reduce_sum(nullp, axis=1)
        ret = seq_condp - seq_nullp

        return [ret, seq_condp, seq_nullp, condp, nullp]

    
    def fusion_decode(self, x, ctx, null_ctx, tm_eos, lm_eos, beam_size, ret_search_detail=False, delay=0, length_penalty_a=0, alpha=0, beta=0, T_nlm=1, T_clm=1, topk=-1, confusion_map=None, PMI_L=-1e9, PMI_clip=1e9, norm_fs=False, topl=0, T_sf=0, T_nsf=0):

        if not hasattr(self, 'op_fusion_beam_search'):
            with self.graph.as_default():
                param_phs = {
                    'null_ctx': tf.placeholder(tf.int32, [None]),
                    'tm_eos': tf.placeholder(tf.int32, []),
                    'lm_eos': tf.placeholder(tf.int32, []),
                    'beam_size': tf.placeholder(tf.int32, []),
                    'delay': tf.placeholder(tf.int32, []),
                    'length_penalty_a': tf.placeholder(tf.float64, []),
                    'alpha': tf.placeholder(tf.float32, []),
                    'beta': tf.placeholder(tf.float32, []),
                    'T_nlm': tf.placeholder(tf.float32, []),
                    'T_clm': tf.placeholder(tf.float32, []),
                    'topk': tf.placeholder(tf.int32, []),
                    'confusion_map': tf.placeholder(tf.int32, [None, None]),
                    'PMI_L': tf.placeholder(tf.float32, []),
                    'PMI_clip': tf.placeholder(tf.float32, []),
                    'norm_fs': tf.placeholder(tf.bool, []),
                    'topl': tf.placeholder(tf.int32, []),
                    'T_sf': tf.placeholder(tf.float32, []),
                    'T_nsf': tf.placeholder(tf.float32, [])
                }
                data_phs = (
                    (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None])),
                    (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None]))
                )

                self.op_fusion_beam_search = self.make_op(
                    self.fn_fusion_decode,
                    data_phs,
                    **param_phs)

        # pmi constraints
        if confusion_map is None:
            confusion_map = [[]] * len(self.vocab.ID2tok)
        else:
            confusion_map = dp.pad_seqs(list(dp.gen_line2IDs(confusion_map, self.vocab)))
        
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
            length_penalty_a = length_penalty_a,
            alpha = alpha,
            beta = beta,
            T_nlm=T_nlm,
            T_clm=T_clm,
            topk=topk,
            confusion_map=confusion_map,
            PMI_L=PMI_L,
            PMI_clip=PMI_clip,
            norm_fs=norm_fs,
            topl=topl,
            T_sf=T_sf,
            T_nsf=T_nsf)

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


    def fn_fusion_decode(self, inputs, null_ctx, tm_eos, lm_eos, beam_size, delay, length_penalty_a, alpha, beta, T_nlm, T_clm, topk, confusion_map, PMI_L, PMI_clip, norm_fs, topl, T_sf, T_nsf, force_decode=False):
        """
            x: '<s> source sentence </s>' * batch_size
            ctx: '<s> context sentence <sep>' * batch_size
            null_ctx:  '<s> <sep>' (shape [None])
            """

        if not force_decode:
            (x, x_len), (ctx, ctx_len) = inputs
        else:
            (x, x_len), (ctx, ctx_len), (ref, ref_len) = inputs

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
            return self.__get_logits_fn_delay(*args, **kwargs, delay=delay, tm_eos=tm_eos, lm_eos=lm_eos, alpha=alpha, beta=beta, T_nlm=T_nlm, T_clm=T_clm, topk=topk, confusion_map=confusion_map, PMI_L=PMI_L, PMI_clip=PMI_clip, norm_fs=norm_fs, topl=topl, T_sf=T_sf, T_nsf=T_nsf)

        def __get_logits_fn_force_decode(*args, **kwargs):
            return self.__get_logits_fn_for_force_decode(*args, **kwargs, delay=delay, tm_eos=tm_eos, lm_eos=lm_eos, pmi_smoothing_a=pmi_smoothing_a, pmi_smoothing_T=pmi_smoothing_T, pmi_constraints=pmi_constraints, pmi_padding_value=pmi_padding_value)

        # Execute
        if not force_decode:
            hypos, scores = fusion_beam_search_delayed_pmi_v2(
                __get_logits_fn,
                cache,
                static_cache,
                init_seq,
                beam_size,
                maxlens,
                eos_id = tm_eos,
                pad_id = self.vocab.PAD_ID,
                length_penalty_a=length_penalty_a,
                normalize_logits = False)

            return hypos, scores
        else:
            cache.update(static_cache)
            scores, ranks = force_decoding(__get_logits_fn_force_decode, cache, ref)
            # padding
            mask = tf.sequence_mask(ref_len - 1, tf.shape(scores)[1])
            scores *= tf.cast(mask, tf.float32)
            ranks *= tf.cast(mask, tf.int32)

            return scores, ranks
        

    def smoothing_methodA_logp(self, logp, a):
        # logp: [batch size, sequence length, vocab size]
        # a: scalar parameter
        p = tf.exp(logp)
        p += a / tf.cast(tf.shape(p)[2], tf.float32)
        p = p / tf.reduce_sum(p, axis=-1, keepdims=True)
        logp = tf.math.log(p)

        return logp

    def smoothing_beta(self, logp, add_logp, b):
        p = tf.exp(logp)
        add_p = tf.exp(add_logp)
        p += add_p * b
        p /= tf.reduce_sum(p, axis=-1, keepdims=True)
        return tf.math.log(p)

    def mask_logp(self, logp, mask, norm):
        logp = tf.where(mask, logp, tf.fill(tf.shape(logp), -40.0))
        if norm:
            logp = tf.math.log_softmax(logp)
        return logp


    def __get_logits_fn_delay(self, dec_inputs, cache, static_cache, _i, prev_pmi_chosen, delay, tm_eos, lm_eos, alpha, beta, T_nlm, T_clm, topk, confusion_map, PMI_L, PMI_clip, T_sf, T_nsf, norm_fs, topl):
        NEGINF = -1e10
        first_pos = tf.equal(0, _i)
        ctx_inputs, y_inputs = tf.cond(
            first_pos,
            lambda: (static_cache['init_ctx'], static_cache['null_ctx']),
            lambda: (dec_inputs, dec_inputs))

        # Shape of dec_inputs: [batch * beam, 1]
        # Logits [batch * beam, 1, vocab]
        pTM = tf.math.log_softmax(self.tm.get_logits_w_cache(dec_inputs, cache['TM']), axis=-1)

        # topl mask
        topl = tf.cond(tf.greater(topl, 0), lambda:topl, lambda:tf.shape(pTM)[-1])
        topl_logp, _ = tf.math.top_k(pTM, topl, sorted=True)
        topl_mask = tf.greater_equal(pTM, topl_logp[:,:,-1:])

        pLM = tf.math.log_softmax(T_nlm * self.lm.get_logits_w_cache(y_inputs, cache['LM'])[:, -1:], axis=-1)
        pLM = replace_logits(pLM, tm_eos, pLM[:,:, lm_eos])
        pLM = replace_logits(pLM, lm_eos, [[-40.0]])
        pLM = self.mask_logp(pLM, topl_mask, True)

        pCTXLM = tf.math.log_softmax(T_clm * self.lm.get_logits_w_cache(ctx_inputs, cache['ctxLM'])[:, -1:], axis=-1)
        pCTXLM = replace_logits(pCTXLM, tm_eos, pCTXLM[:,:,lm_eos])
        pCTXLM = replace_logits(pCTXLM, lm_eos, [[-40.0]])
        pCTXLM = self.mask_logp(pCTXLM, topl_mask, True)

        # smoothing alpha
        pLM = self.smoothing_methodA_logp(pLM, alpha)
        pCTXLM = self.smoothing_methodA_logp(pCTXLM, alpha)

        # smoothing beta
        #tm_top1_i = tf.cast(tf.argmax(pTM, axis=-1), tf.int32) # [B, T]
        #tm_top_LM = tf.batch_gather(pLM, tm_top1_i[:,:,None]) # [B, T, 1]
        #pLM = self.smoothing_beta(pLM, tm_top_LM, beta)
        #pCTXLM = self.smoothing_beta(pCTXLM, tm_top_LM, beta)

        # Choose shallow fusion if T_sf > 0
        pmi = tf.cond(tf.greater(T_sf, 0), lambda:T_sf * pCTXLM, lambda:pCTXLM - pLM)
        # Choose unconditional shallow fusion if T_nsf > 0
        pmi = tf.cond(tf.greater(T_nsf, 0), lambda:T_nsf * pLM, lambda: pmi)
        # bounding
        pmi = tf.maximum(PMI_L, pmi)
        pmi = tf.minimum(pmi, PMI_clip)

        ## PMI vocabulary constraints
        #top1 = tf.math.argmax(pTM, axis=-1)
        #mask = pmi_mask(confusion_map, top1)
        ## if confusion_map' size is 0, no constraints.
        #mask = tf.cond(tf.equal(tf.size(confusion_map), 0), lambda: tf.ones_like(mask), lambda: mask)
        
        # topk mask
        #topk = tf.cond(tf.less_equal(topk, 0), lambda: tf.shape(pTM)[-1], lambda: topk)
        #tm_topk, _ = tf.math.top_k(pTM, topk, sorted=True) # [..., K]
        #topk_mask = tf.greater_equal(pTM, tm_topk[:,:,-1:])

        pmi = self.mask_logp(pmi, topl_mask, False)

        # prev_chosen_pmi: [batch*beam]
        # push [batch*beam, delay]
        pmi_q = tf.concat([cache['pmi_q'], prev_pmi_chosen[:, None]], axis=1)
        # pop [batch*beam], [batch*beam, delay - 1]
        popped, cache['pmi_q'] = pmi_q[:, 0], pmi_q[:, 1:]

        # [batch*beam, 1, 1]
        pmi_to_add = tf.cond(tf.equal(delay, 0), lambda: pmi, lambda: popped[:, None, None])

        p_fusion = pTM + pmi_to_add
        p_fusion = tf.cond(norm_fs, lambda:tf.math.log_softmax(p_fusion), lambda:p_fusion)

        return p_fusion, pmi[:, 0]


    def __get_logits_fn_for_force_decode(
        self,
        _i, dec_inputs, cache,
        delay, tm_eos, lm_eos, pmi_smoothing_a, pmi_smoothing_T, pmi_constraints, pmi_padding_value):

        NEGINF = -1e10
        first_pos = tf.equal(0, _i)
        ctx_inputs, y_inputs = tf.cond(
            first_pos,
            lambda: (cache['init_ctx'], cache['null_ctx']),
            lambda: (dec_inputs, dec_inputs))

        # Shape of dec_inputs: [batch * beam, 1]
        # Logits [batch * beam, 1, vocab]
        pTM = tf.math.log_softmax(self.tm.get_logits_w_cache(dec_inputs, cache['TM']), axis=-1)
        pLM = tf.math.log_softmax(self.lm.get_logits_w_cache(y_inputs, cache['LM'])[:, -1:], axis=-1)
        pCTXLM = tf.math.log_softmax(self.lm.get_logits_w_cache(ctx_inputs, cache['ctxLM'])[:, -1:], axis=-1)

        # smoothing A
        pLM = self.smoothing_methodA_logp(pLM, pmi_smoothing_a)
        pCTXLM = self.smoothing_methodA_logp(pCTXLM, pmi_smoothing_a)

        # Temperature scaling
        pLM = tf.math.log_softmax(pLM * pmi_smoothing_T)
        pCTXLM = tf.math.log_softmax(pCTXLM * pmi_smoothing_T)
        
        pmi = pCTXLM - pLM

        # Move lm_eos to tm_eos and set NEGINF to lm_eos
        _eos_logits = pmi[:, :, lm_eos]
        _neg_fill = tf.fill([tf.shape(pmi)[0], 1], NEGINF)
        pmi = replace_logits(pmi, tm_eos, _eos_logits)
        pmi = replace_logits(pmi, lm_eos, _neg_fill)

        # PMI vocabulary constraints
        top1 = tf.math.argmax(pTM, axis=-1)
        mask = pmi_mask(pmi_constraints, top1)
        # if pmi_constraints' size is 0, no constraints.
        mask = tf.cond(tf.equal(tf.size(pmi_constraints), 0), lambda: tf.ones_like(mask), lambda: mask)
        pmi = tf.where(mask, pmi, tf.ones_like(pmi) * pmi_padding_value)

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
            pmi, x, y, c, null_c
        )


    def cumulative_scores(self, x, y, c, null_c, sos, tm_eos, lm_eos, ret_input_info=False):
        with self.graph.as_default():
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
                    ), **param_phs)
            
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
    
    def pmi(self, c, y, nullc, pmi_smoothing_a=0, pmi_smoothing_T=1, pmi_constraints=None, return_detail=False):
        with self.graph.as_default():
            if not hasattr(self, 'op_pmi'):
                param_phs = {
                    'null_ctx': tf.placeholder(tf.int32, [None]),
                    'pmi_smoothing_a': tf.placeholder(tf.float32, []),
                    'pmi_smoothing_T': tf.placeholder(tf.float32, [])
                }
                self.op_pmi = self.make_op(
                    self.fn_pmi_detail,
                    (
                         (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None])),
                         (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None])),
                    ),
                    **param_phs)
            nullc = self.vocab.line2IDs(nullc)
            data = zip(dp.gen_line2IDs(c, self.vocab), dp.gen_line2IDs(y, self.vocab))
            data = dp.gen_const_capacity_batch_multi_seq(data, self.batch_capacity)

            ret = self.execute_op(
                self.op_pmi,
                list(data),
                null_ctx=nullc,
                pmi_smoothing_a=pmi_smoothing_a,
                pmi_smoothing_T=pmi_smoothing_T)

            if return_detail:
                return ret
            else:
                return ret[0]

    def calculate_fusion_score(self, x, tm_y, c, nullc, lm_y, pmi_smoothing_a=0, pmi_smoothing_T=1, pmi_constraints=None, return_detail=False):
        if not hasattr(self, 'op_fusion_score'):
            param_phs = {
                "nullc": tf.placeholder(tf.int32, [None]),
                'pmi_smoothing_a': tf.placeholder(tf.float32, []),
                'pmi_smoothing_T': tf.placeholder(tf.float32, []),
                'pmi_constraints': tf.placeholder(tf.int32, [None, None])
            }
            self.op_fusion_score = self.make_op(
                self.fn_fusion_score,
                tuple(
                    (tf.placeholder(tf.int32, [None, None]), tf.placeholder(tf.int32, [None]))
                    for i in range(4)),
                **param_phs)

        nullc = self.vocab.line2IDs(nullc)
        data = zip(*(dp.gen_line2IDs(v, vocab) for v,vocab
            in [(x, self.src_vocab), (tm_y, self.vocab), (c, self.vocab), (lm_y, self.vocab)]))
        data = dp.gen_const_capacity_batch_multi_seq(data, self.batch_capacity)

        if pmi_constraints is None:
            pmi_constraints = [[]]
        else:
            pmi_constraints = dp.pad_seqs(list(dp.gen_line2IDs(pmi_constraints, self.vocab)))
        
        ret = self.execute_op(
            self.op_fusion_score, list(data),
            nullc=nullc,
            pmi_smoothing_a=pmi_smoothing_a,
            pmi_smoothing_T=pmi_smoothing_T,
            pmi_constraints=pmi_constraints)

        if return_detail:
            return ret
        else:
            return ret[0]
    
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


def fusion_rerank2(docs, fdec, tm_sos, tm_eos, lm_sos, lm_sep, beam_size=1, length_penalty_a=0, context_limit=1, alpha=0, T=1.0, beta=0, method='fusion_rerank'):
    NSENTS = sum(map(len, docs))
    NDOCS = len(docs)
    if NDOCS == 0:
        return []
    logger.info('#docs: {}, #sents: {}'.format(NDOCS, NSENTS))

    # Create N-best lists
    logger.debug('Creating n-best lists.')

    doc_offset = [0] + list(np.cumsum([len(v) for v in docs]))
    flatten = list(itertools.chain(*docs))
    hypos, scores = fdec.translate_sentences(
        flatten,
        beam_size=beam_size,
        length_penalty_a=length_penalty_a,
        return_search_results=True)

    # restore shape [doc, n_sents, hypos]
    hypos, scores = ([deque(v[doc_offset[i]: doc_offset[i+1]]) for i in range(NDOCS)]
        for v in [hypos, scores])

    # Rerank
    logger.debug('Reranking.')

    # Context queue
    ctx_qs = [deque() for i in range(NDOCS)]

    # translated documents
    translated = [[] for i in range(NDOCS)]
    
    while any(len(v) > 0 for v in hypos):
        cur_hypos, cur_scores, cur_ids = [], [], []
        for i in range(NDOCS):
            if len(hypos[i]) > 0:
                cur_hypos.append(hypos[i].popleft())
                cur_scores.append(scores[i].popleft())
                cur_ids.append(i)

        if len(translated[0]) == 0:
            out = [v[0] for v in cur_hypos]
        else:
            if method == 'fusion_rerank' or method == 'cond_shallow_fusion_rerank':
                # Flatten scores [NDOCS * NHYPOS]
                score = np.array(list(itertools.chain(*cur_scores)))
            elif method == 'pmi_rerank':
                score = np.zeros([NDOCS * beam_size])
            else:
                assert False

            ctx = [[f' {lm_sep} '.join(ctx_qs[i])] * beam_size for i in cur_ids]
            ctx = list(itertools.chain(*ctx))
            fl_hypos = list(itertools.chain(*cur_hypos))

            if method == 'cond_shallow_fusion_rerank':
                score += beta * np.array(fdec.lmi.calculate_cond_log_prob(
                    [f'{lm_sos} {_c} {lm_sep}' for _c in ctx],
                    fl_hypos))
            elif method == 'fusion_rerank' or method == 'pmi_rerank':
                score += np.array(fdec.lmi.calculate_pmi(
                    ctx,
                    fl_hypos,
                    sep=lm_sep,
                    head=lm_sos,
                    alpha=alpha,
                    T=T))
            else:
                assert False
            score = score.reshape([-1, beam_size])
            top1_ids = np.argmax(score, axis=-1)
            out = [v[i] for v,i in zip(cur_hypos, top1_ids)]
        for i, o in zip(cur_ids, out):
            translated[i].append(o)
            ctx_qs[i].append(o)
            if len(ctx_qs[i]) > context_limit:
                ctx_qs[i].popleft()


    return translated

def successively_translate(sents, fdec, tm_sos, tm_eos, lm_sos, lm_sep, beam_size=1, delay=0, length_penalty_a=0, alpha=0, beta=0, T_nlm=1, T_clm=1, topk=-1, confusion_map=None, PMI_L=-1e9, PMI_clip=1e9, norm_fs=False, T_sf=0, T_nsf=0, topl=0, context_limit_type='sentence', context_limit=1, trans_log=None):
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
            length_penalty_a = length_penalty_a,
            alpha = alpha,
            beta = beta,
            T_nlm = T_nlm,
            T_clm = T_clm,
            topk=topk,
            confusion_map=confusion_map,
            PMI_L=PMI_L,
            PMI_clip=PMI_clip,
            norm_fs=norm_fs,
            topl=topl,
            T_sf=T_sf,
            T_nsf=T_nsf)

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
