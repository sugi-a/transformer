import sys, os, re, json, argparse
import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework import nest
from ..components.inference import Inference
from ..components.decoding import length_penalty, beam_search_decode_V2
from ..components.model import align_to_right, remove_offsets, Decoder
from ..components import dataprocessing as dp
from ..language_model import language_model
from ..language_model.inference import Inference as LMInference


class PMIFusionDecoder(Inference):
    def __init__(self, lm_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tm = self.model

        # Creating language model
        self.lmi = LMInference(lm_dir, n_gpus=self.n_gpus, n_cpu_cores=self.n_cpu_cores, batch_capacity=self.batch_capacity//2)
        self.lm = self.lmi.model

        with self.graph.as_default():
            self.op_fusion_decode = self.make_op(self.fn_fusion_decode)
            self.op_fusion_decode2 = self.make_op(self.fn_fusion_decode, 1, {'beam_size':self.ph_beam_size})

            self.op_top_pmi = self.make_op(
                self.fn_top_pmi,
                input_phs=tuple(
                    (tf.placeholder(tf.int32, [None,None]), tf.placeholder(tf.int32, [None]))
                    for i in range(3))
            )


    def make_session(self, *args, **kwargs):
        super().make_session(*args, **kwargs)
        self.lmi.make_session(self.session, load_checkpoint=True)


    def fn_fusion_decode(self, inputs, option=0, config=None):
        (x, x_len), (y, y_len) = inputs
        hypos, scores = self.decode(x, x_len, y, y_len, self.ph_beam_size, option, config)
        return hypos, scores

    
    def fusion_decode(self, x, ctx, beam_size=8, ret_search_detail=False, option=0):
        batch_capacity = 2 * self.batch_capacity // beam_size
        batches = self.make_batches(x, ctx, batch_capacity)

        if option == 0:
            op = self.op_fusion_decode
        elif option == 1:
            op = self.op_fusion_decode2

        candidates, scores = self.execute_op(op, batches, {self.ph_beam_size: beam_size})

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


    def decode(self, x, x_len, ctx, ctx_len, beam_size=8, option=0, config=None):
        """
        Notes:
            Sequences in `y` must have EOS at the end and must not have SOS.
            """
        cache = {}

        batch_size = tf.shape(x)[0]
        init_seq = tf.fill([batch_size, 1], self.tm.params['vocab']['SOS_ID'])

        # TM initialization
        cache['TM'] = self.tm.make_cache(x, x_len, training=False, layer_cache=True)

        # LM initialization
        cache['LM'] = self.lm.make_cache(batch_size, layer_cache=True)
        
        # contextual LM initialization
        # Add sos and remove EOS. [batch, length] -> [batch, length]
        init_ctx = tf.concat([init_seq, ctx], axis=1)[:, :-1]
        init_ctx, offsets = align_to_right(init_ctx, ctx_len)
        
        cache['ctxLM'] = self.lm.make_cache(batch_size, layer_cache=True, offsets=offsets)

        # Cache the context sequence
        cache['init_ctx'] = init_ctx

        # Maximum target length
        maxlens = tf.minimum(self.params['network']['max_length'] - 10, x_len * 3 + 10)
        

        __get_logits_fn = lambda decI, cache: self.__get_logits_fn(decI, cache, option, config)

        # Execute
        hypos, scores = beam_search_decode_V2(
            __get_logits_fn,
            cache,
            init_seq,
            beam_size,
            maxlens,
            self.vocab.EOS_ID,
            self.vocab.PAD_ID,
            params={'length_penalty_a': 0.0})
        
        return hypos, scores


    def __get_logits_fn(self, dec_inputs, cache, option=0, config=None):
        ctx_inputs = tf.cond(
            tf.equal(0, self.tm.decoder.get_layer_cache_length(cache['TM'])),
            lambda: cache['init_ctx'],
            lambda: dec_inputs)
        # Shape of dec_inputs: [batch * beam, 1]
        # Logits [batch * beam, 1, vocab]
        pTM = self.tm.get_logits_w_cache(dec_inputs, cache['TM'])
        pLM = self.lm.get_logits_w_cache(dec_inputs, cache['LM'])
        pCTXLM = self.lm.get_logits_w_cache(ctx_inputs, cache['ctxLM'])[:, -1:]
        
        # Fusion
        if option == 0:
            # Normal fusion by adding logits
            p_fusion = pTM + pCTXLM - pLM
        elif option == 1:
            p_fusion = pTM + pCTXLM - pLM
            TM_top_logits, TM_top_inds = tf.math.top_k(pTM, config['beam_size'], True)
            bias = 1e9 * tf.minimum(tf.sign(pTM - TM_top_logits[:, :, config['beam_size'] - 1]), 0)
            p_fusion += bias


        return p_fusion



    def fn_top_pmi(self, inputs):
        (x, x_len), (y, y_len), (c, c_len) = inputs

        batch_size = tf.shape(y)[0]

        pos0 = tf.fill([batch_size, 1], self.tm.params['vocab']['SOS_ID'])
        
        # Shift c
        c = tf.concat([pos0, c[:, :-1]], axis=1)
        c, offsets = align_to_right(c, c_len)
        # Join c and y
        cy = tf.concat([c, y], axis=1)
        cy_len = c_len + y_len

        # top pmis [batch, y_len, vocab]
        py_logits = self.lm.get_logits(y)
        cond_y_logits = self.lm.get_logits(cy, shift_dec_inputs=False, offsets=offsets
            )[:, tf.shape(c)[1]:]
        pmi_logits = cond_y_logits - py_logits
        top_pmi_logits, pmi_inds = tf.math.top_k(cond_y_logits, self.ph_beam_size)
        pmi = top_pmi_logits - tf.math.reduce_logsumexp(pmi_logits, axis=-1, keepdims=True)

        # top log p(y|x)
        pxy_logits = self.tm.get_logits(x, y, x_len, y_len)
        top_pyx_logits, pyx_inds = tf.math.top_k(pxy_logits, self.ph_beam_size, True)
        logpyx = top_pyx_logits - tf.math.reduce_logsumexp(pxy_logits, axis=-1, keepdims=True)

        # Score
        fusion_logits = pmi_logits + pxy_logits
        top_fusion_logits, fusion_inds = tf.math.top_k(fusion_logits, self.ph_beam_size)
        fusion_score = top_fusion_logits - tf.math.reduce_logsumexp(fusion_logits, axis=-1, keepdims=True)

        # pmi for top logp(y|x)
        top_pyx_pmi = tf.batch_gather(pmi_logits, pyx_inds) - tf.math.reduce_logsumexp(pmi_logits, axis=-1, keepdims=True)


        return pmi, pmi_inds, logpyx, pyx_inds, fusion_score, fusion_inds, top_pyx_pmi


    def top_pmi_analysis(self, x, y, c, k=8):
        x = dp.gen_line2IDs(x, self.src_vocab, put_eos=True)
        y,c = (dp.gen_line2IDs(_, self.vocab, put_eos=True) for _ in (y, c))
        batches = list(dp.gen_multi_padded_batch(
            zip(x, y, c), self.batch_capacity // 128, self.vocab.PAD_ID))
        return self.execute_op(self.op_top_pmi, batches, {self.ph_beam_size: k})


