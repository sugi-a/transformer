import sys, os, re, json, argparse
import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework import nest
from ..components.inference import Inference
from ..components.decoding import length_penalty, beam_search_decode_V2
from ..components.model import align_to_right remove_offsets Decoder


def PMIFusionDecoder:
    def __init__(self, tm, lm):
        self.tm = tm
        self.lm = lm




    def decode(self, x, x_len, ctx, ctx_len, beam_size=8):
        cache = {}

        batch_size = tf.shape(x)[0]
        init_seq = tf.fill([batch_size, 1], self.tm.params['vocab']['SOS_ID'])

        # TM initialization
        cache['TM'] = self.tm.make_cache(x, x_len, training=False, layer_cache=True)

        # LM initialization
        cache['LM'] = self.lm.make_cache(batch_size, layer_cache=True)
        
        # contextual LM initialization
        # Add sos. [batch, length] -> [batch, length + 1]
        init_ctx = tf.concat([tm_init_seq, ctx])
        init_ctx, offsets = align_to_right(init_ctx, ctx_len + 1)
        cache['ctxLM'] = self.lm.make_cache(batch_size, layer_cache=True, offsets=offsets)

        # Cache the context sequence
        cache['init_ctx'] = init_ctx

        # Maximum target length
        maxlens = tf.minimum(tm.params['network']['max_length'] - 10, x_len * 3 + 10)

        # Execute
        hypos, scores = beam_search_decode_V2(
            self.__get_logits_fn,
            cache,
            init_seq,
            beam_size,
            maxlens,
            self.tm.params['EOS_ID'],
            self.tm.params['PAD_ID'],
            params={'length_penalty_a': 0.0})
        

    def __get_logits_fn(self, dec_inputs, cache):
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
        p_fusion = pTM + pLM - pCondLM

        return p_fusion
