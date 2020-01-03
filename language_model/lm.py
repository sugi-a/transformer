from logging import getLogger; logger = getLogger(__file__)
import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np

from ..components import model
from ..components.relative_position import RelativePositionMultiheadSelfAttention

class DecoderLanguageModel(tf.layers.Layer):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params

    
    def  build(self, input_shape):
        self.decoder = model.Decoder(self.params, context=False)
        self.triangle_bias = model.make_attention_bias_triangle(self.params['network']['max_length'])
        self.embedding_layer = model.Embedding_layer(
            self.params['vocab']['vocab_size'],
            self.params['network']['embed_size'])

        super().build(input_shape)


    def make_cache(self, batch_size, layer_cache=False, offsets=None):
        cache = self.decoder.make_cache(batch_size) if layer_cache else {}
        if offsets is not None: cache['offsets'] = offsets
        return cache


    def get_logits_w_cache(self, inputs, cache, training=False):
        return self.decoder(
            inputs,
            self.triangle_bias,
            cache=cache,
            training=training,
            offsets=cache.get('offsets', None))


    def get_logits(self, x, training=False, shift_dec_inputs=True, offsets=None):
        assert self.built

        if shift_dec_inputs:
            x_shift = tf.concat(
                [tf.fill([tf.shape(x)[0], 1], self.params['vocab']['SOS_ID']), x[:,:-1]], axis=1)

        cache = self.make_cache(tf.shape(x)[1], layer_cache=False, offsets=offsets)

        return self.get_logits_w_cache(x_shift, cache, training=training)


    def call(self, inputs, training=False):
        return self.get_logits(inputs, training)


    def instanciate_vars(self):
        with tf.name_scope('dummy_graph'):
            x = tf.placeholder(tf.int32, [2, None])
            self(x)

    def beam_search_decode(self, x, x_len, beam_size, maxlen, sampling_method=None):
        assert self.built

        # Check maxlen < MAXLEN
        assert maxlen < self.params['network']['max_length']

        # Check init sequence length < maxlen
        tf.assert_less(tf.shape(x)[1], maxlen)

        cache = self.make_cache(tf.shape(x)[0])

        # Add SOS to the head. 
        # Note: No need to remove the last token since it's not EOS
        x = tf.concat([tf.fill([tf.shape(x)[0], 1], self.params['vocab']['SOS_ID']), x], axis=1)
        x_len += 1
        
        hypos, scores = model.beam_search_decode(
            self.get_logits_w_cache,
            cache,
            x,
            x_len,
            beam_size,
            maxlen,
            self.params['vocab']['EOS_ID'],
            self.params['vocab']['PAD_ID'])

        return hypos
