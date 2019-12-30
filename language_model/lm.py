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

    
    def __layer_name(self, i):
        return 'layer_{}'.format(i)


    def  build(self, input_shape):
        self.triangle_bias = model.make_attention_bias_triangle(self.params['network']['max_length'])
        self.embedding_layer = model.Embedding_layer(
            self.params['vocab']['vocab_size'],
            self.params['network']['embed_size'])

        if self.params['network']['pos_embedding']:
            self.pos_emb = tf.get_variable(
                'pos_emb',
                [self.params['network']['max_length'], self.params['network']['embed_size']],
                tf.float32)
        
        self.blocks = []
        for i in range(self.params['network']['n_blocks']):
            layer_name = self.__layer_name(i)

            # Self-attention
            if self.params["network"]["relative_position"]:
                self_attn = model.BlockWrapper(
                    RelativePositionMultiheadSelfAttention(
                        self.params['network']['attention_size'],
                        self.params['network']['n_heads'],
                        self.params['network']['dropout_rate'],
                        self.params['network']['rel_pos_max_dist'],
                        self.params['network']['rel_pos_unique_per_head'],
                        name='{}_{}'.format(layer_name, 'self_attention')
                    ),
                    self.params)
            else:
                self_attn = model.BlockWrapper(
                    model.SelfAttention(
                        self.params['network']['attention_size'],
                        self.params['network']['n_heads'],
                        self.params['network']['dropout_rate'],
                        name='{}_{}'.format(layer_name, 'self_attention')),
                    self.params)

            # Feedfoward
            ff = model.BlockWrapper(
                model.Feedforward(
                    self.params['network']['ff_dim'],
                    self.params['network']['dropout_rate'],
                    name='{}_{}'.format(layer_name, 'feedforward')),
                self.params)

            # Register
            self.blocks.append((self_attn, ff))

        self.output_norm = model.Layer_norm()

        super().build(input_shape)


    def make_cache(self, batch_size):
        
        for i in range(self.params['network']['n_blocks']):
            layer_name = self.__layer_name(i)
            cache[layer_name] = {
                'k': tf.zeros([batch_size, 0, self.params['network']['attention_size']]),
                'v': tf.zeros([batch_size, 0, self.params['network']['attention_size']])}
        
        return cache


    def __get_logits_fn(self, inputs, cache=None, training=False):
        cache = cache or {}
        if self.__layer_name(0) in cache:
            l0v_cache = cache[self.__layer_name(0)]['v']
            front = tf.shape(l0v_cache)[1]
        else:
            front = 0

        tail = front + tf.shape(inputs)[1]

        attn_bias = self.triangle_bias[:, :, front:tail, :tail]

        # Decoder embedding
        outputs = self.embedding_layer(inputs)

        # Position information
        _pos_info_count = 0
        if self.params['network']['pos_encoding']:
            # Positional encoding. Take t >= current-front-position
            outputs = outputs + model.positional_encoding(
                tail, self.params["network"]["embed_size"])[front:]
            _pos_info_count += 1
        if self.params['network']['pos_embedding']:
            _pos_info_count += 1
            outputs += self.pos_emb[front: tail]
        if self.params["network"].get("relative_position", False):
            _pos_info_count += 1

        # Alert if there are more than one position information representation used
        if _pos_info_count > 1:
            logger.warning('You are using more than one position info representations in Decoder')
        elif _pos_info_count == 0:
            logger.warning('No position info representation is used!')


        # Dropout
        outputs = tf.layers.dropout(
            outputs,
            self.params['network']['dropout_rate'],
            training=training)

        # Decoder blocks
        for i, (attn, ff) in enumerate(self.blocks):
            layer_name = self.__layer_name(i)
            layer_cache = cache.get(layer_name, None)

            outputs = attn(outputs, attn_bias, training=training, cache=layer_cache)
            outputs = ff(outputs, training=training)

        outputs = self.output_norm(outputs)
        outputs = self.embedding_layer.emb2logits(outputs)

        return outputs


    def get_logits(self, x, training=False):
        assert self.built

        x_shift = tf.concat(
            [tf.fill([tf.shape(x)[0], 1], self.params['vocab']['SOS_ID']), x[:,:-1]], axis=1)

        outputs = self.__get_logits_fn(x_shift)

        return outputs


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
        x = tf.concat([tf.fill([tf.shape(x)[0], 1], self.params['vocab']['SOS_ID']), x, axis=1)
        x_len += 1
        
        hypos, scores = model.beam_search_decode(
            self.__get_logits_fn,
            cache,
            x,
            x_len,
            beam_size,
            maxlen,
            self.params['vocab']['EOS_ID'],
            self.params['vocab']['PAD_ID'],
            self.params['test']['length_penalty_a'],
            sampling_method=sampling_method)

        return hypos
