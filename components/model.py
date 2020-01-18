from logging import getLogger; logger = getLogger(__name__)
import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np

from .relative_position import RelativePositionMultiheadSelfAttention
from .decoding import beam_search_decode, beam_search_decode_V2

NEG_INF = -1e9

def align_to_right(y, y_len, pad=None):
    """ Example.
    Inputs (batch size = 2):
        [[This is my  cat <eos> <pad> <pad>],
         [That is why she hates you   <eos>]]
    Outputs (right-aligned sequences)
        [[<pad> <pad> This is  my    cat <eos>],
         [That  is    why  she hates you <eos>]]
    Returns:
        (aligned sequences), (offsets)
    """
    maxlen = tf.shape(y)[1]
    offsets = maxlen - y_len
    indices = tf.range(maxlen)[None] - offsets[:, None]
    indices = tf.math.maximum(0, indices)
    y_aligned = tf.batch_gather(y, indices)
    if pad is not None:
        y_aligned = tf.where(
            tf.sequence_mask(offsets, maxlen=maxlen),
            tf.broadcast_to(pad, tf.shape(y_aligned)),
            y_aligned)
    
    return y_aligned, offsets


def remove_offsets(y, offsets, pad=None):
    """Returns:
        aligned sequences"""
    maxlen = tf.shape(y)[1]
    indices = tf.range(maxlen)[None] + offsets[:, None]
    indices = tf.math.minimum(indices, maxlen - 1)
    y_aligned = tf.batch_gather(y, indices)
    if pad is not None:
        lengths = maxlen - offsets
        y_aligned = tf.where(
            tf.sequence_mask(lengths, maxlen=maxlen),
            y_aligned,
            tf.broadcast_to(pad, tf.shape(y_aligned)))
    return y_aligned


class Layer_norm(tf.layers.Layer):
    def __init__(self, eps=1e-8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eps = eps

    def build(self, input_shape):
        self.beta = tf.get_variable('beta',
                               input_shape[-1:],
                               initializer=tf.constant_initializer(0))
        self.gamma = tf.get_variable('gamma',
                                input_shape[-1:],
                                initializer=tf.constant_initializer(1))
        super().build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        normalized = (inputs - mean) / tf.sqrt(variance + self.eps)

        return self.gamma * normalized + self.beta


class Embedding_layer(tf.layers.Layer):
    def __init__(self, vocab_size, embedding_size, lookup_table=None, scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lookup_table = lookup_table
        self.scale = scale

    def build(self, input_shape):
        if self.lookup_table is None:
            self.lookup_table = tf.get_variable('lookup_table',
                                                [self.vocab_size, self.embedding_size],
                                                tf.float32)
        super().build(input_shape)

    def call(self, inputs):
        outputs = tf.nn.embedding_lookup(self.lookup_table, inputs)

        if self.scale:
            outputs = outputs * (self.embedding_size ** 0.5)

        return outputs

    def emb2logits(self, inputs):
        # this Layer must be built (as an embedding layer) before used as a projection layer to produce logits at the end of a decoder
        assert self.built
        with tf.name_scope('emb2logits'):
            return tf.tensordot(inputs, self.lookup_table, [[-1],[1]], name='logits')



def positional_encoding(length, emb_size, name='positional_encoding'):
    """Sinusoidal Positional Encoding

    Args:
        length: sentence length (batch.shape[1])
        emb_size: embedding size (batch.shape[-1])

    Returns:
        positional_encoding of shape [seq_length, emb_size]

    """
    # PE(pos, i) = 
    #   sin(pos/(10000^(i/(emb_size/2)))) (0<=i<emb_size/2)
    #   cos(pos/(10000^(i/(emb_size/2)))) (emb_size/2<=i<emb_size)
    with tf.name_scope(name):
        pos = tf.range(tf.cast(length, tf.float32))
        half = emb_size // 2
        i = tf.range(half, dtype=tf.float32)
        scaled_time = pos[:, None] / tf.pow(10000.0, i / half)[None]
        return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

def make_self_attn_bias(lengths, maxlen):
    """
    Args:
        lengths: Tensor of shape [batach_size] with type tf.int32
        maxlen: int
    returns:
        Tensor of shape [batch_size, 1, 1, length] with type tf.float32
    """
    outputs = (1 - tf.sequence_mask(lengths, maxlen, tf.float32)) * NEG_INF
    return tf.expand_dims(tf.expand_dims(outputs, axis=1), axis=1)

def make_attention_bias_triangle(length):
    """
    Args:
        length: length of the longest sequence in the batch
    Returns:
        Tensor of shape [1, 1, length, length]
        """
    valid_locs = tf.matrix_band_part(tf.ones([1,1,length,length]), -1, 0)
    return (1 - valid_locs) * NEG_INF


class MultiheadAttention(tf.layers.Layer):
    def __init__(self, hidden_size, n_heads, dropout_rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.q_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='q_layer')
        self.k_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='k_layer')
        self.v_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='v_layer')
        self.att_out_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='attention_output')
        super().build(input_shape)

    def call(self, query, dictionary, bias, training=False, cache=None):
        head_size = self.hidden_size // self.n_heads
        
        q = self.q_layer(query) # [batch, length, emb_size]
        k = self.k_layer(dictionary)
        v = self.v_layer(dictionary)

        if cache is not None:
            with tf.name_scope('layer_cache_extension'):
                k = tf.concat([cache['k'], k], axis=1)
                v = tf.concat([cache['v'], v], axis=1)
                cache['k'] = k
                cache['v'] = v

        q = tf.stack(tf.split(q, self.n_heads, axis=-1), axis=1) # [batch, nheads, length_q, head_size]
        k = tf.stack(tf.split(k, self.n_heads, axis=-1), axis=1) # [batch, nheads, length_k, head_size]
        v = tf.stack(tf.split(v, self.n_heads, axis=-1), axis=1) # [batch, nheads, length_k, head_size]

        weight = tf.matmul(q, k, transpose_b=True) # [batch, nheads, length_q, length_k]
        weight = weight / (head_size ** 0.5)

        with tf.name_scope('add_bias'):
            weight = weight + bias

        weight = tf.nn.softmax(weight, name='attention_weight')

        weight = tf.layers.dropout(weight, self.dropout_rate, training=training)

        outputs = tf.matmul(weight, v) # [batch, nheads, length_q, head_size]

        outputs = tf.concat(tf.unstack(outputs, axis=1), axis=2) # [batch, length_q, emb_size]

        outputs = self.att_out_layer(outputs)

        return outputs

class SelfAttention(MultiheadAttention):
    def call(self, inputs, *args, **kwargs):
        return super().call(inputs, inputs, *args, **kwargs)

class Feedforward(tf.layers.Layer):
    def __init__(self, n_units, dropout_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_units = n_units
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.relu = tf.layers.Dense(self.n_units, tf.nn.relu, True, name='relu')
        self.linear = tf.layers.Dense(input_shape[-1], use_bias=True, name='linear')
        super().build(input_shape)

    def call(self, inputs, training=False):
        outputs = self.relu(inputs)
        outputs = tf.layers.dropout(outputs, self.dropout_rate, training=training)
        outputs = self.linear(outputs)
        return outputs

def label_smoothing(labels, eps=0.1):
    if eps == 0:
        return labels
    else:
        return (1 - eps) * labels + eps/tf.cast(tf.shape(labels)[-1], tf.float32)

class BlockWrapper(tf.layers.Layer):
    def __init__(self, layer, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.params = params

    def build(self, input_shape):
        # the Layer wrapped by this BlockWrapper must not be built before this BlockWrapper is built
        # in order to make it arranged under the variable scope of this BlockWrapper
        assert not self.layer.built
        self.layer_norm = Layer_norm()
        super().build(input_shape)

    def call(self, x, *args, training=False, **kwargs):
        y = self.layer_norm(x)
        y = self.layer(y, *args, training=training, **kwargs)
        y = tf.layers.dropout(y, self.params["network"]["dropout_rate"], training=training)
        return y + x

class Encoder(tf.layers.Layer):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        self.embedding_layer = Embedding_layer(
            self.params["vocab"]["vocab_size"], self.params["network"]["embed_size"])


    def build(self, input_shape):
        if self.params['network']['pos_embedding']:
            self.pos_emb = tf.get_variable(
                'pos_emb',
                [self.params['network']['max_length'], self.params['network']['embed_size']],
                tf.float32)

        self.blocks = []
        
        for i in range(self.params["network"]["n_blocks"]):
            layer_name = 'layer_{}'.format(i)

            # Self-attention layer
            if self.params["network"].get("relative_position", False):
                self_attn = BlockWrapper(
                    RelativePositionMultiheadSelfAttention(
                        self.params["network"]["attention_size"],
                        self.params["network"]["n_heads"],
                        self.params["network"]["dropout_rate"],
                        self.params["network"]["rel_pos_max_dist"],
                        self.params["network"]["rel_pos_unique_per_head"],
                        name='{}_{}'.format(layer_name, 'self_attention')),
                    self.params)
            else:
                self_attn = BlockWrapper(
                    SelfAttention(
                        self.params["network"]["attention_size"],
                        self.params["network"]["n_heads"],
                        self.params["network"]["dropout_rate"],
                        name='{}_{}'.format(layer_name, 'self_attention')),
                    self.params)

            # Feedforward layer
            ff = BlockWrapper(
                Feedforward(
                    self.params["network"]["ff_size"],
                    self.params["network"]["dropout_rate"],
                    name='{}_{}'.format(layer_name, 'feedforward')),
                self.params)

            # Register
            self.blocks.append((self_attn, ff))

        self.output_norm = Layer_norm()
        super().build(input_shape)

    def call(self, inputs, self_attn_bias, training=False):
        # Embedding [batch, length, emb_size]
        outputs = self.embedding_layer(inputs)
        
        # Position information
        _pos_info_count = 0
        if self.params['network']['pos_encoding']:
            _pos_info_count += 1
            outputs += positional_encoding(
                tf.shape(inputs)[1],
                self.params["network"]["embed_size"])
        if self.params['network']['pos_embedding']:
            _pos_info_count += 1
            outputs += self.pos_emb[:tf.shape(inputs)[1]]
        if self.params["network"].get("relative_position", False):
            _pos_info_count += 1
        # Alert if there are more than one position information representation used
        if _pos_info_count > 1:
            logger.warning('You are using more than one position info representations in Encoder')
        elif _pos_info_count == 0:
            logger.warning('No position information representation is used in Encoder')
        


        outputs = tf.layers.dropout(outputs, self.params["network"]["dropout_rate"], training=training)

        for self_attn, ff in self.blocks:
            outputs = self_attn(outputs, self_attn_bias, training=training)
            outputs = ff(outputs, training=training)

        return self.output_norm(outputs)

class Decoder(tf.layers.Layer):
    def __init__(self, params, embedding_layer=None, context=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        self.embedding_layer = embedding_layer
        self.context = context

    def build(self, input_shape):
        if self.embedding_layer is None:
            self.embedding_layer = Embedding_layer(self.params["vocab"]["vocab_size"], self.params["network"]["embed_size"])
        else:
            # if the embedding layer is owned by another Layer it must be built until now
            # in order to avoid ambiguous variable scope tree
            assert self.embedding_layer.built

        if self.params['network']['pos_embedding']:
            self.pos_emb = tf.get_variable(
                'pos_emb',
                [self.params['network']['max_length'], self.params['network']['embed_size']],
                tf.float32)

        if self.params['network']['pos_encoding']:
            self.pos_enc_table = positional_encoding(
                self.params['network']['max_length'],
                self.params["network"]["embed_size"])
        
        self.blocks = []
        for i in range(self.params["network"]["n_blocks"]):
            layer_name = 'layer_{}'.format(i)

            # Self-attention layer
            if self.params["network"].get("relative_position", False):
                self_attn = BlockWrapper(
                    RelativePositionMultiheadSelfAttention(
                        self.params["network"]["attention_size"],
                        self.params["network"]["n_heads"],
                        self.params["network"]["dropout_rate"],
                        self.params["network"]["rel_pos_max_dist"],
                        self.params["network"]["rel_pos_unique_per_head"],
                        name='{}_{}'.format(layer_name, 'self_attention')),
                    self.params)
            else:
                self_attn = BlockWrapper(
                    SelfAttention(
                        self.params["network"]["attention_size"],
                        self.params["network"]["n_heads"],
                        self.params["network"]["dropout_rate"],
                        name='{}_{}'.format(layer_name, 'self_attention')),
                    self.params)

            # Dec-Enc attention layer
            if self.context:
                ctx_attn = BlockWrapper(
                    MultiheadAttention(
                        self.params["network"]["attention_size"],
                        self.params["network"]["n_heads"],
                        self.params["network"]["dropout_rate"],
                        name='{}_{}'.format(layer_name, 'context_attention')),
                    self.params)    
            else:
                ctx_attn = None

            # Feedforward layer
            ff = BlockWrapper(
                Feedforward(
                    self.params["network"]["ff_size"],
                    self.params["network"]["dropout_rate"],
                    name='{}_{}'.format(layer_name, 'feedforward')),
                self.params)

            # Register
            self.blocks.append((self_attn, ctx_attn, ff))


        self.output_norm = Layer_norm()

        super().build(input_shape)


    def get_layer_cache_length(self, cache):
        cache_l0_v = cache['layer_0']['v']
        return tf.shape(cache_l0_v)[1]


    def call(self, inputs, self_attn_bias, context=None, ctx_attn_bias=None, cache=None, training=False, offsets=None):
        assert (context is None) == (ctx_attn_bias is None) == (not self.context)
        if cache is None: cache = {}

        if 'layer_0' in cache:
            cache_l0_v = cache['layer_0']['v']
            seq_start = tf.shape(cache_l0_v)[1]
            seq_end = seq_start + tf.shape(inputs)[1]
        else:
            seq_start = 0
            seq_end = tf.shape(inputs)[1]

        # ---- Things to give postion information and inter-position interaction ----
        if offsets is not None:
            self_attn_bias = self_attn_bias[:, :, seq_start:seq_end, :seq_end] \
                + NEG_INF * tf.sequence_mask(offsets, seq_end, dtype=tf.float32)[:, None, None]

            indices = tf.range(seq_end - seq_start)[None] + (seq_start - offsets)[:, None]
            if self.params['network']['pos_encoding']:
                pos_enc = tf.gather(self.pos_enc_table, indices)
            else:
                pos_enc = None
            
            if self.params['network']['pos_embedding']:
                pos_emb = tf.gather(self.pos_emb, indices)
            else:
                pos_emb = None
        else:
            self_attn_bias = self_attn_bias[:, :, seq_start:seq_end, :seq_end]

            if self.params['network']['pos_encoding']:
                pos_enc = self.pos_enc_table[seq_start: seq_end]
            else:
                pos_enc = None
            
            if self.params['network']['pos_embedding']:
                pos_emb = self.pos_emb[seq_start: seq_end]
            else:
                pos_emb = None

        # Alert if there are more than one position information representation used
        pos_info_count = [
            self.params['network']['pos_encoding'],
            self.params['network']['pos_embedding'],
            self.params['network']['relative_position']].count(True)
        if pos_info_count > 1:
            logger.warning('You are using more than one position info representations in Decoder')
        elif pos_info_count == 0:
            logger.warning('No position information representation is used in Decoder')


        # ---- Start of dataflow ----
        # Decoder embedding
        outputs = self.embedding_layer(inputs)

        if pos_enc is not None: outputs += pos_enc
        if pos_emb is not None: outputs += pos_emb

        # Dropout
        outputs = tf.layers.dropout(
            outputs, self.params["network"]["dropout_rate"], training=training)

        # Decoder blocks
        for i, (self_attn, ctx_attn, ff) in enumerate(self.blocks):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache.get(layer_name, None)

            outputs = self_attn(outputs, self_attn_bias, training=training, cache=layer_cache)
            if self.context:
                outputs = ctx_attn(outputs, context, ctx_attn_bias, training=training)
            outputs = ff(outputs, training=training)
        
        outputs = self.output_norm(outputs)
        outputs = self.embedding_layer.emb2logits(outputs)
        return outputs


    def make_cache(self, batch_size):
        cache = {}
        for layer in range(self.params["network"]["n_blocks"]):
            layer_name = 'layer_{}'.format(layer)
            with tf.name_scope('cache_{}'.format(layer_name)):
                cache[layer_name] = {
                    'k': tf.zeros([batch_size, 0, self.params["network"]["attention_size"]]),
                    'v': tf.zeros([batch_size, 0, self.params["network"]["attention_size"]])}
        return cache




class Transformer(tf.layers.Layer):
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params

    def build(self, input_shape):
        self.encoder = Encoder(self.params, name='encoder')
        if self.params["network"]["share_embedding"]: 
            self.decoder = Decoder(self.params, self.encoder.embedding_layer, name='decoder')
        else:
            self.decoder = Decoder(self.params, name='decoder')

        self.triangle_bias = make_attention_bias_triangle(self.params['network']['max_length'] + 10)
        
        super().build(input_shape)

    
    def make_cache(self, x, x_len, training=False, layer_cache=False, offsets=None):
        """layer_cache means decoder cache"""
        cache = self.decoder.make_cache(tf.shape(x)[0]) if layer_cache else {}

        with tf.name_scope('encoder_outputs'):
            enc_self_attn_bias = make_self_attn_bias(x_len, tf.shape(x)[1])
            enc_outputs = self.encoder(x, enc_self_attn_bias, training=training)

        cache['enc_outputs'] = enc_outputs
        cache['ctx_attn_bias'] = enc_self_attn_bias

        if offsets is not None: cache['offsets'] = offsets

        return cache


    def call(self, x, x_len, y, y_len, training=False):
        # this method is called only by self.instantiate to instantiate variables of this Layer
        return self.get_logits(x, y, x_len, y_len, training=training)

    def instanciate_vars(self):
        """create dummy graph instance of this Layer in order to place variables in a specific device."""
        if self.built:
            return
        with tf.name_scope('dummy_inputs'):
            x = tf.placeholder(tf.int32, [100, None], 'x')
            x_len = tf.placeholder(tf.int32, [100], 'x_len')
            y = tf.placeholder(tf.int32, [100, None], 'y')
            y_len = tf.placeholder(tf.int32, [100], 'y_len')
        self(x, x_len, y, y_len)


    def get_logits_w_cache(self, dec_inputs, cache, training=False):
        return self.decoder(
            dec_inputs,
            self.triangle_bias,
            context=cache['enc_outputs'],
            ctx_attn_bias=cache['ctx_attn_bias'],
            cache=cache,
            training=training,
            offsets=cache.get('offsets', None))


    def get_logits(self, x, y, x_len, y_len, training=False, shift_dec_inputs=True, offsets=None):
        """Compute logits given inputs for encoder and decoder.
        Args:
            x: inputs for encoder with shape [batch_size, length_enc]
            y: inputs for decoder with shape [batch_size, length_dec]
                `y` is shifted to the right by one step and SOS is added to the beginning
                and the last token is removed. So, `y` should not contain SOS
            x_len: lengths of x with shape [batch_size]
            y_len: lengths of y
            shift_dec_inputs: If true, <SOS> is added to the beginning of each sequence. This argument must be False if `offsets` is not None
            offsets: [batch_size] indicating the offsets of the sequences.

        Returns:
            """
        assert self.built
        assert not (offsets and shift_dec_inputs)

        # add SOS to the beginning and remove the last token
        if shift_dec_inputs:
            dec_inputs = tf.concat(
                [tf.fill([tf.shape(y)[0], 1], self.params["vocab"]["SOS_ID"]), y[:, :-1]],
                axis=1)

        cache = self.make_cache(x, x_len, training, layer_cache=False, offsets=offsets)

        return self.get_logits_w_cache(dec_inputs, cache, training=training)


    def decode(self, x, x_len, beam_size=8, return_search_results=False, init_y=None, init_y_len=None, decode_config=None):
        """Given inputs x, this method produces translation candidates by beam search
        and return the all results if `return_search_results` is True, or the sequence with the MAP otherwise.
        Args:
            x: Source sequence. tf.int32 Tensor of shape [batch, length]
            x_len: lengths of x. tf.int32, [batch]
            return_search_results: Boolean indicating whether to return the whole results or the MAP only.
            init_target_seq: target-side prefix sequence
        Returns:
            If `return_search_results` is True, a tuple of Tensor, ([batch, beam_size, length],
            [batch, beam_size]) is returned. Otherwise a Tensor [batch, length] is returned."""

        assert self.built
        # initial cache
        with tf.name_scope('init_cache'):
            cache = self.make_cache(x, x_len, training=False, layer_cache=True)

        # Initial sequence
        with tf.name_scope('define_init_sequence'):
            init_seq = tf.fill([tf.shape(x)[0], 1], self.params["vocab"]["SOS_ID"])
            if init_y is not None:
                # Remove the last token since it's EOS
                init_seq = tf.concat([init_seq, init_y], axis=1)[:, :-1]
                init_seq_len = init_y_len

        # Maximum target length
        maxlens = tf.minimum(self.params['network']['max_length'] - 10, x_len * 3 + 10)

        hypos, scores = beam_search_decode(
            self.get_logits_w_cache,
            cache,
            init_seq,
            init_seq_len,
            beam_size,
            maxlens,
            self.params["vocab"]["EOS_ID"],
            self.params["vocab"]["PAD_ID"],
            params=decode_config or self.params['test']['decode_config'])

        if return_search_results:
            return hypos, scores
        else:
            top_indices = tf.math.argmax(scores, axis=1)
            top_seqs = tf.batch_gather(hypos, top_indices)
            return top_seqs


    def decode_V2(self, x, x_len, beam_size=8, return_search_results=False, init_y=None, init_y_len=None, decode_config=None):
        """`init_y` MUST have <eos> tokens at the tail,
            which are removed at the start of decoding."""
        assert self.built

        if init_y is None:
            init_y = tf.fill([tf.shape(x)[0], 1], self.params["vocab"]["SOS_ID"])
            cache = self.make_cache(x, x_len, training=False, layer_cache=True)
            offsets = None
        else:
            # Shift 1 to the right. Shape remains to be [batch, length]
            init_y = tf.concat(
                [tf.fill([tf.shape(x)[0], 1], self.params["vocab"]["SOS_ID"]), init_y[:, :-1]], axis=1)

            # Align to the right [batch, len]
            init_y, offsets = align_to_right(init_y, init_y_len, self.params["vocab"]["PAD_ID"])
            cache = self.make_cache(x, x_len, training=False, layer_cache=True, offsets=offsets)

        # Maximum target length
        maxlens = tf.minimum(self.params['network']['max_length'] - 10, x_len * 3 + 10)

        hypos, scores = beam_search_decode_V2(
            self.get_logits_w_cache,
            cache,
            init_y,
            beam_size,
            maxlens,
            self.params["vocab"]["EOS_ID"],
            self.params["vocab"]["PAD_ID"],
            offsets=offsets,
            params=decode_config or self.params['test']['decode_config'])

        # Remove offsets
        if offsets is not None:
            hypos = tf.reshape(
                remove_offsets(
                    tf.reshape(hypos, [tf.shape(hypos)[0], -1]),
                    offsets,
                    self.params['vocab']['PAD_ID']),
                tf.shape(hypos))

        # Remove offsets and remove SOS
        hypos = hypos[:,:, 1:]

        if return_search_results:
            return hypos, scores
        else:
            top_indices = tf.math.argmax(scores, axis=1)
            top_seqs = tf.batch_gather(hypos, top_indices)
            return top_seqs

