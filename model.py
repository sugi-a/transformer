import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np

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
        scaled_time = (
            tf.expand_dims(pos, axis=1) /
            tf.expand_dims(tf.pow(10000.0, i / half), axis=0)
            )
        return tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)

def make_attention_bias_from_seq_mask(lengths, maxlen):
    """
    Args:
        lengths: Tensor of shape [batach_size] with type tf.int32
        maxlen: int
    returns:
        Tensor of shape [batch_size, 1, 1, length] with type tf.float32
    """
    NEG_INF = -1e9
    outputs = (1 - tf.sequence_mask(lengths, maxlen, tf.float32)) * NEG_INF
    return tf.expand_dims(tf.expand_dims(outputs, axis=1), axis=1)

def make_attention_bias_triangle(length):
    """
    Args:
        length: length of the longest sequence in the batch
    Returns:
        Tensor of shape [1, 1, length, length]
        """
    NEG_INF = -1e9
    valid_locs = tf.matrix_band_part(tf.ones([1,1,length,length]), -1, 0)
    return (1 - valid_locs) * NEG_INF


class Multihead_attention(tf.layers.Layer):
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

class SelfAttention(Multihead_attention):
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
    return (1 - eps) * labels + (eps/tf.cast(tf.shape(labels)[-1], tf.float32))

class BlockWrapper(tf.layers.Layer):
    def __init__(self, layer, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer = layer
        self.hparams = hparams

    def build(self, input_shape):
        # the Layer wrapped by this BlockWrapper must not be built before this BlockWrapper is built
        # in order to make it arranged under the variable scope of this BlockWrapper
        assert not self.layer.built
        self.layer_norm = Layer_norm()
        super().build(input_shape)

    def call(self, x, *args, training=False, **kwargs):
        y = self.layer_norm(x)
        y = self.layer(y, *args, training=training, **kwargs)
        y = tf.layers.dropout(y, self.hparams.dropout_rate, training=training)
        return y + x

class Encoder(tf.layers.Layer):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.embedding_layer = Embedding_layer(self.hparams.vocab_size, self.hparams.embed_size)

    def build(self, input_shape):

        self.blocks = []
        for i in range(self.hparams.n_blocks):
            layer_name = 'layer_{}'.format(i)
            self.blocks.append((
                BlockWrapper(SelfAttention(self.hparams.attention_size,
                                                 self.hparams.n_heads,
                                                 self.hparams.dropout_rate,
                                                 name='{}_{}'.format(layer_name, 'self_attention')),
                             self.hparams),
                BlockWrapper(Feedforward(4 * self.hparams.embed_size,
                                         self.hparams.dropout_rate,
                                         name='{}_{}'.format(layer_name, 'feedforward')),
                             self.hparams)
                        ))
        self.output_norm = Layer_norm()
        super().build(input_shape)

    def call(self, inputs, self_attn_bias, training=False):
        outputs = self.embedding_layer(inputs)
        outputs = outputs + positional_encoding(tf.shape(inputs)[1], self.hparams.embed_size)
        outputs = tf.layers.dropout(outputs, self.hparams.dropout_rate, training=training)

        for self_attn, ff in self.blocks:
            outputs = self_attn(outputs, self_attn_bias, training=training)
            outputs = ff(outputs, training=training)

        return self.output_norm(outputs)

class Decoder(tf.layers.Layer):
    def __init__(self, hparams, embedding_layer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.embedding_layer = embedding_layer

    def build(self, input_shape):
        if self.embedding_layer is None:
            self.embedding_layer = Embedding_layer(self.hparams.vocab_size, self.hparams.embed_size)
        else:
            # if the embedding layer is owned by another Layer it must be built until now
            # in order to avoid ambiguous variable scope tree
            assert self.embedding_layer.built
        
        self.blocks = []
        for i in range(self.hparams.n_blocks):
            layer_name = 'layer_{}'.format(i)
            self.blocks.append((
                BlockWrapper(SelfAttention(self.hparams.attention_size,
                                                 self.hparams.n_heads,
                                                 self.hparams.dropout_rate,
                                                 name='{}_{}'.format(layer_name, 'self_attention')),
                             self.hparams),
                BlockWrapper(Multihead_attention(self.hparams.attention_size,
                                                 self.hparams.n_heads,
                                                 self.hparams.dropout_rate,
                                                 name='{}_{}'.format(layer_name, 'context_attention')),
                             self.hparams),
                BlockWrapper(Feedforward(self.hparams.embed_size * 4,
                                         self.hparams.dropout_rate,
                                         name='{}_{}'.format(layer_name, 'feedforward')),
                             self.hparams)
            ))

        self.output_norm = Layer_norm()

        super().build(input_shape)

    def call(self, inputs, enc_outputs, self_attn_bias, ctx_attn_bias, training=False, cache=None):
        if cache is None:
            seq_start = 0
            seq_end = tf.shape(inputs)[1]
        else:
            cache_l0_v = cache['layer_0']['v']
            seq_start = tf.shape(cache_l0_v)[1]
            seq_end = seq_start + tf.shape(inputs)[1]

        outputs = self.embedding_layer(inputs)
        outputs = outputs + positional_encoding(seq_end, self.hparams.embed_size)[seq_start:]
        outputs = tf.layers.dropout(outputs, self.hparams.dropout_rate, training=training)

        for i, (self_attn, ctx_attn, ff) in enumerate(self.blocks):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache[layer_name] if cache is not None else None

            outputs = self_attn(outputs, self_attn_bias, training=training, cache=layer_cache)
            outputs = ctx_attn(outputs, enc_outputs, ctx_attn_bias, training=training)
            outputs = ff(outputs, training=training)
        
        outputs = self.output_norm(outputs)
        outputs = self.embedding_layer.emb2logits(outputs)
        return outputs


class Transformer(tf.layers.Layer):
    def __init__(self, hparams, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams = hparams
        self.config = config

    def build(self, input_shape):
        self.encoder = Encoder(self.hparams, name='encoder')
        if self.hparams.share_embedding: 
            self.decoder = Decoder(self.hparams, self.encoder.embedding_layer, name='decoder')
        else:
            self.decoder = Decoder(self.hparams, name='decoder')
        
        super().build(input_shape)

    def call(self, inputs, lengths, dec_inputs, dec_lengths, training=False):
        # this method is called only by self.instantiate to instantiate variables of this Layer
        with tf.name_scope('enc_self_attn_bias'):
            enc_self_attn_bias = make_attention_bias_from_seq_mask(lengths, tf.shape(inputs)[1])
        enc_outputs = self.encoder(inputs, enc_self_attn_bias, training=training)

        with tf.name_scope('dec_self_attn_bias'):
            dec_self_attn_bias = make_attention_bias_triangle(tf.shape(dec_inputs)[1])
        dec_ctx_attn_bias = enc_self_attn_bias
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_self_attn_bias, dec_ctx_attn_bias, training=training, cache=None)
        return dec_outputs

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

    def get_logits(self, x, y, x_len, y_len, training=False):
        """Compute logits given inputs for encoder and decoder.
        Args:
            x: inputs for encoder with shape [batch_size, length_enc]
            y: inputs for decoder with shape [batch_size, length_dec]
                `y` is shifted to the right by one step and SOS is added to the beginning
                and the last token is removed. So, `y` should not contain SOS
            x_len: lengths of x with shape [batch_size]
            y_len: lengths of y
        Returns:
            """
        assert self.built
        enc_self_attn_bias = make_attention_bias_from_seq_mask(x_len, tf.shape(x)[1])
        enc_outputs = self.encoder(x, enc_self_attn_bias, training=training)

        dec_self_attn_bias = make_attention_bias_triangle(tf.shape(y)[1])
        dec_ctx_attn_bias = enc_self_attn_bias
        # add SOS to the beginning and remove the last token
        batch_size = tf.shape(y)[0]
        dec_inputs = tf.concat([tf.fill([batch_size, 1], self.config.SOS_ID), y[:, :-1]], axis=1)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_self_attn_bias, dec_ctx_attn_bias, training=training)
        return dec_outputs

    def decode(self, x, x_len, beam_size=8, return_search_results=False, init_y=None, init_y_len=None, sampling_method=None):
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
        with tf.name_scope('enc_self_attn_bias'):
            enc_self_attn_bias = make_attention_bias_from_seq_mask(x_len, tf.shape(x)[1])
        enc_outputs = self.encoder(x, enc_self_attn_bias, training=False)

        # initial cache
        with tf.name_scope('init_cache'):
            cache = {
                'enc_outputs': enc_outputs,
                'dec_ctx_attn_bias': enc_self_attn_bias
                }

            batch_size = tf.shape(x)[0]
            for layer in range(self.hparams.n_blocks):
                layer_name = 'layer_{}'.format(layer)
                with tf.name_scope('cache_{}'.format(layer_name)):
                    cache[layer_name] = {'k': tf.zeros([batch_size, 0, self.hparams.attention_size]),
                                         'v': tf.zeros([batch_size, 0, self.hparams.attention_size])}

        with tf.name_scope('max_length'):
            maxlens = tf.math.minimum(512, x_len * 2 + 5)
            maxlen = tf.math.minimum(512, tf.shape(x)[1] * 2 + 5) # hard coding

        with tf.name_scope('dec_self_attn_bias'):
            dec_self_attn_bias = make_attention_bias_triangle(maxlen)

        with tf.name_scope('define_init_sequence'):
            with tf.name_scope('default_prefix'):
                default_prefix = tf.fill([batch_size, 1], self.config.SOS_ID)
                default_len = tf.ones([batch_size], dtype=tf.int32)
            if init_y is not None:
                assert init_y_len is not None
                # make the prefix sequence
                # if init_target_seq's size is 0, use simple SOS initialization
                with tf.name_scope('custom_prefix'):
                    # Add SOS to the beginning and remove the last token
                    custom_prefix = tf.pad(init_y, [[0,0], [1,0]], constant_values=self.config.SOS_ID)[:, :-1]

                no_context = tf.equal(tf.size(init_y), 0)
                init_dec_inputs = tf.cond(no_context, lambda: default_prefix, lambda: custom_prefix)
                init_dec_inputs_len = tf.cond(no_context, lambda: default_len, lambda: init_y_len)
            else:
                init_dec_inputs = default_prefix
                init_dec_inputs_len = default_len


        def get_logits_fn(dec_inputs, cache):
            cache_l0_v = cache['layer_0']['v']
            seq_start = tf.shape(cache_l0_v)[1]
            seq_end = seq_start + tf.shape(dec_inputs)[1]
            length = tf.shape(dec_inputs)[1]
            """
            decoder self-attention is from dec_inputs (shape [batch, n, emb]) to concat(cache, dec_inputs).
            So the bias matrix used here is of shape [n, length] which is a sub part of the real bias matrix
            for the real self-attention (from concat(cache, dec_inputs) to concat(cache, dec_inputs)):
            RMatrix[start:end, 0:end]

            """
            # DEBUG
            _self_attn_bias = dec_self_attn_bias[:, :, seq_start:seq_end, :seq_end]
            outputs = self.decoder(dec_inputs,
                                   cache['enc_outputs'],
                                   _self_attn_bias,
                                   cache['dec_ctx_attn_bias'],
                                   training=False,
                                   cache=cache)
            return outputs

        # Check if the input is empty.
        # Due to the data parallel execution, this graph may recieve an batch with size 0
        # which leads to undefined behavior and errors in the while_loop.
        with tf.name_scope('check_empty_batch'):
            input_is_empty = tf.equal(batch_size, 0)
            def size1_dummy(batch):
                return tf.ones(tf.concat([[1], tf.shape(batch)[1:]], axis=0), dtype=batch.dtype)
            cache, init_dec_inputs, init_dec_inputs_len = tf.cond(input_is_empty,
                lambda: nest.map_structure(size1_dummy, [cache, init_dec_inputs, init_dec_inputs_len]),
                lambda: [cache, init_dec_inputs, init_dec_inputs_len])
            maxlen = tf.cond(input_is_empty, lambda: 3, lambda: maxlen)
            maxlens = tf.cond(input_is_empty, lambda: tf.constant([3],tf.int32), lambda: maxlens)

        beam_candidates, scores = beam_search_decode(get_logits_fn,
                                                     cache,
                                                     init_dec_inputs,
                                                     init_dec_inputs_len,
                                                     beam_size,
                                                     maxlens,
                                                     self.config.EOS_ID,
                                                     self.config.PAD_ID,
                                                     self.hparams.length_penalty_a,
                                                     sampling_method=sampling_method)

        with tf.name_scope('post_check_empty_batch'):
            beam_candidates = tf.cond(input_is_empty, lambda: beam_candidates[:0], lambda: beam_candidates)
            scores = tf.cond(input_is_empty, lambda: scores[:0], lambda: scores)

        if return_search_results:
            return beam_candidates, scores
        else:
            top_indices = tf.math.argmax(scores, axis=1)
            top_seqs = tf.batch_gather(beam_candidates, top_indices)
            return top_seqs


SAMPLING_METHOD_TOPK = 0
SAMPLING_METHOD_SAMPLING = 1
def beam_search_decode(get_logits_fn, init_cache, init_dec_inputs, init_dec_inputs_len, beam_size, maxlens, eos_id, pad_id=0, alpha=1, sampling_method=SAMPLING_METHOD_TOPK):
    """
    Args:
        get_logits_fn: produces logits given decoder inputs and cached inputs
            Args:
                dec_inputs: a Tensor of tf.int32 of shape [batch, 1]
                cache: a dictionary of Tensor's of shape [batch, ..., embed_size]
            Returns:
                logits Tensor of shape [batch, 1, vocab_size]

        init_cache: The initial cache. Each element is of shape [batch_size, ..., embed_size]
        beam_size: int value indicating the beam window width
        maxlens: The maximum length sequences can be. [batch_size]
        eos_id: EOS token ID.
        pad_id: PAD token ID which defaults to 0
        sos_id: Start of sequence ID. It's not necessary when `init_dec_inputs` is specified.
        alpha: Parameter for length normalization (length penalty)
        init_dec_inputs: If None, SOS is used as the first inputs to decoder. Its shape must be [batch_size, 1]
        sampling_method: SAMPLING_METHOD_TOPK or SAMPLING_METHOD_SAMPLING. The former is the
            normal beam search. The latter samples the next token from the categorical
            distribution, in which case the specified `beam_size` is ignored and the beam
            search is performed with beam size 1.
    Returns:
        Beam candidates with shape [batch_size, beam_size, length] and
        beam scores with shape [batch_size, beam_size]


    loop variables
        cache: each element has a shape of [batch_size, batch_size, ...]
        generated_seq: [batch_size, batch_size, None] tf.int32
        log probability of sequence: [batch_size, batch_size] tf.float32
        has_eos: [batch_size, beam_size] tf.bool

        cache, generated_seq, seq_log_prob, has_eos, score

        """
    NEG_INF = -1e9

    if sampling_method == SAMPLING_METHOD_SAMPLING:
        beam_size = 1

    maxlen = tf.reduce_max(maxlens)

    if eos_id is None:
        eos_id = pad_id
    
    with tf.name_scope('batch_size'):
        batch_size = tf.shape(nest.flatten(init_cache)[0])[0]

    def length_penalty(length, alpha):
        return tf.cast(tf.pow((5 + length)/(1 + 5), alpha), dtype=tf.float32)

    def get_shape_keep_last_dim(x):
        orig_shape = x.shape.as_list()
        shape = [None] * len(orig_shape)
        shape[-1] = orig_shape[-1]
        return tf.TensorShape(shape)

    def flatten(batch):
        # [batch_size, b, ...] -> [batch_size*b, ...]
        shape_before = tf.shape(batch)
        shape_after = tf.concat([[shape_before[0] * shape_before[1]], tf.shape(batch)[2:]], axis=0)
        return tf.reshape(batch, shape_after)

    def pack(flat_batch):
        # [batch_size*b, ...] -> [batch_size, b, ...]
        shape_after = tf.concat([[batch_size, beam_size], tf.shape(flat_batch)[1:]], axis=0)
        return tf.reshape(flat_batch, shape_after)

    def fork(batch):
        # [batch_size, b, ...] -> [batch_size, b*beam_size, ...]
        shape_before = tf.shape(batch)
        target_shape = tf.concat([shape_before[:1], shape_before[1:2] * beam_size, shape_before[2:]], axis=0)
        batch = tf.expand_dims(batch, axis=2) # [batch_size, b, 1, ...]
        tile = [1] * len(batch.shape.as_list())
        tile[2] = beam_size
        batch = tf.tile(batch, tile) # [batch_size, b, beam_size, ...]
        batch = tf.reshape(batch, target_shape) # [batch_size, b*beam_size, ...]
        
        return batch

    def cond_fn(loop_vars):
        some_not_ended = tf.logical_not(tf.reduce_all(loop_vars['has_eos']), name='loop_condition')
        shorter_than_maxlen = tf.less(tf.shape(loop_vars['generated_seq'])[2] + dec_inputs_prefix_len - 1, maxlen)
        return tf.logical_and(some_not_ended, shorter_than_maxlen)

    def body_fn(loop_vars):

        with tf.name_scope('loop_body'):
            # The position of the token predicted in this iteration. Starts from 0
            predicting_pos = tf.shape(loop_vars['generated_seq'])[2] + dec_inputs_prefix_len - 1

            # flatten cache and dec_inputs
            with tf.name_scope('flatten_inputs'):
                flat_cache = nest.map_structure(flatten, loop_vars['cache']) # [batch_size*beam_size, ...]
                flat_dec_inputs = flatten(loop_vars['dec_inputs']) # [batch_size*beam_size, length]

            # get the next logits
            # flat_cache is updated in `get_logits_fn`
            with tf.name_scope('get_logits_and_update_layer_cache'):
                # Note: the outputs' length can be greater than 1 because of the initial target-side context
                # so take THE LAST LOGIT
                logits = get_logits_fn(flat_dec_inputs, flat_cache)[:,-1:] # [batch_size*b, 1, vocab_size]

            # restore shape of cache and update
            with tf.name_scope('update_and_restore_structure_of_cache'):
                loop_vars['cache'] = nest.map_structure(pack, flat_cache)

            with tf.name_scope('preliminary_top_ids_and_log_probs'):
                # get the top k=beam_size for each sequence
                # [batch_size*b, 1, beam_size]
                if sampling_method is None or sampling_method == SAMPLING_METHOD_TOPK:
                    top_logits, ids = tf.math.top_k(logits, beam_size, False, name='preliminary_tops') 
                else:
                    assert sampling_method == SAMPLING_METHOD_SAMPLING
                    _logits = logits[:, 0] # [batch_size * b, vocab_size]
                    ids = tf.random.multinomial(_logits, 1) # [batch_size*b, beam_size=1]
                    ids = tf.cast(ids, tf.int32)
                    ids = tf.expand_dims(ids, axis=-1) # [batch_size*b, 1, beam_size=1]
                    top_logits = tf.batch_gather(logits, ids) # [batch_size*b, 1, beam_size=1]

                # get the log probabilities
                with tf.name_scope('logits_to_log_prob'):
                    #[batch_size*b, 1, beam_size] 
                    log_prob = top_logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True) 
                # restore shape of log_prob and ids into forked style (parent nodes -> child nodes)
                with tf.name_scope('restore_shape'):
                    log_prob = tf.reshape(log_prob, [batch_size, beam_size ** 2]) # [batch_size, beam_size^2]
                    ids = tf.reshape(ids, [batch_size, beam_size ** 2, 1]) # [batch_size, beam_size^2, 1]
                with tf.name_scope('check_max_length'):
                    exceeding_maxlens = tf.greater_equal(predicting_pos, maxlens) #[batch]
                    log_prob = tf.where(
                        tf.broadcast_to(exceeding_maxlens[:,None], tf.shape(log_prob)),
                        tf.broadcast_to([[NEG_INF]], tf.shape(log_prob)),
                        log_prob
                        )
                    ids = tf.where(
                        tf.broadcast_to(exceeding_maxlens[:,None,None], tf.shape(ids)),
                        tf.broadcast_to([[[eos_id]]], tf.shape(ids)),
                        ids
                            )

            # fork tensors. tile and reshape tensors into the shape [batch_size, beam_size^2, ...]
            # except 'dec_inputs'
            with tf.name_scope('fork_state_vars'):
                forked_vars = nest.map_structure(fork, {k:v for k,v in loop_vars.items() if k != 'dec_inputs'})

            # calculate updated log probabilities and sequences and scores
            with tf.name_scope('update_sequence'):
                # [batch_size, beam_size**2, 1]
                is_init_seq = tf.expand_dims(tf.less(predicting_pos, forked_init_dec_inputs_len - 1), axis=-1)

                # the (predicting_pos + 1)-th tokens
                # Ensure that its shape is [batch_size, beam_size^2, 1]
                forked_given_next_tokens = forked_init_dec_inputs[:, :, predicting_pos + 1:predicting_pos + 2]
                forked_given_next_tokens = tf.concat([forked_given_next_tokens,
                    tf.zeros([batch_size, beam_size**2, 1 - tf.shape(forked_given_next_tokens)[2]],
                             dtype=tf.int32)],
                    axis=2)
                # [batch_size, beam_size**2, 1]
                is_ended = tf.expand_dims(forked_vars['has_eos'], axis=-1)
                forked_vars['generated_seq'] = tf.concat([
                    forked_vars['generated_seq'],
                    tf.where(
                        is_init_seq,
                        forked_given_next_tokens,
                        tf.where(
                            is_ended,
                            tf.ones_like(ids, dtype=tf.int32) * pad_id,
                            ids)
                    )
                ], axis=2, name='updated_sequence')
                    

            with tf.name_scope('update_log_prob'):
                # seq_log_prob: [batch_size, beam_size^2]
                forked_vars['seq_log_prob'] = tf.where(
                    forked_vars['has_eos'],
                    forked_vars['seq_log_prob'] + table_of_log_prob_to_be_added_to_eos_beam,
                    forked_vars['seq_log_prob'] + log_prob, name='new_log_prob')

            with tf.name_scope('update_scores'):
                forked_vars['score'] = tf.where(forked_vars['has_eos'],
                                                forked_vars['score'] + table_of_log_prob_to_be_added_to_eos_beam,
                                                forked_vars['seq_log_prob'] / length_penalty(tf.shape(forked_vars['generated_seq'])[2], alpha), name='new_scores')

            # update has_eos
            with tf.name_scope('update_eos'):
                is_init_seq = tf.less(predicting_pos, forked_init_dec_inputs_len - 1)
                forked_vars['has_eos'] = tf.where(
                    is_init_seq,
                    tf.fill([batch_size, beam_size**2], tf.constant(False, dtype=tf.bool)),
                    tf.math.logical_or(forked_vars['has_eos'],
                                        tf.equal(eos_id, tf.reshape(ids, [batch_size, beam_size**2])),
                                        name='new_has_eos')
                )
                

            # take top k=beam_size
            with tf.name_scope('choose_top_candidates'):
                top_scores, top_indices = tf.math.top_k(forked_vars['score'], beam_size, True, name='alive_candidates') # [batch_size, beam_size]

                new_vars = nest.map_structure(lambda x:tf.batch_gather(x, top_indices), forked_vars)

            # update dec_inputs
            with tf.name_scope('update_dec_inputs'):
                # [batch_size, beam_size, 1]
                new_vars['dec_inputs'] = new_vars['generated_seq'][:, :, -1:]

        return new_vars


    # initial decoder inputs
    with tf.name_scope('init_dec_inputs'):
        # reshape from [batch_size, length] to [batch_size, 1, length]
        # and tile into [batch_size, beam_size, length]
        init_dec_inputs = tf.tile(tf.expand_dims(init_dec_inputs, axis=1), [1, beam_size, 1])
        # reshape from [batch_size] to [batch_size, 1] and tile into [batch_size, beam_size]
        init_dec_inputs_len = tf.tile(tf.expand_dims(init_dec_inputs_len, axis=1), [1, beam_size])
        with tf.name_scope('forked'):
            forked_init_dec_inputs = fork(init_dec_inputs) # [batch_size, beam_size**2, length]
            forked_init_dec_inputs_len = fork(init_dec_inputs_len) # [batch_size, beam_size**2]

    with tf.name_scope('dec_inputs_prefix'):
        dec_inputs_prefix_len = tf.math.reduce_min(init_dec_inputs_len)
        dec_inputs_prefix = init_dec_inputs[:, :, :dec_inputs_prefix_len] # [batch_size, beam_size, prefix_len]

    with tf.name_scope('table_of_log_prob_to_be_added_to_eos_beams'):
        # [batch_size, beam_size^2]
        table_of_log_prob_to_be_added_to_eos_beam = tf.tile(
            tf.concat([tf.zeros([batch_size, 1]), tf.fill([batch_size, beam_size - 1], NEG_INF)], axis=1),
            [1, beam_size],
            name='table_of_log_prob_to_be_added_to_eos_beams')

    #cache, generated_seq, seq_log_prob, has_eos, score, dec_inputs
    with tf.name_scope('init_loop_vars'):
        init_loop_vars = {
            # Reshape and tile each element in cache from [batch_size, ...] to [batch_size, beam_size, ...]
            'cache': nest.map_structure(lambda x: fork(tf.expand_dims(x, 1)), init_cache),
            'generated_seq': tf.zeros([batch_size, beam_size, 0], dtype=tf.int32),
            # Only one beam for a sample has log probability of 0 and the rest are negative infinity
            'seq_log_prob': tf.concat(
                [tf.zeros([batch_size, 1]), tf.fill([batch_size, beam_size - 1], NEG_INF)], axis=1),
            'has_eos': tf.zeros([batch_size, beam_size], dtype=tf.bool),
            # Only one beam for a sample has log probability of 0 and the rest are negative infinity
            'score': tf.concat(
                [tf.zeros([batch_size, 1]), tf.fill([batch_size, beam_size - 1], NEG_INF)], axis=1),
            'dec_inputs': dec_inputs_prefix
        }


    # shape invariants
    with tf.name_scope('shape_invariants'):
        shape_invariants = {
            'cache': nest.map_structure(get_shape_keep_last_dim, init_loop_vars['cache']),
            'generated_seq': tf.TensorShape([None, None, None]),
            'seq_log_prob': tf.TensorShape([None, None]),
            'has_eos': tf.TensorShape([None, None]),
            'score': tf.TensorShape([None, None]),
            'dec_inputs': tf.TensorShape([None, None, None])
        }

    with tf.name_scope('while_loop'):
        finish_state = tf.while_loop(
            cond_fn,
            body_fn,
            [init_loop_vars],
            shape_invariants,
            back_prop=False,
            maximum_iterations=maxlen,
            parallel_iterations=1
            )


    with tf.name_scope('post_processing'):
        # non-finished sequences get very low score
        finish_state['seq_log_prob'] = tf.where(finish_state['has_eos'],
                                                finish_state['seq_log_prob'],
                                                tf.fill(tf.shape(finish_state['seq_log_prob']), -1e9))
        finish_state['score'] = tf.where(finish_state['has_eos'],
                                         finish_state['score'],
                                         tf.fill(tf.shape(finish_state['score']), -1e9))

        # add EOS at the end of all unfinished sequences
        finish_state['generated_seq'] = tf.concat([
                finish_state['generated_seq'][:,:,:-1],
                tf.fill(tf.concat([tf.shape(finish_state['generated_seq'])[:-1], [1]], axis=0),
                    eos_id)
            ], axis=2)

        # Fianal sort
        with tf.name_scope('final_sort'):
            score, indices = tf.math.top_k(finish_state['score'], beam_size, sorted=True)
            seq = tf.batch_gather(finish_state['generated_seq'], indices)

        # concat with the prefix and remove the first token (usually <SOS>)
        # [batch_size, beam_size, length]
        with tf.name_scope('concat_prefix'):
            seq = tf.concat([dec_inputs_prefix, seq], axis=-1)[:, :, 1:]

    return seq, score
