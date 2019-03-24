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
        pos = tf.range(length, dtype=tf.float32)
        half = emb_size // 2
        i = tf.range(half, dtype=tf.float32)
        scaled_time = (
            tf.expand_dims(pos, axis=1) /
            tf.expand_dims(tf.pow(10000, i / half), axis=0)
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
        self.causality = causality
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.q_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='q_layer')
        self.k_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='k_layer')
        self.v_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='v_layer')
        self.att_out_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='attention_output')
        super().build(input_shape)

    def call(self, query, dictionary, mask, training=False, cache=None):
        head_size = self.hidden_size // self.n_heads
        
        q = self.q_layer(query) # [batch, length, emb_size]
        k = self.k_layer(dictionary)
        v = self.v_layer(dictionary)

        if cache is not None:
            k = tf.concat([cache['k'], v], axis=1)
            v = tf.concat([cache['v'], v], axis=1)
            cache['k'] = k
            cache['v'] = v

        q = tf.stack(tf.split(q, self.n_heads, axis=-1), axis=1) # [batch, nheads, length_q, head_size]
        k = tf.stack(tf.split(k, self.n_heads, axis=-1), axis=1) # [batch, nheads, length_k, head_size]
        v = tf.stack(tf.split(v, self.n_heads, axis=-1), axis=1) # [batch, nheads, length_k, head_size]

        weight = tf.matmul(q, k, transpose_b=True) # [batch, nheads, length_q, length_k]
        weight = weight / (head_size ** 0.5)

        weight = weight + bias
        weight = tf.nn.softmax(weight, name='attention_weight')

        weight = tf.layers.dropout(weight, self.dropout_rate, training=training)

        outputs = tf.matmul(weight, v) # [batch, nheads, length_q, head_size]

        outputs = tf.concat(tf.split(outputs, self.n_heads, axis=1), axis=3)[:,1] # [batch, length_q, emb_size]

        outputs = self.att_out_layer(outputs)

        return outputs

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
    return (1 - eps) * labels + (eps/tf.shape(labels)[-1])

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

    def build(self, input_shape):
        self.embedding_layer = Embedding_layer(self.hparams.vocab_size, self.hparams.embed_size)

        self.blocks = []
        for i in range(self.hparams.n_blocks):
            layer_name = 'layer_{}'.format(i)
            self.blocks.append((
                BlockWrapper(Multihead_attention(self.hparams.attention_size,
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
            outputs = self_attn(outputs, outputs, self_attn_bias, training=training)
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
        else
            # if the embedding layer is owned by another Layer it must be built until now
            # in order to avoid ambiguous variable scope tree
            assert self.embedding_layer.built
        
        self.blocks = []
        for i in range(self.hparams.n_blocks):
            layer_name = 'layer_{}'.format(i)
            self.blocks.append((
                BlockWrapper(Multihead_attention(self.hparams.attention_size,
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
        outputs = self.embedding_layer(inputs)
        outputs = outputs + positional_encoding(tf.shape(inputs)[1], self.hparams.emb_size)
        outputs = tf.layers.dropout(outputs, self.hparams.dropout_rate, training=training)

        self_attn_bias = make_attention_bias_triangle(tf.shape(inputs)[1])

        if cache is not None and cache.get('layer_0') is None:
            # initialize cache
            batch_size = tf.shape(inputs)[0]
            for layer in range(self.blocks):
                layer_name = 'layer_{}'.format(layer)
                cache[layer_name] = {'k': tf.zeros([batch_size, 0, self.hparams.attention_size]),
                                     'v': tf.zeros([batch_size, 0, self.hparams.attention_size])}

        for i, (self_attn, ctx_attn, ff) in enumerate(self.blocks):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache[layer_name] if cache is not None else None

            outputs = self_attn(outputs, outputs, self_attn_bias, training=training, cache=layer_cache)
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
        enc_self_attn_bias = make_attention_bias_from_seq_mask(lengths, tf.shape(inputs)[1])
        enc_outputs = self.encoder(inputs, enc_self_attn_bias, training=training)

        dec_self_attn_bias = make_attention_bias_triangle(tf.shape(dec_inputs)[1])
        dec_ctx_attn_bias = enc_self_attn_bias
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_self_attn_bias, dec_ctx_attn_bias, training=training, cache=None)
        return dec_outputs

    def instanciate_vars(self):
        """create dummy graph instance of this Layer in order to place variables in a specific device."""
        if self.built:
            return
        x = tf.zeros([2,2], dtype=tf.int32)
        x_len = tf.ones([2], dtype=tf.int32)
        self(x, x_len, x, x_len)

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
        enc_outputs = self.encoder(x, x_len, training=training)

        dec_self_attn_bias = make_attention_bias_triangle(tf.shape(y)[1])
        dec_ctx_attn_bias = enc_self_attn_bias
        # add SOS to the beginning and remove the last token
        batch_size = tf.shape(y)[0]
        dec_inputs = tf.concat([tf.fill([batch_size, 1], self.config.SOS_ID), y[:, :-1]], axis=1)
        dec_outputs = self.decoder(dec_inputs, enc_outputs, dec_self_attn_bias, dec_ctx_attn_bias, training=training)
        return dec_outputs

    def decode(self, x, x_len, beam_size=8, return_search_results=False):
        """Given inputs x, this method produces translation candidates by beam search
        and return the all results if `return_search_results` is True, or the sequence with the MAP otherwise.
        Args:
            x: Source sequence. tf.int32 Tensor of shape [batch, length]
            x_len: lengths of x. tf.int32, [batch]
            return_search_results: Boolean indicating whether to return the whole results or the MAP only.
        Returns:
            If `return_search_results` is True, a tuple of Tensor, ([batch, beam_size, length],
            [batch, beam_size]) is returned. Otherwise a Tensor [batch, length] is returned."""

        assert self.built
        enc_self_attn_bias = make_attention_bias_from_seq_mask(x_len, tf.shape(x)[1])
        enc_outputs = self.encoder(x, x_len, training=False)

        cache = {
            'enc_outputs': enc_outputs,
            'dec_ctx_attn_bias': enc_self_attn_bias
            }
        maxlen = tf.shape(x)[1] * 3 # hard coding
        dec_self_attn_bias = make_attention_bias_triangle(maxlen)

        def get_logits_fn(dec_inputs, cache):
            length = tf.shape(dec_inputs)[1]
            outputs = self.decoder(dec_inputs,
                                   cache['enc_outputs'],
                                   dec_self_attn_bias[:,:,:length,:length],
                                   cache['dec_ctx_attn_bias'],
                                   training=False,
                                   cache=cache)
        beam_candidates, scores = beam_search_decode(get_logits_fn,
                                                     cache,
                                                     beam_size,
                                                     maxlen,
                                                     self.config.EOS_ID,
                                                     self.config.PAD_ID,
                                                     self.config.SOS_ID,
                                                     self.hparams.length_penalty_a)

        if return_search_results:
            return beam_candidates, scores
        else:
            top_indices = tf.math.argmax(scores, axis=1)
            top_seqs = tf.batch_gather(beam_candidates, top_indices)
            return top_seqs


def beam_search_decode(get_logits_fn, init_cache, beam_size, maxlen, eos_id, pad_id=0, sos_id=None, alpha=1, init_dec_inputs=None):
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
        maxlen: The maximum length sequences can be
        eos_id: EOS token ID.
        pad_id: PAD token ID which defaults to 0
        sos_id: Start of sequence ID. It's not necessary when `init_dec_inputs` is specified.
        alpha: Parameter for length normalization (length penalty)
        init_dec_inputs: If None, SOS is used as the first inputs to decoder. Its shape must be [batch_size, 1]
    Returns:
        Beam candidates with shape [batch_size, beam_size, length] and
        beam scores with shape [batch_size, beam_size]


    loop variables
        cache: each element has a shape of [batch_size, b, ...]
        generated_seq: [batch_size, b, None] tf.int32
        log probability of sequence: [batch_size, b] tf.float32
        has_eos: [batch_size, b] tf.bool
        b: 1 in the first iteration, beam_size otherwise


        cache, generated_seq, seq_log_prob, has_eos, score

        """
    
    batch_size = tf.shape(nest.flatten(init_cache)[0])[0]

    def length_penalty(length, alpha):
        return tf.pow((5 + length)/(1 + 5), alpha)

    def get_shape_keep_last_dim(x):
        orig_shape = x.shape.as_list()
        shape = [None] * len(orig_shape)
        shape[-1] = orig_shape[-1]
        return tf.TensorShape(shape)

    def flatten(batch):
        shape_after = tf.concat([[-1], tf.shape(batch)[2:]], axis=0)
        return tf.reshape(batch, shape_after)

    def fork(batch):
        target_shape = batch.shape.as_list() # initial shape: [batch_size, b, ...]
        target_shape[1] = -1 # [batch_size, -1, ...]
        batch = tf.expand_dims(batch, axis=2) # [batch_size, b, 1, ...]
        tile = [1] * len(batch.shape.as_list())
        tile[2] = beam_size
        batch = tf.tile(batch, tile) # [batch_size, b, beam_size, ...]
        batch = tf.reshape(batch, target_shape) # [batch_size, b*beam_size, ...]
        
        return batch

    def cond_fn(loop_vars):
        return tf.reduce_all(loop_vars['has_eos'])

    def body_fn(loop_vars):


        # beam duplication number (1 or beam_size)
        b = tf.shape(loop_varsp['generated_seq'])[1]

        # flatten cache and dec_inputs
        flat_cache = flatten(loop_vars['cache'])
        dec_inputs = tf.cond(tf.equal(0, tf.shape(generated_seq)[2]),
                              init_dec_inputs,
                              generated_seq[:, :, -1:])
        flat_dec_inputs = flatten(dec_inputs)

        # get the next logits
        logits = get_logits_fn(flat_dec_inputs, flat_cache) # [batch_size*b, 1, vocab_size]

        # get the top k=beam_size for each sequence
        top_logits, ids = tf.math.top_k(logits, beam_size, False) # [batch_size*b, 1, beam_size]

        # get the log probabilities
        log_prob = top_logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True) #[batch_size*b, 1, beam_size] 
        # restore shape of log_prob and ids into forked style (parent nodes -> child nodes)
        log_prob = tf.reshape(log_prob, [batch_size, b * beam_size]) # [batch_size, b*beam_size]
        ids = tf.reshape(ids, [batch_size, b * beam_size, 1]) # [batch_size, b*beam_size, 1]

        # fork tensors. tile and reshape tensors into the shape [batch_size, b*beam_size, ...]
        forked_vars = nest.map_structure(fork, loop_vars)

        # calculate updated log probabilities and sequences and scores
        forked_vars['generated_seq'] = tf.concat([
            forked_vars['generated_seq'],
            tf.where(tf.expand_dims(forked_vars['has_eos'], axis=-1),
                     tf.ones_like(ids, dtype=tf.int32) * pad_id,
                     ids)
            ], axis=2)
        forked_vars['seq_log_prob'] = tf.where(forked_vars['has_eos'],
                                               forked_vars['seq_log_prob'],
                                               forked_vars['seq_log_prob'] + log_prob)
        forked_vars['score'] = tf.where(forked_vars['has_eos'],
                                        forked_vars['score'],
                                        forked_vars['seq_log_prob'] / length_penalty(tf.shape(forked_vars['generated_seq'])[2], alpha))

        # update has_eos
        forked_vars['has_eos'] = tf.math.logical_or(forked_vars['has_eos'],
                                                    tf.equal(eos_id, tf.reshape(ids, [batch_size, -1])))

        # take top k=beam_size
        top_scores, top_indices = tf.math.top_k(forked_vars['score'], beam_size, True) # [batch_size, beam_size]

        new_vars = nest.map_structure(lambda x:tf.batch_gather(x, top_indices), forked_vars)

        return [new_vars]

    # initial decoder inputs
    if init_dec_inputs is None:
        init_dec_inputs = tf.ones([batch_size, 1, 1], dtype=tf.int32) * sos_id
    #cache, generated_seq, seq_log_prob, has_eos, score
    init_loop_vars = {
        'cache': tf.expand_dims(cache, axis=1),
        'generated_seq': tf.zeros([batch_size, 1, 0]),
        'seq_log_prob': tf.ones([batch_size, 1]),
        'has_eos': tf.zeros([batch_size, 1], dtype=tf.bool),
        'score': tf.zeros([batch_size, 1])

    }

    # shape invariants
    shape_invariants = {
        'cache': nest.map_structure(get_shape_keep_last_dim, init_loop_vars['cache']),
        'generated_seq': tf.TensorShape([None, None, None]),
        'seq_log_prob': tf.TensorShape([None, None]),
        'has_eos': tf.TensorShape(None, None),
        'score': tf.TensorShape([None, None])
    }

    finish_state = tf.while_loop(
        cond_fn,
        body_fn,
        [init_loop_vars],
        [shape_invariants],
        back_prop=False,
        maximum_iterations=maxlen,
        parallel_iterations=1
        )

    # non-finished sequences get very low score
    finish_state['seq_log_prob'] = tf,where(finish_state['has_eos'],
                                            finish_state['seq_log_prob'],
                                            tf.fill(tf.shape(finish_state['seq_log_prob']), -1e9))
    finish_state['score'] = tf.where(finish_state['has_eos'],
                                     finish_state['score'],
                                     tf.fill(tf.shape(finish_state['score']), -1e9))

    # add EOS at the end of all unfinished sequences
    finish_state['generated_seq'] = tf.concat([
            finish_state['generated_seq'][:,:,:-1],
            tf.where(tf.expand_dims(finish_state['has_eos'], axis=-1),
                     finish_state['generated_seq'][:,:,-1:],
                     tf.fill(tf.shape(finish_state['generated_seq'][:,:,-1:]),eos_id)
        ]), axis=2)


    return finish_state['generated_seq'], finish_state['score']
