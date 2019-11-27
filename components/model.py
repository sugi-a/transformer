import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np

def length_penalty(length, alpha):
    return tf.cast(tf.pow((5 + length)/(1 + 5), alpha), dtype=tf.float32)


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

def make_self_attn_bias(lengths, maxlen):
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

        self.blocks = []
        for i in range(self.params["network"]["n_blocks"]):
            layer_name = 'layer_{}'.format(i)
            self.blocks.append((
                BlockWrapper(SelfAttention(self.params["network"]["attention_size"],
                                                 self.params["network"]["n_heads"],
                                                 self.params["network"]["dropout_rate"],
                                                 name='{}_{}'.format(layer_name, 'self_attention')),
                             self.params),
                BlockWrapper(Feedforward(4 * self.params["network"]["embed_size"],
                                         self.params["network"]["dropout_rate"],
                                         name='{}_{}'.format(layer_name, 'feedforward')),
                             self.params)
                        ))
        self.output_norm = Layer_norm()
        super().build(input_shape)

    def call(self, inputs, self_attn_bias, training=False):
        outputs = self.embedding_layer(inputs)
        outputs = outputs + positional_encoding(tf.shape(inputs)[1], self.params["network"]["embed_size"])
        outputs = tf.layers.dropout(outputs, self.params["network"]["dropout_rate"], training=training)

        for self_attn, ff in self.blocks:
            outputs = self_attn(outputs, self_attn_bias, training=training)
            outputs = ff(outputs, training=training)

        return self.output_norm(outputs)

class Decoder(tf.layers.Layer):
    def __init__(self, params, embedding_layer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params
        self.embedding_layer = embedding_layer

    def build(self, input_shape):
        if self.embedding_layer is None:
            self.embedding_layer = Embedding_layer(self.params["vocab"]["vocab_size"], self.params["network"]["embed_size"])
        else:
            # if the embedding layer is owned by another Layer it must be built until now
            # in order to avoid ambiguous variable scope tree
            assert self.embedding_layer.built
        
        self.blocks = []
        for i in range(self.params["network"]["n_blocks"]):
            layer_name = 'layer_{}'.format(i)
            self.blocks.append((
                BlockWrapper(SelfAttention(self.params["network"]["attention_size"],
                                                 self.params["network"]["n_heads"],
                                                 self.params["network"]["dropout_rate"],
                                                 name='{}_{}'.format(layer_name, 'self_attention')),
                             self.params),
                BlockWrapper(Multihead_attention(self.params["network"]["attention_size"],
                                                 self.params["network"]["n_heads"],
                                                 self.params["network"]["dropout_rate"],
                                                 name='{}_{}'.format(layer_name, 'context_attention')),
                             self.params),
                BlockWrapper(Feedforward(self.params["network"]["embed_size"] * 4,
                                         self.params["network"]["dropout_rate"],
                                         name='{}_{}'.format(layer_name, 'feedforward')),
                             self.params)
            ))

        self.output_norm = Layer_norm()

        super().build(input_shape)

    def call(self, inputs, self_attn_bias, cache, training=False):
        """`cache` must contain enc_outputs and ctx_attn_bias"""
        if 'layer_0' in cache:
            cache_l0_v = cache['layer_0']['v']
            seq_start = tf.shape(cache_l0_v)[1]
            seq_end = seq_start + tf.shape(inputs)[1]
        else:
            seq_start = 0
            seq_end = tf.shape(inputs)[1]

        # Decoder embedding
        outputs = self.embedding_layer(inputs)

        # Positional encoding. Take t >= current-front-position
        outputs = outputs + positional_encoding(
            seq_end, self.params["network"]["embed_size"])[seq_start:]

        # Dropout
        outputs = tf.layers.dropout(
            outputs, self.params["network"]["dropout_rate"], training=training)

        # Decoder blocks
        for i, (self_attn, ctx_attn, ff) in enumerate(self.blocks):
            layer_name = 'layer_{}'.format(i)
            layer_cache = cache.get(layer_name, None)

            outputs = self_attn(outputs, self_attn_bias, training=training, cache=layer_cache)
            outputs = ctx_attn(outputs, cache["enc_outputs"], cache["ctx_attn_bias"], training=training)
            outputs = ff(outputs, training=training)
        
        outputs = self.output_norm(outputs)
        outputs = self.embedding_layer.emb2logits(outputs)
        return outputs


    def make_cache(self, enc_outputs, ctx_attn_bias, layer_cache=False):
        cache = {
            'enc_outputs': enc_outputs,
            'ctx_attn_bias': ctx_attn_bias
        }

        if layer_cache:
            batch_size = tf.shape(enc_outputs)[0]
            for layer in range(self.params["network"]["n_blocks"]):
                layer_name = 'layer_{}'.format(layer)
                with tf.name_scope('cache_{}'.format(layer_name)):
                    cache[layer_name] = {
                        'k': tf.zeros([batch_size, 0, self.params["network"]["attention_size"]]),
                        'v': tf.zeros([batch_size, 0, self.params["network"]["attention_size"]])}
        return cache




class Transformer(tf.layers.Layer):
    MAXLEN = 1024
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.params = params

    def build(self, input_shape):
        self.encoder = Encoder(self.params, name='encoder')
        if self.params["network"]["share_embedding"]: 
            self.decoder = Decoder(self.params, self.encoder.embedding_layer, name='decoder')
        else:
            self.decoder = Decoder(self.params, name='decoder')

        self.triangle_bias = make_attention_bias_triangle(Transformer.MAXLEN)
        
        super().build(input_shape)

    def call(self, inputs, lengths, dec_inputs, dec_lengths, training=False):
        # this method is called only by self.instantiate to instantiate variables of this Layer
        with tf.name_scope('enc_self_attn_bias'):
            enc_self_attn_bias = make_self_attn_bias(lengths, tf.shape(inputs)[1])
        enc_outputs = self.encoder(inputs, enc_self_attn_bias, training=training)

        with tf.name_scope('dec_self_attn_bias'):
            dec_self_attn_bias = make_attention_bias_triangle(tf.shape(dec_inputs)[1])

        cache = self.decoder.make_cache(enc_outputs, enc_self_attn_bias)
        dec_outputs = self.decoder(dec_inputs, dec_self_attn_bias, cache, training=training)
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


    def __get_logits_fn(self, dec_inputs, cache):
        if 'layer_0' in cache:
            """
            decoder self-attention is from dec_inputs (shape [batch, n, emb])
            to concat(cache, dec_inputs). So the bias matrix used here is of
            shape [n, length] which is a sub part of the real bias matrix for
            the real self-attention (from concat(cache, dec_inputs) to
            concat(cache, dec_inputs)): RMatrix[start:end, 0:end] 
            """
            l0v_cache = cache['layer_0']['v']
            current_front = tf.shape(l0v_cache)[1]
            current_tail = current_front + tf.shape(dec_inputs)[1]
        else:
            current_front = 0
            current_tail = tf.shape(dec_inputs)[1]

        # Cut out the self-attention bias
        self_attn_bias = self.triangle_bias[:, :, current_front:current_tail, :current_tail]

        # Decoder output
        outputs = self.decoder(dec_inputs, self_attn_bias, cache, training=False)
        return outputs

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
        enc_self_attn_bias = make_self_attn_bias(x_len, tf.shape(x)[1])
        enc_outputs = self.encoder(x, enc_self_attn_bias, training=training)

        #dec_self_attn_bias = make_attention_bias_triangle(tf.shape(y)[1])
        dec_self_attn_bias = self.triangle_bias[:, :, :tf.shape(y)[1], :tf.shape(y)[1]]
        
        # add SOS to the beginning and remove the last token
        dec_inputs = tf.concat(
            [tf.fill([tf.shape(y)[0], 1], self.params["vocab"]["SOS_ID"]), y[:, :-1]], axis=1)

        # Build decoder graph
        cache = self.decoder.make_cache(enc_outputs, enc_self_attn_bias, False)
        dec_outputs = self.decoder(dec_inputs, dec_self_attn_bias, cache, training=training)

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

        # Build encoder graph
        with tf.name_scope('enc_self_attn_bias'):
            enc_self_attn_bias = make_self_attn_bias(x_len, tf.shape(x)[1])
        enc_outputs = self.encoder(x, enc_self_attn_bias, training=False)

        # initial cache
        with tf.name_scope('init_cache'):
            cache = self.decoder.make_cache(enc_outputs, enc_self_attn_bias, True)

        # Initial sequence
        with tf.name_scope('define_init_sequence'):
            init_seq = tf.fill([tf.shape(x)[0], 1], self.params["vocab"]["SOS_ID"])
            if init_y is not None:
                init_seq = tf.concat([init_seq, init_y], axis=1)[:, :-1]
                init_seq_len = init_y_len

        # Check if the input is empty.
        # Due to the data parallel execution, this graph may recieve an batch with size 0
        # which leads to undefined behavior and errors in the while_loop.
        with tf.name_scope('check_empty_batch'):
            zero_batch = tf.equal(tf.shape(x)[0], 0)

            def size1_dummy(batch):
                return tf.ones(tf.concat([[1], tf.shape(batch)[1:]], axis=0), dtype=batch.dtype)

            cache, init_seq, init_seq_len = tf.cond(zero_batch,
                lambda: nest.map_structure(size1_dummy, [cache, init_seq, init_seq_len]),
                lambda: [cache, init_seq, init_seq_len])

        # Maximum target length
        maxlens = tf.minimum(Transformer.MAXLEN - 10, x_len * 3 + 10)

        hypos, scores = beam_search_decode(self.__get_logits_fn,
                                                     cache,
                                                     init_seq,
                                                     init_seq_len,
                                                     beam_size,
                                                     maxlens,
                                                     self.params["vocab"]["EOS_ID"],
                                                     self.params["vocab"]["PAD_ID"],
                                                     self.params["test"]["length_penalty_a"],
                                                     sampling_method=sampling_method)

        with tf.name_scope('post_check_empty_batch'):
            hypos = tf.cond(zero_batch, lambda: hypos[:0], lambda: hypos)
            scores = tf.cond(zero_batch, lambda: scores[:0], lambda: scores)

        if return_search_results:
            return hypos, scores
        else:
            top_indices = tf.math.argmax(scores, axis=1)
            top_seqs = tf.batch_gather(hypos, top_indices)
            return top_seqs


KEY_TOPK = 0
KEY_SAMPLING = 1
def beam_search_decode(get_logits_fn, init_cache, init_seq, init_seq_len, beam_size, maxlens, eos_id, pad_id=0, alpha=1, sampling_method=KEY_TOPK):
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
        sos_id: Start of sequence ID. It's not necessary when `init_seq` is specified.
        alpha: Parameter for length normalization (length penalty)
        init_seq: If None, SOS is used as the first inputs to decoder. Its shape must be [batch_size, 1]
        sampling_method: KEY_TOPK or KEY_SAMPLING. The former is the
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

    maxlen = tf.reduce_max(maxlens)

    if eos_id is None:
        eos_id = pad_id
    
    with tf.name_scope('batch_size'):
        batch_size = tf.shape(nest.flatten(init_cache)[0])[0]

    def get_shape_keep_last_dim(x):
        orig_shape = x.shape.as_list()
        shape = [None] * len(orig_shape)
        shape[-1] = orig_shape[-1]
        return tf.TensorShape(shape)

    def flatten(batch):
        # [batch, n, ...] -> [batch * b, ...]
        shape_before = tf.shape(batch)
        shape_after = tf.concat([[shape_before[0] * shape_before[1]], tf.shape(batch)[2:]], axis=0)
        return tf.reshape(batch, shape_after)

    def pack(flat_batch):
        # [batch * n, ...] -> [batch, n, ...]
        shape_after = tf.concat([[batch_size, beam_size], tf.shape(flat_batch)[1:]], axis=0)
        return tf.reshape(flat_batch, shape_after)

    def fork(batch):
        # [batch, n, ...] -> [batch, n * beam, ...]
        shape_before = tf.shape(batch)
        target_shape = tf.concat([shape_before[:1], shape_before[1:2] * beam_size, shape_before[2:]], axis=0)
        return tf.reshape(fork_inc_dim(batch), target_shape)

    def fork_inc_dim(batch):
        # [bat_size, b, ...] -> [bat_size, b, beam_size, ...]
        batch = tf.expand_dims(batch, axis=2)
        tile = [beam_size if i == 2 else 1 for i in range(len(batch.shape.as_list()))]
        return tf.tile(batch, tile)

    def get_score(log_prob, length):
        return log_prob / length_penalty(length, alpha)

    def cond_fn(loop_vars):
        not_closed = tf.logical_not(tf.reduce_all(loop_vars['has_eos']), name='loop_condition')
        not_long = tf.less(tf.shape(loop_vars['generated_seq'])[2] + com_prefix_len - 1, maxlen)
        return tf.logical_and(not_closed, not_long)

    def body_fn(loop_vars):

        with tf.name_scope('loop_body'):
            # The position of the token predicted in this iteration. Starts from 0
            cur_pos = tf.shape(loop_vars['generated_seq'])[2] + com_prefix_len - 1

            # flatten cache and dec_inputs
            with tf.name_scope('flatten_inputs'):
                # [bat_size, beam_size, ...] -> [batch_size*beam_size, ...]
                flat_cache = nest.map_structure(flatten, loop_vars['cache'])
                # [bat_size, beam_size, length] -> [bat_size * beam_size, length]
                flat_dec_inputs = flatten(loop_vars['dec_inputs'])

            # get the next logits. Layer cache in flat_cache is semi-UPDATED here
            with tf.name_scope('get_logits_and_update_layer_cache'):
                # Note: the outputs' length can be >1 because of the initial target-side context
                # so take THE LAST LOGIT. [bat * beam, out_len, vocab]->[bat * beam, vocab]
                logits = get_logits_fn(flat_dec_inputs, flat_cache)[:,-1] 

            # restore shape of cache. ->[bat_size, beam_size, ...]
            with tf.name_scope('update_and_restore_structure_of_cache'):
                loop_vars['cache'] = nest.map_structure(pack, flat_cache)

            with tf.name_scope('preliminary_top_ids_and_log_probs'):
                # get the top k=beam_size for each sequence.
                # top_logits: [bat * beam, beam], ids: [bat * beam, beam]
                # There are some strategies to choose k=beam words from [bat * beam, vocab]
                if sampling_method is None or sampling_method == KEY_TOPK:
                    # Normal top-k selection
                    top_logits, ids = tf.math.top_k(logits, beam_size, False, name='pre_tops') 
                elif sampling_method == KEY_SAMPLING:
                    # Random sampling based on probability distribution
                    ids = tf.random.multinomial(logits, beam_size) # [bat*beam, beam]
                    ids = tf.cast(ids, tf.int32)[:, None] # [bat*beam, beam]
                    top_logits = tf.batch_gather(logits, ids) # [bat*beam, beam]
                else:
                    assert False

                # get the log probabilities ->[bat * beam, beam] 
                with tf.name_scope('logits_to_log_prob'):
                    log_prob = top_logits - tf.math.reduce_logsumexp(logits, axis=-1, keepdims=True) 
                # Arrange shape of log_prob and ids
                with tf.name_scope('restore_shape'):
                    # log prob. [bat * beam, beam]->[bat, old_beam * new_beam]
                    log_prob = tf.reshape(log_prob, [batch_size, beam_size ** 2]) 

                    # IDs [bat * beam, beam]->[bat, old_beam * new_beam]
                    ids = tf.reshape(ids, [batch_size, beam_size ** 2])

                # Sequence score
                with tf.name_scope('seq_score'):
                    # Fork log probability of sequences. [bat_size, beam * beam]. 
                    forked_seqp = fork(loop_vars['seq_log_prob'])
                    forked_score = fork(loop_vars['score'])

                    # Fork the info of closed paths. [bat, beam]->[bat, beam * beam]
                    forked_ended = fork(loop_vars['has_eos'])

                    # Update sequence log probability [bat, old_beam * new_beam]
                    forked_seqp = forked_seqp + tf.where(forked_ended, eos_mask, log_prob)

                    # Update sequence score [bat, old_beam * new_beam]
                    forked_score = tf.where(forked_ended, forked_score + eos_mask,
                        get_score(forked_seqp, cur_pos + 1))

            with tf.name_scope('get_top_k'):
                # Top k=beam [bat, beam]
                top_score, top_ind = tf.math.top_k(forked_score, beam_size, False)

                # In this top-k selection, you choose top-k=beam paths out of beam^2 paths,
                # which are the new preliminary top paths (PTP).
                # Old beam indicator [bat, old_beam * new_beam] maps an index in PTP to
                # the index of the path's old beam.
                old_beam_i = tf.range(beam_size)[None, :, None]
                old_beam_i = tf.tile(old_beam_i, [batch_size, 1, beam_size])
                old_beam_i = tf.reshape(old_beam_i, [batch_size, beam_size * beam_size])

                # old beam indices [bat, beam]
                old_beam_ind = tf.batch_gather(old_beam_i, top_ind)
                
            with tf.name_scope('update_loop_vars'):
                new_vars = {}

                # UPDATE cache [bat, beam, ...]
                new_vars['cache'] = nest.map_structure(lambda x:
                    tf.batch_gather(x, old_beam_ind), loop_vars['cache'])

                # UPDATE log_probs [bat, beam]
                new_vars['seq_log_prob'] = tf.batch_gather(forked_seqp, top_ind)

                # UPDATE score [bat, beam]
                new_vars['score'] = top_score
                
                # UPDATE `generated_seq`
                # Choosing old branch. [bat, beam, ...]->[bat, beam, len]
                gen_seq = tf.batch_gather(loop_vars["generated_seq"], old_beam_ind)

                # Some sequences can be still in their initial sequence [batch_size]
                is_in_init = tf.less(cur_pos, init_seq_len - 1)
                # Init token at the current position: (pred_pos + 1)-th token. This can be null
                # ([bat, beam, 0]). So ensure shape is [bat, beam, 1] and reshape [bat, beam]
                cur_init_tok = tf.pad(
                    init_seq[:, :, cur_pos+1: cur_pos+2], [[0,0],[0,0],[0,1]])[:, :, 0]

                # If the path is already closed by EOS, new tokens should be PAD. [bat, beam]
                old_ended = tf.batch_gather(loop_vars['has_eos'], old_beam_ind)
                pad_tok = tf.ones([batch_size, beam_size], tf.int32) * pad_id

                # Predicted tokens
                pred = tf.batch_gather(ids, top_ind) # [batch, beam]

                # New token determined out of (init-sequence, padding, predicted) [bat, beam]
                new_tok = tf.where(is_in_init, cur_init_tok, tf.where(old_ended, pad_tok, pred))

                # Append new token. [batch, beam, len]->[batch, beam, len+1]
                new_vars['generated_seq'] = tf.concat([gen_seq, new_tok[:,:, None]], axis=-1)

                # UPDATE dec_inputs. (token input in the next step) [bat, beam, len=1]
                new_vars['dec_inputs'] = new_tok[:, :, None]

                # UPDATE has_eos [batch, beam]
                new_vars['has_eos'] = tf.logical_or(old_ended, tf.equal(new_tok, eos_id))


        return new_vars


    # Initial decoder inputs. Add a beam dimension and replicate along it.
    with tf.name_scope('init_seq_beam_replication'):
        # sequence [bat, len]->[bat, beam, len]
        init_seq = tf.tile(init_seq[:, None], [1, beam_size, 1])

        # length [batch]->[batch, beam]
        init_seq_len = tf.tile(init_seq_len[:, None], [1, beam_size])

    # Initial sequences can differ in length. I denote the common prefix part `common_prefix`
    with tf.name_scope('common_prefix'):
        com_prefix_len = tf.math.reduce_min(init_seq_len)
        common_prefix = init_seq[:, :, :com_prefix_len] # [batch, beam, com_prefix_len]

    # Log probability bias to prevent closed paths from forking
    with tf.name_scope('eos_log_prob_mask'):
        # [batch_size, beam_size^2]
        eos_mask = tf.tile(tf.concat([[0], tf.fill([beam_size - 1], NEG_INF)], axis=0)[None],
            [batch_size, beam_size])

    # Loop variables: cache, generated_seq, seq_log_prob, has_eos, score, dec_inputs
    # Add a beam dim and copy along it. [batch_size, ...] to [batch_size, beam_size, ...]
    with tf.name_scope('init_loop_vars'):
        init_loop_vars = {
            'cache': nest.map_structure(lambda x: fork(x[:, None]), init_cache),
            'generated_seq': tf.zeros([batch_size, beam_size, 0], dtype=tf.int32),
            # Only one beam has log probability of 0 and the rest have negative infinity
            'seq_log_prob': tf.concat([tf.zeros([batch_size, 1]),
                tf.fill([batch_size, beam_size - 1], NEG_INF)], axis=1),
            # Only one beam has log probability of 0 and the rest have  negative infinity
            'score': tf.concat([tf.zeros([batch_size, 1]),
                tf.fill([batch_size, beam_size - 1], NEG_INF)], axis=1),
            'has_eos': tf.zeros([batch_size, beam_size], dtype=tf.bool),
            'dec_inputs': common_prefix
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
            parallel_iterations=2
            )


    with tf.name_scope('post_processing'):
        # non-finished sequences get very low score
        finish_state['seq_log_prob'] = tf.where(finish_state['has_eos'],
                                                finish_state['seq_log_prob'],
                                                tf.fill(tf.shape(finish_state['seq_log_prob']), NEG_INF))
        finish_state['score'] = tf.where(finish_state['has_eos'],
                                         finish_state['score'],
                                         tf.fill(tf.shape(finish_state['score']), NEG_INF))

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
            seq = tf.concat([common_prefix, seq], axis=-1)[:, :, 1:]

    return seq, score
