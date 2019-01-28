import numpy as np
import tensorflow as tf


def layer_norm(input, eps=1e-8, scope="layer_norm", reuse=None):
    """Applies layer normalization and scale and shift over the last axis

    Args:
        inputs: A tensor with 3 dimensions.
            [batch_size, sentence_len, embedding_size]
        epsilon: A floating number. A very small number
            for preventing ZeroDivision Error.
        scope: variable_scope
        reuse:

    Returns:
        A tensor with the same shape and dtype as `inputs`

    """
    with tf.variable_scope(scope, reuse=reuse):
        mean, variance = tf.nn.moments(input, [-1], keep_dims=True)

        params_shape = input.get_shape()[-1:]
        beta = tf.get_variable(
            "beta",
            params_shape,
            initializer=tf.constant_initializer(0),
            dtype=tf.float32)
        gamma = tf.get_variable(
            "gamma",
            params_shape,
            initializer=tf.constant_initializer(1),
            dtype=tf.float32)
        normalized = (input - mean) / tf.sqrt(variance + eps)
        return gamma * normalized + beta


def embedding(input,
              vocab_size,
              embed_size,
              zero_pad=True, scope="embedding", reuse=None, scale=True):
    """Embeds a given tensor

    Args:
        inputs: A Tensor with the shape [batch_size, sentence_len]
            with type int32 or int64 representing the ids
        vocab_size: An int.
        embed_size: number of embedding units
        zero_pad: If True, id 0 is embeded into a 0 vector
        scale: A boolean. If True, the outputs is multiplied by sqrt embed_size.
        scope: variable_scope
        reuse: Boolean

    Returns:
        A 3D Tensor. [batch_size, sentence_len, embed_size]
    """

    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable("lookup_table", (vocab_size, embed_size),
                                       initializer=tf.contrib.layers.xavier_initializer(),
                                       dtype=tf.float32)
        if zero_pad:
            lookup_table = tf.concat((tf.zeros((1, embed_size)), lookup_table[1:]), axis=0)

        outputs = tf.nn.embedding_lookup(lookup_table, input)

        if scale:
            outputs = outputs * (embed_size ** 0.5)

        return outputs


def positional_encoding(input, scope='positional_encoding'):
    """Sinusoidal Positional Encoding

    Args:
        inputs: A 3D Tensor with shape of [batch_size, sentence_len, embed_size]

    Returns:
        A Tensor which is the inputs + positional_encoding

    """
    with tf.name_scope(scope):
        # N:batch size (batch_size), T:sentence length, E:word embedding size
        _, _, E = input.get_shape().as_list()
        T = tf.shape(input)[1]

        position_enc = np.array([
            [pos / np.power(10000, 2.0 * i / E) for i in range(E)]
            for pos in range(1000)], dtype=np.float32)
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        position_enc = tf.convert_to_tensor(position_enc)[:T]  # [T, E]
        position_enc = tf.expand_dims(position_enc, axis=0)  # [1, T, E]

        return input + position_enc  # [N, T, E]


def multihead_attention(dictionary,  # [N, Td, E]
                        dictionary_mask,  # [N, Td]
                        queries,  # [N, Tq, E]
                        n_units,
                        n_heads,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None,
                        dropout_rate=0.1,
                        is_training=True):
    """Multihead Attention

    Args:
        dictionary: a 3d Tensor. [N, Td, E]
        dictionary_mask: a 2d Tensor. [N, Td]. False for padding and True for the other tokens
        queries: a 3d Tensor
        n_units: total attention size
        n_heads: number of attention heads
        causality: If true, future positions in dictionary are masked
        scope: scope
        reuse: reuse
        dropout_rate: dropout rate
        is_training: training or inference


    """
    with tf.variable_scope(scope, reuse=reuse):
        if n_units is None:
            n_units = queries.get_shape().as_list()[-1]
        head_size = n_units // n_heads
        n_queries = tf.shape(queries)[1] #max num of tokens in a sentence in the batch
        batch_size = tf.shape(queries)[0]

        # relu projection to make Query, Key, Value matrices.
        Q = tf.layers.dense(queries, n_units, activation=tf.nn.relu, name='Q')  # [N, Tq, n_units]
        K = tf.layers.dense(dictionary, n_units, activation=tf.nn.relu, name='K')  # [N, Td, n_units]
        V = tf.layers.dense(dictionary, n_units, activation=tf.nn.relu, name='V')  # [N, Td, n_units]

        # linear projection to make multihead
        Q_MH = tf.layers.dense(Q, n_units, name='WQ')  # [N, Tq, n_units]
        K_MH = tf.layers.dense(K, n_units, name='WK')  # [N, Td, n_units]
        V_MH = tf.layers.dense(V, n_units, name='WV')  # [N, Td, n_units]

        # split the last dimension into multiple heads
        Q_MH = tf.concat(tf.split(Q_MH, n_heads, axis=2), axis=0)  # [N*h, Tq, n_units/h]
        K_MH = tf.concat(tf.split(K_MH, n_heads, axis=2), axis=0)  # [N*h, Td, n_units/h]
        V_MH = tf.concat(tf.split(V_MH, n_heads, axis=2), axis=0)  # [N*h, Td, n_units/h]

        # query-key multiplication
        weight = tf.matmul(Q_MH, K_MH, transpose_b=True)  # [N*h, Tq, Td]

        # scaling
        weight = weight / (head_size ** 0.5)

        # dictionary padding masking
        with tf.name_scope('dictionary_masking'):
            padding = tf.fill(tf.shape(weight), -np.inf) #[N*h, Tq, Td]
            mask = tf.tile(
                tf.expand_dims(dictionary_mask, axis=1), #[N, 1, Td]
                [n_heads, n_queries, 1]
            ) #[N, Tq, Td]
            weight = tf.where(mask, weight, padding)

        # Causality masking (in this case Tq = Td : decoder self-attention)
        if causality:
            with tf.name_scope('causality_masking'):
                c_mask = tf.equal(tf.matrix_band_part(tf.ones_like(weight[0]), -1, 0), 1) #[Tq, Td] (Tq == Td)
                c_mask = tf.tile(tf.expand_dims(c_mask, 0), [batch_size * n_heads, 1, 1]) #[N*h, Tq, Td]
                c_padding = tf.fill(tf.shape(weight), -np.inf)
                weight = tf.where(c_mask, weight, c_padding)

        weight = tf.nn.softmax(weight)  # [N*h, Tq, Td]

        # weighted sum
        outputs = tf.matmul(weight, V_MH, name='weighted_sum')  # [N*h, Tq, n_units/h]

        # restore shape
        with tf.name_scope('concatenate_heads'):
            outputs = tf.concat(tf.split(outputs, n_heads, axis=0), axis=2)  # [N, Tq, n_units]

        # linear projection
        outputs = tf.layers.dense(outputs, queries.shape[2], use_bias=False, name='attention_output')  # [N, Tq, E]

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Residual connection
        outputs += queries

        # layer normalization
        outputs = layer_norm(outputs)

        return outputs


def feedforward(input, n_units=1024, scope="feedforward", dropout_rate=0.1, reuse=None, is_training=True):
    with tf.variable_scope(scope, reuse=reuse):
        outputs = tf.layers.dense(input, n_units, activation=tf.nn.relu, use_bias=True)
        outputs = tf.layers.dense(outputs, input.get_shape().as_list()[-1], activation=tf.nn.relu, use_bias=True)

        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        outputs += input
        outputs = layer_norm(outputs)

        return outputs


def label_smoothing(inputs, eps=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - eps) * inputs) + (eps / K)

class Encoder(object):
    """docstring for Encoder"""
    def __init__(self, inputs, lengths, hparams, is_training):
        super(Encoder, self).__init__()

        self.inputs = inputs
        self.lengths = lengths
        self.enc_mask = tf.sequence_mask(lengths, tf.shape(inputs)[1])
        
        with tf.variable_scope("encoder"):
            # Embedding
            self.outputs = embedding(self.inputs,
                                 vocab_size=hparams.vocab_size,
                                 embed_size=hparams.embed_size,
                                 scope="enc_embed",
                                 scale=True,
                                 zero_pad=True)
            # Positional Encoding
            if not hparams.positional_embedding:
                self.outputs = positional_encoding(self.outputs)
            else:
                self.outputs += embedding(
                    tf.tile(
                        tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), axis=0), [tf.shape(self.inputs)[0], 1]),
                    vocab_size=hparams.maxlen,
                    embed_size=hparams.embed_size,
                    scope="enc_coding",
                    scale=False,
                    zero_pad=False)

            # Dropout
            self.outputs = tf.layers.dropout(self.outputs,
                                         rate=hparams.dropout_rate,
                                         training=tf.convert_to_tensor(is_training))

            # Blocks
            for i in range(hparams.n_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    # Multihead Attention
                    self.outputs = multihead_attention(dictionary=self.outputs,  # [N, Td, E]
                                                   dictionary_mask=self.enc_mask,  # [N, Td]
                                                   queries=self.outputs,  # [N, Tq, E]
                                                   n_units=hparams.attention_size,
                                                   n_heads=hparams.n_heads,
                                                   causality=False,
                                                   scope="multihead_attention",
                                                   reuse=None,
                                                   dropout_rate=hparams.dropout_rate,
                                                   is_training=is_training)

                    # Feed Forward
                    self.outputs = feedforward(self.outputs,
                                               n_units=4*hparams.embed_size,
                                               dropout_rate=hparams.dropout_rate,
                                               is_training=is_training)

class Decoder(object):
    """docstring for Decoder"""
    def __init__(self, inputs, lengths, enc_hidden_states, enc_mask, hparams, is_training):
        super(Decoder, self).__init__()
        
        self.inputs = inputs
        self.lengths = lengths
        self.dec_mask = tf.sequence_mask(lengths, tf.shape(inputs)[1])

        with tf.variable_scope("decoder"):
            ## Embedding
            self.outputs = embedding(self.inputs, 
                                  vocab_size=hparams.vocab_size, 
                                  embed_size=hparams.embed_size,
                                  scope="dec_embed",
                                  scale=True,
                                  zero_pad=True)

            ## Positional Encoding
            if not hparams.positional_embedding:
                self.outputs = positional_encoding(self.outputs)
            else:
                self.outputs += embedding(
                    tf.tile(
                        tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), axis=0), [tf.shape(self.inputs)[0], 1]),
                    vocab_size=hparams.maxlen,
                    embed_size=hparams.embed_size,
                    scope="dec_coding",
                    scale=False,
                    zero_pad=False
                 )


            ## Dropout
            self.outputs = tf.layers.dropout(self.outputs, 
                                        rate=hparams.dropout_rate, 
                                        training=tf.convert_to_tensor(is_training))

            ## Blocks
            for i in range(hparams.n_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.outputs = multihead_attention(dictionary=self.outputs,
                                                    dictionary_mask=self.dec_mask,
                                                    queries=self.outputs,
                                                    n_units=hparams.attention_size,
                                                    n_heads=hparams.n_heads,
                                                    causality=True,
                                                    scope="dec_self_attention",
                                                    reuse=None,
                                                    dropout_rate=hparams.dropout_rate,
                                                    is_training=is_training)
                    
                    ## Multihead Attention ( vanilla attention)
                    self.outputs= multihead_attention(dictionary=enc_hidden_states,
                                                    dictionary_mask=enc_mask,
                                                    queries=self.outputs,
                                                    n_units=hparams.attention_size,
                                                    n_heads=hparams.n_heads,
                                                    causality=False,
                                                    scope="dec_vanilla_attention",
                                                    reuse=None,
                                                    dropout_rate=hparams.dropout_rate,
                                                    is_training=is_training)

                    ## Feed Forward
                    self.outputs = feedforward(self.outputs,
                                               n_units=4*hparams.embed_size,
                                               dropout_rate=hparams.dropout_rate,
                                               is_training=is_training)

            # Final linear projection
            with tf.variable_scope("final_leaner_projection"):
                self.logits = tf.layers.dense(self.outputs, hparams.vocab_size, name='logits')
                self.outputs = self.logits
                self.softmax_outputs = tf.nn.softmax(self.logits)


