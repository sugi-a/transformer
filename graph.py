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


def embedding_layer(inputs, weight_matrix, scope="embedding", scale=True):
    """Embeds a given tensor

    Args:
        inputs: A Tensor with the shape [batch_size, sentence_len]
            with type int32 or int64 representing the ids
        scale: A boolean. If True, the outputs is multiplied by sqrt embed_size.
        scope: variable_scope
        weight_matrix: [vocab_size, emb_size]

    Returns:
        A 3D Tensor. [batch_size, sentence_len, embed_size]
    """
    emb_size = weight_matrix.get_shape().as_list()[1]
    outputs = tf.gather(weight_matrix, inputs)
    if scale:
        outputs = outputs * (emb_size ** 0.5)
    return outputs


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
#        if zero_pad:
#            lookup_table = tf.concat((tf.zeros((1, embed_size)), lookup_table[1:]), axis=0)

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

        # linear projection to make multihead
        Q_MH = tf.layers.dense(queries, n_units, use_bias=False, name='WQ')  # [N, Tq, n_units]
        K_MH = tf.layers.dense(dictionary, n_units, use_bias=False, name='WK')  # [N, Td, n_units]
        V_MH = tf.layers.dense(dictionary, n_units, use_bias=False, name='WV')  # [N, Td, n_units]

        # split the last dimension into multiple heads
        Q_MH = tf.stack(tf.split(Q_MH, n_heads, axis=2), axis=0)  # [h, N, Tq, head_size]
        K_MH = tf.stack(tf.split(K_MH, n_heads, axis=2), axis=0)  # [h, N, Tq, head_size]
        V_MH = tf.stack(tf.split(V_MH, n_heads, axis=2), axis=0)  # [h, N, Tq, head_size]

        # query-key multiplication
        weight = tf.matmul(Q_MH, K_MH, transpose_b=True)  # [h, N, Tq, Td]

        # scaling
        weight = weight / (head_size ** 0.5)

        # dictionary padding masking
        with tf.name_scope('dictionary_masking'):
            mask_shape = tf.shape(dictionary_mask)
            bias = tf.expand_dims(
                tf.expand_dims(
                    tf.where(dictionary_mask, tf.fill(mask_shape, 0.0), tf.fill(mask_shape, -np.inf)),
                    axis=1), #[N, 1, Td]
                axis=0) #[1, N, 1, Td]
            weight += bias

        # Causality masking (in this case Tq = Td : decoder self-attention)
        if causality:
            with tf.name_scope('causality_masking'):
                mask_shape = tf.shape(weight[0][0])
                mask = tf.cast(tf.matrix_band_part(tf.ones(mask_shape, tf.float32), -1, 0), tf.bool)
                bias = tf.expand_dims(
                    tf.expand_dims(
                        tf.where(mask, tf.fill(mask_shape, 0.0), tf.fill(mask_shape, -np.inf)),
                        axis=0
                    ), #[1,Tq,Tq]
                    axis=0
                ) #[1, 1, Tq, Tq]
                weight += bias

        # softmax
        weight = tf.nn.softmax(weight)  # [h, N, Tq, Td]

        # attention dropout
        weight = tf.layers.dropout(weight, dropout_rate, training=is_training)

        # weighted sum
        outputs = tf.matmul(weight, V_MH, name='weighted_sum')  # [h, N, Tq, head_size]

        # restore shape
        with tf.name_scope('concatenate_heads'):
            outputs = tf.squeeze(
                tf.concat(
                    tf.split(outputs, n_heads, axis=0), #list of [1, N, Tq, head_size]
                    axis=3), #[1, N, Tq, n_units]
                axis=[0])  # [N, Tq, n_units]

        # linear projection
        outputs = tf.layers.dense(outputs, queries.shape[2], use_bias=False, name='attention_output')  # [N, Tq, E]

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)

        return outputs


def feedforward(input, n_units=1024, scope="feedforward", dropout_rate=0.1, reuse=None, is_training=True):
    with tf.variable_scope(scope, reuse=reuse):
        #relu
        outputs = tf.layers.dense(input, n_units, activation=tf.nn.relu, use_bias=True)
        #dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
        #linear
        outputs = tf.layers.dense(outputs, input.get_shape().as_list()[-1], use_bias=True)

        return outputs


def label_smoothing(inputs, eps=0.1):
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - eps) * inputs) + (eps / K)

class Encoder(object):
    """docstring for Encoder"""
    def __init__(self, inputs, lengths, hparams, is_training):
        super(Encoder, self).__init__()

        with tf.variable_scope("encoder"):
            self.inputs = inputs
            self.lengths = lengths
            self.enc_mask = tf.sequence_mask(lengths, tf.shape(inputs)[1])
            
            # Embedding
            self.embedding_weight = tf.get_variable(
                "embedding_weight",
                [hparams.vocab_size, hparams.embed_size],
                tf.float32)
            self.outputs = embedding_layer(self.inputs, self.embedding_weight)

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
                    outputs_norm = layer_norm(self.outputs, scope="mult_attn_norm")
                    self.outputs = multihead_attention(dictionary=outputs_norm,  # [N, Td, E]
                                                   dictionary_mask=self.enc_mask,  # [N, Td]
                                                   queries=outputs_norm,  # [N, Tq, E]
                                                   n_units=hparams.attention_size,
                                                   n_heads=hparams.n_heads,
                                                   causality=False,
                                                   scope="multihead_attention",
                                                   reuse=None,
                                                   dropout_rate=hparams.dropout_rate,
                                                   is_training=is_training) + self.outputs

                    # Feed Forward
                    outputs_norm = layer_norm(self.outputs, scope="ffn_norm")
                    self.outputs = feedforward(outputs_norm,
                                               n_units=4*hparams.embed_size,
                                               dropout_rate=hparams.dropout_rate,
                                               is_training=is_training) + self.outputs
            self.outputs = layer_norm(self.outputs, scope="output_norm")
            self.hidden_states = self.outputs

class Decoder(object):
    """docstring for Decoder"""
    def __init__(self, inputs, lengths, enc_hidden_states, enc_mask, hparams, is_training, embedding_weight=None):
        super(Decoder, self).__init__()
        
        with tf.variable_scope("decoder"):
            self.inputs = inputs
            self.lengths = lengths
            self.dec_mask = tf.sequence_mask(lengths, tf.shape(inputs)[1])

            ## Embedding
            if embedding_weight is None:
                self.embedding_weight = tf.get_variable("embedding_weight", [hparams.vocab_size, hparams.embed_size], tf.float32)
            else:
                self.embedding_weight = embedding_weight
            self.outputs = embedding_layer(self.inputs, self.embedding_weight)

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
                    outputs_norm = layer_norm(self.outputs, scope="self_attn_norm")
                    self.outputs = multihead_attention(dictionary=outputs_norm,
                                                    dictionary_mask=self.dec_mask,
                                                    queries=outputs_norm,
                                                    n_units=hparams.attention_size,
                                                    n_heads=hparams.n_heads,
                                                    causality=True,
                                                    scope="dec_self_attention",
                                                    reuse=None,
                                                    dropout_rate=hparams.dropout_rate,
                                                    is_training=is_training) + self.outputs
                    
                    ## Multihead Attention ( vanilla attention)
                    outputs_norm = layer_norm(self.outputs, scope="context_attn_norm")
                    self.outputs= multihead_attention(dictionary=enc_hidden_states,
                                                    dictionary_mask=enc_mask,
                                                    queries=outputs_norm,
                                                    n_units=hparams.attention_size,
                                                    n_heads=hparams.n_heads,
                                                    causality=False,
                                                    scope="dec_vanilla_attention",
                                                    reuse=None,
                                                    dropout_rate=hparams.dropout_rate,
                                                    is_training=is_training) + self.outputs

                    ## Feed Forward
                    outputs_norm = layer_norm(self.outputs, scope="ffn_norm")
                    self.outputs = feedforward(outputs_norm,
                                               n_units=4*hparams.embed_size,
                                               dropout_rate=hparams.dropout_rate,
                                               is_training=is_training) + self.outputs

            # Final linear projection
            with tf.variable_scope("leaner_projection"):
                self.outputs = layer_norm(self.outputs, scope="output_norm")
                # outputs: [batch_size, sentence_len, emb_size]. emb_weight: [vocab_size, emb_size]
                self.logits = tf.tensordot(self.outputs, self.embedding_weight, [[2], [1]], name="logits")
                self.outputs = self.logits
                self.softmax_outputs = tf.nn.softmax(self.logits)


