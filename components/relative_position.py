import tensorflow as tf


class RelativePositionMultiheadSelfAttention(tf.layers.Layer):
    def __init__(self, hidden_size, n_heads, dropout_rate, max_relative_dist, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert hidden_size % n_heads == 0

        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_size = self.hidden_size // self.n_heads
        self.dropout_rate = dropout_rate
        self.max_relative_dist = max_relative_dist

    
    def build(self, input_shape):
        self.q_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='q_layer')
        self.k_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='k_layer')
        self.v_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='v_layer')
        self.att_out_layer = tf.layers.Dense(self.hidden_size, use_bias=False, name='attention_output')
        self.pos_emb = tf.get_variable('embedding', [self.max_relative_dist * 2 + 1, self.head_size], tf.float32)

        super().build(input_shape)


    def call(self, query, bias, training=False, cache=None):
        q = self.q_layer(query)

        q = self.q_layer(query) # [batch, length, hidden]
        k = self.k_layer(query) # same as above
        v = self.v_layer(query) # same as above

        
        if cache is not None:
            with tf.name_scope('layer_cache_extension'):
                k = tf.concat([cache['k'], k], axis=1)
                v = tf.concat([cache['v'], v], axis=1)
                cache['k'] = k
                cache['v'] = v

        # Reshape to [batch, nheads, length_q, head_size]
        q = tf.stack(tf.split(q, self.n_heads, axis=-1), axis=1)
        # Reshape to [batch, nheads, length_k, head_size]
        k = tf.stack(tf.split(k, self.n_heads, axis=-1), axis=1)
        v = tf.stack(tf.split(v, self.n_heads, axis=-1), axis=1)

        # Normal q-k multiplication term
        weight = tf.matmul(q, k, transpose_b=True) # [batch, nheads, length_q, length_k]

        with tf.name_scope('relative_position_bias'):
            # Make relative position matrix. [q_len, k_len]
            q_len, k_len = tf.shape(q)[2], tf.shape(k)[2]
            k_indices = tf.range(k_len)
            q_indices = k_indices[-q_len:]
            rel_pos = k_indices[None, :] - q_indices[:, None]

            # Clipping
            rel_pos = tf.clip_by_value(rel_pos, -self.max_relative_dist, self.max_relative_dist)

            # Shift to start from 0
            rel_pos += self.max_relative_dist

            # Make embedding matrix. [q_len, k_len, head_size]
            embeddings = tf.gather(self.pos_emb, rel_pos)

            # Make bias matrix. [bat, nheads, q_len, k_len]
            # q [bat, nheads, q_len, head]->[q_len, bat * nheads, head]
            _q = tf.transpose(q, [2, 0, 1, 3])
            _q = tf.reshape(_q, [q_len, -1, self.head_size])
            # q . emb. [q_len, bat * nheads, k_len]
            _rel_pos_bias = tf.matmul(_q, embeddings, transpose_b=True)
            # Restore shape [q_len, bat * nheads, k_len]->[bat, nheads, q_len, k_len]
            _rel_pos_bias = tf.reshape(_rel_pos_bias, [q_len, -1, self.n_heads, k_len])
            _rel_pos_bias = tf.transpose(_rel_pos_bias, [1,2,0,3])
            # Apply relative postion bias to `weight`
            weight += _rel_pos_bias

        
        weight = weight / (self.head_size ** 0.5)

        with tf.name_scope('add_bias'):
            weight = weight + bias

        weight = tf.nn.softmax(weight, name='attention_weight')

        weight = tf.layers.dropout(weight, self.dropout_rate, training=training)

        outputs = tf.matmul(weight, v) # [batch, nheads, length_q, head_size]

        outputs = tf.concat(tf.unstack(outputs, axis=1), axis=2) # [batch, length_q, emb_size]

        outputs = self.att_out_layer(outputs)

        return outputs

