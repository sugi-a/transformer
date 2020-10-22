import sys
from logging import getLogger; logger = getLogger(__name__)
import tensorflow as tf
from tensorflow import keras
import numpy as np

INF = 1e9
BCT = tf.broadcast_to

def length_penalty(length, alpha):
    """
    Args:
        length: Tensor<any, int32>
        alpha: float
    Returns:
        [shape(length), float32]
    """
    return tf.cast(tf.pow((5 + length)/(1 + 5), alpha), dtype=tf.float32)


def beam_search(
        get_logits_fn,
        update_state_fn,
        sos,
        eos,
        beam_size,
        maxlen,
        pad=0,
        shape_invariants=None,
        length_penalty_fn=None
        ):
    """
    Primitive beam search for autoregressive sequence generation models.
    Mini-batch inputs are supported.
    B := batch size
    K := beam size
    V := vocabulary size
    Args:
        get_logits_fn:
            (dec_input: <[B * K, 1], int32>) => <[B * K, l, V], float32>
            In:
                Pseudo mini-batch consisting of B * K token IDs.
                The i-th element (0 <= i < B * K) is the newest token
                of the (i % K)-th path of the (i // K)-th batch.
            Out:
                Output is the token scores over the vocabulary, which must be
                log-scale score (normalized or unnormalized logits).
                Sequence score is computed as a sum of the token scores.
        update_state_fn:
            (alive_path_ids: <[B * K], int32> => void)
        sos: Integer[B] | <[B], int32>
        maxlen: Integer | Tensor<([]|[B]), int32>
        shape_invariants:
            List<Tuple<<Tensor<any>, TensorShape>>> | None
        length_penalty_fn: callable | None
            callable: (length: <any, int32>) => <shape(length), float32>
    Returns:
        paths: <[B, K], int32>, score: <[B, K], float32>
    """
    B = tf.shape(sos)[0]
    K = beam_size

    if length_penalty_fn is None:
        length_penalty_fn = lambda x: 1

    # [B, K, 1] <- [B]
    paths = BCT(sos[:, None, None], [B, K, 1])
    # [B, K] Sequence log probability
    slogp = tf.concat([tf.fill([B, 1], 0.0), tf.fill([B, K - 1], -INF)], axis=1)
    # [B, K] Sequence score (=slogp if no length penalty is used)
    score = tf.identity(slogp)
    # [B, K]
    closed = tf.fill([B, K], False)
    # [B]
    maxlen = BCT(maxlen, [B])

    i = tf.constant(0)

    shape_inv = [(paths, tf.TensorShape([None, K, None]))] \
        + (shape_invariants if shape_invariants is not None else [])

    while ~tf.math.reduce_all(closed):
        tf.autograph.experimental.set_loop_options(shape_invariants=shape_inv)

        # [B * K, V]
        t_logp = get_logits_fn(tf.reshape(paths, [B * K, -1])[:, -1:])[:, 0]
        # [B, K, V]
        t_logp = tf.reshape(t_logp, [B, K, -1])

        # Force EOS for sequences longer than or equal to their maxlen
        non_eos_bias = tf.concat([
            tf.ones_like(t_logp[:, :, :eos], tf.float32) * (-INF),
            t_logp[:, :, eos: eos + 1],
            tf.ones_like(t_logp[:, :, eos + 1:], tf.float32) * (-INF)
        ], axis=-1)
        t_logp = tf.where(i + 1 >= maxlen[:, None, None], non_eos_bias, t_logp)
        # Set logp=0 for already closed paths
        t_logp = tf.where(closed[:, :, None], 0.0, t_logp)

        # new sequence logp and score
        t_slogp = slogp[:, :, None] + t_logp
        t_score = t_slogp / length_penalty_fn(i + 1)

        # [B, K, V] -> [B, K * V] -> [B, K] Top K
        values, indices = tf.math.top_k(
            tf.reshape(t_score, [B, -1]), k=K, sorted=False)
        # [B, K]
        alive_path_ids = indices // tf.shape(t_score)[-1]
        new_token_ids = indices % tf.shape(t_score)[-1]

        # Update loop variables
        old_close = tf.gather(closed, alive_path_ids, batch_dims=1)
        update_state_fn(
            tf.reshape(alive_path_ids + tf.range(B)[:, None] * K, [-1]))
        paths = tf.concat([
                tf.gather(paths, alive_path_ids, batch_dims=1),
                tf.where(old_close, pad, new_token_ids)[:, :, None]
            ], axis=2)
        slogp = tf.gather(
            tf.reshape(t_slogp, [B, -1]), alive_path_ids, batch_dims=1)
        score = values
        closed = old_close | (new_token_ids == eos)

        i += 1

    # Sort
    score, indices = tf.math.top_k(score, K, sorted=True)
    paths = tf.gather(paths, indices, batch_dims=1)

    return paths, score
