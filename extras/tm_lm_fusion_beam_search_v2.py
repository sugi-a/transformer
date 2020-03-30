import sys, os, argparse, time, json
from logging import basicConfig, getLogger, INFO, DEBUG; logger = getLogger()
from collections import deque
import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework import nest

from ..components.decoding import length_penalty

def fusion_beam_search_delayed_pmi_v2(
    get_logits_fn,
    init_cache,
    static_cache,
    init_seq,
    beam_size,
    maxlens,
    eos_id,
    pad_id=0,
    offsets=None,
    length_penalty_a=0,
    normalize_logits=True):
    """<sos> in `init_seq` is not removed."""
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
        return log_prob / length_penalty(length, length_penalty_a)

    def cond_fn(loop_vars, _i):
        not_closed = tf.logical_not(tf.reduce_all(loop_vars['has_eos']), name='loop_condition')
        not_long = tf.less(tf.shape(loop_vars['generated_seq'])[2] + tf.shape(init_seq)[2] - 1, maxlen)
        return tf.logical_and(not_closed, not_long)

    def body_fn(loop_vars, _i):

        with tf.name_scope('loop_body'):
            # The position of the token predicted in this iteration. Starts from 0 [batch] or [1]
            if offsets is not None:
                cur_pos = tf.shape(loop_vars['generated_seq'])[2] + tf.shape(init_seq)[2] - 1 - offsets
            else:
                cur_pos = (tf.shape(loop_vars['generated_seq'])[2] + tf.shape(init_seq)[2] - 1)[None]

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
                logits, pmi_choice = get_logits_fn(
                    flat_dec_inputs,
                    flat_cache,
                    static_flat_cache,
                    _i,
                    loop_vars['prev_pmi_chosen'])
                logits = logits[:, -1]
                vocab_size = tf.shape(logits)[1]
                if normalize_logits:
                    logits = tf.math.log_softmax(logits)

            with tf.name_scope('update_cache'):
                loop_vars['cache'] = nest.map_structure(pack, flat_cache)


            with tf.name_scope('get_top_k'):
                # [bat*beam, vocab] -> [bat, beam, vocab]
                logits = tf.reshape(logits, [batch_size, beam_size, -1])

                # [bat, beam, vocab]
                eos_bias = tf.broadcast_to(
                    tf.concat([[0], tf.fill([vocab_size - 1], NEG_INF)], axis=0)[None, None],
                    tf.shape(logits))
                # [bat, beam, vocab]
                has_eos = tf.broadcast_to(loop_vars['has_eos'][:, :, None], tf.shape(logits))

                # [bat, beam, 1] + [bat, beam, vocab] -> [bat, beam, vocab]
                seq_logp = loop_vars['seq_log_prob'][:,:, None] + tf.where(
                    has_eos,
                    eos_bias,
                    logits)

                seq_score = tf.where(
                    has_eos,
                    loop_vars['score'][:, :, None] + eos_bias,
                    get_score(seq_logp, cur_pos[:, None, None] + 1))

                # [bat, beam, vocab] -> [bat, beam*vocab]
                seq_score = tf.reshape(seq_score, [batch_size, -1])
                # [bat, beam]
                top_score, top_ind = tf.math.top_k(seq_score, beam_size, False)

                # [bat, beam]
                old_beam_ind = tf.math.floordiv(top_ind, vocab_size)
                pred = tf.floormod(top_ind, vocab_size)


                
            with tf.name_scope('update_loop_vars'):
                new_vars = {}

                # UPDATE cache [bat, beam, ...]
                new_vars['cache'] = nest.map_structure(lambda x:
                    tf.batch_gather(x, old_beam_ind), loop_vars['cache'])

                # UPDATE log_probs [bat, beam]
                new_vars['seq_log_prob'] = tf.batch_gather(tf.reshape(seq_logp, [batch_size, -1]), top_ind)

                # UPDATE score [bat, beam]
                new_vars['score'] = top_score

                new_vars['prev_pmi_chosen'] = tf.reshape(tf.batch_gather(
                    tf.reshape(pmi_choice, [batch_size, -1]),
                    top_ind), [-1])
                
                # UPDATE `generated_seq`
                # Choosing old branch. [bat, beam, ...]->[bat, beam, len]
                gen_seq = tf.batch_gather(loop_vars["generated_seq"], old_beam_ind)

                # If the path is already closed by EOS, new tokens should be PAD. [bat, beam]
                old_ended = tf.batch_gather(loop_vars['has_eos'], old_beam_ind)
                pad_tok = tf.ones([batch_size, beam_size], tf.int32) * pad_id

                # If the sequence length reaches the limit, EOS must come.
                stopping = tf.tile(tf.greater(cur_pos + 2, maxlens)[:, None], [1,beam_size])
                eos_tok = tf.fill([batch_size, beam_size], eos_id)

                # New token to be added
                new_tok = tf.where(old_ended, pad_tok, tf.where(stopping, eos_tok, pred))

                # Append new token. [batch, beam, len]->[batch, beam, len+1]
                new_vars['generated_seq'] = tf.concat([gen_seq, new_tok[:,:, None]], axis=-1)

                # UPDATE dec_inputs. (token input in the next step) [bat, beam, len=1]
                new_vars['dec_inputs'] = new_tok[:, :, None]

                # UPDATE has_eos [batch, beam]
                new_vars['has_eos'] = tf.logical_or(old_ended, tf.equal(new_tok, eos_id))


        return [new_vars, _i + 1]


    # Initial decoder inputs. Add a beam dimension and replicate along it.
    with tf.name_scope('init_seq_beam_replication'):
        # [bat, len] -> [bat, beam, len]
        init_seq = tf.tile(init_seq[:, None], [1, beam_size, 1])


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
            'dec_inputs': init_seq,
            'prev_pmi_chosen': tf.zeros([batch_size* beam_size])
        }

    static_flat_cache = nest.map_structure(lambda x: flatten(fork(x[:, None])), static_cache)

    # shape invariants
    with tf.name_scope('shape_invariants'):
        shape_invariants = [{
            'cache': nest.map_structure(get_shape_keep_last_dim, init_loop_vars['cache']),
            'generated_seq': tf.TensorShape([None, None, None]),
            'seq_log_prob': tf.TensorShape([None, None]),
            'has_eos': tf.TensorShape([None, None]),
            'score': tf.TensorShape([None, None]),
            'dec_inputs': tf.TensorShape([None, None, None]),
            'prev_pmi_chosen': tf.TensorShape([None])
        }, tf.TensorShape([])]

    max_iter = tf.cond(tf.equal(batch_size, 0), lambda: -1, lambda: maxlen)

    with tf.name_scope('while_loop'):
        finish_state, _i = tf.while_loop(
            cond_fn,
            body_fn,
            [init_loop_vars, tf.constant(0, dtype=tf.int32)],
            shape_invariants,
            back_prop=False,
            maximum_iterations=max_iter,
            parallel_iterations=1
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
            # If batch_size==0, top-k op fails. So add pseudo element.
            finish_state['score'] = tf.pad(finish_state['score'], [[0,1],[0,0]])

            # Sorting. score: [batch+1, beam], indices: [batch+1, beam]
            score, indices = tf.math.top_k(finish_state['score'], beam_size, sorted=True)

            # Cut off the pseudo element
            score, indices = score[:-1], indices[:-1]

            # Sort sequences
            seq = tf.batch_gather(finish_state['generated_seq'], indices)

        # concat with the prefix [batch_size, beam_size, init_len + generated_len]
        with tf.name_scope('concat_prefix'):
            seq = tf.concat([init_seq, seq], axis=-1)

    return seq, score

