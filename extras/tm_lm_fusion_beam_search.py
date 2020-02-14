import sys, os, argparse, time, json
from logging import basicConfig, getLogger, INFO, DEBUG; logger = getLogger()
from collections import deque
import tensorflow as tf
import numpy as np

from tensorflow.contrib.framework import nest

from ..components.decoding import length_penalty

def fusion_beam_search(get_logits_fn, init_cache, init_seq, beam_size, maxlens, eos_id, pad_id=0, offsets=None, params=None):
    """<sos> in `init_seq` is not removed."""
    NEG_INF = -1e9

    maxlen = tf.reduce_max(maxlens)

    if eos_id is None:
        eos_id = pad_id

    params = params or {}
    
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
        length_penalty_a = params.get('length_penalty_a', 1.0)
        return log_prob / length_penalty(length, length_penalty_a)

    def cond_fn(loop_vars):
        not_closed = tf.logical_not(tf.reduce_all(loop_vars['has_eos']), name='loop_condition')
        not_long = tf.less(tf.shape(loop_vars['generated_seq'])[2] + tf.shape(init_seq)[2] - 1, maxlen)
        return tf.logical_and(not_closed, not_long)

    def body_fn(loop_vars):

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
                logits, y_logits, cy_logits = (_[:,-1] for _ in get_logits_fn(flat_dec_inputs, flat_cache))

            # restore shape of cache. ->[bat_size, beam_size, ...]
            with tf.name_scope('update_and_restore_structure_of_cache'):
                loop_vars['cache'] = nest.map_structure(pack, flat_cache)

            with tf.name_scope('preliminary_top_ids_and_log_probs'):
                # get the top k=beam_size for each sequence.
                # top_logits: [bat * beam, beam], ids: [bat * beam, beam]
                top_logits, ids = tf.math.top_k(logits, beam_size, False)
                top_mt_y_logits = tf.batch_gather(y_logits, ids)
                top_mt_cy_logits = tf.batch_gather(cy_logits, ids)
                pmi = top_mt_cy_logits - top_mt_y_logits

                log_prob = top_logits

                # Arrange shape of log_prob and ids
                with tf.name_scope('restore_shape'):
                    # log prob. [bat * beam, beam]->[bat, old_beam * new_beam]
                    log_prob = tf.reshape(log_prob, [batch_size, beam_size ** 2]) 
                    pmi = tf.reshape(pmi, [batch_size, beam_size**2])

                    # IDs [bat * beam, beam]->[bat, old_beam * new_beam]
                    ids = tf.reshape(ids, [batch_size, beam_size ** 2])

                # Sequence score
                with tf.name_scope('seq_score'):
                    # Fork log probability of sequences. [bat_size, beam * beam]. 
                    old_forked_seqp = fork(loop_vars['seq_log_prob'])
                    old_forked_score = fork(loop_vars['score'])

                    # Fork the info of closed paths. [bat, beam]->[bat, beam * beam]
                    forked_ended = fork(loop_vars['has_eos'])

                    # Update sequence log probability [bat, old_beam * new_beam]
                    new_forked_seqp = old_forked_seqp + tf.where(forked_ended, eos_mask, log_prob)

                    # Compute score for pruning [bat, old_beam * new_beam]
                    new_mt_score = get_score(new_forked_seqp, cur_pos[:, None] + 1)
                    old_forked_pmi = fork(loop_vars['seq_pmi'])
                    prun_score = old_forked_pmi + new_mt_score

                    # Update PMI
                    new_forked_pmi = old_forked_pmi + tf.where(forked_ended, eos_mask, pmi)
                    new_forked_score = tf.where(forked_ended, old_forked_score + eos_mask, new_forked_pmi + new_mt_score)

            with tf.name_scope('get_top_k'):
                # Top k=beam [bat, beam]
                _, top_ind = tf.math.top_k(prun_score, beam_size, False)

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
                new_vars['seq_log_prob'] = tf.batch_gather(new_forked_seqp, top_ind)

                # UPDATE seq_pmi [bat, beam]
                new_vars['seq_pmi'] = tf.batch_gather(new_forked_pmi, top_ind)

                # UPDATE score [bat, beam]
                new_vars['score'] = tf.batch_gather(new_forked_score, top_ind)
                
                # UPDATE `generated_seq`
                # Choosing old branch. [bat, beam, ...]->[bat, beam, len]
                gen_seq = tf.batch_gather(loop_vars["generated_seq"], old_beam_ind)

                # If the path is already closed by EOS, new tokens should be PAD. [bat, beam]
                old_ended = tf.batch_gather(loop_vars['has_eos'], old_beam_ind)
                pad_tok = tf.ones([batch_size, beam_size], tf.int32) * pad_id

                # If the sequence length reaches the limit, EOS must come.
                stopping = tf.tile(tf.greater(cur_pos + 2, maxlens)[:, None], [1,beam_size])
                eos_tok = tf.fill([batch_size, beam_size], eos_id)

                # Predicted tokens
                pred = tf.batch_gather(ids, top_ind) # [batch, beam]

                # New token to be added
                new_tok = tf.where(old_ended, pad_tok, tf.where(stopping, eos_tok, pred))

                # Append new token. [batch, beam, len]->[batch, beam, len+1]
                new_vars['generated_seq'] = tf.concat([gen_seq, new_tok[:,:, None]], axis=-1)

                # UPDATE dec_inputs. (token input in the next step) [bat, beam, len=1]
                new_vars['dec_inputs'] = new_tok[:, :, None]

                # UPDATE has_eos [batch, beam]
                new_vars['has_eos'] = tf.logical_or(old_ended, tf.equal(new_tok, eos_id))


        return new_vars


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
            'seq_pmi': tf.concat([tf.zeros([batch_size, 1]), tf.fill([batch_size, beam_size - 1], NEG_INF)], axis=1),
            'has_eos': tf.zeros([batch_size, beam_size], dtype=tf.bool),
            'dec_inputs': init_seq
        }


    # shape invariants
    with tf.name_scope('shape_invariants'):
        shape_invariants = {
            'cache': nest.map_structure(get_shape_keep_last_dim, init_loop_vars['cache']),
            'generated_seq': tf.TensorShape([None, None, None]),
            'seq_log_prob': tf.TensorShape([None, None]),
            'seq_pmi': tf.TensorShape([None, None]),
            'has_eos': tf.TensorShape([None, None]),
            'score': tf.TensorShape([None, None]),
            'dec_inputs': tf.TensorShape([None, None, None])
        }

    max_iter = tf.cond(tf.equal(batch_size, 0), lambda: -1, lambda: maxlen)

    with tf.name_scope('while_loop'):
        finish_state = tf.while_loop(
            cond_fn,
            body_fn,
            [init_loop_vars],
            shape_invariants,
            back_prop=False,
            maximum_iterations=max_iter,
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


def fusion_beam_search_delayed_pmi(get_logits_fn, init_cache, init_seq, beam_size, delay, maxlens, eos_id, pad_id=0, offsets=None, params=None):
    """<sos> in `init_seq` is not removed."""
    NEG_INF = -1e9

    maxlen = tf.reduce_max(maxlens)

    if eos_id is None:
        eos_id = pad_id

    params = params or {}
    
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
        length_penalty_a = params.get('length_penalty_a', 1.0)
        return log_prob / length_penalty(length, length_penalty_a)

    def cond_fn(loop_vars):
        not_closed = tf.logical_not(tf.reduce_all(loop_vars['has_eos']), name='loop_condition')
        not_long = tf.less(tf.shape(loop_vars['generated_seq'])[2] + tf.shape(init_seq)[2] - 1, maxlen)
        return tf.logical_and(not_closed, not_long)

    def body_fn(loop_vars):

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
                logits, y_logits, cy_logits = (_[:,-1] for _ in get_logits_fn(flat_dec_inputs, flat_cache))

            # restore shape of cache. ->[bat_size, beam_size, ...]
            with tf.name_scope('update_and_restore_structure_of_cache'):
                loop_vars['cache'] = nest.map_structure(pack, flat_cache)

            with tf.name_scope('preliminary_top_ids_and_log_probs'):
                # get the top k=beam_size for each sequence.
                # top_logits: [bat * beam, beam], ids: [bat * beam, beam]
                top_logits, ids = tf.math.top_k(logits, beam_size, False)
                top_mt_y_logits = tf.batch_gather(y_logits, ids)
                top_mt_cy_logits = tf.batch_gather(cy_logits, ids)
                pmi = top_mt_cy_logits - top_mt_y_logits

                log_prob = top_logits

                # Arrange shape of log_prob and ids
                with tf.name_scope('restore_shape'):
                    # log prob. [bat * beam, beam]->[bat, old_beam * new_beam]
                    log_prob = tf.reshape(log_prob, [batch_size, beam_size ** 2]) 
                    pmi = tf.reshape(pmi, [batch_size, beam_size**2])

                    # IDs [bat * beam, beam]->[bat, old_beam * new_beam]
                    ids = tf.reshape(ids, [batch_size, beam_size ** 2])

                # Sequence score
                with tf.name_scope('seq_score'):
                    # Fork log probability of sequences. [bat_size, beam * beam]. 
                    old_forked_seqp = fork(loop_vars['seq_log_prob'])
                    old_forked_seq_pmi = fork(loop_vars['seq_pmi'])
                    old_forked_pmi_q = fork(loop_vars['pmi_q'])
                    old_forked_ended = fork(loop_vars['has_eos'])

                    # Update sequence log probability [bat, old_beam * new_beam]
                    new_forked_seqp = old_forked_seqp + tf.where(old_forked_ended, eos_mask, log_prob)
                    # Update pmi queue [bat, old_beam * new_beam, delay]
                    new_forked_pmi_q = tf.concat([old_forked_pmi_q, tf.where(old_forked_ended, eos_mask, pmi)[:,:,None]], axis=2)
                    popped = new_forked_pmi_q[:, :, 0]
                    new_forked_pmi_q = new_forked_pmi_q[:,:, 1:]
                    # Update sequence pmi [bat, old_beam * new_beam]
                    new_forked_seq_pmi = old_forked_seq_pmi + popped

                    # Compute score for pruning [bat, old_beam * new_beam]
                    prun_score = new_forked_seq_pmi + new_forked_seqp
                    prun_score = get_score(prun_score, cur_pos[:, None] + 1)

            with tf.name_scope('get_top_k'):
                # Top k=beam [bat, beam]
                _, top_ind = tf.math.top_k(prun_score, beam_size, False)

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
                new_vars['seq_log_prob'] = tf.batch_gather(new_forked_seqp, top_ind)

                # UPDATE pmi queue [bat, beam, delay]
                new_vars['pmi_q'] = tf.batch_gather(new_forked_pmi_q, top_ind)
                
                # UPDATE sequence pmi [bat, beam]
                new_vars['seq_pmi'] = tf.batch_gather(new_forked_seq_pmi, top_ind)

                # UPDATE score [bat, beam].
                new_vars['score'] = tf.batch_gather(prun_score, top_ind)
                
                # UPDATE `generated_seq`
                # Choosing old branch. [bat, beam, ...]->[bat, beam, len]
                gen_seq = tf.batch_gather(loop_vars["generated_seq"], old_beam_ind)

                # If the path is already closed by EOS, new tokens should be PAD. [bat, beam]
                old_ended = tf.batch_gather(loop_vars['has_eos'], old_beam_ind)
                pad_tok = tf.ones([batch_size, beam_size], tf.int32) * pad_id

                # If the sequence length reaches the limit, EOS must come.
                stopping = tf.tile(tf.greater(cur_pos + 2, maxlens)[:, None], [1,beam_size])
                eos_tok = tf.fill([batch_size, beam_size], eos_id)

                # Predicted tokens
                pred = tf.batch_gather(ids, top_ind) # [batch, beam]

                # New token to be added
                new_tok = tf.where(old_ended, pad_tok, tf.where(stopping, eos_tok, pred))

                # Append new token. [batch, beam, len]->[batch, beam, len+1]
                new_vars['generated_seq'] = tf.concat([gen_seq, new_tok[:,:, None]], axis=-1)

                # UPDATE dec_inputs. (token input in the next step) [bat, beam, len=1]
                new_vars['dec_inputs'] = new_tok[:, :, None]

                # UPDATE has_eos [batch, beam]
                new_vars['has_eos'] = tf.logical_or(old_ended, tf.equal(new_tok, eos_id))


        return new_vars


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
        init_logp_template = tf.concat(
            [tf.zeros([batch_size, 1]), tf.fill([batch_size, beam_size - 1], NEG_INF)],
            axis=1)
        init_loop_vars = {
            'cache': nest.map_structure(lambda x: fork(x[:, None]), init_cache),
            'generated_seq': tf.zeros([batch_size, beam_size, 0], dtype=tf.int32),
            'seq_log_prob': init_logp_template,
            'score': init_logp_template,
            'seq_pmi': init_logp_template,
            # Queue of recent PMIs [batch, beam, queue seize=delay]
            'pmi_q': tf.tile(init_logp_template[:,:,None], [1,1,delay]),
            'has_eos': tf.zeros([batch_size, beam_size], dtype=tf.bool),
            'dec_inputs': init_seq
        }


    # shape invariants
    with tf.name_scope('shape_invariants'):
        shape_invariants = {
            'cache': nest.map_structure(get_shape_keep_last_dim, init_loop_vars['cache']),
            'generated_seq': tf.TensorShape([None, None, None]),
            'seq_log_prob': tf.TensorShape([None, None]),
            'seq_pmi': tf.TensorShape([None, None]),
            'pmi_q': tf.TensorShape([None, None, None]),
            'has_eos': tf.TensorShape([None, None]),
            'score': tf.TensorShape([None, None]),
            'dec_inputs': tf.TensorShape([None, None, None])
        }

    max_iter = tf.cond(tf.equal(batch_size, 0), lambda: -1, lambda: maxlen)

    with tf.name_scope('while_loop'):
        finish_state = tf.while_loop(
            cond_fn,
            body_fn,
            [init_loop_vars],
            shape_invariants,
            back_prop=False,
            maximum_iterations=max_iter,
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
            # Pop one from the queue and add
            if offsets is not None:
                cur_pos = tf.shape(finish_state['generated_seq'])[2] + tf.shape(init_seq)[2] - 1 - offsets
            else:
                cur_pos = (tf.shape(finish_state['generated_seq'])[2] + tf.shape(init_seq)[2] - 1)[None]
            popped =tf.concat([finish_state['pmi_q'], tf.zeros([batch_size, beam_size, 1])], axis=2)[:,:,0] 
            finish_state['score'] = get_score(finish_state['seq_log_prob'] + finish_state['seq_pmi'] + popped, cur_pos)

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


