import sys
from logging import getLogger; logger = getLogger(__name__)
import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np

def length_penalty(length, alpha):
    return tf.cast(tf.pow((5 + length)/(1 + 5), alpha), dtype=tf.float32)

BeamSearchKeys = {
    "KEY_TOPK" : 0,
    "KEY_SAMPLING" : 1,
    "KEY_DIVERSE_BEAM_SEARCH" : 2
}

def beam_search_decode(get_logits_fn, init_cache, init_seq, init_seq_len, beam_size, maxlens, eos_id, pad_id=0, alpha=1, params=None):
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
                sampling_method = params.get('sampling_method', None)
                if sampling_method is None or sampling_method == BeamSearchKeys["KEY_TOPK"]:
                    # Normal top-k selection
                    top_logits, ids = tf.math.top_k(logits, beam_size, False, name='pre_tops') 
                elif sampling_method == BeamSearchKeys["KEY_SAMPLING"]:
                    # Random sampling based on probability distribution
                    ids = tf.random.multinomial(logits, beam_size) # [bat*beam, beam]
                    ids = tf.cast(ids, tf.int32)
                    top_logits = tf.batch_gather(logits, ids) # [bat*beam, beam]
                elif sampling_method == BeamSearchKeys["KEY_DIVERSE_BEAM_SEARCH"]:
                    # [Li+ 2016] "A simple, fast diverse decoding algorithm" with
                    # a fixed diversity rate.
                    top_logits, ids = tf.math.top_k(logits, beam_size, False, name='pre_tops') 
                    diversify_bias = tf.cast(tf.range(beam_size), tf.float32) * params["diversity_rate"]
                    top_logits = tf.cond(
                        tf.equal(tf.shape(loop_vars['generated_seq'])[2], 0),
                        lambda: top_logits,
                        lambda: top_logits - diversify_bias[None])
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

                # If the sequence length reaches the limit, EOS must come.
                stopping = tf.tile(tf.greater(cur_pos + 2, maxlens)[:, None], [1,beam_size])
                eos_tok = tf.fill([batch_size, beam_size], eos_id)

                # Predicted tokens
                pred = tf.batch_gather(ids, top_ind) # [batch, beam]

                # New token to be added
                new_tok = tf.where(is_in_init, cur_init_tok,
                    tf.where(old_ended, pad_tok,
                        tf.where(stopping, eos_tok, pred)))

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

        # concat with the prefix and remove the first token (usually <SOS>)
        # [batch_size, beam_size, length]
        with tf.name_scope('concat_prefix'):
            seq = tf.concat([common_prefix, seq], axis=-1)[:, :, 1:]

    return seq, score

