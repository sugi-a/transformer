import tensorflow as tf
import numpy as np
from tensorflow.contrib.framework import nest

class CumulativeAverage(object):
    """
    init_op: initializes the average
    update_op: evaluates inputs and update the average
    average: current average
        """
    def __init__(self, inputs, weight, name='cumulative_average'):
        with tf.variable_scope(name):
            flat_inputs = nest.flatten(inputs)
            self.weighted_sums = [tf.get_local_variable(x.name.split(':')[0], x.shape, dtype=tf.float32) for x in flat_inputs]
            self.total_weight = tf.get_local_variable('weight', [], dtype=tf.float32)

            self.init_op = tf.group([tf.assign(x, tf.zeros_like(x)) for x in self.weighted_sums]
                + [tf.assign(self.total_weight, 0.0)])

            self.update_op = tf.group([tf.assign(var, var + weight*x) for var,x in zip(self.weighted_sums, flat_inputs)]
                + [tf.assign(self.total_weight, self.total_weight + weight)])

            flat_average = [v / self.total_weight for v in self.weighted_sums]
            
            self.average = nest.pack_sequence_as(inputs, flat_average)


def compute_parallel_and_average(model_fn, inputs_list, averaging_device=None, *args, **kwargs):
    """
    Args:
        model_fn:
            Args:
                inputs: dictionary or list of tensors
                *args, **kwargs
            Returns:
                tuple of (outputs, weight). outputs is a nested structure of Tensors
        inputs_list: List of nested structure (dict, list) of tensors which is input to model_fn
    Returns:
        averaged outputs of model_fn and the total weight
        """
    outputs_list = []
    weight_list = []
    for i, inputs in enumerate(inputs_list):
        with tf.device('/gpu:{}'.format(i)), tf.name_scope('tower_{}'.format(i)):
            outputs, weight = model_fn(inputs, *args, **kwargs)
            outputs_list.append(outputs)
            weight_list.append(weight)
    
    with tf.device(averaging_device):
        sum_weight = tf.add_n(weight_list)
        norm_weight_list = [w / sum_weight for w in weight_list]

        flat_outputs_list = [nest.flatten(outputs) for outputs in outputs_list]
        flat_w_outputs_list = [[x * w for x in outputs] for outputs, w in zip(flat_outputs_list, norm_weight_list)]
        flat_outputs_avg = [tf.add_n(tensor) for tensor in zip(*flat_w_outputs_list)]
        outputs_avg = nest.pack_sequence_as(outputs_list[0], flat_outputs_avg)

    return outputs_avg, sum_weight

def non_even_split(inputs, n):
    """
    Args:
        inputs: nested structure
    Returns:
        list of outputs (nested structure).
        Each structure contains elements with the same batch size B. Different structure can have different B
        """
    flat_inputs = nest.flatten(inputs)
    with tf.name_scope('split_along_batch'):
        batch_size = tf.shape(flat_inputs[0])[0]
        remainder = tf.floormod(batch_size, n, name='remainder')
        quotient = tf.floor_div(batch_size, n, name='quotient')
        split_shape = tf.concat(
            [tf.fill([remainder], quotient + 1), tf.fill([n - remainder], quotient)],
            axis=0, name='split_shape')

        # list of list of a split tensor
        flat_inputs_split = [tf.split(x, split_shape, axis=0, num=n) for x in flat_inputs]
        # list of nested structure
        list_inputs = [nest.pack_sequence_as(inputs, x) for x in zip(*flat_inputs_split)]

    return list_inputs

def compute_parallel_and_concat(model_fn, inputs, n_parallel, split_device=None, concat_device=None, *args, **kwargs):
    # inputs can be a nested structure. In that case, every element in the structure must be a batch with the
    # same batch size.

    with tf.device(split_device):
        list_inputs = non_even_split(inputs, n_parallel)    

    list_outputs = []
    for i, inputs in enumerate(list_inputs):
        with tf.device('/gpu:{}'.format(i)), tf.name_scope('tower_{}'.format(i)):
            list_outputs.append(model_fn(inputs, *args, **kwargs))
    
    with tf.device(concat_device):
        list_flat_outputs = [nest.flatten(x) for x in list_outputs]
        flat_outputs_concat = [tf.concat(tensors, axis=0) for tensors in zip(*list_flat_outputs)]
        outputs_concat = nest.pack_sequence_as(inputs, flat_outputs_concat)

    return outputs_concat

def compute_parallel(model_fn, inputs_list, *args, **kwargs):
    """
    Args:
        inputs_list: list of nested structure
    Returns:
        list of the same nested structure as inputs_list
        """
    outputs_list = []
    for i, inputs in enumerate(inputs_list):
        with tf.device('/gpu:{}'.format(i)), tf.name_scope('tower_{}'.format(i)):
            outputs_list.append(model_fn(inputs, *args, **kwargs))
    
    return outputs_list
    

def custom_summary(summary_dict):
    """
    Args:
        tag_value: list of (tag, value)"""
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in summary_dict.items()])

def tf_restorable_vars(checkpoint, var_list=None, unexist_ok=None):
    """
    Args:
        checkpoint: checkpoint name
        var_list: total set of the variables you desire to restore
        unexist_ok: list of variables which can be unloaded if not present in the checkpoint
    Returns:
        A & ~(~C & B), ~C & B (for A:var_list, B:unexist_ok, C:exist)

    """
    var_list = var_list or tf.global_variables()
    unexist_ok = unexist_ok or []
    reader = tf.train.NewCheckpointReader(name)

    ignored = set((v for v in unexist_ok if not reader.has_tensor(v.op)))
    if type(var_list) == dict:
        ret = {k: v for k, v in var_list.items() if not (v in ignored)}
    elif type(var_list) == list:
        ret = [v for v in var_list if not (v in ignored)]
    elif:
        assert False
            
    return ret, ignored
