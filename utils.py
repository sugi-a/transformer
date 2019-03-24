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
            self.weighted_sums = [tf.get_local_variable(x.name, x.shape, dtype=tf.float32) for x in flat_inputs]
            self.total_weight = tf.get_local_variable('weight', [], dtype=tf.float32)

            self.init_op = tf.group([tf.assign(x, tf.zeros_like(x)) for x in self.weighted_sums]
                + [tf.assign(self.total_weight, 0.0)])

            self.update_op = tf.group([tf.assign(var, var + x) for var,x in zip(self.weighted_sums, flat_inputs)]
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
        flat_outputs_avg = [tf.add_n(tensor) for tensor in zip(flat_w_outputs_list)]
        outputs_avg = nest.pack_sequence_as(outputs_list[0], flat_outputs_avg)

    return outputs_avg, sum_weight

def compute_parallel_and_concat(model_fn, inputs_list, concat_device=None):
    outputs_list = []
    for i, inputs in enumerate(inputs_list):
        with tf.device('/gpu:{}'.format(i)), tf.name_scope('tower_{}'.format(i)):
            outputs = model_fn(inputs, *args, **kwargs)
            outputs_list.append(outputs)
    
    with tf.device(concat_device):
        flat_outputs_list = [nest.flatten(outputs) for outputs in outputs_list]
        flat_outputs_concat = [tf.concat(tensors, axis=0) for tensors in zip(flat_outputs_list)]
        outputs_concat = nest.pack_sequence_as(outputs_list[0], flat_outputs_concat)

    return outputs_concat

    

def custom_summary(summary_dict):
    """
    Args:
        tag_value: list of (tag, value)"""
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in summary_dict.items()])
