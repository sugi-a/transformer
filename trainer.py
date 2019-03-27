import os
import sys
import codecs
import argparse
import time
from logging import getLogger, StreamHandler, DEBUG, basicConfig
logger = getLogger(__name__)
logger.setLevel(DEBUG)
import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np

from utils import *
from model import *
import dataprocessing


def get_train_info(y, y_len, dec_outputs):
    """returns loss, accuracy, ntokens"""

    is_target = tf.sequence_mask(y_len, tf.shape(y)[1], dtype=tf.float32)
    ntokens = tf.reduce_sum(is_target)
    y_onehot = tf.one_hot(y, tf.shape(dec_outputs)[-1])
    prediction = tf.cast(tf.math.argmax(dec_outputs, axis=-1), tf.int32)
    #loss = tf.losses.softmax_cross_entropy(y_onehot, dec_outputs, is_target, label_smoothing=0.1)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=label_smoothing(y_onehot),
        logits=dec_outputs)
    loss = tf.reduce_sum(loss * is_target) / ntokens
    
    accuracy = tf.reduce_sum(tf.cast(tf.equal(y, prediction), dtype=tf.float32) * is_target) / ntokens

    return loss, accuracy, ntokens

def get_learning_rate(step, warm_up_step, embed_size):
    WARM_UP_STEP = tf.cast(warm_up_step, tf.float32) 
    step = tf.cast(step, tf.float32)
    rate = embed_size ** (-0.5) * tf.minimum(tf.rsqrt(step),
                                                step * tf.pow(WARM_UP_STEP, -1.5))
    return rate

def train():
    basicConfig()
    logger.info('Start.')
    print('start')
    
    parser = argparse.ArgumentParser('train transformer')
    parser.add_argument('--model_dir', required=True,
        help="Path of the directory which has model_config.py (required)")
    parser.add_argument('--n_cpu_cores', default=8, type=int,
        help="Number of CPU cores this script can use when preprocessing dataset")
    parser.add_argument('--n_gpus', default=1, type=int,
        help="Number of GPUs.")
    args = parser.parse_args()

#    parser = argparse.ArgumentParser()
#    parser.add_argument('--model_dir',
#                        required=True,
#                        help='Path to the directory containing `model_config.py`')
#    parser.add_argument('--n_cpu_cores',
#                        default=8,
#                        type=int,
#                        help='Number of cpu cores used by `tf.data.Dataset.map`')
#    parser.add_argument('--n_gpus',
#                        default=1,
#                        type=int,
#                        help='Number of GPUs available')
#    args = parser.parse_args()

    sys.path.insert(0, args.model_dir)
    import model_config
    logger.info('model_config has been loaded from {}'.format(model_config.__file__))
    from model_config import Hyperparams
    from model_config import Config

    from translate import Inference

    # train dataset
    train_data = (dataprocessing.make_dataset_source_target(
                    Config.source_train_tok,
                    Config.target_train_tok,
                    Config.vocab_source,
                    Config.vocab_target,
                    UNK_ID=Config.UNK_ID,
                    EOS_ID=Config.EOS_ID,
                    shuffle_size=2000*1000,
                    ncpu=args.n_cpu_cores)
        .filter(lambda x,y: tf.logical_and(tf.greater(Hyperparams.maxlen, x[1]),
                                         tf.greater(Hyperparams.maxlen, y[1])))
        .padded_batch(Hyperparams.batch_size // args.n_gpus,
                      (([None], []), ([None], [])),
                      ((Config.PAD_ID, 0), (Config.PAD_ID, 0)))
        .prefetch(args.n_gpus * 2))

    # dev dataset
    dev_data = dataprocessing.make_dataset_source_target(
                    Config.source_dev_tok,
                    Config.target_dev_tok,
                    Config.vocab_source,
                    Config.vocab_target,
                    UNK_ID=Config.UNK_ID,
                    EOS_ID=Config.EOS_ID,
                    ncpu=args.n_cpu_cores
                )\
        .padded_batch(model_config.Hyperparams.batch_size // args.n_gpus,
                      (([None], []), ([None], [])),
                      ((Config.PAD_ID, 0), (Config.PAD_ID, 0)))\
        .prefetch(args.n_gpus * 2)

    # train/dev iterators and input tensors
    train_iterator = train_data.make_initializable_iterator()
    dev_iterator = dev_data.make_initializable_iterator()
    
    train_parallel_inputs = [train_iterator.get_next() for i in range(args.n_gpus)]
    dev_parallel_inputs = [dev_iterator.get_next() for i in range(args.n_gpus)]
    
    # model
    model = Transformer(Hyperparams, Config, name='transformer')
    
    # place variables on device:CPU
    with tf.device('/cpu:0'):
        model.instanciate_vars()

    # train ops and info
    global_step_var = tf.train.get_or_create_global_step()
    lr = get_learning_rate(global_step_var, Hyperparams.warm_up_step, Hyperparams.embed_size)
    optimizer = tf.train.AdamOptimizer(lr)
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model.scope_name)

    def _train_info(inputs):
        (x, x_len), (y, y_len) = inputs
        logits = model.get_logits(x, y, x_len, y_len, True)
        loss, accuracy, ntokens = get_train_info(y, y_len, logits)
        grad_vars = optimizer.compute_gradients(loss, var_list=train_vars)
        grads, _ = zip(*grad_vars)

        logger.debug(str(all([x is y for x,y in zip(train_vars, _)])))

        return {'loss': loss, 'accuracy': accuracy, 'gradient': grads}, ntokens

    train_info, _ = compute_parallel_and_average(_train_info, train_parallel_inputs, averaging_device='/cpu:0')
    
    # updating parameters
    with tf.device('/cpu:0'):
        grad_vars = [(grad, v) for grad, v in zip(train_info['gradient'], train_vars)]
        train_info['train_op'] = optimizer.apply_gradients(grad_vars, global_step=global_step_var)

    # dev ops and info
    def _dev_info(inputs):
        (x, x_len), (y, y_len) = inputs
        logits = model.get_logits(x, y, x_len, y_len, False)
        loss, accuracy, ntokens = get_train_info(y, y_len, logits)
        return {'loss': loss, 'accuracy': accuracy}, ntokens

    dev_info, dev_ntokens = compute_parallel_and_average(_dev_info, dev_parallel_inputs, averaging_device='/cpu:0')
    dev_info_avg = CumulativeAverage(dev_info, dev_ntokens, name='dev_info_avg')

    # For BLEU evaluation
    inference = Inference(model, n_gpus=args.n_gpus, n_cpu_cores=args.n_cpu_cores)

    # Saver and Summary
    summary_dir = Config.logdir + '/summary'
    checkpoint_dir = Config.logdir + '/checkpoint'
    dev_bleu_dir = Config.logdir + '/dev_bleu_log'
    for p in [Config.logdir, summary_dir, checkpoint_dir, dev_bleu_dir]:
        if not os.path.exists(p):
            os.mkdir(p)

    saver = tf.train.Saver(max_to_keep=12)

    summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())
    
    train_summary_op = tf.summary.merge([
        tf.summary.scalar('accuracy', train_info['accuracy']),
        tf.summary.scalar('loss', train_info['loss']),
        tf.summary.scalar('learning_rate', lr)
        ])

    dev_summary_op = tf.summary.merge([
        tf.summary.scalar('dev_accuracy', dev_info_avg.average['accuracy']),
        tf.summary.scalar('dev_loss', dev_info_avg.average['loss'])
    ])

    # initializer
    init_op = tf.group([tf.tables_initializer(), tf.global_variables_initializer()])

    # Session config
    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True

    with tf.Session(config=sess_config) as sess:
        logger.info('Session started.')

        # initialization
        logger.info('Initializing.')
        sess.run(init_op)
        logger.info('Initialization done.')

        # loading checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            logger.info('Checkpoint was found: {}'.format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
        else:
            logger.info('No checkpoint was found. Training starts from the beginning.')

        max_bleu = -100
        no_improvement_count = 0

        global_step = sess.run(global_step_var)

        # Training epoch loop
        while True:

            # BLEU evaluation
            logger.info('Computing BLEU score')
            score = inference.BLEU_evaluation(Config.source_dev_tok,
                                              Config.target_dev_tok,
                                              beam_size=1,
                                              session=sess,
                                              result_file_prefix=dev_bleu_dir+'/step_{}'.format(global_step))
            logger.info('Computing BLEU score done. {}'.format(score))
            no_improvement_count = no_improvement_count + 1 if score < max_bleu else 0
            max_bleu = max(score, max_bleu)
            if no_improvement_count > 4:
                break

            # add BLEU score to summary
            summary_writer.add_summary(custom_summary({'BLEU': score}), global_step)

            # Initialize train dataset Iterator
            sess.run(train_iterator.initializer)

            # Epoch-local step counter
            local_step = 0
            step_time = time.time()
            sec_per_step = 10

            logger.info('New epoch starts.')
            # Training loop
            while True:
                try:
                    if global_step % (200) == 0:
                        train_summary, global_step, _ = sess.run([train_summary_op,
                                                                  global_step_var,
                                                                  train_info['train_op']])

                        # write summary of train data
                        summary_writer.add_summary(train_summary, global_step)

                        # dev
                        sess.run(dev_info_avg.init_op)
                        sess.run(dev_iterator.initializer)
                        while True:
                            try:
                                sess.run(dev_info_avg.update_op)
                            except tf.errors.OutOfRangeError:
                                break
                        dev_summary = sess.run(dev_summary_op)
                        summary_writer.add_summary(dev_summary, global_step)

                    else:
                        global_step, _ = sess.run([global_step_var, train_info['train_op']])

                    local_step += 1
                    _ = step_time
                    step_time = time.time()
                    sec_per_step = sec_per_step * 0.9 + (step_time - _) * 0.1
                    sys.stderr.write('{} sec/step. local step: {}     \r'.format(sec_per_step, local_step))
                except tf.errors.OutOfRangeError:
                    break

            # Save parameters
            logger.info('Saving parameters. Global step: {}'.format(global_step))
            saver.save(sess, checkpoint_dir + '/' + Config.model_name, global_step=global_step)

if __name__ == '__main__':
    train()

