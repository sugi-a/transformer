import os, sys, codecs, argparse, time, json
from pathlib import Path
from logging import getLogger, StreamHandler, DEBUG, basicConfig
logger = getLogger(__name__)
logger.setLevel(DEBUG)
import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np
np.random.seed(seed=0)

from transformer.utils import *
from transformer.model import *
from transformer import dataprocessing
from transformer.inference import Inference

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
    basicConfig(level=DEBUG)
    logger.info('Start.')
    print('start')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True,
                        help='Path to the directory containing `model_config.py`')
    parser.add_argument('--n_cpu_cores', default=8, type=int,
                        help='Number of cpu cores used by `tf.data.Dataset.map`')
    parser.add_argument('--n_gpus', default=1, type=int,
                        help='Number of GPUs available')
    parser.add_argument('--central_device_data_parallel', default='None',
                        help='"None" is converted to `None`. Default is "None"')
    parser.add_argument('--random_seed', default=0, type=int)

    parser.add_argument('--inherit_no_bleu_improve', default=10, type=int)

    args = parser.parse_args()
    if args.central_device_data_parallel == 'None':
        args.central_device_data_parallel = None

    sys.path.insert(0, args.model_dir)
    import model_config
    params = model_config.params

    # train dataset
    if params["train"]["batch"]["fixed_capacity"]:
        logger.info('Calling make_dataset_source_target_const_capacity_batch')
        train_data = dataprocessing.make_dataset_source_target_const_capacity_batch(
            params["train"]["data"]["source_train"],
            params["train"]["data"]["target_train"],
            params["vocab"]["source_dict"],
            params["vocab"]["target_dict"],
            UNK_ID=params["vocab"]["UNK_ID"],
            EOS_ID=params["vocab"]["EOS_ID"],
            PAD_ID=params["vocab"]["PAD_ID"],
            maxlen=params["train"]["data"]["maxlen"],
            batch_capacity=params["train"]["batch"]["capacity"] // args.n_gpus,
            order_mode='sort' if params["train"]["batch"]["sort"] else None,
            allow_skip=True)
        train_data = train_data.prefetch(args.n_gpus * 2)
    else:
        train_data = (dataprocessing.make_dataset_source_target(
                        params["train"]["data"]["source_train"],
                        params["train"]["data"]["target_train"],
                        params["vocab"]["source_dict"],
                        params["vocab"]["target_dict"],
                        UNK_ID=params["vocab"]["UNK_ID"],
                        EOS_ID=params["vocab"]["EOS_ID"],
                        shuffle_size=2000*1000,
                        ncpu=args.n_cpu_cores)
            .filter(lambda x,y: tf.logical_and(
                tf.greater(params["train"]["data"]["maxlen"], x[1]),
                tf.greater(params["train"]["data"]["maxlen"], y[1])))
            .padded_batch(params["train"]["batch"]["size"] // args.n_gpus,
                  (([None], []), ([None], [])),
                  ((params["vocab"]["PAD_ID"], 0), (params["vocab"]["PAD_ID"], 0)))
            .prefetch(args.n_gpus * 2))

    # dev dataset
    dev_data = dataprocessing.make_dataset_source_target(
                    params["train"]["data"]["source_dev"],
                    params["train"]["data"]["target_dev"],
                    params["vocab"]["source_dict"],
                    params["vocab"]["target_dict"],
                    UNK_ID=params["vocab"]["UNK_ID"],
                    EOS_ID=params["vocab"]["EOS_ID"],
                    ncpu=args.n_cpu_cores
                )\
        .padded_batch(params["train"]["batch"]["size"] // args.n_gpus,
                      (([None], []), ([None], [])),
                      ((params["vocab"]["PAD_ID"], 0), (params["vocab"]["PAD_ID"], 0)))\
        .prefetch(args.n_gpus * 2)

    # train/dev iterators and input tensors
    train_iterator = train_data.make_initializable_iterator()
    dev_iterator = dev_data.make_initializable_iterator()
    
    train_parallel_inputs = [train_iterator.get_next() for i in range(args.n_gpus)]
    dev_parallel_inputs = [dev_iterator.get_next() for i in range(args.n_gpus)]
    
    # model
    tf.set_random_seed(args.random_seed)
    model = Transformer(params, name='transformer')
    
    # place variables on device:CPU
    with tf.device(args.central_device_data_parallel):
        model.instanciate_vars()

    # train ops and info
    global_step_var = tf.train.get_or_create_global_step()
    lr = get_learning_rate(global_step_var, params["train"]["warm_up_step"], params["network"]["embed_size"])
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

    if args.n_gpus == 1:
        train_info, _ = _train_info(train_parallel_inputs[0])
    else:
        train_info, _ = compute_parallel_and_average(_train_info,
                                                    train_parallel_inputs,
                                                    averaging_device=args.central_device_data_parallel)
    
    # updating parameters
    with tf.device(args.central_device_data_parallel):
        grad_vars = [(grad, v) for grad, v in zip(train_info['gradient'], train_vars)]
        train_info['train_op'] = optimizer.apply_gradients(grad_vars, global_step=global_step_var)

    # dev ops and info
    def _dev_info(inputs):
        (x, x_len), (y, y_len) = inputs
        logits = model.get_logits(x, y, x_len, y_len, False)
        loss, accuracy, ntokens = get_train_info(y, y_len, logits)
        return {'loss': loss, 'accuracy': accuracy}, ntokens

    if args.n_gpus == 1:
        dev_info, dev_ntokens = _dev_info(dev_parallel_inputs[0])
    else:
        dev_info, dev_ntokens = compute_parallel_and_average(_dev_info,
                                                            dev_parallel_inputs,
                                                            averaging_device=args.central_device_data_parallel)
    dev_info_avg = CumulativeAverage(dev_info, dev_ntokens, name='dev_info_avg')

    # For BLEU evaluation
    inference = Inference(model, n_gpus=args.n_gpus, n_cpu_cores=args.n_cpu_cores)

    # Saver and Summary
    summary_dir = params["train"]["logdir"] + '/summary'
    checkpoint_dir = params["train"]["logdir"] + '/checkpoint'
    dev_bleu_dir = params["train"]["logdir"] + '/dev_bleu_log'
    for p in [summary_dir, checkpoint_dir, dev_bleu_dir]:
        if not os.path.exists(p):
            os.mkdir(p)

    saver = tf.train.Saver(max_to_keep=20)

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
        last_check_step = 0
        # Check the past BLEUs in summary
        summary_paths = None
        try:
            summary_paths = list(map(str, Path(summary_dir).glob('*')))
        except:
            logger.info('No existing summary file.')
        if summary_paths:
            bleu_record = []
            for summary_path in summary_paths:
                summary_itr = tf.train.summary_iterator(summary_path)
                try:
                    for e in summary_itr:
                        for v in e.summary.value:
                            if v.tag == 'BLEU':
                                bleu_record.append((e.step, v.simple_value))
                except:
                    pass
            bleu_record.sort(key=lambda x:x[0])
            for rec in bleu_record:
                if rec[1] < max_bleu:
                    no_improvement_count += 1
                else:
                    no_improvement_count = 0
                max_bleu = max(max_bleu, rec[1])
            no_improvement_count = min(no_improvement_count, args.inherit_no_bleu_improve)
            logger.info('No improve count succeeded: {}'.format(no_improvement_count))

        global_step = sess.run(global_step_var)

        # Training epoch loop
        while True:

            if no_improvement_count >= 10:
                logger.info('Early stopping.')
                break

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
                    if global_step >= last_check_step + 5000:
                        last_check_step = global_step
                        # evaluation
                        if hasattr(model_config, 'validation_metric'):
                            # You can define a function `train_time_metric` in model_config
                            # Args:
                            #   inference: instance of Inference
                            #   session: current session whose graph includes inference and the all parameters have been
                            #   loaded.
                            # Returns:
                            #   score (float)
                            logger.info('Evaluating by the custom metric')
                            score = model_config.validation_metric(global_step, inference)
                        else:
                            logger.info('Custom metric was not found.')
                            
                            # negative loss on development data as score
                            sess.run(dev_info_avg.init_op)
                            sess.run(dev_iterator.initializer)
                            while True:
                                try:
                                    sess.run(dev_info_avg.update_op)
                                except tf.errors.OutOfRangeError:
                                    break
                            score = sess.run(dev_info_avg.average['loss'])
                        no_improvement_count = no_improvement_count + 1 if score < max_bleu else 0
                        if no_improvement_count >= 10:
                            break
                        max_bleu = max(score, max_bleu)
                        # add BLEU score to summary
                        summary_writer.add_summary(custom_summary({'dev score': score}), global_step)

                        # Save parameters
                        logger.info('Saving parameters. Global step: {}, no improvement: {}'.format(global_step, no_improvement_count))
                        saver.save(sess, checkpoint_dir + '/model', global_step=global_step)


                    if global_step % 400 == 0:
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

if __name__ == '__main__':
    train()

