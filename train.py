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
    with codecs.open(params["train"]["logdir"] + '/config.json', 'w') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)

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
            order_mode='sort' if params["train"]["batch"]["sort"] else "shuffle",
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

    # epoch variable
    epoch_var = tf.get_variable('epoch', dtype=tf.int32, initializer=0)
    epoch_var_ph = tf.placeholder(tf.int32, [])
    epoch_var_setter = epoch_var.assign(epoch_var_ph)

    # validation score variable
    score_var = tf.get_variable('dev_score', dtype=tf.float32, initializer=-1e20)
    score_var_ph = tf.placeholder(tf.float32, [])
    score_var_setter = score_var.assign(score_var_ph)

    # For BLEU evaluation
    inference = Inference(model_config, model, n_gpus=args.n_gpus, n_cpu_cores=args.n_cpu_cores)

    # Saver and Summary directories
    summary_dir = params["train"]["logdir"] + '/summary'
    checkpoint_dir = params["train"]["logdir"] + '/checkpoint'
    sup_checkpoint_dir = params["train"]["logdir"] + '/sup_checkpoint'
    for p in [summary_dir, checkpoint_dir, sup_checkpoint_dir]:
        if not os.path.exists(p):
            os.mkdir(p)

    # Savers. sup_saver is the one to save parameters with good results
    saver = tf.train.Saver(max_to_keep=5)
    sup_saver = tf.train.Saver(max_to_keep=5)

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
        # checkpoint with the best score
        latest_sup_checkpoint = tf.train.latest_checkpoint(sup_checkpoint_dir)
        max_step, max_epoch, max_score = 1e10, 1e10, -1e10
        if latest_sup_checkpoint is not None:
            sup_saver.restore(sess, latest_sup_checkpoint)
            max_step, max_epoch = sess.run((global_step_var, epoch_var))
            max_score = sess.run(score_var)

        # latest checkpoint
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is not None:
            logger.info('Checkpoint was found: {}'.format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            global_step, epoch = sess.run((global_step_var, epoch_var))
            last_save_step = global_step
        else:
            logger.info('No checkpoint was found. Training starts from the beginning.')
            global_step, epoch, max_score = sess.run((global_step_var, epoch_var, score_var))

        def stop_test(epoch, step, max_epoch, max_step):
            conf = params["train"]["stop"]
            if conf["limit"]["type"] == "epoch" and conf["limit"]["n"] < epoch:
                    logger.info('stop for the epoch limit'); return True
            if conf["limit"]["type"] == "step" and conf["limit"]["n"] < step:
                logger.info('stop for the step limit'); return True

            conf = conf["early_stopping"]
            if conf["type"] == "epoch" and conf["n"] < epoch - max_epoch:
                logger.info('Early stopping by epoch limit') ; return True
            elif conf["type"] == "step" and conf["n"] < step - max_step:
                logger.info('Early stopping by global step limit'); return True

        logger.debug('{},{},{},{}'.format(epoch, global_step, max_epoch, max_step))
        # Training epoch loop
        should_stop = False
        while not should_stop:
            epoch += 1
            sess.run(epoch_var_setter, feed_dict={epoch_var_ph: epoch})

            # stop test
            should_stop = stop_test(epoch, global_step, max_epoch, max_step)
            if should_stop: break

            # Initialize train dataset Iterator
            sess.run(train_iterator.initializer)

            # Epoch-local step counter
            local_step = 0
            step_time = time.time()
            sec_per_step = 10

            logger.info('New epoch starts.')
            # Training loop
            while not should_stop:
                # check step limit
                should_stop = stop_test(epoch, global_step, max_epoch, max_step)
                if should_stop: break
                try:
                    if global_step % 1500 == 0:
                        # validation
                        if hasattr(model_config, 'validation_metric'):
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
                            score = - sess.run(dev_info_avg.average['loss'])
                        # store score into score variable
                        sess.run(score_var_setter, feed_dict={score_var_ph: score})

                        if score > max_score:
                            max_step = global_step
                            max_epoch = epoch
                            max_score = score

                            # save superior checkpoint
                            sup_saver.save(sess, sup_checkpoint_dir + '/model', global_step=global_step)

                        should_stop = stop_test(epoch, global_step, max_epoch, max_step)
                        if should_stop: break

                        # add score to summary
                        summary_writer.add_summary(custom_summary({'dev score': score}), global_step)

                        # Save parameters
                        logger.info('Saving parameters. Global step: {}, no improvement: {}'
                            .format(global_step, global_step - max_step))
                        saver.save(sess, checkpoint_dir + '/model', global_step=global_step)
                        last_save_step = global_step


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

                    # calculate time
                    local_step += 1
                    _ = step_time
                    step_time = time.time()
                    sec_per_step = sec_per_step * 0.9 + (step_time - _) * 0.1
                    sys.stderr.write('{} sec/step. local step: {}     \r'.format(sec_per_step, local_step))
                except tf.errors.OutOfRangeError:
                    break
        # last parameter save
        if last_save_step != global_step:
            logger.info('saving parameters on finishing training')
            saver.save(sess, checkpoint_dir + '/model', global_step=global_step)
            last_save_step = global_step



if __name__ == '__main__':
    train()

