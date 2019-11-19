import os, sys, codecs, argparse, time, json
from pathlib import Path
from logging import getLogger, StreamHandler, DEBUG, basicConfig
logger = getLogger(__name__)
logger.setLevel(DEBUG)
import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np
np.random.seed(seed=0)

from components.utils import *
from components.model import *
from components import dataprocessing
from inference import Inference

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

class StatusVar:
    def __init__(self, name, init, dtype):
        with tf.variable_scope(name):
            self.var = tf.get_variable(name, dtype=dtype, initializer=init, trainable=False)
            self.ph = tf.placeholder(dtype, self.var.shape)
            self.assign = self.var.assign(self.ph)

    def set(self, value):
        tf.get_default_session().run(self.assign, feed_dict={self.ph: value})

class ValidationScore:
    def __init__(self, name, early_stop_step, early_stop_epoch):
        with tf.variable_scope(name):
            self.keys = ['score', 'step', 'epoch']
            self.svs = [StatusVar(n, init, dt) for n,init,dt in
                zip(self.keys, (-np.inf, 2**30, 2**30), (tf.float32, tf.int32, tf.int32))]
        self.step_limit, self.epoch_limit = early_stop_step, early_stop_epoch
        self.early_stop_step, self.early_stop_epoch = early_stop_step, early_stop_epoch

    def get(self):
        return tf.get_default_session().run([sv.var for sv in self.svs])

    def update(self, score, step, epoch):
        mscore, mstep, mepoch = self.get()

        if score > mscore:
            tf.get_default_session().run([sv.assign for sv in self.svs],
                feed_dict={sv.ph: value for sv, value in zip(self.svs, (score, step, epoch))})
            return True

        return False

    def should_stop(self, step, epoch):
        mscore, mstep, mepoch = self.get()

        # early stopping test
        if step - mstep > self.early_stop_step or epoch - mepoch > self.early_stop_epoch:
            return True

        return False

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
    parser.add_argument('--central_device_data_parallel', default=None,
                        help='device where NN parameters are placed')
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--inherit_no_bleu_improve', default=10, type=int)
    parser.add_argument('--basedir', default=None, type=str)

    args = parser.parse_args()

    # model's working directory
    model_dir = os.path.abspath(args.model_dir)

    # load model_config.py
    sys.path.insert(0, model_dir)
    import model_config
    params = model_config.params

    # log directory MODELDIR/log
    logdir = args.model_dir + '/log'
    os.makedirs(logdir, exist_ok=True)
    with codecs.open(logdir + '/config.json', 'w') as f:
        json.dump(params, f, ensure_ascii=False, indent=4)

    # Change directory if specified in the commad line params or config
    os.chdir(args.basedir or params["basedir"] or '.')

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

    # epoch status
    epoch_sv = StatusVar('epoch', 0, tf.int32)

    # validation status
    with tf.variable_scope('validation_status'):
        conf = params['train']['stop']['early_stopping']
        if conf['type'] == 'step':
            s, e = conf['n'], 2**30
        else:
            s, e = 2**30, conf['n']
        metric_score = ValidationScore('metric_score', s, e)
        loss_score = ValidationScore('loss', s, e)

    # when to stop training
    def should_stop(step, epoch):
        # stop for iteration limit
        conf = params["train"]["stop"]["limit"]
        if conf["type"] == "step" and conf["n"] < step:
            logger.info('stop for the step limit'); return True
        elif conf["type"] == "epoch" and conf["n"] < epoch:
                logger.info('stop for the epoch limit'); return True

        # early stopping
        if metric_score.should_stop(step, epoch) and loss_score.should_stop(step, epoch):
            logger.info('early stopping') ; return True

        return False


    # For BLEU evaluation
    inference = Inference(model_config, model, n_gpus=args.n_gpus, n_cpu_cores=args.n_cpu_cores)

    # Saver and Summary directories
    summary_dir = logdir + '/summary'
    checkpoint_dir = logdir + '/checkpoint'
    sup_checkpoint_dir = logdir + '/sup_checkpoint'
    for p in [summary_dir, checkpoint_dir, sup_checkpoint_dir]:
        if not os.path.exists(p):
            os.mkdir(p)

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

        # set inference's session
        inference.make_session(sess)

        # initialization
        logger.info('Initializing.')
        sess.run(init_op)
        logger.info('Initialization done.')

        # Savers (restorer, periodic saver, max_valid saver)
        # - restorer
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            restorer = None; logger.debug('Checkpoint was not found.')
        else:
            v_list, ignore = tf_restorable_vars(latest_checkpoint, unexist_ok=None)
            restorer = tf.train.Saver(v_list, max_to_keep=1)
            if len(ignore) > 0:
                logger.info('Variables not restored: {}'.format(', '.join(v.op for v in ignore)))
            logger.info('Restoring from: {}'.format(latest_checkpoint))
            restorer.restore(sess, latest_checkpoint)
        # - periodic saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        # - max validation score saver
        sup_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # step and epoch
        global_step = global_step_var.eval()
        epoch = epoch_sv.var.eval()

        # stop test
        _should_stop =  should_stop(global_step, epoch)

        # Training epoch loop
        last_validation_step = global_step
        while not _should_stop:
            # increment epoch
            epoch += 1 ; epoch_sv.set(epoch)

            # Initialize train dataset Iterator
            sess.run(train_iterator.initializer)

            # processing time
            step_time, sec_per_step = time.time(), 10

            logger.info('New epoch starts.')
            # Training loop
            while not _should_stop:
                # Validation, checkpoint and determine whether to stop training
                if global_step % params["train"]["stop"]["early_stopping"]["test_period"] == 0 \
                    and global_step != last_validation_step:
                    last_validation_step = global_step

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

                    # update max score
                    if metric_score.update(score, global_step, epoch):
                        # save current states into supremum checkpoint
                        sup_saver.save(sess, sup_checkpoint_dir + '/model', global_step=global_step)
                    # add score to summary
                    summary_writer.add_summary(custom_summary({'dev score': score}), global_step)

                    # Save parameters
                    logger.info('Saving params. step: {}, score: {}'.format(global_step, score))
                    saver.save(sess, checkpoint_dir + '/model', global_step=global_step)

                    # stopping test
                    _should_stop = should_stop(global_step, epoch)
                    if _should_stop: break

                try:


                    if global_step % 600 == 0:
                        _, train_summary, global_step = sess.run([train_info['train_op'],
                                                                train_summary_op,
                                                                global_step_var])

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

                        # update dev loss score
                        dev_loss_score = - sess.run(dev_info_avg.average['loss'])
                        loss_score.update(dev_loss_score, global_step, epoch)

                        # stop test by dev loss
                        _should_stop = should_stop(global_step, epoch)
                        if _should_stop:
                            # save the lastest state
                            logger.info('Saving params. step: {}, score: {}'.format(global_step, score))
                            saver.save(sess, checkpoint_dir + '/model', global_step=global_step)

                            break


                    else:
                        global_step, _ = sess.run([global_step_var, train_info['train_op']])

                    # calculate time
                    _, step_time = step_time, time.time()
                    sec_per_step = sec_per_step * 0.99 + (step_time - _) * 0.01
                    sys.stderr.write('{} s/step. epoch: {}, glb: {}\t\r'.format(
                        sec_per_step, epoch, global_step))
                    sys.stderr.flush()
                except tf.errors.OutOfRangeError:
                    break


if __name__ == '__main__':
    train()


"""
notes

Saver and checkpoints
    - 3 savers are used
        - for restoration of the latest checkpoint
        - for saving of periodic checkpoints
        - for saving of best states which maximizes the validation score

"""
