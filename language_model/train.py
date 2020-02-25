import os, sys, argparse, time, json
from logging import getLogger, DEBUG, basicConfig; logger = getLogger(__name__)

import tensorflow as tf
from tensorflow.contrib.framework import nest
import numpy as np

from . import datasetloader
from .language_model import DecoderLanguageModel, load_model_config
from ..components.utils import *
from ..components import dataprocessing
from ..components.model import label_smoothing

class Train:
    def __init__(self, model_dir, n_gpus=1, n_cpu_cores=4, random_seed=0):

        # model's working directory
        self.model_dir = model_dir

        # log directory MODELDIR/log
        self.logdir = self.model_dir + '/log'
        os.makedirs(self.logdir, exist_ok=True)

        # load model_config.py
        self.config = load_model_config(model_dir, 'lm_config.py')
        self.params = self.config['params']

        # Computation options
        self.n_gpus = n_gpus
        self.n_cpu_cores = n_cpu_cores
        self.random_seed = random_seed

        # Log the config
        with open(self.logdir + '/config.json', 'w') as f:
            json.dump(self.params, f, ensure_ascii=False, indent=4)


    def __get_loss(self, x, x_len, dec_outputs):
        """
        Args:
            x: [BATCH_SIZE, MAXLEN]. Label (reference) sequences.
            x_len: [BATCH_SIZE]. Lengths of `x`
            dec_outputs: [BATCH_SIZE, MAXLEN, VOCAB_SIZE]. Decoder output sequences.
        returns:
            loss, accuracy and ntokens Tensors"""

        # Target mask indicating non-<PAD>
        is_target = tf.sequence_mask(x_len, tf.shape(x)[1], dtype=tf.float32)

        # Number of non-<PAD> tokens
        ntokens = tf.cast(tf.reduce_sum(x_len), tf.float32)

        # One-hot representation of the ref. sequences [BATCH_SIZE, MAXLEN, NVOCAB]
        x_onehot = tf.one_hot(x, tf.shape(dec_outputs)[-1])

        # Label smoothing
        pred_smoothed = label_smoothing(x_onehot, self.params["train"]["label_smoothing"])

        # Top-1 prediction by the model [BATCH_SIZE, MAXLEN]
        prediction = tf.cast(tf.math.argmax(dec_outputs, axis=-1), tf.int32)

        # Loss function
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=pred_smoothed,
            logits=dec_outputs)
        loss = tf.reduce_sum(loss * is_target) / ntokens

        # Top-1 prediction accuracy
        accuracy = tf.reduce_sum(tf.cast(tf.equal(x, prediction), dtype=tf.float32) * is_target) / ntokens

        return loss, accuracy, ntokens
        

    def __get_train_info(self, inputs):
        (x, x_len) = inputs
        x_in, x_out = x[:, :-1], x[:, 1:]
        io_len = x_len - 1

        # Logits [BATCH_SIZE, MAXLEN, NVOCAB]
        logits = self.model.get_logits(x_in, True)

        # Loss info
        loss, accuracy, ntokens = self.__get_loss(x_out, io_len, logits)

        # Gradient
        grad_vars = self.optimizer.compute_gradients(loss, var_list=self.train_vars)
        grads, _ = zip(*grad_vars)

        return {'loss': loss, 'accuracy': accuracy, 'gradient': grads}, ntokens


    def __get_dev_info(self, inputs):
        (x, x_len) = inputs
        x_in, x_out = x[:, :-1], x[:, 1:]
        io_len = x_len - 1

        logits = self.model.get_logits(x_in, False)
        loss, accuracy, ntokens = self.__get_loss(x_out, io_len, logits)
        return {'loss': loss, 'accuracy': accuracy}, ntokens

        
    def __get_learning_rate(self, step):
        step = tf.cast(step, tf.float32)
        warm_up_step = tf.cast(self.params["train"]["warm_up_step"], tf.float32)
        embed_size = self.params["network"]["embed_size"]
        rate = embed_size ** (-0.5) * tf.minimum(tf.rsqrt(step), step * tf.pow(warm_up_step, -1.5))
        return rate


    # when to stop training
    def __should_stop(self, step, epoch):
        # stop for iteration limit
        conf = self.params["train"]["stop"]["limit"]
        if conf["type"] == "step" and conf["n"] < step:
            logger.info('stop for the step limit'); return True
        elif conf["type"] == "epoch" and conf["n"] < epoch:
                logger.info('stop for the epoch limit'); return True

        # early stopping
        if self.metric_score.should_stop(step, epoch) and self.loss_score.should_stop(step, epoch):
            logger.info('early stopping') ; return True

        return False

    def train(self):
        logger.info('Transformer training.')
        
        np.random.seed(0)
        params = self.params

        # train dataset
        self.vocab = dataprocessing.Vocabulary(
            params['vocab']['dict'],
            PAD_ID=params['vocab']['PAD_ID'],
            EOS_ID=params['vocab']['EOS_ID'],
            UNK_ID=params['vocab']['UNK_ID'])

        if params['train']['batch']['mode'] == 'random_sliding_window':
            bconf = params['train']['batch']['config']

            DATASET_STATE_FILE = os.path.join(self.logdir, 'train_data_loading_state.json')
            train_data_gen = datasetloader.RandomSlidingWindow(
                params['train']['data']['train'],
                self.vocab,
                bconf['window_size'],
                keep_remainder_larger_equal=bconf['keep_remainder_larger_equal'],
                random=True,
                state_log_file=DATASET_STATE_FILE)

            train_data = tf.data.Dataset.from_generator(
                    train_data_gen,
                    tf.int32,
                    tf.TensorShape([None])) \
                .shuffle(bconf['shuffle_buf_size']) \
                .map(lambda x: (x, tf.shape(x)[0])) \
                .padded_batch(
                    bconf['batch_size'] // self.n_gpus,
                    (tf.TensorShape([bconf['window_size']]), tf.TensorShape([])),
                    (params['vocab']['PAD_ID'], 0),
                    False) \
                .prefetch(2 * self.n_gpus)
        else:
            assert False

        # dev dataset
        dev_data_gen = datasetloader.RandomSlidingWindow(
            [params['train']['data']['dev']],
            self.vocab,
            bconf['window_size'],
            keep_remainder_larger_equal=1,
            random=False)

        dev_data = tf.data.Dataset.from_generator(
                dev_data_gen,
                tf.int32,
                tf.TensorShape([None])) \
            .map(lambda x: (x, tf.shape(x)[0])) \
            .padded_batch(
                bconf['batch_size'] // self.n_gpus,
                (tf.TensorShape([None]), tf.TensorShape([])),
                (params['vocab']['PAD_ID'], 0),
                False) \
            .prefetch(2 * self.n_gpus)

        # train/dev iterators and input tensors
        train_iterator = train_data.make_initializable_iterator()
        dev_iterator = dev_data.make_initializable_iterator()
        
        train_parallel_inputs = [train_iterator.get_next() for i in range(self.n_gpus)]
        dev_parallel_inputs = [dev_iterator.get_next() for i in range(self.n_gpus)]
        
        # model
        tf.set_random_seed(self.random_seed)
        model = DecoderLanguageModel(params)
        self.model = model
        
        # initialize variables
        model.instanciate_vars()

        # train ops and info
        global_step_var = tf.train.get_or_create_global_step()
        lr = self.__get_learning_rate(global_step_var)
        self.optimizer = tf.train.AdamOptimizer(lr)
        self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=model.scope_name)

        if self.n_gpus == 1:
            train_info, _ = self.__get_train_info(train_parallel_inputs[0])
            dev_info, dev_ntokens = self.__get_dev_info(dev_parallel_inputs[0])
        else:
            train_info, _ = compute_parallel_and_average(self.__get_train_info, train_parallel_inputs)
            dev_info, dev_ntokens = compute_parallel_and_average(self.__get_dev_info, dev_parallel_inputs)
        
        # updating parameters
        grad_vars = [(grad, v) for grad, v in zip(train_info['gradient'], self.train_vars)]
        train_info['train_op'] = self.optimizer.apply_gradients(grad_vars, global_step=global_step_var)

        # dev ops and info
        dev_info_avg = CumulativeAverage(dev_info, dev_ntokens, name='dev_info_avg')

        # epoch status
        epoch_sv = TFVar('epoch', 0, tf.int32)

        # validation status
        with tf.variable_scope('validation_status'):
            conf = params['train']['stop']['early_stopping']
            if conf['type'] == 'step':
                s, e = conf['n'], 2**30
            else:
                s, e = 2**30, conf['n']
            self.metric_score = ValidationScore('metric_score', s, e)
            self.loss_score = ValidationScore('loss', s, e)


        # Saver and Summary directories
        summary_dir = self.logdir + '/summary'
        checkpoint_dir = self.logdir + '/checkpoint'
        sup_checkpoint_dir = self.logdir + '/sup_checkpoint'
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
        sess_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        sess_config.allow_soft_placement = True

        with tf.Session(config=sess_config) as sess:
            logger.info('Session started.')

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
            sup_saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=1)

            # step and epoch
            global_step = global_step_var.eval()
            epoch = epoch_sv.var.eval()
            score = 0

            # stop test
            _should_stop =  self.__should_stop(global_step, epoch)

            # Training epoch loop
            last_validation_step = global_step

            # processing time
            step_time, sec_per_step = time.time(), 10

            while not _should_stop:
                # increment epoch
                epoch += 1 ; epoch_sv.set(epoch)

                # Initialize train dataset Iterator
                sess.run(train_iterator.initializer)

                logger.info('New epoch starts.')
                # Training loop
                while not _should_stop:
                    # Validation, checkpoint and determine whether to stop training
                    if global_step % params["train"]["stop"]["early_stopping"]["test_period"] == 0 \
                        and global_step != last_validation_step:
                        last_validation_step = global_step

                        # validation
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
                        if self.metric_score.update(score, global_step, epoch):
                            # save current states into supremum checkpoint
                            sup_saver.save(sess, sup_checkpoint_dir + '/model', global_step=global_step)
                        # add score to summary
                        summary_writer.add_summary(custom_summary({'dev score': score}), global_step)

                        # Save parameters
                        logger.info('Saving params. step: {}, score: {}'.format(global_step, score))
                        saver.save(sess, checkpoint_dir + '/model', global_step=global_step)

                        # stopping test
                        _should_stop = self.__should_stop(global_step, epoch)
                        if _should_stop: break

                    try:


                        if global_step % 1000 == 0:
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
                            self.loss_score.update(dev_loss_score, global_step, epoch)

                            # stop test by dev loss
                            _should_stop = self.__should_stop(global_step, epoch)
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
                        sys.stderr.write('{:5.3} s/step. epoch: {:3}, glb: {:8}\t\r'.format(
                            sec_per_step, epoch, global_step))
                        sys.stderr.flush()
                    except tf.errors.OutOfRangeError:
                        break

class TFVar:
    def __init__(self, name, init, dtype):
        with tf.variable_scope(name):
            self.var = tf.get_variable(name, dtype=dtype, initializer=init, trainable=False)
            self.ph = tf.placeholder(dtype, self.var.shape)
            self.assign = self.var.assign(self.ph)

    def set(self, value):
        tf.get_default_session().run(self.assign, feed_dict={self.ph: value})

    def get(self):
        return tf.get_default_session().run(self.var)

class ValidationScore:
    def __init__(self, name, early_stop_step, early_stop_epoch):
        with tf.variable_scope(name):
            self.keys = ['score', 'step', 'epoch']
            self.svs = [TFVar(n, init, dt) for n,init,dt in
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

def main():
    basicConfig(level=DEBUG)

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '--dir', '-d', type=str, default='.',
        help='Path to the directory containing `model_config.py`')
    parser.add_argument('--n_cpu_cores', default=4, type=int,
                        help='Number of cpu cores used by `tf.data.Dataset.map`')
    parser.add_argument('--n_gpus', default=1, type=int,
                        help='Number of GPUs available')
    parser.add_argument('--random_seed', default=0, type=int)
    args = parser.parse_args()


    Train(args.model_dir, args.n_gpus, args.n_cpu_cores, args.random_seed).train()

if __name__ == '__main__':
    main()
