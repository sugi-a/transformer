from logging import getLogger; logger = getLogger(__name__)
from logging import DEBUG, INFO, basicConfig
import os, sys, argparse, time, json
import itertools
import random
from collections import deque

import tensorflow as tf
from tensorflow import keras, nest
import numpy as np

from .layers import DecoderLanguageModel 
from ..utils import multi_gpu as mg
from ..custom_text_data_pipeline import core as dp
from ..custom_text_data_pipeline.vocabulary import Vocabulary
from ..vanilla.train import \
    sparse_softmax_xent_loss, \
    get_mask, \
    deco_function_oneshot_shape_inv, \
    get_visible_gpus, \
    Stats, \
    count_toks, \
    get_output_specs_shape_inv, \
    weighted_avg, \
    learning_rate


TShape = tf.TensorShape


def get_vocabs_from_config(config):
    return Vocabulary(
        vocab_file=config['dict'],
        PAD_ID=config['PAD_ID'],
        EOS_ID=config['EOS_ID'],
        UNK_ID=config['UNK_ID'],
        SOS_ID=config['SOS_ID'])


def gen_doc_from_lines(seq_iterable):
    doc = []
    for seq in seq_iterable:
        if len(seq) == 0:
            if len(doc) > 0:
                yield doc
            doc = []
        else:
            doc.append(seq)
    if len(doc) > 0:
        yield doc


def gen_front_aligned_segment_from_docs(doc_iterable, window_size, stride):
    for doc in doc_iterable:
        for idx in range(0, len(doc), stride):
            buf = []
            for i in range(idx, len(doc), stride):
                buf.extend(doc[i])
                if len(buf) > window_size:
                    break
            if len(buf) < window_size:
                yield buf
                break
            else:
                yield buf[:window_size]


class Train:
    def __init__(
            self,
            model,
            vocab,
            train_config,
            logdir,
            gpus=None,
            accums=None):
        self.logdir = logdir
        self.model = model
        self.train_config = train_config
        self.vocab = vocab

        self.dev_loss = keras.metrics.Mean()

        vis_gpus = get_visible_gpus()
        self.gpus = vis_gpus if gpus is None else gpus
        logger.info(f'Number of GPUs: {self.gpus}')

        self.accums = 1 if accums is None else accums


    def calc_loss(self, x, training):
        x_i, x_o = x[:, :-1], x[:, 1:]

        mask = get_mask(x_o)
        ntoks = tf.reduce_sum(mask)
            
        if ntoks > 0:
            lgts = self.model(x_i, training=training)
        
            losses = sparse_softmax_xent_loss(
                x_o, lgts, self.train_config['label_smoothing'])

            loss = tf.reduce_sum(losses * mask) / ntoks
        else:
            loss = tf.constant(0.0)
        
        return loss


    def calc_grad(self, x):
        with tf.GradientTape() as tape:
            loss = self.calc_loss(x, True)
        
        grad = tape.gradient(loss, self.model.trainable_variables)
        
        return grad 
    

    def get_batch_weight(self, batch):
        return count_toks(batch[:, 1:])

    
    @deco_function_oneshot_shape_inv
    def train_step(self, inputs):
        count_fn = lambda b: count_toks(b[:, 1:], tf.float32)
        fn = lambda b: (self.calc_grad(b), count_fn(b))
        o_specs = get_output_specs_shape_inv(fn, inputs[0][0])

        def accum_fn(batches):
            g, ntoks = zip(*mg.sequential_map(fn, batches, o_specs))
            return weighted_avg(g, ntoks)

        g, ntoks = zip(*mg.distr_map(accum_fn, inputs))
        g, _ = weighted_avg(g, ntoks)
        
        self.optimizer.apply_gradients(zip(g, self.model.trainable_variables))
        

    @deco_function_oneshot_shape_inv
    def dev_step(self, inputs):
        count_fn = lambda b: count_toks(b[:, 1:], tf.float32)
        fn = lambda b: (self.calc_loss(b, False), count_fn(b))
        o_specs = get_output_specs_shape_inv(fn, inputs[0][0])

        def accum_fn(batches):
            o, ntoks = zip(*mg.sequential_map(fn, batches, o_specs))
            return weighted_avg(o, ntoks)

        o, ntoks = zip(*mg.distr_map(accum_fn, inputs))
        o, w = weighted_avg(o, ntoks)

        self.dev_loss.update_state(o, w)


    def update_dev_metrics(self, dev_dataset):
        self.dev_loss.reset_states()

        for data in dev_dataset:
            self.dev_step(data)


    def write_dev_metrics(self, writer, step):
        with writer.as_default():
            tf.summary.scalar(
                'dev_loss',
                self.dev_loss.result(),
                step=tf.cast(step, tf.int64))
    

    def create_train_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']

        w, h = self.accums, self.gpus
        n = w * h

        batch_size = bc['batch_size'] // n

        return (
            dp.ChainableGenerator(lambda: dc['train'])
            .trans(dp.gen_random_sample)
            .trans(dp.gen_line_from_files)
            .trans(dp.gen_line2IDs, self.vocab)
            .trans(gen_doc_from_lines)
            .trans(
                gen_front_aligned_segment_from_docs,
                window_size=bc['sampling']['window_size'],
                stride=bc['sampling']['stride_sentences'])
            .trans(dp.gen_random_sample, bufsize=bc['shuffle_buf_size'])
            .map(lambda seq: (seq,))
            .trans(dp.gen_batch_multi, batch_size)
            .trans(dp.gen_pad_batch_multi)
            .map(lambda batch: batch[0])
            .map(lambda x: dp.list2numpy_nested(x))
            .trans(dp.gen_fold, n, np.zeros([0, 0]))
            .map(lambda b: tuple(b[i: i+w] for i in range(0, n, w)))
        )
    

    def create_dev_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']

        w, h = self.accums, self.gpus
        n = w * h

        batch_size = bc['batch_size'] // n

        return (
            dp.ChainableGenerator(lambda: dp.gen_line_from_file(dc['dev']))
            .trans(dp.gen_line2IDs, self.vocab)
            .trans(gen_doc_from_lines)
            .trans(
                gen_front_aligned_segment_from_docs,
                window_size=bc['sampling']['window_size'],
                stride=bc['sampling']['window_size'])
            .map(lambda seq: (seq,))
            .trans(dp.gen_batch_multi, batch_size)
            .trans(dp.gen_pad_batch_multi)
            .map(lambda batch: batch[0])
            .map(lambda x: dp.list2numpy_nested(x))
            .trans(dp.gen_fold, n, np.zeros([0, 0]))
            .map(lambda b: tuple(b[i: i+w] for i in range(0, n, w)))
        )
    

    def dataset_from_gen(self, gen):
        w, h = self.accums, self.gpus
        n = w * h
        structure = ((None,) * w,) * h
        dtype = nest.map_structure(lambda x: tf.int32, structure)
        shape = nest.map_structure(lambda x: TShape([None,]*2), structure)
        return tf.data.Dataset.from_generator(gen, dtype, shape)


    def train(self):
        tc = self.train_config

        # Random
        def set_random_seed(seed):
            """Reset the random seed of Python, Tensorflow and Numpy"""
            random.seed(seed)
            tf.random.set_seed(seed)
            np.random.seed(seed)

        set_random_seed(tc['random_seed'])
        rnd = random.Random(tc['random_seed'])

        # Dataset
        train_dataset = \
            self.dataset_from_gen(self.create_train_data_gen()).prefetch(1)
        dev_dataset = \
            self.dataset_from_gen(self.create_dev_data_gen()).prefetch(1)

        # Step Counters
        epoch = tf.Variable(0, dtype=tf.int32)
        step = tf.Variable(0, dtype=tf.int32)
        loc_step = tf.Variable(0, dtype=tf.int32) # Steps reset in every epoch

        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(
            lambda: learning_rate(self.model.d_model, step, tc['warm_up_steps']),
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        # Early Stopping
        best_epoch = tf.Variable(0, dtype=tf.int32)
        best_score = tf.Variable(-1e10)

        # Checkpoint and Managers

        # Main checkpoint
        ckpt = tf.train.Checkpoint(
            epoch=epoch,
            step=step,
            loc_step=loc_step,
            optimizer=self.optimizer,
            model=self.model,
            best_epoch=best_epoch,
            best_score=best_score
        )

        # Main manager
        manager = tf.train.CheckpointManager(
            ckpt,
            directory=f'{self.logdir}/checkpoint',
            max_to_keep=1,
            step_counter=step)
        
        # Manger for long-term history
        manager_hist = tf.train.CheckpointManager(
            ckpt,
            directory=f'{self.logdir}/checkpoint_history',
            max_to_keep=None,
            step_counter=step)

        # Checkpoint for recording the best epoch
        ckpt_best = tf.train.Checkpoint(
            epoch=epoch,
            step=step,
            model=self.model
        )

        manager_best = tf.train.CheckpointManager(
            ckpt_best,
            directory=f'{self.logdir}/checkpoint_best',
            max_to_keep=3,
            step_counter=step)
        
        if manager.latest_checkpoint:
            logger.info(f'Restoring from {manager.latest_checkpoint}')
            ckpt.restore(manager.latest_checkpoint)
            logger.info(
                f'Restored\n'
                f'Epoch: {epoch.numpy()},\n'
                f'Step: {step.numpy()}\n'
                f'Best Checkpoint: \n'
                f'\tEpoch: {best_epoch.numpy()}\n'
                f'\tScore: {best_score.numpy()}\n'
            )
        else:
            logger.info('Checkpoint was not found')

        start_epoch = epoch.numpy()
        start_loc_step = loc_step.numpy()
        
        # Summary
        writer = tf.summary.create_file_writer(f'{self.logdir}/summary')

        # Training Loop
        logger.debug('Train Loop Starts')
        for epoch_ in range(tc['max_epoch']):
            if epoch_ < start_epoch:
                continue
            
            set_random_seed(rnd.randrange(0xFFFF))
            
            # Epoch Loop
            t = time.time()
            for loc_step_, data in enumerate(train_dataset):
                if loc_step_ < start_loc_step:
                    continue
                elif loc_step_ == start_loc_step:
                    start_loc_step = -1
                
                self.train_step(data)

                step.assign_add(1)

                # Print processing speed
                t_ = time.time()
                t, dt = t_, t_ - t
                sys.stdout.write(f'Step: {step.numpy()}, Time elapsed: {dt}\n')
                sys.stdout.flush()

                # Summary
                if step.numpy() % tc['summary_interval'] == 0:
                    self.update_dev_metrics(dev_dataset)
                    self.write_dev_metrics(writer, step)
                
            epoch.assign_add(1)
            loc_step.assign(0)

            # Epoch Summary
            # Basic summary
            self.update_dev_metrics(dev_dataset)
            self.write_dev_metrics(writer, step)
            loss = self.dev_loss.result()
            logger.info(f'Epoch {epoch.numpy()}, Loss: {loss}')
            
            # Early Stopping
            score = -loss
            logger.debug(
                f'Last Best: {best_score.numpy()}, This time: {score}')

            should_early_stop = False
            if score > best_score.numpy():
                best_score.assign(score)
                best_epoch.assign(epoch)

                logger.info('Updating the best checkpoint')
                manager_best.save(step)
            elif epoch - best_epoch > tc['early_stopping_patience']:
                should_early_stop = True

            # Checkpoint
            logger.info('Checkpointing')
            manager.save(step)

            # History
            _t = epoch.numpy()
            if int(_t ** 0.5) ** 2 == _t:
                logger.info('Saving as long-term checkpoint')
                manager_hist.save(step)

            if should_early_stop:
                logger.info('Early Stopping')
                break


    def check_dataset(self, dataset):
        @tf.function(input_signature=[dataset.element_spec])
        def toks_(batch):
            return tf.math.add_n([count_toks(x) for x in nest.flatten(batch)])

        @tf.function(input_signature=[dataset.element_spec])
        def sents_(batch):
            return tf.math.add_n([tf.shape(x)[0] for x in nest.flatten(batch)])

        @tf.function(input_signature=[dataset.element_spec])
        def capacity_(batch):
            return tf.math.add_n([tf.size(x) for x in nest.flatten(batch)])

        @tf.function(input_signature=[dataset.element_spec])
        def longest_(batch):
            lens = [tf.shape(x)[1] for x in nest.flatten(batch)]
            return tf.math.reduce_max(lens)

        metrics = ['Sec', 'Tokens', 'Sents', 'Capacity','Longest']
        stats = [Stats() for i in range(len(metrics))]
        
        last_t = time.time()
        i = 0
        for data in dataset:
            t = time.time()
            last_t, dt = t, t - last_t
            scores = [
                dt, toks_(data).numpy(), sents_(data).numpy(),
                capacity_(data).numpy(), longest_(data).numpy()]
            for sts, score in zip(stats, scores):
                sts.update(score)
            i += 1
            if i % 100 == 0:
                print(i)
                sys.stdout.flush()

        print(f'Steps: {i}')
        for m, sts in zip(metrics, stats):
            res = sts.summarize()
            print(f'{m}/Step')
            for label, score in res.items():
                print(f'{label}: {score}')
            print()


    def debug(self):
        gen = self.create_train_data_gen()
        for d in gen():
            a = d


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default='.')
    parser.add_argument('--n_gpus', type=int)
    parser.add_argument('--accums', type=int)
    parser.add_argument('--mode', type=str,
        choices=['train', 'check_data', 'debug'], default='train')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_base_class', action='store_true')
    parser.add_argument('--debug_post_split', action='store_true')
    parser.add_argument('--debug_eager_function', action='store_true')
    
    args = parser.parse_args(argv)

    basicConfig(level=DEBUG if args.debug else INFO)

    if args.debug_eager_function:
        tf.config.run_functions_eagerly(True)

    if args.mode == 'train' or args.mode == 'debug':
        # Configs
        with open(f'{args.dir}/model_config.json') as f:
            model_config = json.load(f)
        
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)
        
        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)

        # Transformer Model
        model = DecoderLanguageModel.from_config(model_config)

        # Vocabulary
        vocab = get_vocabs_from_config(vocab_config)

        # Directory for logging
        logdir = f'{args.dir}'

        trainer = Train(
            model,
            vocab=vocab,
            train_config=train_config,
            logdir=logdir,
            gpus=args.n_gpus,
            accums=args.accums)
        
        if args.mode == 'train':
            trainer.train()
        else:
            trainer.debug()

    elif args.mode == 'check_data':
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)

        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)
        
        vocab = get_vocabs_from_config(vocab_config)
        
        trainer = Train(
            model=None,
            vocab=vocab,
            train_config=train_config,
            logdir=None,
            gpus=args.n_gpus,
            accums=args.accums)
        
        train_dataset = trainer.dataset_from_gen(
            trainer.create_train_data_gen()).prefetch(1)
        dev_dataset = trainer.dataset_from_gen(
            trainer.create_dev_data_gen()).prefetch(1)
        print('Train Dataset')
        trainer.check_dataset(train_dataset)
        print('Dev Dataset')
        trainer.check_dataset(dev_dataset)


if __name__ == '__main__':
    main(sys.argv[1:])
