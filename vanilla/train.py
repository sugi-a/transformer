from logging import getLogger; logger = getLogger(__name__)
from logging import basicConfig, ERROR, WARNING, INFO, DEBUG, NOTSET
import sys
import random
import argparse
import json
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow import nest
from tensorflow.nest import map_structure
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

from .layers import Transformer, label_smoothing
from ..utils import multi_gpu as mg
from ..custom_text_data_pipeline import core as dp

from ..custom_text_data_pipeline.vocabulary import Vocabulary


TSpec = tf.TensorSpec
TCastI = lambda x: tf.cast(x, tf.int32)
TCastI64 = lambda x: tf.cast(x, tf.int64)
TCastF = lambda x: tf.cast(x, tf.float32)


def get_mask(y, dtype=tf.float32):
    return tf.cast(y != 0, dtype)


def sparse_softmax_xent_loss(labels, logits, eps):
    """Sparse softmax crossentropy loss with label smoothing
    An implementation which is theoretically efficient.
    Args:
        labels: [B, L]
        logits: [B, L, V]
        eps: float
    Returns:
        [B, L]
        SXENT(P, Q) = E_P[-log Q]
        P = smooth(onehot(labels))
        log(Q) = log_softmax(logits)
    """
    logp = tf.nn.log_softmax(logits)
    if eps == 0:
        return -tf.gather(logp, labels, batch_dims=2)
    else:
        a = tf.reduce_mean(logp, axis=-1)
        b = tf.gather(logp, labels, batch_dims=2)
        return -(eps * a + (1 - eps) * b)


def loss_additive(logits, y, ls_eps):
    """Unnormalized (not divided by the number of tokens) loss.
    Args:
        logits: <[B, L, V]>
        y: <[B, L]>
    Returns:
        loss: <[], tf.float32>
    """
    #V = tf.shape(logits)[-1]
    #label = label_smoothing(tf.one_hot(y, V), ls_eps)
    #loss = tf.nn.softmax_cross_entropy_with_logits(label, logits)
    loss = sparse_softmax_xent_loss(y, logits, ls_eps)
    return tf.reduce_sum(loss * get_mask(y))


def loss_norm(logits, y, ls_eps):
    """Normalized (per-token) loss.
    Args:
        logits: <[B, L, V]>
        y: <[B, L]>
    Returns:
        loss: <[], tf.float32>
    """
    return loss_additive(logits, y, ls_eps) / count_toks(y)


def count_toks(seqs, dtype=tf.float32):
    return tf.reduce_sum(get_mask(seqs, dtype))


def count_corr(pred, y, dtype=tf.float32):
    pred = tf.cast(pred, tf.int32)
    return tf.reduce_sum(tf.cast((pred == y) &(y != 0), dtype))


def count_corr_from_logits(logits, y):
    """
    Args:
        logits: <[B, L, V]>
    Returns:
        <[B, L]>
    """
    pred = tf.argmax(logits, axis=-1)
    return count_corr(pred, y)


def accuracy(logits, y):
    return count_corr_from_logits(logits, y) / count_toks(y)


def distributed_map_reduce_sum(fn, inputs):
    add_fn = lambda *x: tf.math.add_n(x)
    return map_structure(add_fn , *mg.distr_map(fn, inputs))


def learning_rate(d_model, step, warmup):
    d_model, step, warmup = map(TCastF, (d_model, step, warmup))
    return d_model ** (-0.5) * tf.minimum(step ** -0.5, step * (warmup ** -1.5))


def set_random_seed(seed):
    """Reset the random seed of Python, Tensorflow and Numpy"""
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def weighted_avg(nesteds, weights):
    W = tf.math.add_n(weights)
    ws = [w / W for w in weights]
    weighted_nesteds = [
        map_structure(lambda v: v*w, nst) for nst, w in zip(nesteds, ws)]
    fn_add = lambda *x: tf.math.add_n(x)
    return map_structure(fn_add, *weighted_nesteds), W


def get_visible_gpus():
    return len(tf.config.experimental.list_physical_devices('GPU')) \
        + len(tf.config.experimental.list_physical_devices('XLA_GPU'))


def get_vocabs_from_config(vocab_config):
    vc = vocab_config
    vocab_src = Vocabulary(
        vc['source_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    vocab_trg = Vocabulary(
        vc['target_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    return vocab_src, vocab_trg


class Stats:
    def __init__(self):
        self.reset()


    def reset(self):
        self.sum = 0
        self.x2 = 0
        self.m = 1e10
        self.M = -1e10
        self.n = 0
        self.mean = 0
        self.var = 0
        self.std = 0


    def update(self, x):
        self.sum += x
        self.x2 += x ** 2
        self.m = min(self.m, x)
        self.M = max(self.M, x)
        self.n += 1


    def summarize(self):
        self.mean = self.sum / self.n
        self.var = self.x2 / self.n - self.mean ** 2
        self.std = self.var ** 0.5
        return {
            'mean': self.mean,
            'sum': self.sum,
            'min': self.m,
            'max': self.M,
            'var': self.var,
            'std': self.std,
            'n': self.n
        }
    

class Train:
    def __init__(
            self,
            model,
            source_vocab,
            target_vocab,
            train_config,
            logdir):
        self.logdir = logdir

        self.model = model

        self.train_config = train_config

        self.vocab_src = source_vocab
        self.vocab_trg = target_vocab

        def calc_loss_(logits, y, loss):
            return loss

        def calc_accuracy_(logits, y, loss):
            return accuracy(logits, y)

        self.metrics = {
            'loss': {
                'calc': calc_loss_,
                'train_mean': keras.metrics.Mean(),
                'dev_mean': keras.metrics.Mean()
            },
            'accuracy': {
                'calc': calc_accuracy_,
                'train_mean': keras.metrics.Mean(),
                'dev_mean': keras.metrics.Mean()
            }
        }

        # Dataset generator pipeline settings
        pfn = self.pipeline_fns = {}
        bc = self.train_config['batch']

        # How to read lines from multiple files (default: no interleaving)
        pfn['line_from_files_multi'] = dp.gen_line_from_files_multi

        # Length smoothing
        if bc['length_smoothing'] is None:
            pfn['length_smoothing'] = lambda x: x
            pfn['post_ls_shuffle'] = lambda x: x
        elif bc['length_smoothing']['method'] == 'segsort':
            pfn['length_smoothing'] = lambda x: dp.gen_segment_sort(
                x,
                segsize=bc['length_smoothing']['segsize'],
                key=lambda seqs: len(seqs[0]))
            pfn['post_ls_shuffle'] = lambda x: dp.gen_random_sample(
                x,
                bufsize=bc['length_smoothing']['post_shuf_buf_size'])
        else:
            assert False
        
        # Batching
        if bc['constraint'] == 'size':
            pfn['batching'] = lambda x: dp.gen_batch_multi(x, bc['size'])
        else:
            pfn['batching'] = \
                lambda x: dp.gen_batch_of_capacity_multi(x, bc['size'])


    def calc_metrics(self, batch):
        x, y = batch
        if tf.size(x) > 0:
            y_i, y_o = y[:, :-1], y[:, 1:]
            logits = self.model(x, y_i, training=False)
            loss = loss_norm(
                logits, y_o, ls_eps=self.train_config['label_smoothing'])
            metrics = {k: v['calc'](logits, y_o, loss)
                for k, v in self.metrics.items()}
        else:
            metrics = {k: 0.0 for k in self.metrics.keys()}

        return metrics


    def calc_grad_metrics(self, batch):
        """Compute gradient, metrics and #tokens given a batch.
        Args:
            batch: (x: <[B, L_x]>, y: <[B, L_y]>)
        Returns:
            (grad: Gradient, metrics: list<Tensor>, n_tokens: tf.int32)
        """
        x, y = batch
        if tf.size(x) > 0:
            tc = self.train_config
            y_i, y_o = y[:, :-1], y[:, 1:]
            with tf.GradientTape() as tape:
                logits = self.model(x, y_i, training=True)
                loss = loss_norm(logits, y_o, ls_eps=tc['label_smoothing'])

            grad = tape.gradient(loss, self.model.trainable_variables)

            metrics = {k: v['calc'](logits, y_o, loss)
                for k, v in self.metrics.items()}
        else:
            grad = map_structure(lambda x: 0.0, self.model.trainable_variables)
            metrics = {k: 0.0 for k in self.metrics.keys()}

        return grad, metrics


    @tf.function(input_signature=[(TSpec([None, None], tf.int32),) * 2])
    def train_step(self, inputs):
        """
        Args:
            inputs: (x, y)
                x: Tensor<[B, L_src], int32>
                y: Tensor<[B, L_trg], int32>
        """
        g, metrics = self.calc_grad_metrics(inputs)

        self.optimizer.apply_gradients(zip(g, self.model.trainable_variables))
        
        ntoks = count_toks(inputs[1][:, 1:])
        for k,v in self.metrics.items():
            v['train_mean'].update_state(metrics[k], ntoks)
    

    @tf.function(input_signature=[(TSpec([None, None], tf.int32),) * 2])
    def dev_step(self, inputs):
        metrics = self.calc_metrics(inputs)

        for k,v in self.metrics.items():
            v['dev_mean'].update_state(metrics[k], count_toks(inputs[1][:,1:]))
    

    def update_dev_metrics(self, dev_dataset):
        for metric in self.metrics.values():
            metric['dev_mean'].reset_states()

        for data in dev_dataset:
            self.dev_step(data)
    

    def write_dev_metrics(self, writer, step):
        with writer.as_default():
            for name, m in self.metrics.items():
                tf.summary.scalar(f'{name}_dev', m['dev_mean'].result(), step=TCastI64(step))


    def write_and_reset_train_metrics(self, writer, step):
        with writer.as_default():
            for name, m in self.metrics.items():
                tf.summary.scalar(f'{name}_train', m['train_mean'].result(), step=TCastI64(step))
                m['train_mean'].reset_states()


    def translate_batch(self, x):
        """
        Args:
            x: [B, L]
        Returns:
            [B, L_OUT]
        """
        if tf.size(x) > 0:
            # [B, 1, L]
            paths, scores = self.model.beam_search_decode_with_prefix(
                x,
                prefix_or_sos=self.vocab_trg.SOS_ID,
                eos=self.vocab_trg.EOS_ID,
                beam_size=1)
            # [B, L] <- [B, 1, L]
            return paths[:, 0]
        else:
            return x


    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int32)])
    def translate_step(self, x):
        """
        Args:
            x: [B, L]
        """
        return self.translate_batch(x)


    def compute_bleu(self, dataset):
        """Compute BLEU on subwords"""

        refs, hyps = [], []

        for batch in dataset:
            x, y = batch
            pred = self.translate_step(x)
            refs.extend(self.vocab_trg.IDs2text(y.numpy()))
            hyps.extend(self.vocab_trg.IDs2text(pred.numpy()))
        
        refs = [[line.split()] for line in refs]
        hyps = [line.split() for line in hyps]

        return corpus_bleu(refs, hyps)


    def create_train_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
        pfn = self.pipeline_fns
                    
        return (
            dp.ChainableGenerator(
                lambda: zip(dc['source_train'], dc['target_train']))
            .trans(dp.gen_random_sample)
            .trans(pfn['line_from_files_multi'])
            .trans(dp.gen_line2IDs_multi, (self.vocab_src, self.vocab_trg))
            .trans(dp.gen_random_sample, bufsize=bc['shuffle_buffer_size'])
            .trans(pfn['length_smoothing'])
            .trans(pfn['batching'])
            .trans(dp.gen_pad_batch_multi)
            .trans(pfn['post_ls_shuffle'])
        )


    def create_dev_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
        pfn = self.pipeline_fns
        
        return (
            dp.ChainableGenerator.zip(
                lambda: dp.gen_line_from_file(dc['source_dev']),
                lambda: dp.gen_line_from_file(dc['target_dev']))
            .trans(dp.gen_line2IDs_multi, (self.vocab_src, self.vocab_trg))
            .trans(pfn['batching'])
            .trans(dp.gen_pad_batch_multi)
        )
    

    def dataset_from_gen(self, gen, structure=None):
        structure = (None, None) if structure is None else structure
        dtype = map_structure(lambda x: tf.int32, structure)
        shape = map_structure(lambda x: tf.TensorShape([None, None]), structure)
        return tf.data.Dataset.from_generator(gen, dtype, shape)


    def unit_test(self, i):
        ckpt = tf.train.Checkpoint(model=self.model)
        lst_ckpt = tf.train.latest_checkpoint(f'{self.logdir}/checkpoint_best')
        logger.debug('Restoring' f'{lst_ckpt}')
        ckpt.restore(lst_ckpt)
        logger.debug('Restored')
        if i == 0:
            dataset = self.dataset_from_gen(self.create_dev_data_gen(), (None, None))

            refs, hyps = [], []

            t = time.time()
            for batch in dataset:
                x, y = batch
                pred = self.translate_step(x)
                hyps.extend(self.vocab_trg.IDs2text(pred.numpy()))
                refs.extend(self.vocab_trg.IDs2text(y.numpy()))
                for hyp in hyps:
                    print(hyp)
                t, dt = time.time(), time.time() - t
                logger.debug(dt)

            
            refs = [[line.split()] for line in refs]
            hyps = [line.split() for line in hyps]
            logger.info(corpus_bleu(refs, hyps))


    def train(self):
        tc = self.train_config

        # Random
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
            lambda: learning_rate(self.model.d_model, step, tc['warm_up_step']),
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        # Early Stopping
        best_epoch = tf.Variable(0, dtype=tf.int32)
        best_score = tf.Variable(-1e10)

        # Checkpoint and Managers
        ckpt = tf.train.Checkpoint(
            epoch=epoch,
            step=step,
            loc_step=loc_step,
            optimizer=self.optimizer,
            model=self.model,
            best_epoch=best_epoch,
            best_score=best_score
        )

        ckpt_best = tf.train.Checkpoint(
            epoch=epoch,
            step=step,
            model=self.model
        )

        manager = tf.train.CheckpointManager(
            ckpt,
            directory=f'{self.logdir}/checkpoint',
            max_to_keep=None,
            step_counter=step)

        manager_best = tf.train.CheckpointManager(
            ckpt_best,
            directory=f'{self.logdir}/checkpoint_best',
            max_to_keep=2,
            step_counter=step)
        
        if manager.latest_checkpoint:
            logger.info(f'Restoring from {manager.latest_checkpoint}')
            ckpt.restore(manager.latest_checkpoint)
            logger.info(
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

                t_ = time.time()
                t, dt = t_, t_ - t
                sys.stdout.write(f'Step: {step.numpy()}, Time elapsed: {dt}\n')
                sys.stdout.flush()

                # Summary
                if step.numpy() % tc['summary_interval'] == 0:
                    self.update_dev_metrics(dev_dataset)
                    self.write_dev_metrics(writer, step)
                    self.write_and_reset_train_metrics(writer, step)
                
            epoch.assign_add(1)
            loc_step.assign(0)

            # Epoch Summary
            self.update_dev_metrics(dev_dataset)
            self.write_dev_metrics(writer, step)
            loss = self.metrics['loss']['dev_mean'].result().numpy()
            logger.info('Computing BLEU')
            bleu = self.compute_bleu(dev_dataset)
            with writer.as_default():
                tf.summary.scalar('BLEU', bleu, step=TCastI64(step))

            logger.info(f'Epoch {epoch.numpy()}, Loss: {loss}, BLEU: {bleu}')
            
            # Early Stopping
            if tc['early_stopping_criterion'] == 'loss':
                score_ = -loss
            else:
                score_ = bleu

            # Checkpoint depending on early stopping test
                logger.debug(
                    f'Last Best: {best_score.numpy()}, This time: {score_}')
            if score_ > best_score.numpy():
                best_score.assign(score_)
                best_epoch.assign(epoch)

                logger.info('Updating the best checkpoint')
                manager_best.save(step)
            elif epoch - best_epoch > tc['early_stopping_patience']:
                manager.save(step)
                logger.info('Early Stopping')
                break

            logger.info('Checkpointing')
            manager.save(step)
    

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

        print(f'Steps: {i}')
        for m, sts in zip(metrics, stats):
            res = sts.summarize()
            print(f'{m}/Step')
            for label, score in res.items():
                print(f'{label}: {score}')
            print()


class TrainMultiGPULegacy(Train):
    def __init__(self, *args, gpus=None, accums=None,
            split_type='small_minibatch', **kwargs):
        super().__init__(*args, **kwargs)

        vis_gpus = get_visible_gpus()
        self.gpus = vis_gpus if gpus is None else gpus
        logger.debug(f'Number of GPUs: {self.gpus}')
            
        self.accums = 1 if accums is None else accums

        self.split_type = split_type

        if split_type == 'small_minibatch':
            pfn = self.pipeline_fns
            n = self.gpus * self.accums
            bc = self.train_config['batch']
            if bc['constraint'] == 'size':
                pfn['batching'] = lambda x: dp.gen_batch_multi(x, bc['size'] // n)
            else:
                pfn['batching'] = \
                    lambda x: dp.gen_batch_of_capacity_multi(x, bc['size'] // n)
        elif split_type == 'divide_batch':
            pass
        else:
            raise Exception('Invalid parameter')
    

    def dataset_from_gen(self, gen):
        nsplit = self.gpus * self.accums

        if self.split_type == 'small_minibatch':
            gen = gen \
                .map(lambda x: dp.list2numpy_nested(x)) \
                .trans(dp.gen_fold, nsplit, (np.zeros([0, 0]),) * 2)
            return super().dataset_from_gen(gen, ((None, None),) * nsplit)
        else:
            split = lambda *x: mg.non_even_split(x, nsplit)
            return super().dataset_from_gen(gen).map(split)


    def train_step(self, inputs):
        """
        Args:
            inputs:
                list<pair<x, y>, N_accum * N_gpu>
                x: Tensor<[B, L_src], int32>
                y: Tensor<[B, L_trg], int32>
        """
        spec = TSpec([None, None], tf.int32),
        @tf.function(input_signature=[((spec,) * 2) * self.gpus * self.accums])
        def train_step_(inputs):
            inputs = [inputs[i: i + self.accums]
                for i in range(0, self.accums * self.gpus, self.accums)]

            def accum_fn(batches):
                g_ms = mg.sequential_map(self.calc_grad_metrics, batches)
                ntoks = mg.sequential_map(lambda b: count_toks(b[0][:, 1:]), batches)
                return weighted_avg(g_ms, ntoks)

            g_ms, ntoks = distributed_map_reduce_sum(accum_fn, inputs)
            (grad, metrics), ntok = weighted_avg(g_ms, ntoks)

            # Update parameters
            self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
            
            # Update train metrics
            for name, m in self.metrics.items():
                m['train_mean'].update_state(metrics[name])

        self.train_step = train_step_
        self.train_step(inputs)
    

    def dev_step(self, inputs):
        spec = TSpec([None, None], tf.int32),

        @tf.function(input_signature=[((spec,) * 2) * self.gpus * self.accums])
        def dev_step_(inputs):
            inputs = [inputs[i: i + self.accums]
                for i in range(0, self.accums * self.gpus, self.accums)]

            def accum_fn(batches):
                ms = mg.sequential_map(self.calc_metrics, batches)
                ntoks = mg.sequential_map(lambda b: count_toks(b[0][:, 1:]), batches)
                return weighted_avg(ms, ntoks)

            ms, ntoks = distributed_map_reduce_sum(accum_fn, inputs)
            metrics, ntok = weighted_avg(ms, ntoks)

            for name, m in self.metrics.items():
                m['dev_mean'].update_state(metrics[name])

        self.dev_step = dev_step_
        self.dev_step(inputs)
    

    def translate_step(self, xs):
        """
        Args:
            xs: <[B, L]>[N_gpu * N_accum]
        """
        spec = TSpec([None, None], tf.int32),

        @tf.function(input_signature=[((spec,) * 2) * self.gpus * self.accums])
        def translate_step(xs):
            xs = [xs[i: i + self.accums]
                for i in range(0, self.accums * self.gpus, self.accums)]

            def accum_fn(xs):
                return mg.sequential_map(self.translate_batch, xs)
            accum_fn = lambda x: mg.split_sequential_map_concat(fn, x, self.accums)
            y = mg.split_distr_map_concat(accum_fn, x, self.gpus)
            return y

        self.translate_step = translate_step_
        return self.translate_step(xs)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default='.')
    parser.add_argument('--n_gpus', type=int)
    parser.add_argument('--n_accums', type=int)
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

    if args.mode == 'train':
        # Configs
        with open(f'{args.dir}/model_config.json') as f:
            model_config = json.load(f)
        
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)
        
        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)

        # Transformer Model
        model = Transformer.from_config(model_config)

        # Vocabulary
        vocab_src, vocab_trg = get_vocabs_from_config(vocab_config)

        # Directory for logging
        logdir = f'{args.dir}'

        if args.debug_base_class:
            trainer = Train(
                model,
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=logdir)
        else:
            trainer = TrainMultiGPULegacy(
                model,
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=logdir,
                gpus=args.n_gpus,
                accums=args.n_accums,
                split_type='divide_batch' \
                    if args.debug_post_split else 'small_minibatch')
        
        trainer.train()
    elif args.mode == 'check_data':
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)

        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)
        
        vocab_src, vocab_trg = get_vocabs_from_config(vocab_config)
        
        if args.debug_base_class:
            trainer = Train(
                model=None,
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=None)
        else:
            trainer = TrainMultiGPULegacy(
                model=None,
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=None,
                gpus=args.n_gpus,
                accums=args.n_accums,
                split_type='divide_batch' \
                    if args.debug_post_split else 'small_minibatch')
        
        train_dataset = trainer.dataset_from_gen(
            trainer.create_train_data_gen()).prefetch(1)
        dev_dataset = trainer.dataset_from_gen(
            trainer.create_dev_data_gen()).prefetch(1)
        print('Train Dataset')
        trainer.check_dataset(train_dataset)
        print('Dev Dataset')
        trainer.check_dataset(dev_dataset)
    elif args.mode == 'debug':
        with open(f'{args.dir}/model_config.json') as f:
            model_config = json.load(f)

        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)

        with open(f'{args.dir}/vocab_config.json') as f:
            vocab_config = json.load(f)
        
        vocab_src, vocab_trg = get_vocabs_from_config(vocab_config)
        
        if args.debug_base_class:
            trainer = Train(
                model=Transformer.from_config(model_config),
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=args.dir)
        else:
            trainer = TrainMultiGPULegacy(
                model=Transformer.from_config(model_config),
                source_vocab=vocab_src,
                target_vocab=vocab_trg,
                train_config=train_config,
                logdir=args.dir,
                gpus=args.n_gpus,
                accums=args.n_accums,
                split_type='divide_batch' \
                    if args.debug_post_split else 'small_minibatch')
        
        trainer.unit_test(0)
        
    else:
        raise Exception('Invalid parameter')


if __name__ == '__main__':
    main(sys.argv[1:])
