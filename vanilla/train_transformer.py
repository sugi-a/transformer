from logging import getLogger; logger = getLogger(__name__)
from logging import basicConfig, ERROR, WARNING, INFO, DEBUG, NOTSET
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
from ..utils.multi_gpu import *
from ..custom_text_data_pipeline.core import *
from ..custom_text_data_pipeline.vocabulary import Vocabulary


@tf.function
def get_mask(y, dtype=tf.float32):
    return tf.cast(y != 0, dtype)


def loss_additive(logits, y, ls_eps):
    """Unnormalized (not divided by the number of tokens) loss.
    Args:
        logits: <[B, L, V]>
        y: <[B, L]>
    Returns:
        loss: <[], tf.float32>
    """
    V = tf.shape(logits)[-1]
    label = label_smoothing(tf.one_hot(y, V), ls_eps)
    loss = tf.nn.softmax_cross_entropy_with_logits(label, logits)
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
    return tf.reduce_sum(tf.cast(pred == y & y != 0, dtype))


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
    count_corr_from_logits(logits, y) / count_toks(y)


def sequential_map_reduce_sum(fn, inputs):
    add_fn = lambda (*x): tf.math.add_n(x)
    return map_structure(add_fn, *sequential_map(fn, inputs))
    

def distributed_map_reduce_sum(fn, inputs):
    add_fn = lambda (*x): tf.math.add_n(x)
    return map_structure(add_fn , *distr_map(fn, inputs))


def learning_rate(d_model, step, warmup):
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
    return len(tf.config.experimental.list_physical_devices('GPU'))
    

class Train:
    def __init__(
            self, transformer_model, train_config, logdir):
        self.logdir = logdir

        self.model = transformer_model

        self.train_config = train_config

        vc = train_config['vocab']
        self.vocab_src = Vocabulary(
            vc['source_dict'],
            PAD_ID=vc['PAD_ID'],
            SOS_ID=vc['SOS_ID'],
            EOS_ID=vc['EOS_ID'],
            UNK_ID=vc['UNK_ID'])
        self.vocab_trg = Vocabulary(
            vc['target_dict'],
            PAD_ID=vc['PAD_ID'],
            SOS_ID=vc['SOS_ID'],
            EOS_ID=vc['EOS_ID'],
            UNK_ID=vc['UNK_ID'])

        self.metrics = [
            'loss': {
                'calc': lambda (logits, y, loss): loss,
                'train_mean': keras.metrics.Mean(),
                'dev_mean': kearas.metrics.Mean()
            },
            'accuracy': {
                'calc': lambda (logits, y, loss): accuracy(logits, y),
                'train_mean': keras.metrics.Mean(),
                'dev_mean': keras.metrics.Mean()
            }
        ]

        # Dataset generator pipeline settings
        pfn = self.pipeline_fns = {}
        bc = self.train_config['batch']

        # How to read lines from multiple files (default: no interleaving)
        pfn['line_from_files_multi'] = gen_line_from_files_multi

        # Length smoothing
        if bc['length_smoothing'] is None:
            pfn['length_smoothing'] = lambda x: x
            pfn['post_ls_shuffle'] = lambda x: x
        elif bc['length_smoothing'] == 'segsort':
            pfn['length_smoothing'] = lambda x: gen_segment_sort(
                x,
                segsize=bc['length_smoothing']['segsize'],
                key=lambda seqs: len(seqs[0]))
            pfn['post_ls_shuffle'] = lambda x: gen_random_sample(
                x,
                bufsize=bc['length_smoothing']['post_shuf_buf_size'])
        else:
            assert False
        
        # Batching
        if bc['constraint'] == 'size':
            pfn['batching'] = lambda x: gen_batch_multi(x, bc['size'])
        else:
            pfn['batching'] = \
                lambda x: gen_batch_of_capacity_multi(x, bc['size'])


    def calc_metrics(self, batch):
        if tf.size(batch) > 0:
            x, y = batch
            y_i, y_o = y[:, :-1], y[:, 1:]
            logits = self.model(x, y_i, training=False)
            loss = loss_norm(logits, y_o, ls_eps=self.eps)
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
        if tf.size(batch) > 0:
            tc = self.train_config
            x, y = batch
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


    @tf.function
    def train_step(self, inputs):
        """
        Args:
            inputs: (x, y)
                x: Tensor<[B, L_src], int32>
                y: Tensor<[B, L_trg], int32>
        """
        g, metrics = self.calc_grad_metrics(inputs)

        self.optimizer.apply_gradients(zip(g, self.model.trainable_variables))
        
        for k,v in self.metrics.items():
            v['train_mean'].update_state(metrics[k])
    

    @tf.function()
    def dev_step(self, inputs):
        metrics = self.calc_metrics(inputs)

        for k,v in self.metrics.items():
            v['dev_mean'].update_state(metrics[k])
    

    def update_dev_metrics(self, dev_dataset):
        for metric in self.metrics.values():
            metric['dev_mean'].reset_states()

        for data in dev_dataset:
            self.dev_step(data)
    

    def write_dev_metrics(self, writer, step):
        with writer.as_default:
            for name, m in self.metrics.items():
                tf.summary.scalar(f'{name}_dev', m['dev_mean'], step=step)


    def write_and_reset_train_metrics(self, writer, step):
        with writer.as_default:
            for name, m in self.metrics.items():
                tf.summary.scalar(f'{name}_train', m['train_mean'], step=step)
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


    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def translate_step(self, x):
        """
        Args:
            x: [B, L]
        """
        return self.translate_batch(x)


    def compute_bleu(self, dataset):
        """Compute BLEU on subwords"""

        refs, hyps = []

        for batch in dataset:
            x, y = batch
            pred = self.translate_step(x)
            refs.extend(self.vocab_trg.IDs2text(y).numpy())
            hyps.extend(self.vocab_trg.IDs2text(pred).numpy())
        
        refs = [[line.split()] for lien in refs]
        hyps = [line.split() for line in hyps]

        return corpus_bleu(refs, hyps)


    def create_train_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
        pfn = self.pipeline_fns
                    
        return (
            ChainableGenerator(
                lambda: zip(dc['source_train'], dc['target_train']))
            .trans(gen_random_sample)
            .trans(pfn['line_from_files_multi'])
            .trans(gen_line2IDs_multi, (self.vocab_src, self.vocab_trg))
            .trans(gen_random_sample, bufsize=bc['shuffle_buffer_size'])
            .trans(pfn['length_smoothing'])
            .trans(pfn['batching'])
            .trans(gen_pad_batch_multi)
            .trans(pfn['post_ls_shuffle'])
        )


    def create_dev_data_gen(self):
        bc = self.train_config['batch']
        dc = self.train_config['data']
        pfn = self.pipeline_fns
        
        return (
            ChainableGenerator(
                lambda: zip(dc['source_test'], dc['target_test']))
            .trans(gen_line_from_files_multi)
            .trans(gen_line2IDs_multi, (self.vocab_src, self.vocab_trg))
            .trans(pfn['batching'])
            .trans(gen_pad_batch_multi)
        )
    

    def dataset_from_gen(self, gen, structure=None):
        structure = (None, None) if structure is None else structure
        dtype = map_structure(lambda x: tf.int32, structure)
        shape = map_structure(lambda x: tf.TensorShape([None, None]), structure)
        return tf.data.Dataset.from_generator(gen, dtype, shape)


    def train():
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
        optimizer = tf.keras.optimizers.Adam(
            lambda: learning_rate(step, self.model.d_model, tc['warm_up_step']),
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        
        # Early Stopping
        early_stopping = {
            'best_epoch': tf.Variable(0, dtype=tf.int32),
            'best_score': tf.Variable(0.0)
        }

        # Checkpoint and Managers
        ckpt = tf.train.Checkpoint(
            epoch=epoch,
            step=step,
            loc_step=loc_step,
            optimizer=optimizer,
            model=self.model
        )

        manager = tf.train.CheckpointManager(
            ckpt, f'{self.logdir}/ckpt', step_counter=step)
        
        if manager.latest_checkpoint:
            logger.info(f'Restoring from {manager.latest_checkpoint}')
            ckpt.restore(manager.latest_checkpoint)
        else:
            logger.info('Checkpoint was not found')
        
        # Summary
        writer = tf.summary.create_file_writer(f'{self.logdir}/summary')

        # Training Loop
        for epoch_ in range(tc['max_epoch']):
            if epoch_ < epoch:
                continue
            
            set_random_seed(rnd.random())
            
            # Epoch Loop
            for loc_step_, data in enumerate(train_dataset):
                if loc_step_ < loc_step:
                    continue
                
                self.train_step(data, optimizer)

                step.assign_add(1)
                loc_step.assign_add(1)

                # Summary
                if step.numpy() % tc['summary_interval'] == 0:
                    self.update_dev_metrics(dev_dataset)
                    self.write_dev_metrics(writer, step)
                    self.write_and_reset_train_metrics(writer, step)
                
            epoch.assign_add(1)

            # Epoch Summary
            self.update_dev_metrics(dev_dataset)
            self.write_dev_metrics(writer, step)
            bleu = self.compute_bleu(dev_dataset)
            with writer.as_default():
                tf.summary.scalar('BLEU', bleu)
            
            # Early Stopping
            if tc['early_stopping_criterion'] == 'loss':
                score_ = self.metrics['loss']['dev_mean']
            else:
                score_ = bleu
            if score_ > early_stopping['best_score']:
                early_stopping['best_score'].assign(score_)
                early_stopping['best_epoch'].assign(epoch)
            elif epoch - early_stopping['best_epoch'] \
                    > tc['early_stopping_patience']:
                logger.info('Early Stopping')
                manager.save(step)
                break

            # Checkpoint
            manager.save(step)
    

    def check_dataset(self):
        train_dataset = \
            self.dataset_from_gen(self.create_train_data_gen()).prefetch(1)
        dev_dataset = \
            self.dataset_from_gen(self.create_dev_data_gen()).prefetch(1)
        
        @tf.function
        def toks_(batch):
            return tf.math.add_n([count_toks(x) for x in nest.flatten(batch)])
        
        def check_dataset_(dataset):
            last_t = start_t = time.time()
            t2, m, M = 0, 1e10, 0
            n, n2, n_m, n_M = 0, 1e10, 0
            i = 0
            for data in dataset:
                t = time.time()
                last_t, dt = t, t - last_t
                t2 += dt ** 2
                m, M = min(m, dt), max(M, dt)
                toks = toks_(data).numpy()
                n += toks
                n2 += toks ** 2
                n_m, n_M = min(n_m, toks), max(n_M, toks)
                i += 1
            end_t = time.time()
            tps = (end_t - start_t) / i

            for l in [
                f'#Steps: {i}',
                f'Time elapsed: {end_t - start_t}',
                f'Sec/Step: {tps}',
                f'Std[ Sec/Step ]: {(t2/i - tps ** 2) ** 0.5}',
                f'Max[ Sec/Step ]: {M}',
                f'Min[ Sec/Step ]: {m}',
                f'Total Tokens: {n}',
                f'Tokens/Step: {n / i}',
                f'Std[ Tokens/Step ]: {(n2/i - (n/i)**2) ** 0.5}'
                f'Max[ Tokens/Step ]: {n_M}',
                f'Min[ Tokens/Step ]: {n_m}',
                f'Token/Sec: {n / i / tps}',
            ]:
                print(l)
        
        print('Train Dataset')
        check_dataset(train_dataset)
        print()
        print('Dev Dataset')
        check_dataset(dev_dataset)


class TrainMultiGPULegacy(Train):
    def __init__(self, *args, **kwargs, gpus=None, accums=None,
            split_type='pre_split'):
        super().__init__(*args, **kwargs)

        vis_gpus = get_visible_gpus()
        self.gpus = vis_gpus if gpus is None else gpus
        logger.debug(f'Number of GPUs: {self.gpus}')
            
        self.accums = 1 if accums is None else accums

        self.split_type = split_type

        if split_type == 'pre_split':
            pfn = self.pipeline_fns
            n = self.gpus * self.accums
            if bc['constraint'] == 'size':
                pfn['batching'] = lambda x: gen_batch_multi(x, bc['size'] // n)
            else:
                pfn['batching'] = \
                    lambda x: gen_batch_of_capacity_multi(x, bc['size'] // n)
        elif split_type == 'post_split':
            pass
        else:
            raise Exception('Invalid parameter')
    

    def dataset_from_gen(self, gen):
        nsplit = self.gpus * self.accums

        if self.split_type == 'pre_split':
            gen = gen.trans(gen_fold, nsplit,
                padding_for_remainder=(np.zeros([0, 0]),) * 2)
            return super().dataset_from_gen(gen, ((None, None),) * nsplit)
        else:
            split = lambda x: non_even_split(x, nsplit)
            return super().dataset_from_gen(gen).map(split)


    @tf.function
    def train_step(self, inputs):
        """
        Args:
            inputs:
                list<pair<x, y>, N_accum * N_gpu>
                x: Tensor<[B, L_src], int32>
                y: Tensor<[B, L_trg], int32>
        """
        inputs = [inputs[i: i + self.accums]
            for i in range(0, self.accums * self.gpus, self.accums)]

        def accum_fn(batches):
            g_ms = sequential_map(self.calc_grad_metrics, batches)
            ntoks = sequential_map(lambda b: count_toks(b[0][:, 1:]), batches)
            return weighted_avg(g_ms, ntoks)

        g_ms, ntoks = distributed_map_reduce_sum(accum_fn, inputs)
        (grad, metrics), ntok = weighted_avg(g_ms, ntoks)

        # Update parameters
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))
        
        # Update train metrics
        for name, m in self.metrics.items():
            m['train_mean'].update_state(metrics[name])
    

    @tf.function
    def dev_step(self, inputs):
        inputs = [inputs[i: i + self.accums]
            for i in range(0, self.accums * self.gpus, self.accums)]

        def accum_fn(batches):
            ms = sequential_map(self.calc_metrics, batches)
            ntoks = sequential_map(lambda b: count_toks(b[0][:, 1:]), batches)
            return weighted_avg(ms, ntoks)

        ms, ntoks = distributed_map_reduce_sum(accum_fn, inputs)
        metrics, ntok = weighted_avg(ms, ntoks)

        for name, m in self.metrics.items():
            m['dev_mean'].update_state(metrics[name])
    

    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def translate_step(self, xs):
        """
        Args:
            xs: <[B, L]>[N_gpu * N_accum]
        """
        xs = [xs[i: i + self.accums]
            for i in range(0, self.accums * self.gpus, self.accums)]

        def accum_fn(xs):
            return sequential_map(self.translate_batch, xs)
        accum_fn = lambda x: split_sequential_map_concat(fn, x, self.accums)
        y = split_distr_map_concat(accum_fn, x, self.gpus)
        return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', '-d', type=str, default='.')
    parser.add_argument('--n_gpus', type=int)
    parser.add_argument('--n_accums', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument(
        '--mode', type=str, choices=['train', 'check_data'], default='train')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_base_class', action='store_true')
    parser.add_argument('--debug_post_split', action='store_true')
    
    args = parser.parse_args()

    basicConfig(level=DEBUG if args.debug else INFO)

    if args.mode == 'train':
        # Configs
        with open(f'{args.dir}/model_config.json') as f:
            model_config = json.load(f)
        
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)

        model = Transformer.from_config(model_config)

        logdir = f'{args.dir}/log'

        if args.debug_base_class:
            trainer = Train(model, train_config, logdir)
        else:
            sp_type = 'post_split' if args.debug_post_split else 'pre_split'
            trainer = TrainMultiGPULegacy(
                model, train_config, logdir,
                gpus=args.n_gpus, accums=args.n_accums, split_type=sp_type)
        
        trainer.train()
    elif args.mode == 'check_data':
        with open(f'{args.dir}/train_config.json') as f:
            train_config = json.load(f)
        
        if args.debug_base_class:
            trainer = Train(None, train_config, None)
        else:
            sp_type = 'post_split' if args.debug_post_split else 'pre_split'
            trainer = TrainMultiGPULegacy(
                None, train_config, None,
                gpus=args.n_gpus, accums=args.n_accums, split_type=sp_type)
        
        trainer.check_dataset()
    else:
        raise Exception('Invalid parameter')


if __name__ == '__main__':
    main()