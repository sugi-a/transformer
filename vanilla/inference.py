from logging import getLogger; logger = getLogger(__name__)
from logging import DEBUG, INFO, basicConfig
import argparse
import sys
import json

import tensorflow as tf
from tensorflow import nest, keras
from tensorflow.nest import map_structure
import numpy as np

from .layers import Transformer, transfer_padding_to_left
from ..utils.multi_gpu import *
from ..utils.beam_search import length_penalty
from ..custom_text_data_pipeline.core import *
from ..custom_text_data_pipeline.vocabulary import Vocabulary

TSpec = tf.TensorSpec
TI32 = tf.int32

def get_len_penalty_fn(alpha):
    return lambda l: length_penalty(l, alpha)


def create_mask(y, dtype=tf.float32):
    return tf.cast(y != 0, dtype)


def count_toks(y, dtype=tf.float32):
    return tf.reduce_sum(create_mask, dtype)


def deco_cache_provider(f):
    cache = {}
    def f_(*args):
        f(*args, cache)
    
    return f_


def is_tensor_spec(nested):
    return all(isinstance(a, TSpec) for a in nest.flatten(nested))


def get_signed_func_wrapper(f, signature):
    tspec_idx = [i for i, s in enumerate(signature) if is_tensor_spec(s)]
    tensor_sig = [sig[idx] for idx in tspec_idx]

    @tf.function(input_signature=tensor_sig)
    def f_(*tensors):
        args = sig[:]
        for tensor, idx in zip(tensors, tensor_idx):
            args[idx] = tensor
        return f(*args)
    
    def wrapper(*args):
        assert all(a is s for i,(a,s) in
            enumerate(zip(args, signature)) if i not in tspec_idx)
        return f_(*[args[idx] for idx in tensor_idx])
    
    return wrapper


class InferenceBase:
    def __init__(
            self, transformer_model, vocab_config):
        
        self.model = transformer_model

        vc = vocab_config
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
        
    
    def create_data_gen_multi(self, x_multi, vocabs):
        return (
            ChainableGenerator(lambda: iter(x_multi))
            .trans(gen_line2IDs_multi, vocabs)
            .trans(gen_batch_of_capacity_multi, self.config['batch_capacity'])
            .trans(gen_pad_batch_multi)
        )


    def dataset_from_gen(self, gen, structure):
        dtype = map_structure(lambda x: tf.int32, structure)
        shape = map_structure(lambda x: tf.tensorShape([None, None]), structure)
        return tf.data.Dataset.from_generator(gen, dtype, shape)


    def create_dataset_multi(self, xs, vocabs):
        gen = self.create_data_gen_multi(zip(*xs), vocabs)
        return self.dataset_from_gen(gen, (None,) * len(vocabs))


    @deco_cache_provider
    def comp_translate(self, x, pfx, beam_size, _cache=None):
        def f(x, pfx, beam_size):
            if tf.size(x) > 0:
                # [B, K, L], [B, K]
                paths, scores = self.model.beam_search_decode_with_prefix(
                    x,
                    prefix_or_sos=self.vocab_trg.SOS_ID if pfx is None else pfx,
                    eos=self.vocab_trg.EOS_ID,
                    beam_size=beam_size)
            else:
                B = tf.shape(x)[0]
                paths = tf.zeros([B, beam_size, 0], tf.int32)
                scores = tf.zeros([B, beam_size], tf.float32)

            return paths, scores

        if len(_cache) == 0:
            M0 = TSpec([], tf.int32)
            M2 = TSpec([None, None], tf.int32)
            _cache[True] = get_signed_func_wrapper(f, [M2, None, M0])
            _cache[False] = get_signed_func_wrapper(f, [M2, M2, M0])

        return _cache[pfx is None](x, pfx, beam_size)
    

    @deco_cache_provider
    def comp_token_logp(self, x, y_in, y_out, offsets=None, _cache=None):
        def f(x, y_in, y_out, offsets):
            # [B, L-1, V]
            logits = self.model(x, y_in, training=False, offsets=offsets)
            logp_dist = tf.nn.softmax(logits)

            # [B, L-1] <- [B, L-1, V]
            logp = tf.gather(logp_dist, y_out, batch_dims=2)

            return logp * create_mask(y_out)

        if len(_cache) == 0:
            M1 = TSpec([None], tf.int32)
            M2 = TSpec([None, None], tf.int32)
            _cache[True] = get_signed_func_wrapper(f, [M2, M2, M2, None])
            _cache[False] = get_signed_func_wrapper(f, [M2, M2, M2, M1])

        return _cache[offsets is None](x, y_in, y_out, offsets)


    @tf.function(input_signature=[TSpec([None, None], TI32)]*2)
    def comp_seq_logp(self, x, y):
        tok_logp = self.comp_token_logp(x, y[:, :-1], y[:, 1:])
        return tf.reduce_sum(tok_logp, axis=1)
    

    @tf.function(input_signature=[TSpec([None, None], TI32)]*3)
    def comp_seq_logp_conditional(self, x, y, prefix):
        pfx, offsets = transfer_padding_to_left(prefix)
        L_pfx = tf.shape(pfx)[1]

        # [B, L_pfx + L_y]
        conc = tf.concat([pfx, y], axis=1)
        tok_logp = self.comp_token_logp(x, conc[:, :-1], conc[:, 1:], offsets)
        return tf.reduce_sum(tok_logp[:, L_pfx - 1:], axis=1)
    

    def gen_sents2hypotheses(
            self, x, beam_size, length_penalty=None, prefix=None):
        """
        Returns:
            yield hypos_and_scores: (str[], float[])
        """
        src_v, trg_v = self.vocab_src, self.vocab_trg

        if prefix is None:
            dataset = (
                self.create_dataset_multi((x,), (src_v,))
                .map(lambda *b: self.comp_translate(b[0], None, beam_size))
            )
        else:
            dataset = (
                self.create_dataset_multi((x, prefix), (src_v, trg_v))
                .map(lambda *b: self.comp_translate(b[0], b[1], beam_size))
            )
        
        dataset = dataset.prefech(1).unbatch().prefech(1)

        for hypos, scores in dataset:
            yield trg_v.IDs2text(hypos), scores
                

    def gen_sents2sents(self, x, beam_size=1, length_penalty=None, prefix=None):
        """
        Returns:
            yield str
        """
        for hypos, scores in self.gen_sents2hypotheses(
                x_iter, beam_size, length_penalty, prefix):
            yield hypos[0]
    

    def gen_sents2logps(self, x, y):
        dataset = (
            self.create_dataset_multi((x, y), (self.vocab_src, self.vocab_trg))
            .prefech(1)
            .map(lambda *b: self.comp_seq_logp(b[0], b[1]))
            .unbatch().prefech(1)
        )
        yield from dataset


    def sents2ppl(self, x, y):
        def fn_(x, y):
            logp = tf.reduce_sum(self.comp_seq_logp(x, y))
            toks = count_toks(y[:, 1:]) # num of toks to be predicted
            return logp, toks

        dataset = (
            self.create_dataset_multi((x, y), (self.vocab_src, self.vocab_trg))
            .map(fn_).prefech(1)
        )
        reduce_fn = lambda a,b: (a[0] + b[0], a[1] + b[1])
        logp, toks = dataset.reduce((0.0, 0.0), reduce_fn)
        return np.exp(- logp.numpy() / toks.numpy())


    def gen_sents2conditional_logps(self, x, prefix, y):
        src_v, trg_v = self.vocab_src, self.vocab_trg
        dataset = (
            self.create_dataset_multi((x, y, prefix), (src_v, trg_v, trg_v))
            .prefech(1)
            .map(lambda *b: self.comp_seq_logp_conditional(b[0], b[1], b[2]))
            .unbatch().prefech()
        )
        yield from dataset
    


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', '-d', type=str, default='.')
    p.add_argument('--checkpoint', type=str)
    p.add_argument('--capacity', type=int, default=16384)
    p.add_argument('--mode', choices=['translate', 'ppl', 'logp'],
        default='translate')
    p.add_argument('--prefix', action='store_true')
    p.add_argument('--beam_size', type=int, default=1)
    p.add_argument('--length_penalty', type=float)
    p.add_argument('--debug', action='store_true')
    args = p.parse_args()

    if args.debug:
        basicConfig(level=DEBUG)

    # Config
    with open(f'{args.dir}/model_config.json') as f:
        model_config = json.load(f)
    
    with open(f'{args.dir}/vocab_config.json') as f:
        vocab_config = json.load(f)
    
    # Transformer Model
    model = Transformer.from_config(model_config)

    # Inference Class
    inference = InferenceBase(model, vocab_config)

    if args.mode == 'translate':
        if args.length_penalty is None:
            lp_fn = None
        else:
            lp_fn = get_len_penalty_fn(args.length_penalty)
        
        if args.prefix:
            x, prefix = fork_iterable((l.split('\t') for l in sys.stdin), 2)
        else:
            x, prefix = sys.stdin, None

        for line in inference.gen_sents2sents(
                x,
                beam_size=args.beam_size,
                length_penalty=lp_fn,
                prefix=prefix):
            print(line)

if __name__ == '__main__':
    main()
