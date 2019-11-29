import sys, os, re, json, argparse
import tensorflow as tf
import numpy as np

from ..components.inference import Inference
from tensorflow.contrib.framework import nest

class DecodeAnalysis(Inference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.ph_alt_topk = tf.placeholder(tf.int32, [])
        self.op_topk_alts = self.make_op(self.fn_topk_alternatives)
        self.make_session()

    def fn_topk_alternatives(self, inputs):
        (x, x_len), (y, y_len) = inputs
        logits = self.model.get_logits(x, y, x_len, y_len, False)
        is_target = tf.sequence_mask(y_len, tf.shape(y)[1], dtype=tf.float32)

        probs = tf.math.softmax(logits)

        # top_probs: [batch, len, k]
        top_probs, inds = tf.math.top_k(probs, self.ph_alt_topk, True)

        return [inds, top_probs]



    def get_topk_alts(self, sources, targets, k=None):
        batches = self.make_batches(sources, targets)
        k = k or 8

        # [#sequence, len, k]
        ids, probs = self.execute_op(self.op_topk_alts, batches, {self.ph_alt_topk: k})
        
        target_toks = [t.split() + ['<eos>'] for t in targets]

        toks = nest.map_structure(lambda x:self.vocab.ID2tok[x], ids)

        output = ''
        for i in range(len(ids)):
            output += sources[i]
            output += '{}\n'.format('\t\t'.join(target_toks[i]))
            for bi in range(k):
                for t in range(len(target_toks[i])):
                    output += '{:.3f}\t{}\t'.format(probs[i][t][bi], toks[i][t][bi])
                output += '\n'
            output += '\n'

        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', '--dir', '-d', type=str, required=True)
    parser.add_argument('--n_gpus', type=int, default=None)
    sub_parsers = parser.add_subparsers(dest='mode')

    parser_alts = sub_parsers.add_parser('alts')
    parser_alts.add_argument('source_file', type=str)
    parser_alts.add_argument('target_file', type=str)
    parser_alts.add_argument('-k', type=int, default=None)

    args = parser.parse_args()

    dec_analysis = DecodeAnalysis(args.model_dir, n_gpus=args.n_gpus)

    if args.mode == 'alts':
        with open(args.source_file, 'r') as f:
            sources = f.readlines()
        with open(args.target_file, 'r') as f:
            targets = f.readlines()
        
        print(dec_analysis.get_topk_alts(sources, targets, args.k))
        


if __name__ == '__main__':
    main()
