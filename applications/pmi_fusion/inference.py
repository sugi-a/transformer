from logging import getLogger, DEBUG, basicConfig; logger = getLogger(__name__)
import argparse
import sys
import json
import time
from collections import deque

import tensorflow as tf
from tensorflow import keras, nest
import numpy as np

from ...utils.beam_search import beam_search
from ...vanilla import layers as vl
from ...language_model import layers as ll
from ...custom_text_data_pipeline import core as dp
from ...custom_text_data_pipeline.vocabulary import Vocabulary

SF0 = tf.TensorSpec([], tf.float32)
SI0 = tf.TensorSpec([], tf.int32)
SI2 = tf.TensorSpec([None, None], tf.int32)


def gen_lines_to_docs(line_iterable):
    doc = []
    for line in line_iterable:
        line = line.strip()
        if len(line) == 0:
            if len(doc) > 0:
                yield doc
                doc = []
        else:
            doc.append(line)
            
    if len(doc) > 0:
        yield doc


def gen_docs_to_lines(docs_iterable):
    for i, doc in enumerate(docs_iterable):
        if i > 0:
            yield ''
        yield from doc


def recursive_update(target, ref):
    """Recursively update the nested structure `target` by `ref`"""
    containers = (list, dict) # tuple is not allowed
    assert isinstance(ref, containers)
    
    if isinstance(ref, list):
        assert len(ref) == len(target)
        for i in range(len(ref)):
            if isinstance(ref[i], containers):
                recursive_update(target[i], ref[i])
            else:
                target[i] = ref[i]
    else:
        assert target.keys() == ref.keys()
        for k in ref.keys():
            if isinstance(ref[k], containers):
                recursive_update(target[k], ref[k])
            else:
                target[k] = ref[k]


def create_mask(seq, dtype=tf.float32):
    return tf.cast(seq != 0, dtype)


def count_left_padding(seq, dtype=tf.int32):
    pads = tf.cast(seq == 0, dtype)
    left_pads = tf.math.cumprod(pads, axis=1)
    return tf.reduce_sum(left_pads, axis=1)


def create_stateful_decoder_TM(model, x, ntiles=None):
    enc_out, enc_pad_bias = model.encode(x, training=False)

    if ntiles is not None:
        enc_out, enc_pad_bias = nest.map_structure(
            lambda x: tf.repeat(x, ntiles, axis=0),
            (enc_out, enc_pad_bias))

    state, shape_inv = model.create_cache(tf.shape(enc_out)[0])
    return (
        model.create_decoder(enc_out, enc_pad_bias, state),
        state,
        shape_inv
    )


def create_stateful_decoder_LM(model, B, ntiles=None, offsets=None):
    if ntiles is not None:
        B *= ntiles
    state, shape_inv = model.create_cache(B)
    def f(y):
        if hasattr(model, 'id_substitutions_'):
            for f, t in model.id_substitutions_:
                y = tf.where(y == f, t, y)
        return model(y, training=False, cache=state, offsets=None)
    return f, state, shape_inv


def create_stateful_decoder_CtxLM(model, c, ntiles=None):
    if ntiles is not None:
        c = tf.repeat(c, ntiles, axis=0)
    B = tf.shape(c)[0]
    offsets = count_left_padding(c)
    f, state, shape_inv = create_stateful_decoder_LM(model, B, offsets=offsets)

    f(c) # Update state

    return f, state, shape_inv


def get_fscore_logits_from_decoders(tm, lm, dlm, y_in):
    tm_logp = tf.nn.log_softmax(tm(y_in))
    lm_logp = tf.nn.log_softmax(lm(y_in))
    dlm_logp = tf.nn.log_softmax(dlm(y_in))

    return tm_logp + dlm_logp - lm_logp


def get_tok_fscore_from_decoders(tm, lm, dlm, y):
    y_in, y_out = y[:, :-1], y[:, 1:]

    # [B, L, V]
    fscore_logits = get_fscore_logits_from_decoders(tm, lm, dlm, y_in)

    # [B, L]
    return tf.gather(fscore, y_out) * create_mask(y_out)


def get_tok_fscore(tm, lm, x, y, c):
    B = tf.shape(x)[0]
    dec_tm, _, _ = create_stateful_decoder_TM(tm, x)
    dec_lm, _, _ = create_stateful_decoder_LM(lm, B)
    dec_clm, _, _ = create_stateful_decoder_CtxLM(lm, c)
    return get_tok_fscore_from_decoders(dec_tm, dec_lm, dec_clm, y)


def get_seq_fscore(tm, lm, x, y, c):
    tok_fscore = get_tok_fscore(tm, lm, x, y, c)
    return tf.reduce_sum(tok_fscore, axis=1)


def ctx_aware_beam_search(tm, lm, x, c, beam_size, sos, eos, maxlen):
    B = tf.shape(x)[0]
    K = beam_size
    sos = tf.broadcast_to(sos, [B])
    dec_tm, state_tm, sinv_tm = create_stateful_decoder_TM(tm, x, K)
    dec_lm, state_lm, sinv_lm = create_stateful_decoder_LM(lm, B, K)
    dec_clm, state_clm, sinv_clm = create_stateful_decoder_CtxLM(lm, c, K)
    
    state = [state_tm, state_lm, state_clm]
    shape_inv = [sinv_tm, sinv_lm, sinv_clm]

    def get_logits_fn_(y):
        return get_fscore_logits_from_decoders(
            dec_tm,
            dec_lm,
            dec_clm,
            y)
    
    def perm_batch_fn_(permutation):
        tm.permute_cache(state_tm, permutation)
        lm.permute_cache(state_lm, permutation)
        lm.permute_cache(state_clm, permutation)
    
    def get_state_fn_():
        return state
    
    def put_controlled_state_fn_(state_):
        recursive_update(state, state_)
    
    paths, scores = beam_search(
        get_logits_fn=get_logits_fn_,
        perm_batch_fn=perm_batch_fn_,
        sos=sos,
        eos=eos,
        beam_size=beam_size,
        maxlen=maxlen,
        pad=0,
        get_state_fn=get_state_fn_,
        put_controlled_state_fn=put_controlled_state_fn_,
        shape_invariants=shape_inv,
        length_penalty_fn=None)
    
    return paths, scores
    

class Inference:
    def __init__(
            self, 
            transformer_model,
            decoder_language_model,
            vocab_source,
            vocab_target,
            batch_capacity):

        self.tm = transformer_model
        self.lm = decoder_language_model
        
        self.vocab_src = vocab_source
        self.vocab_trg = vocab_target
        
        self.batch_capacity = batch_capacity

        # ID substitution for the language model
        # Must be improved
        self.lm.id_substitutions_ = [
            (self.vocab_trg.SOS_ID, self.vocab_trg.EOS_ID)
        ]


    def create_data_gen_multi(self, x_multi, vocabs):
        return (
            dp.ChainableGenerator(lambda: iter(x_multi))
            .trans(dp.gen_line2IDs_multi, vocabs)
            .trans(dp.gen_batch_of_capacity_multi, self.batch_capacity)
            .trans(dp.gen_pad_batch_multi)
        )


    def dataset_from_gen(self, gen, structure):
        dtype = map_structure(lambda x: tf.int32, structure)
        shape = map_structure(lambda x: tf.TensorShape([None, None]), structure)
        return tf.data.Dataset.from_generator(gen, dtype, shape)


    def create_dataset_multi(self, xs, vocabs):
        gen = self.create_data_gen_multi(zip(*xs), vocabs)
        return self.dataset_from_gen(gen, (None,) * len(vocabs))


    @tf.function(input_signature=[SI2, SI2, SI0, SF0])
    def comp_translate(self, x, c, beam_size, maxlen_ratio):
        with tf.device('/gpu:0'):
            if tf.size(x) > 0:
                src_lens = tf.reduce_sum(create_mask(x), axis=1)
                maxlen = tf.cast(maxlen_ratio * src_lens + 10.0, tf.int32)
                maxlen = tf.where(src_lens == 0.0, 0, maxlen)
                # [B, K, L], [B, K]

                paths, scores = ctx_aware_beam_search(
                    self.tm,
                    self.lm,
                    x,
                    c,
                    beam_size=beam_size,
                    sos=self.vocab_trg.SOS_ID,
                    eos=self.vocab_trg.EOS_ID,
                    maxlen=maxlen)
            else:
                B = tf.shape(x)[0]
                paths = tf.zeros([B, beam_size, 0], tf.int32)
                scores = tf.zeros([B, beam_size], tf.float32)

            return paths, scores


    @tf.function(input_signature=[SI2, SI2, SI2])
    def comp_seq_logp(self, x, y, c):
        with tf.device('/gpu:0'):
            return get_seq_fscore(self.tm, self.lm, x, y, c)
    

    def translate_doc(self, doc, n_ctx, beam_size, maxlen_ratio=1.5):
        src_v, trg_v = self.vocab_src, self.vocab_trg
        def make_x_(src):
            return tf.constant([src_v.line2IDs(src)], tf.int32)
        
        def make_c_(ctx):
            if len(ctx) == 0:
                return tf.constant([[]], tf.int32)
            else:
                return tf.constant([trg_v.tokens2IDs(ctx)], tf.int32)

        out = []
        ctx_q = deque()
        len_q = deque()
        t = time.time()

        for i, sent in enumerate(doc):
            if i % 10 == 0 and i > 0:
                logger.debug(f'{i}/{len(doc)}\t{(time.time() - t) / (i + 1)}')
            x = make_x_(sent)
            c = make_c_(ctx_q)
            paths, _ = self.comp_translate(x, c, beam_size, maxlen_ratio)
            paths = paths.numpy()
            trans_toks = trg_v.IDs2tokens1D(paths[0][0])

            out.append(' '.join(trans_toks))

            len_q.append(1 + len(trans_toks))
            ctx_q.append(trg_v.ID2tok[trg_v.EOS_ID])
            ctx_q.extend(trans_toks)

            if len(len_q) > n_ctx:
                l = len_q.popleft()
                for _ in range(l):
                    ctx_q.popleft()

        return out


    def translate_docs(
            self, docs, n_ctx, beam_size, maxlen_ratio=1.5):
        return [self.translate_doc(doc, n_ctx, beam_size, maxlen_ratio)
            for doc in docs]


    def translate_docs_batch(self, docs, n_ctx, beam_size, maxlen_ratio=1.5):
        src_v, trg_v = self.vocab_src, self.vocab_trg
        def make_batches_(src_sents, ctx_sents):
            o = dp.gen_line2IDs_multi(
                zip(src_sents, ctx_sents),
                (src_v, trg_v))
            o = dp.gen_batch_of_capacity_multi(o, self.batch_capacity)
            o = dp.gen_pad_batch_multi(o)
            o = dp.gen_list2numpy_nested(o)
            return o

        docs = [deque(doc) for doc in docs]
        out = [[] for i in range(len(docs))]
        ctx_q = [deque() for i in range(len(docs))]
        len_q = [deque() for i in range(len(docs))]

        t = time.time()
        ntotal = sum(map(len, docs))
        nprocessed = 0

        while True:
            idx = [i for i, doc in enumerate(docs) if len(doc) > 0]

            if len(idx) == 0:
                break

            src_sents = [docs[i].popleft() for i in idx]
            ctx_sents = [' '.join(ctx_q[i]) for i in idx]

            x_cs = make_batches_(src_sents, ctx_sents)
            os = []
            for x, c in x_cs:
                paths, _ = self.comp_translate(x, c, beam_size, maxlen_ratio)
                paths = paths[:, 0].numpy()
                trans_toks = trg_v.IDs2tokens2D(paths)
                os.extend(trans_toks)

            assert len(os) == len(idx)
            for i, o in zip(idx, os):
                out[i].append(' '.join(o))
                len_q[i].append(1 + len(o))
                ctx_q[i].append(trg_v.ID2tok[trg_v.EOS_ID])
                ctx_q[i].extend(o)
                
                if len(len_q[i]) > n_ctx:
                    l = len_q[i].popleft()
                    for _ in range(l):
                        ctx_q[i].popleft()

            nprocessed += len(idx)
            logger.debug(f'{nprocessed}/{ntotal}\t{(time.time()-t)/nprocessed}')

        return out

    def gen_fscore(self, x, y, c):
        """
            c: </s> AAA BBB </s> CCC DDD
            y : </s> EEE FFF </s>
        """
        dataset = (
            self.create_dataset_multi(
                (x, y, c),
                (self.vocab_src, self.vocab_trg, self.vocab_trg))
            .prefetch(1)
            .map(lambda *b: self.comp_seq_logp(b[0], b[1], b[2]))
            .unbatch().prefetch(1)
        )
        yield from dataset


    def unit_test(self):
        t = time.time()
        with open('./inf_test/dev.ru') as f:
            for i, line in enumerate(self.gen_sents2sents( f, beam_size=1)):
                a = line
                if i % 100 == 0 and i != 0:
                    print(i, (time.time() - t)/i)


def load_tm(tm_dir, checkpoint):
    # Translation model Config
    with open(f'{tm_dir}/model_config.json') as f:
        model_config = json.load(f)
    
    # Transformer Model
    model = vl.Transformer.from_config(model_config)
    ckpt = tf.train.Checkpoint(model=model)
    if checkpoint is None:
        ckpt_path = tf.train.latest_checkpoint(f'{tm_dir}/checkpoint_best')
    else:
        ckpt_path = checkpoint
    assert ckpt_path is not None
    ckpt.restore(ckpt_path)
    logger.info(f'Checkpoint: {ckpt_path}')

    return model


def load_lm(lm_dir, checkpoint):
    # Translation model Config
    with open(f'{lm_dir}/model_config.json') as f:
        model_config = json.load(f)
    
    # Transformer Model
    model = ll.DecoderLanguageModel.from_config(model_config)
    ckpt = tf.train.Checkpoint(model=model)
    if checkpoint is None:
        ckpt_path = tf.train.latest_checkpoint(f'{lm_dir}/checkpoint_best')
    else:
        ckpt_path = checkpoint
    assert ckpt_path is not None
    ckpt.restore(ckpt_path)
    logger.info(f'Checkpoint: {ckpt_path}')

    return model


def main(argv, in_fp):
    p = argparse.ArgumentParser()
    p.add_argument('tm_dir', type=str)
    p.add_argument('lm_dir', type=str)
    p.add_argument('--tm_checkpoint', '--tm_ckpt', type=str)
    p.add_argument('--lm_checkpoint', '--lm_ckpt', type=str)
    p.add_argument('--capacity', type=int, default=16384)
    p.add_argument('--mode', choices=['translate', 'fscore', 'test'],
        default='translate')
    p.add_argument('--beam_size', type=int, default=1)
    p.add_argument('--n_ctx', type=int, default=3)
    p.add_argument('--length_penalty', type=float)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--debug_eager_function', action='store_true')
    p.add_argument('--progress_frequency', type=int, default=10**10)
    args = p.parse_args(argv)

    if args.debug:
        basicConfig(level=DEBUG)

    if args.debug_eager_function:
        tf.config.run_functions_eagerly(True)

    tm = load_tm(args.tm_dir, args.tm_checkpoint)
    lm = load_lm(args.lm_dir, args.lm_checkpoint)

    with open(f'{args.tm_dir}/vocab_config.json') as f:
        vc = json.load(f)

    vocab_src = Vocabulary(
        args.tm_dir + '/' + vc['source_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    vocab_trg = Vocabulary(
        args.tm_dir + '/' + vc['target_dict'],
        PAD_ID=vc['PAD_ID'],
        SOS_ID=vc['SOS_ID'],
        EOS_ID=vc['EOS_ID'],
        UNK_ID=vc['UNK_ID'])
    
    # Inference Class
    inference = Inference(tm, lm, vocab_src, vocab_trg, args.capacity)

    if args.mode == 'translate':
        docs = gen_lines_to_docs(in_fp)
        out_docs = inference.translate_docs_batch(
            docs,
            n_ctx=args.n_ctx,
            beam_size=args.beam_size,
            maxlen_ratio=1.5)
        for line in gen_docs_to_lines(out_docs):
            print(line)
    elif args.mode == 'test':
        inference.unit_test()

if __name__ == '__main__':
    main(sys.argv[1:], sys.stdin)
