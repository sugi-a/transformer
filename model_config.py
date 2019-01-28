import os
import subprocess

import numpy as np
import tensorflow as tf
import codecs

import sentencepiece as spm

class Hyperparams:

    # training
    batch_size = 64
    lr = 0.0001
    n_epochs = 100

    # model
    maxlen = 100
    n_blocks = 6
    n_heads = 8
    attention_size = 512
    embed_size = 512
    dropout_rate = 0.1
    vocab_size = 16000
    positional_embedding = False

class Config:
    #sentencepiece setting for this model
    _SP_model_dir = '/disk/sugi/dataset/ASPEC/sp16000'
    _SP_model_prefix_source = _SP_model_dir + '/sp16000'
    _SP_model_prefix_target = _SP_model_prefix_source #vocabulary is shared
    _SP_model_file_source = _SP_model_prefix_source + ".model"
    _SP_model_file_target = _SP_model_prefix_target + ".model"

    # data
    source_train = '/disk/sugi/dataset/ASPEC/train/train-1.en.txt'
    target_train = '/disk/sugi/dataset/ASPEC/train/train-1.ja.txt'
    source_dev = '/disk/sugi/dataset/ASPEC/dev/dev.en.txt'
    target_dev = '/disk/sugi/dataset/ASPEC/dev/dev.ja.txt'
    source_test = '/disk/sugi/dataset/ASPEC/test/test.en.txt'
    target_test = '/disk/sugi/dataset/ASPEC/test/test.ja.txt'

    # tokenized dataset
    source_train_tok = _SP_model_dir + '/train/train-1.en.tok'
    target_train_tok = _SP_model_dir + '/train/train-1.ja.tok'
    source_dev_tok = _SP_model_dir + '/dev/dev.en.tok'
    target_dev_tok = _SP_model_dir + '/dev/dev.ja.tok'
    source_test_tok = _SP_model_dir + '/test/test.en.tok'
    target_test_tok = _SP_model_dir + '/test/test.ja.tok'

    #vocabulary file.
    vocab_source = _SP_model_prefix_source + '.vocab'
    vocab_target = _SP_model_prefix_target + '.vocab'

    # working directory
    model_name = "model"
    logdir = os.path.dirname(__file__) + "/log"
    
    # special IDs for tokenization
    PAD_ID = 0
    UNK_ID = 1
    SOS_ID = 2
    EOS_ID = 3

    def tokenize_source(source_sents):
        """tokenize sentences into sequences of IDs
        
        Args:
            source_sents: list of str
            
        Returns:
            list of 1D numpy array of int"""
        
        #in this model_config.py source=en
        sp = spm.SentencePieceProcessor()
        sp.Load(_SP_model_file_source)
        return [np.array(sp.EncodeAsIds(sent)) for sent in source_sents]

    def tokenize_target(target_sents):
        """tokenize target sentences into sequences of IDs
        
        Args:
            target_sents: list of str
            
        Returns:
            list of 1D numpy array of int"""
        #in this model_config.py target=ja
        sp = spm.SentencePieceProcessor()
        sp.Load(_SP_model_file_target)
        return [np.array(sp.EncodeAsIds(sent)) for sent in target_sents]

    def detokenize_source(source_seqs):
        """detokenize source sequence of IDs to strings
        
        Args:
            source_seqs: list of 1D numpy array of int
        
        Returns:
            list of str"""

        #in this model_config.py source=en
        sp = spm.SentencePieceProcessor()
        sp.Load(_SP_model_file_source)
        return [sp.DecodeIds(seq) for seq in source_seqs]
    
    def detokenize_target(target_seqs):
        """detokenize target sequence of IDs to strings
        
        Args:
            target_seqs: list of 1D numpy array of int
        
        Returns:
            list of str"""

        #in this model_config.py target=en
        sp = spm.SentencePieceProcessor()
        sp.Load(_SP_model_file_target)
        return [sp.DecodeIds(seq) for seq in target_seqs]

    def blue_tokenize_source(source_sents):
        """tokenize source sentences for BLEU evaluation.
        Japanese texts are tokenized by kytea.
        Texts in languages which use space as delimiter are tokenized
        by splitting by space
        
        Args:
            source_sents: list of str
            
        Returns:
            list of list of str.
            ret[i] is the list of str for the i-th sentence"""

        #in this model_config.py source=en. so split by " "
        return [sent.split() for sent in source_sents]

    def blue_tokenize_target(target_sents):
        """tokenize target sentences for BLEU evaluation.
        Japanese texts are tokenized by kytea.
        Texts in languages which use space as delimiter are tokenized
        by splitting by space
        
        Args:
            target_sents: list of str
            
        Returns:
            list of list of str.
            ret[i] is the list of str for the i-th sentence"""

        #in this model_config.py target=ja. so split by kytea
        kytea_input = "\n".join(target_sents)
        kytea_output = subprocess.run(["kytea", "-notags"], 
                        input=kytea_input,
                        stdout=subprocess.PIPE,
                        text=True).stdout
        return [line.split() for line in kytea_output.split("\n")]

# preprocessing
if __name__ == '__main__':
    #train sentencepiece by train data
    os.makedirs(Config._SP_model_dir, exist_ok=True)
    #!model dependent: in this model_config.py vocabulary is shared by source and target
    print("Training sentencepiece")
    spm.SentencePieceTrainer.Train(
        '--input={} '\
        '--model_prefix={} '\
        '--vocab_size={} '\
        '--character_coverage={} '\
        '--pad_id={} '\
        '--bos_id={} '\
        '--eos_id={} '\
        '--unk_id={} '\
        .format(Config.source_train + "," + Config.target_train,
                Config._SP_model_prefix_source,
                Hyperparams.vocab_size,
                0.9995,
                Config.PAD_ID,
                Config.SOS_ID,
                Config.EOS_ID,
                Config.UNK_ID
                ))
    
    #tokenize train/dev/test data
    #both source and target data are tokenized by the same model
    sp = spm.SentencePieceProcessor()
    sp.Load(Config._SP_model_file_source)
    for src, dest in zip([Config.source_train,
                          Config.target_train,
                          Config.source_dev,
                          Config.target_dev,
                          Config.source_test,
                          Config.target_test],
                          [Config.source_train_tok,
                          Config.target_train_tok,
                          Config.source_dev_tok,
                          Config.target_dev_tok,
                          Config.source_test_tok,
                          Config.target_test_tok]):
        #make directory for data
        with codecs.open(src, 'r', 'utf-8') as src_file:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with codecs.open(dest, 'w', 'utf-8') as dest_file:
                print("Tokenizing {} to {}".format(src, dest))
                for line in src_file:
                    encoded = sp.EncodeAsPieces(line.strip())
                    dest_file.write(" ".join(encoded) + "\n")
            

