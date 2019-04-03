# Templete of model_config.py

import os, sys
import subprocess
from logging import getLogger, DEBUG, basicConfig
logger = getLogger(__name__)

import numpy as np
import tensorflow as tf
import codecs

import sentencepiece as spm

sys.path.append('/home/sugi/nlp/transformer')
import model_config_template

class Hyperparams(model_config_template.Hyperparams):

    # training
    batch_size = 128
    warm_up_step = 4000
    n_epochs = 20

    # model
    maxlen = 64
    n_blocks = 6
    n_heads = 8
    attention_size = 512
    embed_size = 512
    dropout_rate = 0.1
    vocab_size = 16000
    share_embedding = True
    positional_embedding = False

    #beam search
    length_penalty_a = 1

class Config(model_config_template.Config):
    _SP_model_dir = '/disk/sugi/dataset/ASPEC/preprocessed/sp16k_t1500k_shared'
    _SP_model_prefix_source = _SP_model_dir + '/sp16k'
    _SP_model_prefix_target = _SP_model_prefix_source #vocabulary is shared
    _SP_model_file_source = _SP_model_prefix_source + ".model"
    _SP_model_file_target = _SP_model_prefix_target + ".model"

    #Preprocessed dataset
    source_train = '/disk/sugi/dataset/ASPEC/preprocessed/train/train-1500k.en.txt'
    target_train = '/disk/sugi/dataset/ASPEC/preprocessed/train/train-1500k.ja.txt'
    source_dev = '/disk/sugi/dataset/ASPEC/preprocessed/dev/dev.en.txt' 
    target_dev = '/disk/sugi/dataset/ASPEC/preprocessed/dev/dev.ja.txt' 
    source_test = '/disk/sugi/dataset/ASPEC/preprocessed/test/test.en.txt' 
    target_test = '/disk/sugi/dataset/ASPEC/preprocessed/test/test.ja.txt' 

    # tokenized dataset
    source_train_tok = _SP_model_dir + '/train/train-1500k.en.tok'
    target_train_tok = _SP_model_dir + '/train/train-1500k.ja.tok'
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

    # Overriding preprocess_(source/target)
    @classmethod
    def preprocess_source(cls, text):
        return subprocess.run(['/disk/sugi/dataset/ASPEC/preprocess.sh', 'en'],
            input=text.encode(), stdout=subprocess.PIPE).stdout.decode()

    @classmethod
    def preprocess_target(cls, text):
        return subprocess.run(['/disk/sugi/dataset/ASPEC/preprocess.sh', 'ja'],
            input=text.encode(), stdout=subprocess.PIPE).stdout.decode()

    @classmethod
    def text2tokens_BLEU(cls, texts):
        """
        Args:
            texts: list of str
            type: "source" or "target"
            
        Returns:
            list of list of str."""
            
        output = [line.strip().split() for line in texts]
        return output


# preprocessing: Train sentencepiece with vocabulary size 32k shared by source/target
if __name__ == '__main__':
    basicConfig()
    logger.setLevel(DEBUG)
    #Preprocessing
    if not os.path.exists(Config.source_train):
        logger.info("Preprocessed data ({}) was not found. Preprocessing the corpus.".format(Config.source_train))

        #####################################################
        # CHOOSE ONE BELOW
        #####################################################


        #####################################################
        # Preprocessing Type 1
        # en: Tokenization + Truecasing by Moses
        # ja: Tokenization by Kytea
        subprocess.run(['/disk/sugi/dataset/ASPEC/preprocess.sh', 'init'])
        #####################################################
        
        #####################################################
        # Preprocessing Type 1
        # en: Tokenization + Truecasing by Moses
        # ja: Nothing
        #subprocess.run(["/disk/sugi/dataset/ASPEC/preprocess2.sh", "init"])
        #####################################################

        logger.info('Preprocessing done.')

    else:
        logger.info('Preprocessed corpus already exists.')


    if not os.path.exists(Config.source_train_tok):
        logger.info('Subword corpus ({}) was not found.'.format(Config.source_train_tok))
        logger.info('Creating subword corpus')
        #train sentencepiece with the preprocessed train data
        #!model dependent: in this model_config.py vocabulary is shared by source and target
        os.makedirs(Config._SP_model_dir, exist_ok=True)
        logger.info("Training sentencepiece")
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
        #!both source and target data are tokenized by the same model
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
                    logger.info("Tokenizing {} into {}".format(src, dest))
                    for line in src_file:
                        encoded = sp.EncodeAsPieces(line.strip())
                        dest_file.write(" ".join(encoded) + "\n")
            
    else:
        logger.info('Subword format corpus already exists.')

    logger.info('train/dev/test data are prepaired.')
