# Templete of model_config.py

import os
import subprocess
from logging import getLogger, DEBUG, basicConfig
logger = getLogger(__name__)

import numpy as np
import tensorflow as tf
import codecs

import sentencepiece as spm

class Hyperparams:

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

"""

# About data processing and BLEU evaluation

## Data processing for Encoder

There are 4 format of text

1. Raw text
    - Can include any character
2. Preprocessed text
    - truecased
    - tokenized (Moses, Mecab, Kytea etc)
    - not necessarily normalized
3. Subword text
    - subwords sequence
    - each subword corresponds to a ID
    - by default SentencePiece is used
4. BLEU-tokenized text
    - tokenized to be evaluated with BLEU
    - in most cases the same as 2.

General corpora are distributed in the format 1. Format 1 texts are converted into ID sequences through the following pipeline and feed to the transformer encoder.

Raw text -> Preprocessed text -> Subwords -> IDs

In this framework, the train/dev/test dataset are cached in the format 3. It's to avoid the expensive process of converting raw texts into subwords in every training time. So, before running trainer.py, train/dev/(test) dataset in the format 3 needs to be prepared.

Note: Currently on-the-fly tokenization methods like subword regularization are not supported.

## Data processing of Decoder outputs
Sequence of IDs output by Decoder is back-converted to a text in the following process

IDs -> subwords -> format 2 text

The final output the user gets is in the same sytle as the preprocessed data which is not necessary the natural style of writing. For example, the initial letter of the first word in a English sentence might not be upper case, or Japanese texts can be in the style where words are separated by space.

## BLEU evaluation
translations and references are made by the following process.

### Translations

Decoder output IDs --(ID to token)--> subwords --(detokenization)--> format 2 texts --(tokenization for BLEU)--> format 4 texts

### References

Subwords --(detokenization)--> format 2 texts --(tokenization for BLEU)--> format 4 texts

Note: Although we initailly have the reference texts in the format 2, we don't use it and instead we detokenize the reference texts in the format 3 to produce ones in the format 2. This is because text normalization in this framework is done by SentencePiece. Reference texts need to undergo SentencePiece's conversion.

# Data files used by the training/test/inference scripts (training.py, inference.py)

- tokenized texts of train/dev data for training
- vocabulary files for token-to-ID transformation

# Methods of this class (type: source/target)

- preprocess(texts, type): preprocess texts. don't override this method. Instead override preprocess_source and preprocess_target
- text2tokens(texts, type): tokenization
- text2IDs(texts, type): tokenization (directly into IDs)
- IDs2text(seqs, type): detokenization from IDs to text
- tokens2text(tokens, type): detokenization from tokens to text
- text2tokens_BLEU(texts, type): tokenization for BLEU evaluation. 

"""
class Config:
    # -------------- internal settings of data processing -------------
    #variables here won't be referred from train/test/prediction scripts

    #sentencepiece setting for this model
    #make sure the vocabulary size is consistent with Hyperparams.vocab_size
    _SP_model_dir = '/disk/sugi/dataset/ASPEC/preprocessed/sp16k_t1500k_shared'
    _SP_model_prefix_source = _SP_model_dir + '/sp16k'
    _SP_model_prefix_target = _SP_model_prefix_source #vocabulary is shared
    _SP_model_file_source = _SP_model_prefix_source + ".model"
    _SP_model_file_target = _SP_model_prefix_target + ".model"


    #Special tokens. Make sure these are consistent with the tokenization model settings.
    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    # Constants
    SOURCE = 'source'
    TARGET = 'target'

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


    @classmethod
    def preprocess_source(cls, text):
        """
        Args:
            text: a str containing lines representing source sentences to preprocess.
        Returns:
            str. preprocessed source sentences.
            """
        return subprocess.check_output(
            ["/disk/sugi/dataset/ASPEC/preprocess.sh", 'en'],
            input=text.encode()
        ).decode('utf-8')

    @classmethod
    def preprocess_target(cls, text):
        """
        Args:
            text: str. target sentences to preprocess.
        Returns:
            str. preprocessed target sentences.
            """
        return subprocess.check_output(
            ["/disk/sugi/dataset/ASPEC/preprocess.sh", 'ja'],
            input=text.encode()
        ).decode('utf-8')

    @classmethod
    def text2tokens_BLEU(cls, texts):
        """Tokenize sentences for BLEU evaluation.
        Japanese texts are tokenized by kytea.
        Texts in languages which use space as delimiter are tokenized
        by splitting by space
        
        Args:
            texts: list of str
            type: "source" or "target"
            
        Returns:
            list of list of str."""
            
        ###################################################
        ## CHOOSE ONE METHOD TO PRODUCE `output` BELOW
        ###################################################
            

        ####################################################
        # Lang: ja
        # Preprocesing Type: 2 (Not using kytea to tokenize)
#        kytea_input = ("\n".join(texts) + "\n").encode()
#        kytea_output = subprocess.check_output(["kytea", "-out", "tok"],
#                                               input=kytea_input).decode()
#        output = [line.strip().split() for line in kytea_output.strip().split("\n")]
        #####################################################
        #####################################################
        # Lang: ja
        # Preprocessing Type: 1 (Using kytea to tokenize)
        # - Simply split by SPACE
        output = [line.strip().split() for line in texts]
        #####################################################
        #####################################################
        # Lang: en
        # Preprocessing Type: Tokenization + Truecasing by MOSES toolkit
        # - Simply split by SPACE
#        output = [line.strip().split() for line in texts]
        #####################################################

        return output
    
    @classmethod
    def _load_sp(cls):
        cls._sp_source = spm.SentencePieceProcessor()
        cls._sp_source.Load(cls._SP_model_file_source)
        cls._sp_target = spm.SentencePieceProcessor()
        cls._sp_target.Load(cls._SP_model_file_target)

    @classmethod
    def text2tokens(cls, sents, type):
        """tokenize sentences into sequences of tokens
        
        Args:
            sents: list of str
            type: "source" or "target"
            
        Returns:
            list of list of str"""
        
        if not hasattr(cls, '_sp_source'): cls._load_sp()
        sp = cls._sp_source if type==cls.SOURCE else cls._sp_target
        return [sp.EncodeAsPieces(sent) for sent in sents]

    @classmethod
    def text2IDs(cls, sents, type):
        """tokenize sentences into sequences of IDs
        
        Args:
            sents: list of str
            
        Returns:
            list of list of int"""
        #in this model_config.py target=ja
        if not hasattr(cls, '_sp_source'): cls._load_sp()
        sp = cls._sp_source if type==cls.SOURCE else cls._sp_target
        return [sp.EncodeAsIds(sent) for sent in sents]

    @classmethod
    def tokens2text(cls, tokens, type):
        """detokenize tokens into strings
        Args:
            tokens: list of list of str
            type: "source" or "target"
        
        Returns:
            list of str"""
        if not hasattr(cls, '_sp_source'): cls._load_sp()
        sp = cls._sp_source if type==cls.SOURCE else cls._sp_target
        return [sp.DecodePieces(tok) for tok in tokens]

    @classmethod
    def IDs2text(cls, seqs, type):
        """detokenize sequence of IDs into strings
        Args:
            source_seqs: list of list of int
        
        Returns:
            list of str"""
        #in this model_config.py source=en
        if not hasattr(cls, '_sp_source'): cls._load_sp()
        sp = cls._sp_source if type==cls.SOURCE else cls._sp_target
        return [sp.DecodeIds(seq) for seq in seqs]

    @classmethod
    def IDs2tokens(cls, seqs, type):
        """Convert id sequences into subword sequences
        NOTE: The output of this method is equivalent to text2tokens(IDs2text(seqs, type))
        So, all the control symbols (SOS, EOS, PAD etc) are ignored.

        Args:
            seqs: list of list of int
        Returns:
            list of list of str.
            """
        if not hasattr(cls, '_sp_source'): cls._load_sp()
        sp = cls._sp_source if type==cls.SOURCE else cls._sp_target
        return [[sp.id_to_piece(id) for id in seq if not sp.is_control(id)] for seq in seqs]

    @classmethod
    def preprocess(cls, texts, type, in_file=None, out_file=None):
        """Preprocess texts
        Args:
            texts: input text. list of str. if in_file is specified this argument is ignored.
            type: "source" or "target"
            in_file: You can specify a file as the source instead of list of text. Default is None.
            out_file: If None, this method returns list of str. If not None, outputs will be written into out_file. Default is None
        Returns:
            List of str if out_file is None, otherwise None"""
        
        if in_file is not None:
            with codecs.open(in_file, "r") as f:
                input_text = f.read()
        else:
            input_text = "\n".join(texts) + "\n"

        # Project-dependent preprocessing
        if type == cls.SOURCE:
            output_text = cls.preprocess_source(input_text)
        else:
            assert type == cls.TARGET
            output_text = cls.preprocess_target(input_text)


        if out_file is not None:
            with codecs.open(out_file, "w") as f:
                f.write(output_text)
        else:
            return output_text.strip().split("\n")
