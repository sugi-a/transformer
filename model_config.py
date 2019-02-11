import os
import subprocess

import numpy as np
import tensorflow as tf
import codecs

import sentencepiece as spm

class Hyperparams:

    # training
    batch_size = 128
    lr = 0.0001
    n_epochs = 20

    # model
    maxlen = 64
    n_blocks = 6
    n_heads = 8
    attention_size = 512
    embed_size = 512
    dropout_rate = 0.1
    vocab_size = 16000
    positional_embedding = False

    #beam search
    length_penalty_a = 1

"""
# About data processing and BLEU evaluation

## Data processing for Encoder

To be input into Encoder of Tranformer, the original text (raw corpus text) is converted into sequence of IDs through 2 steps.

1. (almost) Irreversible preprocessing is applied to the original text producing "preprocessed text". Example:

    - MOSES tokenizer for English text
    - MOSES Trucaser for English text
    - Lowercasing all words in English text
    - Kytea for Japanese text (this can be regarded as reversible)

2. Reversible tokenization is applied to the preprocessed data producing tokenized text. Example:
    
    - sentencepiece tokenizer

3. Converting tokenized data into IDs. This is done through Tensorflow's data loading pipeline directly connected to Encoder. This phase includes the following.

    - OOV tokens are mapped to Config.UNK_ID
    - Sequences longer than Hyperparams.max_len are skipped
    - The EOS ID (Config.EOS_ID) is added at the end of each sequence

In short, the corpus is processed in the following flow.
Original -> Preprocessed -> Tokenized -> IDs

Note: Currently on-the-fly tokenization methods like subword regularization are not supported. Static vocabulary files are required in the phase 3.

## Data processing of Decoder outputs
Sequence of IDs output by Decoder is back-converted to a text in the following 2 steps

1. ID to token
2. Detokenization

Decoder output IDs --(ID to token)--> tokens --(detokenization)--> text

The final output the user gets is in the same sytle as the preprocessed data which is not necessary the natural style of writing. For example, the initial letter of the first word in a English sentence might not be upper case, or Japanese texts can be in the style where words are separated by space.

## BLEU evaluation
To perform BLEU evaluation, the final output text needs to be tokenized. This tokenization can be different from which is done to create Encoder inputs described above. For example, in general, Japanese texts are tokenized by Kytea for BLEU evaluation.
In short, the flow of creating list of tokens for BLEU evaluation is as follows.

- For translations: Decoder output IDs --(ID to token)--> tokens text --(detokenization)--> preprocessed text --(tokenization for BLEU)--> tokens for BLEU
- For references: tokenized text --(detokenization)--> preprocessed text --(tokenization for BLEU)--> tokens for BLEU

# Data files used by the training/test/inference scripts (training.py, inference.py)

- tokenized texts of train/dev data for training
- preprocessed text of test data for BLEU evaluation
- vocabulary files for token-to-ID transformation

Note: The original corpus and the first intermediate data files (preprocessed text) aren't used by those scripts.

# Methods of this class (type: source/target)

- preprocess(texts, type): preprocess texts
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
    _SP_model_dir = '/disk/sugi/dataset/ASPEC/preprocessed/sp16k_t1_shared'
    _SP_model_prefix_source = _SP_model_dir + '/sp16k'
    _SP_model_prefix_target = _SP_model_prefix_source #vocabulary is shared
    _SP_model_file_source = _SP_model_prefix_source + ".model"
    _SP_model_file_target = _SP_model_prefix_target + ".model"

    #original dataset
    _source_train_orig = '/disk/sugi/dataset/ASPEC/extracted/train/train-1.en.txt'
    _target_train_orig = '/disk/sugi/dataset/ASPEC/extracted/train/train-1.ja.txt'
    _source_dev_orig = '/disk/sugi/dataset/ASPEC/extracted/dev/dev.en.txt'
    _target_dev_orig = '/disk/sugi/dataset/ASPEC/extracted/dev/dev.ja.txt'
    _source_test_orig = '/disk/sugi/dataset/ASPEC/extracted/test/test.en.txt'
    _target_test_orig = '/disk/sugi/dataset/ASPEC/extracted/test/test.ja.txt'


    #Special tokens. Make sure these are consistent with the tokenization model settings.
    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    #Preprocessed dataset
    source_train = '/disk/sugi/dataset/ASPEC/preprocessed/train/train-1.en.txt'
    target_train = '/disk/sugi/dataset/ASPEC/preprocessed/train/train-1.ja.txt'
    source_dev = '/disk/sugi/dataset/ASPEC/preprocessed/dev/dev.en.txt' 
    target_dev = '/disk/sugi/dataset/ASPEC/preprocessed/dev/dev.ja.txt' 
    source_test = '/disk/sugi/dataset/ASPEC/preprocessed/test/test.en.txt' 
    target_test = '/disk/sugi/dataset/ASPEC/preprocessed/test/test.ja.txt' 

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
    
    def preprocess(texts, type, in_file=None, out_file=None):
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

        t = "en" if type=="source" else "ja"
        output_text = subprocess.check_output(
            ["/disk/sugi/dataset/ASPEC/preprocess.sh", t],
            input=input_text.encode()).decode()
            
        if out_file is not None:
            with codecs.open(out_file, "w") as f:
                f.write(output_text)
        else:
            return output_text.strip().split("\n")


    def text2tokens(sents, type):
        """tokenize sentences into sequences of tokens
        
        Args:
            sents: list of str
            type: "source" or "target"
            
        Returns:
            list of list of str"""
        
        sp = spm.SentencePieceProcessor()
        model = Config._SP_model_file_source if type == "source" else Config._SP_model_file_target
        sp.Load(model)
        return [sp.EncodeAsPieces(sent) for sent in sents]

    def text2IDs(sents, type):
        """tokenize sentences into sequences of IDs
        
        Args:
            sents: list of str
            
        Returns:
            list of list of int"""
        #in this model_config.py target=ja
        sp = spm.SentencePieceProcessor()
        model = Config._SP_model_file_source if type == "source" else Config._SP_model_file_target
        sp.Load(model)
        return [sp.EncodeAsIds(sent) for sent in sents]

    def tokens2text(tokens, type):
        """detokenize tokens into strings
        Args:
            tokens: list of list of str
            type: "source" or "target"
        
        Returns:
            list of str"""
        sp = spm.SentencePieceProcessor()
        model = Config._SP_model_file_source if type == "source" else Config._SP_model_file_target
        sp.Load(model)
        return [sp.DecodePieces(tok) for tok in tokens]

    def IDs2text(seqs, type):
        """detokenize sequence of IDs into strings
        Args:
            source_seqs: list of list of int
        
        Returns:
            list of str"""
        #in this model_config.py source=en
        sp = spm.SentencePieceProcessor()
        model = Config._SP_model_file_source if type == "source" else Config._SP_model_file_target
        sp.Load(model)
        return [sp.DecodeIds(seq) for seq in seqs]

    def text2tokens_BLEU(texts):
        """Tokenize sentences for BLEU evaluation.
        Japanese texts are tokenized by kytea.
        Texts in languages which use space as delimiter are tokenized
        by splitting by space
        
        Args:
            texts: list of str
            type: "source" or "target"
            
        Returns:
            list of list of str."""
            
        #in this model_config.py target=ja AND the preprocessed text is already tokenized by kytea. so just split the text by space is enough.
        return [line.split() for line in texts]
            


# preprocessing: Train sentencepiece with vocabulary size 32k shared by source/target
if __name__ == '__main__':
    #Preprocessing
    if not os.path.exists(Config.source_train):
        print("preprocessing the corpus")
        subprocess.run(["/disk/sugi/dataset/ASPEC/preprocess.sh", "init"])

    if not os.path.exists(Config.source_train_tok):
        #train sentencepiece with the preprocessed train data
        #!model dependent: in this model_config.py vocabulary is shared by source and target
        os.makedirs(Config._SP_model_dir, exist_ok=True)
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
                    print("Tokenizing {} to {}".format(src, dest))
                    for line in src_file:
                        encoded = sp.EncodeAsPieces(line.strip())
                        dest_file.write(" ".join(encoded) + "\n")
            

