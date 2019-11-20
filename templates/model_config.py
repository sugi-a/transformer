import os, json
"""
What this module must have:
    params: configuration of the Transformer and training. (dict)
What this module can have:
    IDs2tokens: method to convert IDs [nlines, maxlen] into tokens (function)
    validation_metric: validation function
"""

with open(os.path.dirname(__file__) + '/' + 'model_config.json') as f:
    params = json.load(f)

if 'basedir' in params: # add prefix to the dataset paths
    _p = params["basedir"] + '/'
    for k in ["source_train", "target_train"]:
        for i in range(len(params["train"]["data"][k])):
            params["train"]["data"][k][i] = _p + params["train"]["data"][k][i]
    params["train"]["data"]["source_dev"] = _p + params["train"]["data"]["source_dev"]
    params["train"]["data"]["target_dev"] = _p + params["train"]["data"]["target_dev"]
    params["vocab"]["source_dict"] = _p + params["vocab"]["source_dict"]
    params["vocab"]["target_dict"] = _p + params["vocab"]["target_dict"]

# Following two functions can be customly defined
"""
def IDs2tokens(IDs, lang):
    '''IDs: list of list of int'''
    pass

def validation_metric(global_step, inference):
    return <score>
"""

# Some examples:

"""
import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.Load(<path to spm file>)

def IDs2tokens(IDs):
    '''IDs: list of list of int'''
    return [' '.join(sp.id_to_piece(id) for id in sent if not sp.is_control(id)) for sent in IDs]

# def IDs2tokens(IDs):
#     '''IDs: list of list of int'''
#     return [sp.DecodeIds(sent) for sent in IDs]

with open(<dev_src>) as f:
    __src_lines = [line.strip() for line in f]
with open(<dev_trg>) as f:
    __trg_lines = [line.strip() for line in f]

    # decode, split, wrap
    refs = [sp.decode_pieces(line.split()) for line in __trg_lines]
    refs = [line.split() for line in refs]
    refs = [[ref] for ref in refs]


from nltk.translate.bleu_score import corpus_bleu

def validation_metric(global_step, inference):
    # translate, decode, split
    outs = inference.translate_sentences(__src_lines, 1)
    outs = [sp.decode_pieces(line.split()) for line in outs]
    outs = [line.split() for line in outs]

    return corpus_bleu(refs, outs)

"""
