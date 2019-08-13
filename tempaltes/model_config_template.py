# Templete of model_config.py
assert False

import os, subprocess, codecs
from logging import getLogger, DEBUG, basicConfig; logger = getLogger(__name__)

params = {
    "train": {
        "batch": {
            "fixed_capacity": False,
            "size": 128,
            "capacity": 128 * 64,
            "sort": False
        },
        "warm_up_step": 4000,
        "stop": {
            "type": "step", # "step", "epoch", "early_stopping"
            "max": 300,
            "wait_steps": 40000 # if "type"=="early_stopping", training stops when validation score hasn't improve for this amount of steps.
        },
        "data": {
            "maxlen": 64,
            "source_train": [],
            "target_train": [],
            "source_dev": "",
            "target_dev": ""
        },
        "logdir": ""
    },
    "test": {
        "length_penalty_a": 1.0
    },
    "network": {
        "n_blocks": 6,
        "n_heads": 8,
        "attention_size": 512,
        "embed_size": 512,
        "dropout_rate": 0.1,
        "vocab_size": 16000,
        "share_embedding": True,
        "positional_embedding": False
    },
    "vocab": {
        "PAD_ID": 0,
        "SOS_ID": 1,
        "EOS_ID": 2,
        "UNK_ID": 3,
        "source_dict": "",
        "target_dict": ""
    }
    
}

"""
def IDs2tokens(IDs, lang):
    '''IDs: list of list of int'''
    pass
"""

"""
def validation_metric(global_step, inference):
    pass

"""
