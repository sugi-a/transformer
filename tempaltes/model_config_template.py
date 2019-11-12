# Templete of model_config.py
assert False

import os, subprocess, json
from logging import getLogger, DEBUG, basicConfig; logger = getLogger(__name__)

with open(os.path.dirname(__file__) + '/' + 'model_config.json') as f:
    params = json.load(f)
"""
def IDs2tokens(IDs, lang):
    '''IDs: list of list of int'''
    pass
"""

"""
def validation_metric(global_step, inference):
    pass

"""
