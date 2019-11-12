#!/bin/bash -e

MODEL_CONFIG_JSON="./model_config.json"


SRC=""
TRG=""
MODEL_PREFIX=""
VOCAB_SIZE="$(cat MODEL_CONFIG_JSON | jq '.vocab.vacab_size')"
PAD_ID="$(cat MODEL_CONFIG_JSON | jq '.vocab.PAD_ID')"
SOS_ID="$(cat MODEL_CONFIG_JSON | jq '.vocab.SOS_ID')"
EOS_ID="$(cat MODEL_CONFIG_JSON | jq '.vocab.EOS_ID')"
UNK_ID="$(cat MODEL_CONFIG_JSON | jq '.vocab.UNK_ID')"
USER_SYMBOLS=""


spm_train \
    --input=$SRC,$TRG \
    --model_prefix=$MODEL_PREFIX \
    --vocab_size=$VOCAB_SIZE \
    --character_coverage=0.9995 \
    --pad_id=$PAD_ID \
    --bos_id=$BOS_ID \
    --eos_id=$EOS_ID \
    --unk_id=$UNK_ID \
    --user_defined_symbols=$USER_SYMBOLS || { echo 'spm error'; exit 1; }
