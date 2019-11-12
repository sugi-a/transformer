#!/bin/bash -e

SRC=""
TRG=""
MODEL_PREFIX=""
VOCAB_SIZE=""




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
