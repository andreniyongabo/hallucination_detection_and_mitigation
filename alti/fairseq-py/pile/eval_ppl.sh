#!/bin/bash

DATAROOT=/private/home/myleott/data/data-bin/ThePile
DATASET=$1
MODELPATH=$2
SAVE_PREFIX=$3
shift
shift
shift
D=$DATAROOT/$DATASET
SAVE_PREFIX_FULL=$SAVE_PREFIX.$DATASET
mkdir -p $SAVE_DIR

python -m fairseq_cli.eval_lm $D --path $MODELPATH \
  --tokens-per-sample 1024  --max-sentences 1  \
  --results-path "$SAVE_PREFIX_FULL".results.json \
  "$@" > >(tee "$SAVE_PREFIX_FULL".stdout.log) 2> >(tee "$SAVE_PREFIX_FULL".stderr.log >&2)
