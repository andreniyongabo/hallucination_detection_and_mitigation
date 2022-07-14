#!/bin/bash

if [ $# -ne 2 ]; then
    echo "usage: $0 [DATASET (e.g., BookCorpus)] [MODELPATH (/path/to/model.pt)]"
    exit
fi

DATAROOT=/private/home/myleott/data/ThePile/data-gpt-newlines-bin

DATASET=$1
MODELPATH=$2
mkdir -p $MODELPATH.results
OUTPUT_NAME=$MODELPATH.results/ThePile.$DATASET.test_ppl

python -m fairseq_cli.eval_lm $DATAROOT/$DATASET --path $MODELPATH --tokens-per-sample 1024 &> $OUTPUT_NAME
