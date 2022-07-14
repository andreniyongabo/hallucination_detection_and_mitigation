#!/bin/bash

if [ $# -ne 2 ]; then
    echo "usage: $0 [DATASET (e.g., BookCorpus)] [MODELPATH (/path/to/model.pt)]"
    exit
fi

DATAROOT=/private/home/myleott/data/data-bin/ThePile

DATASET=$1
MODELPATH=$2
mkdir -p $MODELPATH.results
OUTPUT_NAME=$MODELPATH.results/ThePile.$DATASET.test_ppl

cd ~/src/fairseq-py
python -m fairseq_cli.eval_lm $DATAROOT/$DATASET --path $MODELPATH --tokens-per-sample 1024 \
  --model-parallel-size 8 --distributed-world-size 8 --batch-size 8 &> $OUTPUT_NAME
