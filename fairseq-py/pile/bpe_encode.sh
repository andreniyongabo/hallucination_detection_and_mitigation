#!/bin/bash

INPUT_NAME=$1
OUTPUT_NAME="${INPUT_NAME}.bpe"

echo "BPE encoding $INPUT_NAME"

python /private/home/myleott/src/fairseq-py/examples/roberta/multiprocessing_bpe_encoder.py --encoder-json /private/home/myleott/src/gpt2_bpe/encoder.json --vocab-bpe /private/home/myleott/src/gpt2_bpe/vocab.bpe --inputs $INPUT_NAME --outputs $OUTPUT_NAME --keep-empty
