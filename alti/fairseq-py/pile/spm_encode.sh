#!/bin/bash

INPUT_NAME=$1
shift
OUTPUT_NAME="${INPUT_NAME}.bpe"
m="/large_experiments/flores/namangoyal/cc100_combined/spm_256000.model"
DICT=/large_experiments/moe/cc100_xl/bin/shard0/dict.txt
echo "SPM encoding $INPUT_NAME"

python $fdir/scripts/spm_encode.py  \
  --model $m \
  --inputs $INPUT_NAME \
  --outputs $OUTPUT_NAME \
  --output_format piece $@
