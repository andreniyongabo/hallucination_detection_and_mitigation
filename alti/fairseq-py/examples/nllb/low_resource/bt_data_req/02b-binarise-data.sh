#!/bin/bash

LANGS=(asm zul amh tel urd vie ita fra deu)
SIZES=(50k 25k 10k 5k 1k)
MONO_SIZES=(150k 300k)

PREPROCESS=~jeanm/src/fairseq-py/fairseq_cli/preprocess.py
DATA_DIR=/checkpoint/jeanm/nllb/bt_data_req

for lang in "${LANGS[@]}"; do
  # Sort the pair of languages
  sorted_langs=($(for l in $lang eng; do echo $l; done | sort))
  src=${sorted_langs[0]}
  tgt=${sorted_langs[1]}

  dict=$DATA_DIR/data-bin/dicts/$lang.txt

  for size in "${SIZES[@]}"; do
    echo Binarising $src-$tgt $size seed data...
    mkdir -p $DATA_DIR/data-bin/seed/$src-$tgt.$size
    testpref=$DATA_DIR/data/test/test
    trainpref=$DATA_DIR/data/train/$src-$tgt.$size
    validpref=$DATA_DIR/data/valid/valid
    python $PREPROCESS --source-lang=$src --target-lang=$tgt \
      --trainpref $trainpref --validpref $validpref  \
      --testpref $testpref --thresholdtgt 0 --thresholdsrc 0 --workers 20 \
      --srcdict $dict --joined-dictionary \
      --destdir $DATA_DIR/data-bin/seed/$src-$tgt.$size
  done

  for mono_size in "${MONO_SIZES[@]}"; do
    mono_data=$DATA_DIR/data/monolingual/$mono_size
    dest_dir=$DATA_DIR/data-bin/monolingual/$lang.${mono_size}
    echo Binarising $lang $mono_size monolingual data...
    python $PREPROCESS --only-source --source-lang $lang --target-lang eng \
      --joined-dictionary --srcdict $dict --workers 20 \
      --testpref $mono_data \
      --destdir $dest_dir
    cp $dict $dest_dir/dict.eng.txt
    done
done
