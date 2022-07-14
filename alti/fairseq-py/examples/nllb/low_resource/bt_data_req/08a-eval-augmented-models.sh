#!/bin/bash

LANGS=(asm zul amh tel urd vie ita fra deu)
SIZES=(1k 5k 10k 25k 50k)

DATA_DIR=/checkpoint/jeanm/nllb/bt_data_req
SPM_MODEL=/large_experiments/flores/namangoyal/cc100_combined/spm_256000.model

for lang in "${LANGS[@]}"; do
  # Sort the pair of languages
  sorted_langs=($(for l in $lang eng; do echo $l; done | sort))
  src=${sorted_langs[0]}
  tgt=${sorted_langs[1]}
  for size in "${SIZES[@]}"; do
    data=$DATA_DIR/data-bin/seed/$src-$tgt.$size
    for split in valid test; do
      for checkpoint in $DATA_DIR/checkpoints/eng-$lang.$size+*; do
        if [ -f $checkpoint/out_$split ]; then
          echo $checkpoint... already computed
          continue
        fi
        echo $checkpoint...
        fairseq-generate $data --path $checkpoint/checkpoint_best.pt \
          --task translation_multi_simple_epoch --target-lang $lang --source-lang eng \
          --langs "$lang,eng" --lang-pairs "eng-$lang" --gen-subset $split \
          --decoder-langtok --encoder-langtok src \
          --bpe 'sentencepiece' --sentencepiece-model $SPM_MODEL \
          --beam 4 --max-tokens 5000 --sacrebleu \
          --fp16 --fp16-no-flatten-grads > $checkpoint/out_$split
      done
    done
  done
done
