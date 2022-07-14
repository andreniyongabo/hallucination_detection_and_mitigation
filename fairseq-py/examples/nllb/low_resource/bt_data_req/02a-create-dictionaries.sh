#!/bin/bash

# Creates language-specific joint dicts with good coverage of bitext and mono data.

LANGS=(asm zul amh tel urd vie ita fra deu)
MAX_BITEXT_SIZE_k=50
MAX_MONO_SIZE_k=300

PREPROCESS=~jeanm/src/fairseq-py/fairseq_cli/preprocess.py
DATA_DIR=/checkpoint/jeanm/nllb/bt_data_req

for lang in "${LANGS[@]}"; do
  # Sort the pair of languages
  sorted_langs=($(for l in $lang eng; do echo $l; done | sort))
  src=${sorted_langs[0]}
  tgt=${sorted_langs[1]}

  echo Creating a joint eng+$lang dictionary...

  # Pretend we're binarising a dataset. This is just to get a dictionary out of
  # Fairseq - the dataset itself will be discarded. As training data we concatenate the
  # bitext training data and the mono data, to make sure coverage is comprehensive.

  mono_data=$DATA_DIR/data/monolingual/${MAX_MONO_SIZE_k}k
  bitext_train=$DATA_DIR/data/train/$src-$tgt.${MAX_BITEXT_SIZE_k}k
  bitext_valid=$DATA_DIR/data/valid/valid
  bitext_test=$DATA_DIR/data/test/test

  dicts_dir=$DATA_DIR/data-bin/dicts
  mkdir -p $dicts_dir/$lang
  rm -rf $dicts_dir/$lang/*
  cat $bitext_train.$lang $mono_data.$lang > $dicts_dir/$lang/train.$lang
  cat $bitext_train.eng $mono_data.eng > $dicts_dir/$lang/train.eng
  ln -f -s $bitext_valid.$lang $dicts_dir/$lang/valid.$lang
  ln -f -s $bitext_valid.eng $dicts_dir/$lang/valid.eng
  ln -f -s $bitext_test.$lang $dicts_dir/$lang/test.$lang
  ln -f -s $bitext_test.eng $dicts_dir/$lang/test.eng

  python $PREPROCESS --source-lang=$src --target-lang=$tgt --joined-dictionary \
    --trainpref $dicts_dir/$lang/train --validpref $dicts_dir/$lang/valid  \
    --testpref $dicts_dir/$lang/test --thresholdtgt 0 --thresholdsrc 0 --workers 20 \
    --destdir $dicts_dir/$lang/
  mv $dicts_dir/$lang/dict.eng.txt $dicts_dir/$lang.txt
  rm -r $dicts_dir/$lang
done
