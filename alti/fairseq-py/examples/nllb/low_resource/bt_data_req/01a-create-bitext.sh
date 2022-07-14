#!/bin/bash

LANG_ISO=(asm zul amh tel urd vie ita fra deu)
LANG_FLORES=(as zu am te ur vi it fr de)

FAIRSEQ=~/src/fairseq-py
FLORES=/large_experiments/flores/training_data/binarized
FLORES_DEV=/large_experiments/flores/flores101_new_spm_matrix_binarized/spm_encoded/flores101
DATA_DIR=/checkpoint/jeanm/nllb/bt_data_req

# Provides consistent random data so that `shuf` selects the same set of
# lines from the two files making up the parallel corpus
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
  </dev/zero 2>/dev/null
}

for i in "${!LANG_ISO[@]}"; do
  # Sort the pair of languages.
  # NB: this code is somewhat fragile - the two sets of language codes must be strictly
  # positive monotonic for this to work.
  lang_iso=${LANG_ISO[$i]}
  lang_flores=${LANG_FLORES[$i]}
  sorted_iso=($(for l in $lang_iso eng; do echo $l; done | sort))
  sorted_flores=($(for l in $lang_flores en; do echo $l; done | sort))

  flores_src_pref=$FLORES/train.spm.${sorted_flores[0]}-${sorted_flores[1]}
  train_out_pref=$DATA_DIR/data/train/${sorted_iso[0]}-${sorted_iso[1]}

  echo Truncating $flores_src_pref.en...
  cat $flores_src_pref.*.en | shuf -n 50000 --random-source=<(get_seeded_random 42) > $train_out_pref.50k.eng
  head -n 25000 $train_out_pref.50k.eng > $train_out_pref.25k.eng
  head -n 10000 $train_out_pref.50k.eng > $train_out_pref.10k.eng
  head -n 5000 $train_out_pref.50k.eng > $train_out_pref.5k.eng
  head -n 1000 $train_out_pref.50k.eng > $train_out_pref.1k.eng
  # Dev and devtest data is copied as-is
  cp $FLORES_DEV.dev.spm.en  $DATA_DIR/data/valid/valid.eng
  cp $FLORES_DEV.devtest.spm.en  $DATA_DIR/data/test/test.eng

  echo Truncating $flores_src_pref.$lang_flores...
  cat $flores_src_pref.*.$lang_flores | shuf -n 50000 --random-source=<(get_seeded_random 42) > $train_out_pref.50k.$lang_iso
  head -n 25000 $train_out_pref.50k.$lang_iso > $train_out_pref.25k.$lang_iso
  head -n 10000 $train_out_pref.50k.$lang_iso > $train_out_pref.10k.$lang_iso
  head -n 5000 $train_out_pref.50k.$lang_iso > $train_out_pref.5k.$lang_iso
  head -n 1000 $train_out_pref.50k.$lang_iso > $train_out_pref.1k.$lang_iso
  # Dev and devtest data is copied as-is
  cp $FLORES_DEV.dev.spm.$lang_flores  $DATA_DIR/data/valid/valid.$lang_iso
  cp $FLORES_DEV.devtest.spm.$lang_flores  $DATA_DIR/data/test/test.$lang_iso
done
