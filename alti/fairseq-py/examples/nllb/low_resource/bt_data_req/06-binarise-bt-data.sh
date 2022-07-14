#!/bin/bash

LANGS=(asm zul amh tel urd vie ita fra deu)
MONO_SIZES=(150k 300k)
SIZES=(50k 25k 10k 5k 1k)

PREPROCESS=~jeanm/src/fairseq-py/fairseq_cli/preprocess.py
DATA_DIR=/checkpoint/jeanm/nllb/bt_data_req
EXTRACT=~jeanm/src/fairseq-py/examples/backtranslation/extract_bt_data.py

for lang in "${LANGS[@]}"; do
  # Sort the pair of languages
  sorted_langs=($(for l in $lang eng; do echo $l; done | sort))
  src=${sorted_langs[0]}
  tgt=${sorted_langs[1]}
  dict=$DATA_DIR/data-bin/dicts/$lang.txt
  for size in "${SIZES[@]}"; do
    for mono_size in "${MONO_SIZES[@]}"; do
      input_data=$DATA_DIR/data-bin/monolingual/$lang.${mono_size}
      output_extracted_data=$DATA_DIR/data/bt/${lang}.${mono_size}
      output_bin_data=$DATA_DIR/data-bin/bt/${lang}.${mono_size}
      pair=$src-$tgt
      combined_data=$DATA_DIR/data-bin/seed+bt/$pair.${size}+${mono_size}
      bitext=$DATA_DIR/data-bin/seed/$pair.$size

      mkdir -p $output_extracted_data
      mkdir -p $output_bin_data/$mono_size
      mkdir -p $combined_data

      echo
      echo Processing $mono_size.$lang backtranslated with $size seed model
      echo

      echo Extracting to $output_extracted_data/$size
      python $EXTRACT --minlen 1 --maxlen 250 --ratio 1.5 --srclang eng --tgtlang $lang \
        --output $output_extracted_data/$size $input_data/backtranslated.$size.out

      echo Binarising to $output_bin_data/$size
      python $PREPROCESS --source-lang=$src --target-lang=$tgt --joined-dictionary \
        --srcdict $dict --trainpref $output_extracted_data/$size \
        --workers 20 --destdir $output_bin_data/$size

      echo Combining in $combined_data
      for bitext_lang in eng $lang; do
        ln -f -s $dict $combined_data/dict.$bitext_lang.txt
        for ext in bin idx; do
          ln -f -s $bitext/train.$pair.$bitext_lang.$ext $combined_data/train.$pair.$bitext_lang.$ext
          ln -f -s $output_bin_data/$size/train.$pair.$bitext_lang.$ext $combined_data/train1.$pair.$bitext_lang.$ext
          ln -f -s $bitext/valid.$pair.$bitext_lang.$ext $combined_data/valid.$pair.$bitext_lang.$ext
          ln -f -s $bitext/test.$pair.$bitext_lang.$ext $combined_data/test.$pair.$bitext_lang.$ext
        done
      done
    done
  done
done
