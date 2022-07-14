#!/bin/bash

LANG_ISO=(asm zul amh tel urd)
LANG_MOSES=(as zu am te ur)

INPUT_DIR=/large_experiments/mmt/lidruns/2021-07-08-16-51-minimining0/result-merged
RAW_OUTPUT_DIR=/checkpoint/jeanm/nllb/2021-07-08-16-51-minimining0-filtered
OUTPUT_DIR=/checkpoint/jeanm/nllb/bt_data_req/data/monolingual
MOSES_SCRIPTS=~jeanm/src/mosesdecoder/scripts
NORMPUNC=$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl
REMNP=$MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl
SENTSPLIT=$MOSES_SCRIPTS/ems/support/split-sentences.perl
SPM_ENCODE=~jeanm/src/sentencepiece/build/src/spm_encode
SPM_MODEL=/large_experiments/flores/namangoyal/cc100_combined/spm_256000.model

for i in "${!LANG_ISO[@]}"; do
  input_data=$INPUT_DIR/${LANG_ISO[$i]}.predict-prob.txt
  lang=${LANG_ISO[$i]}

  echo Filtering $lang...

  condition="(\$1==\"__label__$lang\"&&\$2>0.5){print\$(NF)}"
  awk -F '\t' $condition $input_data | \
    iconv -f utf-8 -t utf-8 -c | \
    perl $NORMPUNC ${LANG_MOSES[$i]} | perl $REMNP | \
    sort | uniq -u | shuf | \
    $SPM_ENCODE --model=$SPM_MODEL > $RAW_OUTPUT_DIR/$lang.txt

  head -n 300000 $RAW_OUTPUT_DIR/$lang.txt > $OUTPUT_DIR/300k.$lang
  head -n 150000 $RAW_OUTPUT_DIR/$lang.txt > $OUTPUT_DIR/150k.$lang
done
