#!/bin/bash

# We create the monolingual datasets from CC100 data, which needs to be segmented.
# The data is split over two directories.

LANG_ISO=(vie)
LANG_CC=(vi_VN)
LANG_MOSES=(vi)

CC100=/large_exp/flores/namangoyal/cc100_xl
MOSES_SCRIPTS=~jeanm/src/mosesdecoder/scripts
NORMPUNC=$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl
REMNP=$MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl
SENTSPLIT=$MOSES_SCRIPTS/ems/support/split-sentences.perl
SPM_ENCODE=~jeanm/src/sentencepiece/build/src/spm_encode
SPM_MODEL=/large_experiments/flores/namangoyal/cc100_combined/spm_256000.model

for i in "${!LANG_ISO[@]}"; do
    cc_src=$CC100/${LANG_CC[$i]}.txt
    outdir=/checkpoint/jeanm/nllb/bt_data_req/data/monolingual
    lang=${LANG_ISO[$i]}

    echo Creating $lang monolingual corpus...

    # We get rid of invalid bytes, empty lines, lines with >6k tokens, and remove the
    # last line as it may have been truncated, leading to decoding errors or UNKs

    head -c 500M $cc_src | awk NF | iconv -f utf-8 -t utf-8 -c | \
      head -n -1 | \
      perl $NORMPUNC ${LANG_MOSES[$i]} | perl $REMNP | \
      perl $SENTSPLIT -l ${LANG_MOSES[$i]} | sort | uniq -u | shuf | \
      $SPM_ENCODE --model=$SPM_MODEL | \
      awk -F' ' 'NF<6000 && $NF!=""' > $outdir/500MB.$lang
    head -n 300000 $outdir/500MB.$lang > $outdir/300k.$lang
    head -n 150000 $outdir/500MB.$lang > $outdir/150k.$lang
done

# Next, we deal with the languages that are in the other directory.
# These are already segmented so we can skip a step.

LANG_ISO=(ita fra deu eng)
LANG_CC=(it_IT fr_XX de_DE en_XX)
LANG_MOSES=(it fr de en)

CC100=/datasets01/cc100/031720/
MOSES_SCRIPTS=~jeanm/src/mosesdecoder/scripts
NORMPUNC=$MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl
REMNP=$MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl
SENTSPLIT=$MOSES_SCRIPTS/ems/support/split-sentences.perl
SPM_ENCODE=~jeanm/src/sentencepiece/build/src/spm_encode
SPM_MODEL=/large_experiments/flores/namangoyal/cc100_combined/spm_256000.model

for i in "${!LANG_ISO[@]}"; do
    cc_src=$CC100/${LANG_CC[$i]}.txt
    outdir=/checkpoint/jeanm/nllb/bt_data_req/data/monolingual
    lang=${LANG_ISO[$i]}

    echo Creating $lang monolingual corpus...

    # We get rid of invalid bytes, empty lines, lines with >6k tokens, and remove the
    # last line as it may have been truncated, leading to decoding errors or UNKs

    head -c 500M $cc_src | awk NF | iconv -f utf-8 -t utf-8 -c | \
      head -n -1 | \
      perl $NORMPUNC ${LANG_MOSES[$i]} | perl $REMNP | \
      perl $SENTSPLIT -l ${LANG_MOSES[$i]} | sort | uniq -u | shuf | \
      $SPM_ENCODE --model=$SPM_MODEL | \
      awk -F' ' 'NF<6000 && $NF!=""' > $outdir/500MB.$lang
    head -n 300000 $outdir/500MB.$lang > $outdir/300k.$lang
    head -n 150000 $outdir/500MB.$lang > $outdir/150k.$lang
done
