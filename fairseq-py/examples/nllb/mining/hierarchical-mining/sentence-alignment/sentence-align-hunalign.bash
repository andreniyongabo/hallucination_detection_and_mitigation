#!/bin/bash

MOSES=/private/home/pkoehn/mosesdecoder
SCRIPT_NAME=$(readlink -f "$0")
SENT_ALIGN=$(dirname "$SCRIPT_NAME")

DOCS=$1
SENT=$2
LANGUAGE_CODE=$3
SENT=`echo $SENT | sed 's/\.xz$//'`

TOKENIZER_SRC="$MOSES/scripts/tokenizer/tokenizer.perl -b -l en"
TOKENIZER_TRG="$MOSES/scripts/tokenizer/tokenizer.perl -b -l $LANGUAGE_CODE"
SPLITTER_SRC="$MOSES/scripts/ems/support/split-sentences.perl -b -l en"
SPLITTER_TRG="$MOSES/scripts/ems/support/split-sentences.perl -b -l $LANGUAGE_CODE"
DICT=$SENT_ALIGN/hunalign-dict/en-$LANGUAGE_CODE.dic
TMP_DIR=$SENT.tmp
mkdir -p $TMP_DIR

# xzcat $original_docs.xz | $LIB/process-sentence-piece.perl -language $language -base64 | xz - > $docs.xz

touch $SENT.processing
test ! -e $SENT.scheduled || rm $SENT.scheduled

paste <(xzcat $DOCS | cut -f 1-2 ) \
      <(xzcat $DOCS | cut -f 3 \
          | $SENT_ALIGN/util/base64-sentence-split-wrapper.perl "$SPLITTER_SRC") \
      <(xzcat $DOCS | cut -f 4 \
	  | $SENT_ALIGN/util/base64-sentence-split-wrapper.perl "$SPLITTER_TRG") \
      <(xzcat $DOCS | cut -f 3 \
          | $SENT_ALIGN/util/base64-sentence-split-wrapper.perl "$SPLITTER_SRC" \
          | $SENT_ALIGN/util/base64-tokenizer-wrapper.perl "$TOKENIZER_SRC") \
      <(xzcat $DOCS | cut -f 4 \
          | $SENT_ALIGN/util/base64-sentence-split-wrapper.perl "$SPLITTER_TRG" \
          | $SENT_ALIGN/util/base64-tokenizer-wrapper.perl "$TOKENIZER_TRG") \
  | python3 $SENT_ALIGN/util/bitextor_align_segments.py \
       --hunalign $SENT_ALIGN/hunalign/src/hunalign/hunalign \
       -d $DICT \
       -t $TMP_DIR \
  | xz - > $SENT.xz

rmdir $TMP_DIR
rm $SENT.processing

