#!/bin/bash

set -euxo pipefail

DOCS=$1
SENT=$2
LANGUAGE_CODE=$3

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate vecalign
export CFLAGS=-I$HOME/.conda/envs/vecalign/lib/python3.7/site-packages/numpy/core/include

# LASER
export LASER=/private/home/pkoehn/project/laser
model_dir="${LASER}/models"
encoder="${model_dir}/bilstm.93langs.2018-12-26.pt"
bpe_codes="${model_dir}/93langs.fcodes"

#export SENTENCE_ALIGN=/private/home/pkoehn/project/paracrawl/sentence-alignment
SCRIPT_NAME=$(readlink -f "$0")
SENTENCE_ALIGN=$(dirname "$SCRIPT_NAME")
export VECALIGN=$SENTENCE_ALIGN/vecalign

DOCS=`echo $DOCS | sed 's/\.xz$//'` # just in case
SENT=`echo $SENT | sed 's/\.xz$//'` # just in case

touch $SENT.processing
test ! -e $SENT.scheduled || rm $SENT.scheduled

$SENTENCE_ALIGN/util/prepare-data-for-vecalign.perl $DOCS $SENT $LANGUAGE_CODE

# create joint sentences to be embedded
export TMPDIR=$SENT.tmpdir
mkdir $TMPDIR

$SENTENCE_ALIGN/util/make_overlap_paracrawl.py $SENT.tmp.sent.txt 3 $SENT.tmp.txt

# embed with LASER
cat $SENT.tmp.sent.txt.en \
  | python ${LASER}/source/embed.py \
    --encoder ${encoder} \
    --token-lang en \
    --bpe-codes ${bpe_codes} \
    --output $SENT.tmp.sent.txt.en.emb \
    --verbose

cat $SENT.tmp.sent.txt.$LANGUAGE_CODE \
  | python ${LASER}/source/embed.py \
    --encoder ${encoder} \
    --token-lang $LANGUAGE_CODE \
    --bpe-codes ${bpe_codes} \
    --output $SENT.tmp.sent.txt.$LANGUAGE_CODE.emb \
    --verbose

# generate alignment
$SENTENCE_ALIGN/util/vecalign_multi.py $SENT.tmp.txt $SENT.tmp.sent.txt -a 3 > $SENT.tmp.aligned

rmdir $TMPDIR

$SENTENCE_ALIGN/util/postprocess-vecalign.perl $SENT $LANGUAGE_CODE

rm $SENT.tmp.sent.txt.en
rm $SENT.tmp.sent.txt.en.emb
rm $SENT.tmp.sent.txt.$LANGUAGE_CODE
rm $SENT.tmp.sent.txt.$LANGUAGE_CODE.emb
rm $SENT.tmp.txt
rm $SENT.tmp.aligned
rm $SENT.processing

