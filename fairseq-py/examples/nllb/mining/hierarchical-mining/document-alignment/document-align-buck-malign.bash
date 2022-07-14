#!/bin/bash
set -e

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate malign_python3

SCRIPT_NAME=$(readlink -f "$0")
MALIGN_DOCALIGN=$(dirname "$SCRIPT_NAME")
LIB=$MALIGN_DOCALIGN/perl
MOSES=/private/home/pkoehn/mosesdecoder

lett=$1
docs=$2
language=$3

DOCS=`echo $lett | sed 's/\.xz$//'` # just in case
SENT=`echo $docs | sed 's/\.xz$//'` # just in case

# legacy
if [ -e $lett.gz ] && [ ! -e $lett.xz ]; then
  zcat $lett.gz | xz - > $lett.xz && rm $lett.gz
fi  

# permanent files created
txt=$docs.$language.txt
english=$docs.en.txt
translated=$docs.$language.translated
matches=$docs.matches

touch $docs.processing

# extract foreign and english text
xzcat $lett.xz | \
    python3 $MALIGN_DOCALIGN/utils/extract_lett.py \
    --langs en,$language \
    --splitter $MALIGN_DOCALIGN/utils/split-sentences2.perl \
    --prune_type "words" \
    --prune 1000 \
    --output_prefix $docs.en-$language. \
    --output_dir /

extracted_e=$docs.en-$language.en.extracted
extracted_f=$docs.en-$language.$language.extracted
extracted_translated=$docs.en-$language.$language.translated

# treanslate
zcat $extracted_f.gz | xz - > $txt.xz && rm $extracted_f.gz
zcat $extracted_e.gz | xz - > $english.xz
$LIB/translate-foreign.perl $txt.xz $language | xz - > $translated.xz
rm $txt.xz.dedup $txt.xz.dedup.moses.log $txt.xz.dedup.translated
paste <(xzcat $txt.xz | cut -f 1) <(xzcat $translated.xz) | gzip - > $extracted_translated.gz

# Compute matches
python3 $MALIGN_DOCALIGN/utils/compute_matches.py \
    --english $extracted_e.gz \
    --translated $extracted_translated.gz \
    --output_matches $matches \
    --threshold 0.0 \
    --batch_size 1000
rm $extracted_e.gz
rm $extracted_translated.gz

# Outputting the document-aligned data in Bitextor format"
xzcat $lett.xz | \
  python3 $MALIGN_DOCALIGN/utils/build_docs.py \
  --matches $matches \
  --threshold 0.0 | \
  xz - > $docs.xz

rm -f $docs.processing
xz $matches
