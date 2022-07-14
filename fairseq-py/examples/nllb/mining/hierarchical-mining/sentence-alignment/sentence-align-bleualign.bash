#!/bin/bash

set -euxo pipefail

SCRIPT_NAME=$(readlink -f "$0")
SENT_ALIGN=$(dirname "$SCRIPT_NAME")
MALIGN_DOCALIGN=$SENT_ALIGN/../document-alignment
LIB=$MALIGN_DOCALIGN/perl

docs=$1
sent=$2
language=$3

docs=`echo $docs | sed 's/\.xz$//'` # just in case
sent=`echo $sent | sed 's/\.xz$//'` # just in case

# permanent files created
txt=$sent.$language.txt
english=$sent.en.txt
translated=$sent.$language.translated
matches=$sent.matches

touch $sent.processing
test ! -e $sent.scheduled || rm $sent.scheduled

# extract foreign and english text
xzcat $docs.xz | cut -f 1-2 | sed 's/^/0\t/' | xz - > $matches.xz
xzcat $docs.xz | cut -f 1,3 | $SENT_ALIGN/util/get-sentences-from-docs.perl | xz - > $english.xz
xzcat $docs.xz | cut -f 2,4 | $SENT_ALIGN/util/get-sentences-from-docs.perl | xz - > $txt.xz

# translate
$LIB/translate-foreign.perl $txt.xz $language | xz - > $translated.xz
rm $txt.xz.dedup $txt.xz.dedup.moses.log $txt.xz.dedup.translated

# reformat translations info bleualign format and run bleualign

$SENT_ALIGN/util/format-translations-for-bleualign.perl $sent $language \
  | ./bleualign-cpp/build/bleualign_cpp \
  | xz - > $sent.xz

rm $txt.xz $english.xz $translated.xz $matches.xz
rm $sent.processing
