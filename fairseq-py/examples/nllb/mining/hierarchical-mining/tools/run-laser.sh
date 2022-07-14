#!/bin/bash

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh

conda activate laser

FILE_E=$1
FILE_F=$2
L_E=$3
L_F=$4
OUT=$5
MODEL=$6

export LASER="${HOME}/project/laser"
date > $OUT.log
cat $FILE_E | python $LASER/source/embed.py --encoder $MODEL --bpe-codes $LASER/models/93langs.fcodes --output $FILE_E.emb --token-lang $L_E --verbose > $FILE_E.emb.log
cat $FILE_F | python $LASER/source/embed.py --encoder $MODEL --bpe-codes $LASER/models/93langs.fcodes --output $FILE_F.emb --token-lang $L_F --verbose > $FILE_F.emb.log
python $LASER/source/mine_bitexts.py --src-lang $L_E --trg-lang $L_F --output $OUT --mode score --retrieval max --margin ratio -k 4 --verbose --gpu --unify --src-embeddings $FILE_E.emb --trg-embeddings $FILE_F.emb $FILE_E $FILE_F >> $OUT.log
date >> $OUT.log
