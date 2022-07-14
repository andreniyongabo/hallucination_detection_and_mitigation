#!/bin/bash


EXPERIMENT_NAME="2021-04-29-14-31-en-fr-tr-clean1"
EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER


DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER


PLAIN_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/plain"
TRAIN_DIR="$DATA_FOLDER/train"
VALID_DIR="$DATA_FOLDER/valid"
mkdir -p $TRAIN_DIR
mkdir -p $VALID_DIR


VALID_SIZE=100000


ALL_LANGS=(en fr tr)

for LANG in ${ALL_LANGS[@]}
do
    PLAIN_FILE="$PLAIN_DIR/$LANG.mono.txt"
    NUMBER_OF_LINES=$(cat $PLAIN_FILE | wc -l)
    echo "$LANG => $NUMBER_OF_LINES"

    TRAIN_NB_LINES=$(expr $NUMBER_OF_LINES - $VALID_SIZE)

    echo "TRAIN_NB_LINES = $TRAIN_NB_LINES"

    TRAIN_FILE="$TRAIN_DIR/$LANG.mono.txt"
    VALID_FILE="$VALID_DIR/$LANG.mono.txt"

    head -n $TRAIN_NB_LINES $PLAIN_FILE > $TRAIN_FILE
    tail -n $VALID_SIZE $PLAIN_FILE > $VALID_FILE
done

cat $TRAIN_DIR/*.mono.txt | shuf > "$TRAIN_DIR/all.txt"
cat $VALID_DIR/*.mono.txt | shuf > "$VALID_DIR/all.txt"

rm $TRAIN_DIR/*.mono.txt
rm $VALID_DIR/*.mono.txt


FASTTEXT_BIN="/private/home/celebio/nlp/nllb_lid/fastText/fasttext"


echo "Training fastText:"
$FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model \
     -minn 2 -maxn 5 -minCount 10 -epoch 10 -dim 16 -loss softmax -bucket 20000000

RESULT_GATHER="$FASTTEXT_BIN test-label $RESULT_FOLDER/model.bin $VALID_DIR/all.txt"
RESULT_TXT="$RESULT_FOLDER/result.test-label.txt"
$RESULT_GATHER > $RESULT_TXT

echo "$RESULT_GATHER"
cat $RESULT_TXT









