#!/bin/bash


EXPERIMENT_NAME="2021-10-14-16-21-quantize-ft"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

RESULT_FOLDER="$EXPERIMENT_FOLDER/result"
mkdir -p $RESULT_FOLDER

FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"

TRAINED_LID_PATH="/large_experiments/nllb/mmt/lidruns/2021-10-12-11-33-multifilter3/"



quantize_ft_model() {
    $FASTTEXT_BIN quantize
}


quantize_ft_model
