#!/bin/bash


EXPERIMENT_NAME="2021-07-08-16-51-minimining0"
EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

RESULT_FOLDER="$EXPERIMENT_FOLDER/result"
RESULT_FOLDER_MERGED="$EXPERIMENT_FOLDER/result-merged"

mkdir -p $RESULT_FOLDER
mkdir -p $RESULT_FOLDER_MERGED
echo "RESULT_FOLDER is $RESULT_FOLDER"


MINIMINING_CC_DIR="/checkpoint/vishrav/minimining_v0"

LANGS=("amh" "asm" "ben" "cym" "ful" "gle" "guj" "hau" "hin" "ibo" "isl" "kam" "kan" "lin" "lug" "luo" "mal" "mar" "npi" "nso" "nya" "orm" "ory" "pan" "sna" "snd" "som" "swh" "tam" "tel" "umb" "urd" "wol" "xho" "yor" "zul")


FASTTEXT_BIN="/private/home/celebio/nlp/nllb_lid/fastText/fasttext"
LID_MODEL="/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/model.7_with_fula.bin"
FASTTEXT_PREDICT_PY="/private/home/celebio/nlp/nllb_lid/misc/fasttext_predict.py"

predict_lang() {
    LANG=$1
    echo "Computing for $LANG"
    LANG_FILE="$MINIMINING_CC_DIR/$LANG.txt.xz"
    RESULT_FILE="$RESULT_FOLDER_MERGED/$LANG.predict-prob.txt"

    FILE2_TAB="$RESULT_FOLDER/$LANG.txt.tab"
    xzcat $LANG_FILE > $FILE2_TAB

    cat $FILE2_TAB | $FASTTEXT_PREDICT_PY --model $LID_MODEL > $RESULT_FILE
}

predict_langs() {
    for LANG in "${LANGS[@]}"; do
        predict_lang $LANG &
    done

    wait
}


predict_langs

