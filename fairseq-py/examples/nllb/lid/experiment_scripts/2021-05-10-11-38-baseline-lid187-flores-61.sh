#!/bin/bash


EXPERIMENT_NAME="2021-05-10-11-38-baseline-lid187-flores-61"
EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER


FLORES_DIR="/checkpoint/angelafan/flores_preliminary_data"
PLAIN_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/plain"
TRAIN_DIR="$DATA_FOLDER/train"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TRAIN_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR





BASELINE_LID_187="/private/home/egrave/lid/lid.187.bin"



prepare_flores_dev_data () {
    ALLMOST_ALL_FLORES=(af am as bg bn cs cx da de en et fa fr gu ha he hi hr hu hy id ig it jv km kn ko ku lg lo lt mg mk ml mn mr ms my ne nl om or pa si su sv sw ta te tl tn tr ur xh yo zu)

    CONCAT_FILE="$TEST_DIR/concat.all.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-dev.txt"
    rm $CONCAT_FILE

    for FLORES_LANG in ${ALLMOST_ALL_FLORES[@]}
    do
        if [ ${flores_to_jw300[$FLORES_LANG]} ]; then
            JW300_LANG=${flores_to_jw300[$FLORES_LANG]}
        else
            JW300_LANG="$FLORES_LANG"
        fi;

        cat $FLORES_DIR/${FLORES_LANG}_*.dev | awk -v lang="$JW300_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
    done

    shuf $CONCAT_FILE > $TEST_FINAL_FILE
    rm $CONCAT_FILE
}

prepare_flores_devtest_data () {
    ALLMOST_ALL_FLORES=(af am as bg bn cs cx da de en et fa fr gu ha he hi hr hu hy id ig it jv km kn ko ku lg lo lt mg mk ml mn mr ms my ne nl om or pa si su sv sw ta te tl tn tr ur xh yo zu)

    CONCAT_FILE="$TEST_DIR/concat.all.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-devtest.txt"
    rm $CONCAT_FILE

    for FLORES_LANG in ${ALLMOST_ALL_FLORES[@]}
    do
        if [ ${flores_to_jw300[$FLORES_LANG]} ]; then
            JW300_LANG=${flores_to_jw300[$FLORES_LANG]}
        else
            JW300_LANG="$FLORES_LANG"
        fi;

        cat $FLORES_DIR/${FLORES_LANG}_*.devtest | awk -v lang="$JW300_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
    done

    shuf $CONCAT_FILE > $TEST_FINAL_FILE
    rm $CONCAT_FILE
}


FASTTEXT_BIN="/private/home/celebio/nlp/nllb_lid/fastText/fasttext"


eval_valid () {
    RESULT_GATHER="$FASTTEXT_BIN test-label $BASELINE_LID_187 $VALID_DIR/all.txt"
    RESULT_TXT="$RESULT_FOLDER/result.test-label.txt"
    $RESULT_GATHER > $RESULT_TXT

    echo "$RESULT_GATHER"
    cat $RESULT_TXT
}

eval_flores_dev () {
    RESULT_GATHER="$FASTTEXT_BIN test-label $BASELINE_LID_187 $TEST_DIR/flores-dev.txt"
    RESULT_TXT="$RESULT_FOLDER/result.test-label-flores-dev.txt"
    $RESULT_GATHER > $RESULT_TXT

    echo "$RESULT_GATHER"
    cat $RESULT_TXT
}
eval_flores_devtest () {
    RESULT_GATHER="$FASTTEXT_BIN test-label $BASELINE_LID_187 $TEST_DIR/flores-devtest.txt"
    RESULT_TXT="$RESULT_FOLDER/result.test-label-flores-devtest.txt"
    $RESULT_GATHER > $RESULT_TXT

    echo "$RESULT_GATHER"
    cat $RESULT_TXT
}


prepare_flores_dev_data
prepare_flores_devtest_data

eval_valid
eval_flores_dev
eval_flores_devtest