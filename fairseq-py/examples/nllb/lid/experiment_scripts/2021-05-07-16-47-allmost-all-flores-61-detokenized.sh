#!/bin/bash


EXPERIMENT_NAME="2021-05-07-16-47-allmost-all-flores-61-detokenized"
EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER


tools="/private/home/schwenk/tools"
moses_tok="$tools/mosesdecoder/scripts/tokenizer/tokenizer.perl"
mose_detok="$tools/mosesdecoder/scripts/tokenizer/detokenizer.perl"


FLORES_DIR="/checkpoint/angelafan/flores_preliminary_data"
PLAIN_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/plain"
TRAIN_DIR="$DATA_FOLDER/train"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TRAIN_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR



declare -A flores_to_jw300=(["cx"]="ceb" ["ku"]="kmr" ["ms"]="zlm")




prepare_data () {
    VALID_SIZE=100000

    ALLMOST_ALL_FLORES=(af am as bg bn cs cx da de en et fa fr gu ha he hi hr hu hy id ig it jv km kn ko ku lg lo lt mg mk ml mn mr ms my ne nl om or pa si su sv sw ta te tl tn tr ur xh yo zu)

    for FLORES_LANG in ${ALLMOST_ALL_FLORES[@]}
    do
        if [ ${flores_to_jw300[$FLORES_LANG]} ]; then
            JW300_LANG=${flores_to_jw300[$FLORES_LANG]}
        else
            JW300_LANG="$FLORES_LANG"
        fi;
        # echo "JW300_LANG=$JW300_LANG FLORES_LANG=$FLORES_LANG"

        PLAIN_FILE="$PLAIN_DIR/$JW300_LANG.mono.txt"
        NUMBER_OF_LINES=$(cat $PLAIN_FILE | wc -l)
        echo "$JW300_LANG => $NUMBER_OF_LINES"

        if [ "$NUMBER_OF_LINES" -lt "$VALID_SIZE" ]; then
            VALID_SIZE=$(expr $NUMBER_OF_LINES / 10)
            echo "    /!\ specific valid size for this language= $VALID_SIZE"
        fi

        SHUFFLED_FILE="$TRAIN_DIR/$JW300_LANG.shuffled.txt"

        cat $PLAIN_FILE | perl $mose_detok -q -l $JW300_LANG | awk -v lang="$JW300_LANG" '{print "__label__"lang " " $0}' | shuf > $SHUFFLED_FILE



        TRAIN_NB_LINES=$(expr $NUMBER_OF_LINES - $VALID_SIZE)
        echo "    TRAIN_NB_LINES = $TRAIN_NB_LINES"

        TRAIN_FILE="$TRAIN_DIR/$JW300_LANG.mono.txt"
        VALID_FILE="$VALID_DIR/$JW300_LANG.mono.txt"

        head -n $TRAIN_NB_LINES $SHUFFLED_FILE > $TRAIN_FILE
        tail -n $VALID_SIZE $SHUFFLED_FILE > $VALID_FILE
    done

    cat $TRAIN_DIR/*.mono.txt | shuf > "$TRAIN_DIR/all.txt"
    cat $VALID_DIR/*.mono.txt | shuf > "$VALID_DIR/all.txt"

    rm $TRAIN_DIR/*.mono.txt
    rm $VALID_DIR/*.mono.txt
}


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

train () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1 -epoch 1 -dim 16 -loss softmax -bucket 5000000 -thread 10

}

eval_valid () {
    RESULT_GATHER="$FASTTEXT_BIN test-label $RESULT_FOLDER/model.bin $VALID_DIR/all.txt"
    RESULT_TXT="$RESULT_FOLDER/result.test-label.txt"
    $RESULT_GATHER > $RESULT_TXT

    echo "$RESULT_GATHER"
    cat $RESULT_TXT
}

eval_flores_dev () {
    RESULT_GATHER="$FASTTEXT_BIN test-label $RESULT_FOLDER/model.bin $TEST_DIR/flores-dev.txt"
    RESULT_TXT="$RESULT_FOLDER/result.test-label-flores-dev.txt"
    $RESULT_GATHER > $RESULT_TXT

    echo "$RESULT_GATHER"
    cat $RESULT_TXT
}
eval_flores_devtest () {
    RESULT_GATHER="$FASTTEXT_BIN test-label $RESULT_FOLDER/model.bin $TEST_DIR/flores-devtest.txt"
    RESULT_TXT="$RESULT_FOLDER/result.test-label-flores-devtest.txt"
    $RESULT_GATHER > $RESULT_TXT

    echo "$RESULT_GATHER"
    cat $RESULT_TXT
}



prepare_data
prepare_flores_dev_data
prepare_flores_devtest_data
train

eval_valid
eval_flores_dev
eval_flores_devtest
