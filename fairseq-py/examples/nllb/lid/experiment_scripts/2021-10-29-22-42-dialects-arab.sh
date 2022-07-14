#!/bin/bash


EXPERIMENT_NAME="2021-10-29-22-42-dialects-arab"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER

FLORES_DEV_DIR="/large_experiments/mmt/flores101/dev"
FLORES_DEVTEST_DIR="/large_experiments/mmt/flores101/devtest"
JW300_DETOK_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/detok"
LID187_DATA_DIR="/private/home/celebio/lid187_data_2"

TRAIN_DIR="$DATA_FOLDER/train"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TRAIN_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR

echo "DATA_FOLDER is $DATA_FOLDER"


ARAB_DIALECT__LANGS=("ara" "ara-IQ" "ara-Latn" "ara-LB" "ara-MA" "ara-SA" "ara-TN" "ara-YE")


FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"


prepare_data_1 () {

    for i in "${!ARAB_DIALECT__LANGS[@]}"; do
        CUR_LANG=${ARAB_DIALECT__LANGS[i]}


        FLORES_DEV_FILE="$FLORES_DEV_DIR/$CUR_LANG.dev"

        MONO_CAT_FILE="$TRAIN_DIR/$CUR_LANG.cat.txt"
        > $MONO_CAT_FILE

        cat $FLORES_DEV_FILE | awk -v lang="$CUR_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
    done
}

prepare_data_combine() {
    UNSHUFFLED_TRAIN_FILE="$TRAIN_DIR/all.unshuf.txt"
    > $UNSHUFFLED_TRAIN_FILE
    UNSHUFFLED_VALID_FILE="$VALID_DIR/all.unshuf.txt"
    > $UNSHUFFLED_VALID_FILE
    ALL_TRAIN_FILE="$TRAIN_DIR/all.txt"
    > $ALL_TRAIN_FILE
    ALL_VALID_FILE="$VALID_DIR/all.txt"
    > $ALL_VALID_FILE


    for i in "${!ARAB_DIALECT__LANGS[@]}"; do
        CUR_LANG=${ARAB_DIALECT__LANGS[i]}

        VALID_SIZE=100000
        VALID_COMPARE_SIZE=$(expr $VALID_SIZE \* 4)

        MONO_CAT_FILE="$TRAIN_DIR/$CUR_LANG.cat.txt"

        echo "MONO_CAT_FILE=$MONO_CAT_FILE"
        NUMBER_OF_LINES=$(cat $MONO_CAT_FILE | wc -l)

        echo "    $CUR_LANG NUMBER_OF_LINES = $NUMBER_OF_LINES"

        if [ "$NUMBER_OF_LINES" -lt "$VALID_COMPARE_SIZE" ]; then
            VALID_SIZE=$(expr $NUMBER_OF_LINES / 10)
            echo "        /!\ specific valid size for this language ($CUR_LANG) = $VALID_SIZE"
        fi
        TRAIN_NB_LINES=$(expr $NUMBER_OF_LINES - $VALID_SIZE)
        echo "VALID_SIZE = $VALID_SIZE"
        echo "    $CUR_LANG TRAIN_NB_LINES = $TRAIN_NB_LINES"


        TRAIN_FILE="$TRAIN_DIR/$CUR_LANG.train.mono.txt"
        VALID_FILE="$VALID_DIR/$CUR_LANG.valid.mono.txt"

        echo "head -n $TRAIN_NB_LINES $MONO_CAT_FILE > $TRAIN_FILE"
        head -n $TRAIN_NB_LINES $MONO_CAT_FILE > $TRAIN_FILE
        tail -n $VALID_SIZE $MONO_CAT_FILE > $VALID_FILE

        cat $TRAIN_FILE >> $UNSHUFFLED_TRAIN_FILE
        cat $VALID_FILE >> $UNSHUFFLED_VALID_FILE
    done
    shuf $UNSHUFFLED_TRAIN_FILE > $ALL_TRAIN_FILE
    shuf $UNSHUFFLED_VALID_FILE > $ALL_VALID_FILE
}

prepare_flores_devtest_data () {
    CONCAT_FILE="$TEST_DIR/concat.all.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-devtest.txt"
    > $CONCAT_FILE

    for i in "${!ARAB_DIALECT__LANGS[@]}"; do
        CUR_LANG=${ARAB_DIALECT__LANGS[i]}

        if [ ! -z "$CUR_LANG" ]
        then
            FILES_FOUND=$(ls -1 $FLORES_DEVTEST_DIR/${CUR_LANG}.devtest | head -n 1)
            FILES_FOUND_NUM=$(ls -1 $FLORES_DEVTEST_DIR/${CUR_LANG}.devtest | wc -l)

            # echo "LANG=$CUR_LANG" "$CUR_LANG" "$FILES_FOUND_NUM"
            # echo "$FILES_FOUND"
            cat $FILES_FOUND | awk -v lang="$CUR_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
        fi


    done

    shuf $CONCAT_FILE > $TEST_FINAL_FILE
    rm $CONCAT_FILE
}

train_fasttext_8_1 () {
    echo "Training fastText:"
        # -wordNgrams 2 -lr 0.6 -minn 1 -maxn 8 -minCount 1 -epoch 400 -dim 16 -loss softmax -bucket 1000000
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.8.1 \
        -thread 5 \
        -autotune-validation $VALID_DIR/all.txt -autotune-duration 600

        # -autotune-metric precisionAtRecall:80

}

eval_valid_fasttext_variants_8 () {
    GOLD="$RESULT_FOLDER/valid.gold"
    TEST_FILE="$VALID_DIR/all.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 9`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-valid.fasttext.8.$i.txt"

        LID_MODEL="$RESULT_FOLDER/model.8.$i.bin"

        if [ -s $LID_MODEL ]
        then
            echo "LID_MODEL $LID_MODEL"
            PREDICTIONS="$RESULT_FOLDER/valid.fasttext.predictions.8.$i"

            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
            $RESULT_GATHER > $PREDICTIONS

            CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
            $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
        fi
    done
}

eval_flores_devtest_fasttext_variants_8 () {
    GOLD="$RESULT_FOLDER/flores-devtest.gold"
    TEST_FILE="$TEST_DIR/flores-devtest.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 9`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-devtest.fasttext.8.$i.txt"

        LID_MODEL="$RESULT_FOLDER/model.8.$i.bin"

        if [ -s $LID_MODEL ]
        then
            echo "LID_MODEL $LID_MODEL"
            PREDICTIONS="$RESULT_FOLDER/flores-devtest.fasttext.predictions.8.$i"

            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
            $RESULT_GATHER > $PREDICTIONS

            CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
            $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
        fi
    done
}

# prepare_data_1

# prepare_data_combine
# prepare_flores_devtest_data

train_fasttext_8_1
eval_valid_fasttext_variants_8
eval_flores_devtest_fasttext_variants_8



