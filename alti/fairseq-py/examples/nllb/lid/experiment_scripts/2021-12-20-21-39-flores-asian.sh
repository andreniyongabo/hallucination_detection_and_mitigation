#!/bin/bash


EXPERIMENT_NAME="2021-12-20-21-39-flores-asian"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER

FLORES_DEV_DIR="/large_experiments/nllb/mmt/flores101/dev"
FLORES_DEVTEST_DIR="/large_experiments/nllb/mmt/flores101/devtest"
JW300_DETOK_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/detok"
LID187_DATA_DIR="/private/home/celebio/lid187_data_2"
FBSEED_DATA_DIR="/private/home/celebio/fbseed20211130_data"

TRAIN_DIR="$DATA_FOLDER/train"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TRAIN_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR


FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"


NLLB_DEC__NLLB_LANGS=("jpn" "yue" "zho_Hans" "zho_Hant")
NLLB_DEC__FLORES_LANGS=("jpn" "yue" "zho_simpl" "zho_trad")
NLLB_DEC__FBSEED_LANGS=("" "" "" "")
NLLB_DEC__JW300_LANGS=("" "" "" "")
NLLB_DEC__LID_187_LANGS=("ja" "yue" "" "")
NLLB_DEC__NLLB_LANG_SCRIPTS=("" "" "" "")

prepare_data_1 () {
    for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
        NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}
        FLORES_LANG=${NLLB_DEC__FLORES_LANGS[i]}
        FBSEED_LANG=${NLLB_DEC__FBSEED_LANGS[i]}
        JW300_LANG=${NLLB_DEC__JW300_LANGS[i]}
        LID_187_LANG=${NLLB_DEC__LID_187_LANGS[i]}

        if [ -z "$NLLB_LANG" ]
        then
            echo "Empty lang code"
            exit 1
        fi

        MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"
        > $MONO_CAT_FILE

        chosen_source=""

        if [ -z "$chosen_source" ]
        then
            FLORES_DEV_FILE="$FLORES_DEV_DIR/$FLORES_LANG.dev"
            if [ -f "$FLORES_DEV_FILE" ]; then
                chosen_source="${chosen_source}-floresdev"

                cat $FLORES_DEV_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
            fi
        fi
        printf "%-20s %s \n" $NLLB_LANG $chosen_source

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

    for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
        NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}

        VALID_SIZE=100000
        VALID_COMPARE_SIZE=$(expr $VALID_SIZE \* 4)


        MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"
        # MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"
        MONO_SHUF_FILE="$TRAIN_DIR/$NLLB_LANG.shuf.txt"

        shuf $MONO_CAT_FILE > $MONO_SHUF_FILE

        NUMBER_OF_LINES=$(cat $MONO_CAT_FILE | wc -l)


        if [ "$NUMBER_OF_LINES" -lt "$VALID_COMPARE_SIZE" ]; then
            VALID_SIZE=$(expr $NUMBER_OF_LINES / 10)
        fi
        TRAIN_NB_LINES=$(expr $NUMBER_OF_LINES - $VALID_SIZE)

        # if [ "$TRAIN_NB_LINES" -gt "400000" ]; then
        #     TRAIN_NB_LINES="400000"
        # fi

        TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.txt"
        VALID_FILE="$VALID_DIR/$NLLB_LANG.valid.mono.txt"

        # echo "head -n $TRAIN_NB_LINES $MONO_CAT_FILE > $TRAIN_FILE"
        # shuf $MONO_CAT_FILE | head -n $TRAIN_NB_LINES > $TRAIN_FILE

        cat $MONO_SHUF_FILE > $TRAIN_FILE
        echo " " > $VALID_FILE
        # head -n $TRAIN_NB_LINES $MONO_SHUF_FILE > $TRAIN_FILE
        # tail -n $VALID_SIZE $MONO_SHUF_FILE > $VALID_FILE

        rm $MONO_SHUF_FILE

        printf "%-20s train size: %-10s    valid size: %-10s \n" $NLLB_LANG $TRAIN_NB_LINES $VALID_SIZE

        cat $TRAIN_FILE >> $UNSHUFFLED_TRAIN_FILE
        cat $VALID_FILE >> $UNSHUFFLED_VALID_FILE
    done
    shuf $UNSHUFFLED_TRAIN_FILE > $ALL_TRAIN_FILE
    shuf $UNSHUFFLED_VALID_FILE > $ALL_VALID_FILE
}

prepare_flores_filled_data () {
    CONCAT_FILE="$TEST_DIR/concat.flores-filled.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-filled.txt"
    > $CONCAT_FILE


    for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
        NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}
        FLORES_LANG=${NLLB_DEC__FLORES_LANGS[i]}
        FBSEED_LANG=${NLLB_DEC__FBSEED_LANGS[i]}
        JW300_LANG=${NLLB_DEC__JW300_LANGS[i]}
        LID_187_LANG=${NLLB_DEC__LID_187_LANGS[i]}

        chosen_source=""
        if [ ! -z "$FLORES_LANG" ]
        then
            FILES_FOUND=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | head -n 1)
            FILES_FOUND_NUM=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | wc -l)

            # echo "LANG=$NLLB_LANG" "$FLORES_LANG" "$FILES_FOUND_NUM"
            # echo "$FILES_FOUND"
            chosen_source="${chosen_source}-flores"

            cat $FILES_FOUND | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
        else


            # echo "Flores not available for $NLLB_LANG"
            VALID_FILE="$VALID_DIR/$NLLB_LANG.valid.mono.txt"
            if [ -f "$VALID_FILE" ]; then
                chosen_source="${chosen_source}-fromvalid"
                # echo "will use $VALID_FILE"
                shuf $VALID_FILE | head -n 1000 >> $CONCAT_FILE
            else
                echo "ERROR: File not found $VALID_FILE"
                return 1
            fi
        fi

        printf "%-20s %s \n" $NLLB_LANG $chosen_source
    done

    shuf $CONCAT_FILE > $TEST_FINAL_FILE
    rm $CONCAT_FILE
}

train_fasttext_8_1 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.8.1 \
        -lr 0.5 -minn 1 -maxn 5 -minCount 1 -epoch 500 -dim 256 -loss softmax -bucket 2000000 -thread 40
}

eval_flores_filled_fasttext_variants_8 () {
    GOLD="$RESULT_FOLDER/flores-filled.gold"
    TEST_FILE="$TEST_DIR/flores-filled.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 9`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.8.$i.txt"

        LID_MODEL="$RESULT_FOLDER/model.8.$i.bin"

        if [ -s $LID_MODEL ]
        then
            echo "LID_MODEL $LID_MODEL"
            PREDICTIONS="$RESULT_FOLDER/flores-filled.fasttext.predictions.8.$i"

            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
            $RESULT_GATHER > $PREDICTIONS

            CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
            $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
        fi
    done
}


# prepare_data_1
# prepare_data_combine

# prepare_flores_filled_data

train_fasttext_8_1
eval_flores_filled_fasttext_variants_8




