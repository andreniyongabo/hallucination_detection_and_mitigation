#!/bin/bash


EXPERIMENT_NAME="2022-02-08-15-26-increase-flores-eval"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER


DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER

FLORES_DEV_DIR="/large_experiments/nllb/mmt/flores101/dev"
FLORES_DEVTEST_DIR="/large_experiments/nllb/mmt/flores101/devtest"
FLORES_BETA_DEVTEST_DIR="/large_experiments/nllb/mmt/flores101_beta/devtest"
JW300_DETOK_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/detok"
LID187_DATA_DIR="/private/home/celebio/lid187_data_2"
FBSEED_DATA_DIR="/private/home/celebio/fbseed20211130_data"

TRAIN_DIR="$DATA_FOLDER/train"
TRAIN_FILTER_DIR="$DATA_FOLDER/train_filter"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TRAIN_DIR
mkdir -p $TRAIN_FILTER_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR

echo "DATA_FOLDER is $DATA_FOLDER"


FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"
ORIGINAL_RUN="/large_experiments/nllb/mmt/lidruns/2021-12-27-14-48-multifilter6-ndup/"


get_flores_filled_from_original () {
    ORIGINAL_TEST_FLORES_FILLED="$TEST_DIR/flores-filled.original.txt"
    NEW_TEST_FLORES_FILLED="$TEST_DIR/flores-filled.new.txt"
    > $NEW_TEST_FLORES_FILLED

    cat "$ORIGINAL_RUN/data/test/flores-filled.txt" | sort > $ORIGINAL_TEST_FLORES_FILLED
    cat $ORIGINAL_TEST_FLORES_FILLED | cut -f 1 -d" " | sort | uniq > "$TEST_DIR/labels.txt"

    ORIGIN_RESULT="$TEST_DIR/lang.origins.txt"
    > $ORIGIN_RESULT
    cat "$TEST_DIR/labels.txt" | while read label;
    do
        NLLB_LANG=${label:9}
        CHECK_HEAD="10"

        TMP_ORIGINAL_CONTENT="$TEST_DIR/content.original.$NLLB_LANG.txt"
        TMP_FLORES_CONTENT="$TEST_DIR/content.flores.$NLLB_LANG.txt"

        cat $ORIGINAL_TEST_FLORES_FILLED | grep "^__label__${NLLB_LANG} " | cut -f 2- -d" " | head -n $CHECK_HEAD \
            > $TMP_ORIGINAL_CONTENT


        FLORES_LANG="$NLLB_LANG"

        if [ $NLLB_LANG == "kas_Arab" ]; then
            FLORES_LANG="kas-Arab"
        fi

        FLORES_FILE="$FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest"
        FLORES_BETA_FILE="$FLORES_BETA_DEVTEST_DIR/${FLORES_LANG}.devtest"
        chosen_source="???"

        if [ -s $FLORES_FILE ]
        then
            cat $FLORES_FILE | sort | head -n $CHECK_HEAD > $TMP_FLORES_CONTENT

            DIFF=$(diff $TMP_ORIGINAL_CONTENT $TMP_FLORES_CONTENT)

            if [ -z "$DIFF" ]
            then
                chosen_source="flores"
            else
                chosen_source="can_be_flores_now"
            fi

            rm $TMP_FLORES_CONTENT

        else
            if [ -s $FLORES_BETA_FILE ]
            then
                cat $FLORES_BETA_FILE | sort | head -n $CHECK_HEAD > $TMP_FLORES_CONTENT

                DIFF=$(diff $TMP_ORIGINAL_CONTENT $TMP_FLORES_CONTENT)

                if [ -z "$DIFF" ]
                then
                    chosen_source="flores_beta"
                else
                    chosen_source="can_be_flores_beta_now"
                fi

                rm $TMP_FLORES_CONTENT
            fi
        fi

        rm $TMP_ORIGINAL_CONTENT

        printf "%-20s %s \n" $NLLB_LANG $chosen_source >> $ORIGIN_RESULT

        if [ $chosen_source == 'flores' ]; then
            cat $FLORES_FILE | sort | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $NEW_TEST_FLORES_FILLED
        elif [ $chosen_source == 'can_be_flores_now' ]; then
            cat $FLORES_FILE | sort | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $NEW_TEST_FLORES_FILLED
        elif [ $chosen_source == 'flores_beta' ]; then
            cat $FLORES_BETA_FILE | sort | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $NEW_TEST_FLORES_FILLED
        elif [ $chosen_source == 'can_be_flores_beta_now' ]; then
            cat $FLORES_BETA_FILE | sort | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $NEW_TEST_FLORES_FILLED
        else
            cat $ORIGINAL_TEST_FLORES_FILLED | grep "^__label__${NLLB_LANG} " >> $NEW_TEST_FLORES_FILLED
        fi

    done

    cat $ORIGIN_RESULT

}


eval_flores_filled_fasttext_variants_8 () {
    GOLD="$RESULT_FOLDER/flores-filled.gold"
    TEST_FILE="$TEST_DIR/flores-filled.new.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 20`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.8.$i.txt"

        LID_MODEL="$ORIGINAL_RUN/result/model.8.$i.bin"

        if [ -s $LID_MODEL ] && [ ! -s $RESULT_TXT ]
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





get_flores_filled_from_original
eval_flores_filled_fasttext_variants_8


