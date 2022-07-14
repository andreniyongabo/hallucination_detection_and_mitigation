#!/bin/bash


EXPERIMENT_NAME="2022-02-02-17-45-labelupdate"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER


echo "RESULT_FOLDER is $RESULT_FOLDER"

FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"


replace_gold_language_codes () {
    OLD_RESULT_FOLDER="/large_experiments/nllb/mmt/lidruns/2021-12-27-14-48-multifilter6-ndup/result"

    AUX_FILE="$RESULT_FOLDER/flores-filled.gold.aux"
    GOLD="$RESULT_FOLDER/flores-filled.gold"

    cat $OLD_RESULT_FOLDER/flores-filled.gold > "$AUX_FILE"
    cat $AUX_FILE > "$GOLD"

    declare -A MYMAP
    MYMAP["ayr"]="aym"
    MYMAP["fuv"]="ful"
    MYMAP["min_Latn"]="min"
    MYMAP["nob"]="nor"
    MYMAP["srp_Cyrl"]="srp"
    MYMAP["tat_Cyrl"]="tat"
    MYMAP["ton"]="tog"
    MYMAP["zho_Hans"]="zho"


    for K in "${!MYMAP[@]}"; do
        echo $K --- ${MYMAP[$K]};
        cat "$GOLD" | sed s/__label__${MYMAP[$K]}/__label__${K}/g > "$AUX_FILE"
        mv "$AUX_FILE" "$GOLD"
    done

    # cat "$GOLD" | sed s/__label__min/__label__min_Latn/g > "$AUX_FILE"
    # mv "$AUX_FILE" "$GOLD"
    # cat "$GOLD" | sed s/__label__aym/__label__ayr/g > "$AUX_FILE"
    # mv "$AUX_FILE" "$GOLD"
    # cat "$GOLD" | sed s/__label__ayr/__label__aym/g > "$AUX_FILE"
    # mv "$AUX_FILE" "$GOLD"



}

eval_flores_filled_fasttext_variants_8 () {
    OLD_TEST_DIR="/large_experiments/nllb/mmt/lidruns/2021-12-27-14-48-multifilter6-ndup/data/test"

    GOLD="$RESULT_FOLDER/flores-filled.gold"
    TEST_FILE="$OLD_TEST_DIR/flores-filled.txt"
    # cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 20`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.8.$i.txt"

        LID_MODEL="$RESULT_FOLDER/model.8.$i.labelupdate.bin"

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

# replace_gold_language_codes

eval_flores_filled_fasttext_variants_8
