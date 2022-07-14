#!/bin/bash


EXPERIMENT_NAME="2021-07-01-00-17-compare-on-common4"
EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER

FLORES_DIR="/checkpoint/angelafan/flores_preliminary_data"
DETOK_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/detok"
TRAIN_DIR="$DATA_FOLDER/train"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TRAIN_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR

echo "DATA_FOLDER is $DATA_FOLDER"



# Languages that exist in flores99
ISO_639_3_LANGS_FLORES99=("nya" "afr" "amh" "ara" "hye" "asm" "ast" "aze" "ben" "bos" "bul" "mya" "cat" "ceb" "ckb" "zho" "hrv" "ces" "dan" "nld" "eng" "est" "fin" "fra" "ful" "glg" "lug" "kat" "deu" "guj" "hau" "heb" "hin" "hun" "isl" "ibo" "ind" "ita" "jpn" "jav" "kea" "kan" "kaz" "khm" "kor" "kur" "kir" "lao" "lav" "lin" "lit" "luo" "ltz" "mkd" "mlg" "msa" "mal" "mlt" "mar" "ell" "mon" "npi" "nso" "ory" "orm" "pus" "fas" "pol" "por" "pan" "ron" "rus" "srp" "sna" "snd" "sin" "slk" "slv" "som" "spa" "sun" "swh" "swe" "tgl" "tam" "tel" "tha" "tsn" "tur" "ukr" "urd" "uzb" "vie" "cym" "wol" "xho" "yor" "zul" "bel")
JW300_LANGS_FLORES99=("nya" "af" "am" "ar" "hy" "as" "" "az" "bn" "" "bg" "my" "cat" "ceb" "" "" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "kea" "kn" "kk_Cyrl" "km" "ko" "" "ky" "lo" "lv" "ln" "lt" "luo" "" "mk" "mg" "zlm" "ml" "mt" "mr" "el" "mn" "ne" "nso" "or" "om" "" "fa" "pl" "pt" "pa" "ro" "ru" "sr_Cyrl" "sn" "" "si" "sk" "sl" "" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz_Latn" "vi" "cy" "" "xh" "yo" "zu" "")
FLORES_101_LANGS_FLORES99=("ny" "af" "am" "ar" "hy" "as" "ast" "az" "bn" "bs" "bg" "my" "ca" "cx" "cb" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "ff" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "q3" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "qy" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "ns" "or" "om" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz" "vi" "cy" "wo" "xh" "yo" "zu" "be")
LID_187_LANGS_FLORES99=("" "af" "am" "ar" "hy" "as" "ast" "az" "bn" "bs" "bg" "my" "ca" "ceb" "ckb" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "" "or" "om" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz" "vi" "cy" "wo" "xh" "yo" "zu" "be")
CLD3_LANGS_FLORES99=("ny" "af" "am" "ar" "hy" "" "" "az" "bn" "bs" "bg" "my" "ca" "ceb" "" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "" "ka" "de" "gu" "ha" "iw" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "" "lt" "" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "" "" "" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "fil" "ta" "te" "th" "" "tr" "uk" "ur" "uz" "vi" "cy" "" "xh" "yo" "zu" "be")


ISO_639_3_LANGS_COMMON4=("afr" "ara" "ben" "bul" "cat" "zho" "hrv" "ces" "dan" "nld" "eng" "est" "fin" "fra" "deu" "guj" "heb" "hin" "hun" "ind" "ita" "jpn" "kan" "kor" "lav" "lit" "mkd" "mal" "mar" "ell" "npi" "fas" "pol" "por" "pan" "ron" "rus" "slk" "slv" "som" "spa" "swh" "swe" "tgl" "tam" "tel" "tha" "tur" "ukr" "urd" "vie" "cym")
JW300_LANGS_COMMON4=("af" "ar" "bn" "bg" "cat" "" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "de" "gu" "he" "hi" "hu" "id" "it" "ja" "kn" "ko" "lv" "lt" "mk" "ml" "mr" "el" "ne" "fa" "pl" "pt" "pa" "ro" "ru" "sk" "sl" "" "es" "sw" "sv" "tl" "ta" "te" "th" "tr" "uk" "ur" "vi" "cy")
FLORES_101_LANGS_COMMON4=("af" "ar" "bn" "bg" "ca" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "de" "gu" "he" "hi" "hu" "id" "it" "ja" "kn" "ko" "lv" "lt" "mk" "ml" "mr" "el" "ne" "fa" "pl" "pt" "pa" "ro" "ru" "sk" "sl" "so" "es" "sw" "sv" "tl" "ta" "te" "th" "tr" "uk" "ur" "vi" "cy")
LID_187_LANGS_COMMON4=("af" "ar" "bn" "bg" "ca" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "de" "gu" "he" "hi" "hu" "id" "it" "ja" "kn" "ko" "lv" "lt" "mk" "ml" "mr" "el" "ne" "fa" "pl" "pt" "pa" "ro" "ru" "sk" "sl" "so" "es" "sw" "sv" "tl" "ta" "te" "th" "tr" "uk" "ur" "vi" "cy")
CLD3_LANGS_COMMON4=("af" "ar" "bn" "bg" "ca" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "de" "gu" "iw" "hi" "hu" "id" "it" "ja" "kn" "ko" "lv" "lt" "mk" "ml" "mr" "el" "ne" "fa" "pl" "pt" "pa" "ro" "ru" "sk" "sl" "so" "es" "sw" "sv" "fil" "ta" "te" "th" "tr" "uk" "ur" "vi" "cy")
LANGDETECT_LANGS_COMMON4=("af" "ar" "bn" "bg" "ca" "zh-cn" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "de" "gu" "he" "hi" "hu" "id" "it" "ja" "kn" "ko" "lv" "lt" "mk" "ml" "mr" "el" "ne" "fa" "pl" "pt" "pa" "ro" "ru" "sk" "sl" "so" "es" "sw" "sv" "tl" "ta" "te" "th" "tr" "uk" "ur" "vi" "cy")


prepare_data_1 () {
    for i in "${!ISO_639_3_LANGS_FLORES99[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS_FLORES99[i]}
        JW300_LANG=${JW300_LANGS_FLORES99[i]}
        LID_187_LANG=${LID_187_LANGS_FLORES99[i]}

        # JW300
        # -----------

        JW300_DETOK_FILE="$DETOK_DIR/$JW300_LANG.detok.txt"
        if [ -f "$JW300_DETOK_FILE" ]; then
            NUMBER_OF_LINES=$(cat $JW300_DETOK_FILE | wc -l)
            echo -e "$ISO_639_3_LANG \t jw300 lines : $NUMBER_OF_LINES"


            MONO_JW300_FILE="$TRAIN_DIR/$ISO_639_3_LANG.jw300.mono.txt"
            cat $JW300_DETOK_FILE | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' > $MONO_JW300_FILE
        else
            echo -e "lang:$ISO_639_3_LANG \t $JW300_LANG does not exist: $JW300_DETOK_FILE"
        fi


        # LID187
        # -----------

        echo -e "$ISO_639_3_LANG \t $LID_187_LANG "

        if [ "$LID_187_LANG" != "" ]; then
            MONO_LID187_FILE="$TRAIN_DIR/$ISO_639_3_LANG.lid187.mono.txt"

            OLD_LID_TRAIN="/private/home/egrave/lid/train.txt"
            cat $OLD_LID_TRAIN | grep "__label__${LID_187_LANG} " | cut -f 2- -d" " | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' > $MONO_LID187_FILE

            MONO_LID187_FILE_NB_LINES=$(wc -l $MONO_LID187_FILE)

            echo -e "$ISO_639_3_LANG \t $LID_187_LANG \t : $MONO_LID187_FILE_NB_LINES"
        fi

    done
}

prepare_data_2 () {
    VALID_SIZE=100000
    VALID_COMPARE_SIZE=$(expr $VALID_SIZE \* 4)

    for i in "${!ISO_639_3_LANGS_FLORES99[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS_FLORES99[i]}
        JW300_LANG=${JW300_LANGS_FLORES99[i]}
        LID_187_LANG=${LID_187_LANGS_FLORES99[i]}

        MONO_JW300_FILE="$TRAIN_DIR/$ISO_639_3_LANG.jw300.mono.txt"
        MONO_LID187_FILE="$TRAIN_DIR/$ISO_639_3_LANG.lid187.mono.txt"

        MONO_CAT_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cat.txt"

        cat $MONO_JW300_FILE $MONO_LID187_FILE | shuf > $MONO_CAT_FILE

        NUMBER_OF_LINES=$(cat $MONO_CAT_FILE | wc -l)
        if [ "$NUMBER_OF_LINES" -lt "$VALID_COMPARE_SIZE" ]; then
            VALID_SIZE=$(expr $NUMBER_OF_LINES / 10)
            echo "    /!\ specific valid size for this language ($ISO_639_3_LANG) = $VALID_SIZE"
        fi

        TRAIN_NB_LINES=$(expr $NUMBER_OF_LINES - $VALID_SIZE)
        echo "    $ISO_639_3_LANG TRAIN_NB_LINES = $TRAIN_NB_LINES"


        TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.train.mono.txt"
        VALID_FILE="$VALID_DIR/$ISO_639_3_LANG.valid.mono.txt"

        head -n $TRAIN_NB_LINES $MONO_CAT_FILE > $TRAIN_FILE
        tail -n $VALID_SIZE $MONO_CAT_FILE > $VALID_FILE

    done

    cat $TRAIN_DIR/*.train.mono.txt | shuf > "$TRAIN_DIR/all.txt"
    cat $VALID_DIR/*.valid.mono.txt | shuf > "$VALID_DIR/all.txt"

    rm $TRAIN_DIR/*.cat.txt
    rm $TRAIN_DIR/*.train.mono.txt
    rm $VALID_DIR/*.valid.mono.txt
}

prepare_flores_dev_data () {
    CONCAT_FILE="$TEST_DIR/concat.all.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-dev.txt"
    rm $CONCAT_FILE

    for i in "${!ISO_639_3_LANGS_COMMON4[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS_COMMON4[i]}
        FLORES_101_LANG=${FLORES_101_LANGS_COMMON4[i]}

        if [ $FLORES_101_LANG == 'ast' ]
        then
            FILES_FOUND="$FLORES_DIR/ast.dev"
            FILES_FOUND_NUM="1"
        else
            FILES_FOUND=$(ls -1 $FLORES_DIR/${FLORES_101_LANG}_*.dev | head -n 1)
            FILES_FOUND_NUM=$(ls -1 $FLORES_DIR/${FLORES_101_LANG}_*.dev | wc -l)
        fi

        if [ "$FILES_FOUND_NUM" -gt 1 ]; then
            echo "/!\ Warning found: $FILES_FOUND_NUM $ISO_639_3_LANG"
        fi

        cat $FILES_FOUND | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE

    done

    shuf $CONCAT_FILE > $TEST_FINAL_FILE
    rm $CONCAT_FILE
}



prepare_flores_devtest_data () {
    CONCAT_FILE="$TEST_DIR/concat.all.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-devtest.txt"
    rm $CONCAT_FILE

    for i in "${!ISO_639_3_LANGS_COMMON4[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS_COMMON4[i]}
        FLORES_101_LANG=${FLORES_101_LANGS_COMMON4[i]}

        if [ $FLORES_101_LANG == 'ast' ]
        then
            FILES_FOUND="$FLORES_DIR/ast.devtest"
            FILES_FOUND_NUM="1"
        else
            FILES_FOUND=$(ls -1 $FLORES_DIR/${FLORES_101_LANG}_*.devtest | head -n 1)
            FILES_FOUND_NUM=$(ls -1 $FLORES_DIR/${FLORES_101_LANG}_*.devtest | wc -l)
        fi

        if [ "$FILES_FOUND_NUM" -gt 1 ]; then
            echo "/!\ Warning found: $FILES_FOUND_NUM $ISO_639_3_LANG"
        fi

        cat $FILES_FOUND | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE

    done

    shuf $CONCAT_FILE > $TEST_FINAL_FILE
    rm $CONCAT_FILE
}


FASTTEXT_BIN="/private/home/celebio/nlp/nllb_lid/fastText/fasttext"

train_fasttext () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1 -epoch 1 -dim 16 -loss softmax -bucket 5000000 -thread 10
}

eval_dev_fasttext () {
    PREDICTIONS="$RESULT_FOLDER/flores-dev.fasttext.predictions"
    GOLD="$RESULT_FOLDER/flores-dev.gold"
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-dev.fasttext.txt"

    TEST_FILE="$TEST_DIR/flores-dev.txt"
    RESULT_GATHER="$FASTTEXT_BIN predict $RESULT_FOLDER/model.bin $TEST_FILE"
    $RESULT_GATHER > $PREDICTIONS

    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb_lid/misc/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT

    cat $RESULT_TXT
}


LANGDETECT_BIN="/private/home/celebio/nlp/nllb_lid/cld3_scripts/run_langdetect.py"

eval_dev_langdetect () {
    PREDICTIONS="$RESULT_FOLDER/flores-dev.langdetect.predictions"
    GOLD="$RESULT_FOLDER/flores-dev.gold"

    TEST_FILE="$TEST_DIR/flores-dev.txt"
    cat "$TEST_FILE" | $LANGDETECT_BIN > $PREDICTIONS
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-dev.langdetect.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb_lid/misc/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT

    cat $RESULT_TXT
}




prepare_data_1
prepare_data_2
prepare_flores_dev_data
prepare_flores_devtest_data

train_fasttext
eval_dev_fasttext
eval_dev_langdetect


