#!/bin/bash


EXPERIMENT_NAME="2021-06-10-11-38-flores-99-match-cld3"
EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER

FLORES_DIR="/checkpoint/angelafan/flores_preliminary_data"
DETOK_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/detok"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TEST_DIR

echo "DATA_FOLDER is $DATA_FOLDER"


ISO_639_3_LANGS=("nya" "afr" "amh" "ara" "hye" "asm" "ast" "aze" "ben" "bos" "bul" "mya" "cat" "ceb" "ckb" "zho" "hrv" "ces" "dan" "nld" "eng" "est" "fin" "fra" "ful" "glg" "lug" "kat" "deu" "guj" "hau" "heb" "hin" "hun" "isl" "ibo" "ind" "ita" "jpn" "jav" "kea" "kan" "kaz" "khm" "kor" "kur" "kir" "lao" "lav" "lin" "lit" "luo" "ltz" "mkd" "mlg" "msa" "mal" "mlt" "mar" "ell" "mon" "npi" "nso" "ory" "orm" "pus" "fas" "pol" "por" "pan" "ron" "rus" "srp" "sna" "snd" "sin" "slk" "slv" "som" "spa" "sun" "swh" "swe" "tgl" "tam" "tel" "tha" "tsn" "tur" "ukr" "urd" "uzb" "vie" "cym" "wol" "xho" "yor" "zul" "bel")
JW300_LANGS=("nya" "af" "am" "ar" "hy" "as" "" "az" "bn" "" "bg" "my" "cat" "ceb" "" "" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "kea" "kn" "kk_Cyrl" "km" "ko" "" "ky" "lo" "lv" "ln" "lt" "luo" "" "mk" "mg" "zlm" "ml" "mt" "mr" "el" "mn" "ne" "nso" "or" "om" "" "fa" "pl" "pt" "pa" "ro" "ru" "sr_Cyrl" "sn" "" "si" "sk" "sl" "" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz_Latn" "vi" "cy" "" "xh" "yo" "zu" "")
FLORES_101_LANGS=("ny" "af" "am" "ar" "hy" "as" "ast" "az" "bn" "bs" "bg" "my" "ca" "cx" "cb" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "ff" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "q3" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "qy" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "ns" "or" "om" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz" "vi" "cy" "wo" "xh" "yo" "zu" "be")
LID_187_LANGS=("" "af" "am" "ar" "hy" "as" "ast" "az" "bn" "bs" "bg" "my" "ca" "ceb" "ckb" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "" "or" "om" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz" "vi" "cy" "wo" "xh" "yo" "zu" "be")
CLD3_LANGS=("ny" "af" "am" "ar" "hy" "" "" "az" "bn" "bs" "bg" "my" "ca" "ceb" "" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "" "ka" "de" "gu" "ha" "iw" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "" "lt" "" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "" "" "" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "fil" "ta" "te" "th" "" "tr" "uk" "ur" "uz" "vi" "cy" "" "xh" "yo" "zu" "be")


list_langs () {

    for i in "${!ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
        JW300_LANG=${JW300_LANGS[i]}
        FLORES_101_LANG=${FLORES_101_LANGS[i]}
        LID_187_LANG=${LID_187_LANGS[i]}
        CLD3_LANG=${CLD3_LANGS[i]}

        printf "%s \t %s \t %s \t %s \t %s\n" $ISO_639_3_LANG $JW300_LANG $FLORES_101_LANG $LID_187_LANG $CLD3_LANG
    done
}


prepare_flores_dev_data () {
    CONCAT_FILE="$TEST_DIR/concat.all.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-dev.txt"
    rm $CONCAT_FILE

    for i in "${!ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
        JW300_LANG=${JW300_LANGS[i]}
        FLORES_101_LANG=${FLORES_101_LANGS[i]}
        LID_187_LANG=${LID_187_LANGS[i]}
        CLD3_LANG=${CLD3_LANGS[i]}

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

    for i in "${!ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
        JW300_LANG=${JW300_LANGS[i]}
        FLORES_101_LANG=${FLORES_101_LANGS[i]}
        LID_187_LANG=${LID_187_LANGS[i]}
        CLD3_LANG=${CLD3_LANGS[i]}

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


CLD3_BIN="/private/home/celebio/nlp/nllb_lid/cld3_scripts/run_cld3.py"

eval_cld3 () {
    PREDICTIONS="$RESULT_FOLDER/flores-dev.predictions"
    GOLD="$RESULT_FOLDER/flores-dev.gold"
    cat "$TEST_DIR/flores-dev.txt" | $CLD3_BIN > $PREDICTIONS
    cat "$TEST_DIR/flores-dev.txt" | cut -f 1 -d" " > $GOLD

    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-dev.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb_lid/misc/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT

    cat $RESULT_TXT
}


prepare_flores_dev_data
prepare_flores_devtest_data

eval_cld3



