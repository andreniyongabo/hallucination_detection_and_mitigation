#!/bin/bash


EXPERIMENT_NAME="2021-05-24-22-35-flores-99-match-lid187-only"
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



ISO_639_3_LANGS=("bel" "nya" "afr" "amh" "ara" "hye" "asm" "ast" "aze" "ben" "bos" "bul" "mya" "cat" "ceb" "ckb" "zho" "hrv" "ces" "dan" "nld" "eng" "est" "fin" "fra" "ful" "glg" "lug" "kat" "deu" "guj" "hau" "heb" "hin" "hun" "isl" "ibo" "ind" "ita" "jpn" "jav" "kea" "kan" "kaz" "khm" "kor" "kur" "kir" "lao" "lav" "lin" "lit" "luo" "ltz" "mkd" "mlg" "msa" "mal" "mlt" "mar" "ell" "mon" "nep" "nso" "ory" "orm" "pus" "fas" "pol" "por" "pan" "ron" "rus" "srp" "sna" "snd" "sin" "slk" "slv" "som" "spa" "sun" "swh" "swe" "tgl" "tam" "tel" "tha" "tsn" "tur" "ukr" "urd" "uzb" "vie" "cym" "wol" "xho" "yor" "zul")
JW300_LANGS=("" "nya" "af" "am" "ar" "hy" "as" "" "az" "bn" "" "bg" "my" "cat" "ceb" "" "" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "kea" "kn" "kk_Cyrl" "km" "ko" "" "ky" "lo" "lv" "ln" "lt" "luo" "" "mk" "mg" "zlm" "ml" "mt" "mr" "el" "mn" "ne" "nso" "or" "om" "" "fa" "pl" "pt" "pa" "ro" "ru" "sr_Cyrl" "sn" "" "si" "sk" "sl" "" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz_Latn" "vi" "cy" "" "xh" "yo" "zu")
FLORES_101_LANGS=("be" "ny" "af" "am" "ar" "hy" "as" "ast" "az" "bn" "bs" "bg" "my" "ca" "cx" "cb" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "ff" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "q3" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "qy" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "ns" "or" "om" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz" "vi" "cy" "wo" "xh" "yo" "zu")
LID_187_LANGS=("be" "" "af" "am" "ar" "hy" "as" "ast" "az" "bn" "bs" "bg" "my" "ca" "ceb" "ckb" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "" "or" "om" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz" "vi" "cy" "wo" "xh" "yo" "zu")

list_langs () {

    for i in "${!ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
        JW300_LANG=${JW300_LANGS[i]}
        FLORES_101_LANG=${FLORES_101_LANGS[i]}
        LID_187_LANG=${LID_187_LANGS[i]}

        printf "%s \t %s \t %s \t %s\n" $ISO_639_3_LANG $JW300_LANG $FLORES_101_LANG $LID_187_LANG
    done
}

prepare_data_1 () {
    for i in "${!ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
        JW300_LANG=${JW300_LANGS[i]}
        LID_187_LANG=${LID_187_LANGS[i]}



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

    for i in "${!ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
        JW300_LANG=${JW300_LANGS[i]}
        LID_187_LANG=${LID_187_LANGS[i]}

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

    for i in "${!ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
        JW300_LANG=${JW300_LANGS[i]}
        FLORES_101_LANG=${FLORES_101_LANGS[i]}
        LID_187_LANG=${LID_187_LANGS[i]}

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

train () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1 -epoch 1 -dim 16 -loss softmax -bucket 5000000 -thread 10

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


prepare_data_1
prepare_data_2
prepare_flores_dev_data
prepare_flores_devtest_data
train

eval_flores_dev
eval_flores_devtest