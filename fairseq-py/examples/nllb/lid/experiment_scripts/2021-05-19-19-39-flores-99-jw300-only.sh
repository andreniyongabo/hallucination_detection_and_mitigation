#!/bin/bash


EXPERIMENT_NAME="2021-05-19-19-39-flores-99-jw300-only"
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

ISO_639_3_LANGS=("bel" "nya" "est" "mkd" "lav" "slv" "lin" "bos" "lit" "glg" "mon" "ckb" "hye" "kat" "lao" "kir" "mlg" "dan" "fin" "lug" "hrv" "slk" "bul" "srp" "sna" "cat" "heb" "wol" "hun" "swe" "kaz" "ell" "ces" "tsn" "nso" "aze" "kur" "asm" "ful" "ceb" "afr" "khm" "sin" "som" "msa" "xho" "orm" "nld" "nep" "ron" "uzb" "zul" "ibo" "sun" "snd" "ukr" "mal" "pol" "yor" "mya" "pus" "tgl" "fas" "kan" "amh" "tha" "guj" "ita" "jav" "hau" "vie" "kor" "pan" "tam" "tur" "tel" "mar" "swh" "jpn" "deu" "urd" "ind" "por" "rus" "ben" "ara" "fra" "spa" "hin" "zho" "eng" "cym" "ast" "isl" "mlt" "kea" "luo" "ltz" "ory")
JW300_LANGS=("be" "ny" "et" "mk" "lv" "sl" "ln" "bs" "lt" "gl" "mn" "cb" "hy" "ka" "lo" "ky" "mg" "da" "fi" "lg" "hr" "sk" "bg" "sr" "sn" "ca" "he" "wo" "hu" "sv" "kk" "el" "cs" "tn" "ns" "az" "ku" "as" "ff" "cx" "af" "km" "si" "so" "ms" "xh" "om" "nl" "ne" "ro" "uz" "zu" "ig" "su" "sd" "uk" "ml" "pl" "yo" "my" "ps" "tl" "fa" "kn" "am" "th" "gu" "it" "jv" "ha" "vi" "ko" "pa" "ta" "tr" "te" "mr" "sw" "ja" "de" "ur" "id" "pt" "ru" "bn" "ar" "fr" "es" "hi" "zh" "en" "cy" "ast" "is" "mt" "q3" "qy" "lb" "or")
FLORES_101_LANGS=("be" "ny" "et" "mk" "lv" "sl" "ln" "bs" "lt" "gl" "mn" "cb" "hy" "ka" "lo" "ky" "mg" "da" "fi" "lg" "hr" "sk" "bg" "sr" "sn" "ca" "he" "wo" "hu" "sv" "kk" "el" "cs" "tn" "ns" "az" "ku" "as" "ff" "cx" "af" "km" "si" "so" "ms" "xh" "om" "nl" "ne" "ro" "uz" "zu" "ig" "su" "sd" "uk" "ml" "pl" "yo" "my" "ps" "tl" "fa" "kn" "am" "th" "gu" "it" "jv" "ha" "vi" "ko" "pa" "ta" "tr" "te" "mr" "sw" "ja" "de" "ur" "id" "pt" "ru" "bn" "ar" "fr" "es" "hi" "zh" "en" "cy" "ast" "is" "mt" "q3" "qy" "lb" "or")
ALLMOST_ALL_FLORES=(af am as bg bn cs cx da de en et fa fr gu ha he hi hr hu hy id ig it jv km kn ko ku lg lo lt mg mk ml mn mr ms my ne nl om or pa si su sv sw ta te tl tn tr ur xh yo zu)

for i in "${!ISO_639_3_LANGS[@]}"; do
    ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
    JW300_LANG=${JW300_LANGS[i]}
    FLORES_101_LANG=${FLORES_101_LANGS[i]}

    # printf "%s \t %s \t %s \t %s\n" $ISO_639_3_LANG $JW300_LANG $FLORES_101_LANG $LID_187_LANG

done

prepare_data_1 () {
    for i in "${!ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
        JW300_LANG=${JW300_LANGS[i]}
        FLORES_101_LANG=${FLORES_101_LANGS[i]}

        # JW300
        # -----------

        JW300_DETOK_FILE="$DETOK_DIR/$JW300_LANG.detok.txt"
        if [ -f "$JW300_DETOK_FILE" ]; then
            NUMBER_OF_LINES=$(cat $JW300_DETOK_FILE | wc -l)
            echo -e "$ISO_639_3_LANG \t jw300 lines : $NUMBER_OF_LINES"


            MONO_JW300_FILE="$TRAIN_DIR/$ISO_639_3_LANG.jw300.mono.txt"
            cat $JW300_DETOK_FILE | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' > $MONO_JW300_FILE
        else
            echo -e "lang:$ISO_639_3_LANG \t $JW300_LANG does not exist."
        fi


        # LID187
        # -----------

        # not using lid 187 in this experiment


    done
}

prepare_data_2 () {
    VALID_SIZE=100000

    for i in "${!ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS[i]}
        JW300_LANG=${JW300_LANGS[i]}
        FLORES_101_LANG=${FLORES_101_LANGS[i]}

        MONO_JW300_FILE="$TRAIN_DIR/$ISO_639_3_LANG.jw300.mono.txt"

        MONO_CAT_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cat.txt"

        cat $MONO_JW300_FILE | shuf > $MONO_CAT_FILE

        NUMBER_OF_LINES=$(cat $MONO_CAT_FILE | wc -l)
        if [ "$NUMBER_OF_LINES" -lt "$VALID_SIZE" ]; then
            VALID_SIZE=$(expr $NUMBER_OF_LINES / 10)
            echo "    /!\ specific valid size for this language= $VALID_SIZE"
        fi

        TRAIN_NB_LINES=$(expr $NUMBER_OF_LINES - $VALID_SIZE)
        echo "    TRAIN_NB_LINES = $TRAIN_NB_LINES"


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

