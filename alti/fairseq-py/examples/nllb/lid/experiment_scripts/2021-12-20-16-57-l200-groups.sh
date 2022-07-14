#!/bin/bash


EXPERIMENT_NAME="2021-12-20-16-57-l200-groups"
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

echo "DATA_FOLDER is $DATA_FOLDER"


INITIAL_XP_FOLDER="/large_experiments/nllb/mmt/lidruns/2021-11-26-15-16-l200/"
INITIAL_TRAIN_FOLDER="$INITIAL_XP_FOLDER/data/train"
INITIAL_TEST_FOLDER="$INITIAL_XP_FOLDER/data/test"

# NLLB_DEC__NLLB_LANGS=("afr" "aka" "amh" "ara_Arab" "acm" "apc" "asm" "ast" "ayr" "azj" "bam" "bel" "ben" "bos" "bul" "cat" "ceb" "ces" "ckb" "cym" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "fuv" "gla" "gle" "glg" "guj" "hat" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ilo" "ind" "isl" "ita" "jav" "jpn" "kac" "kam" "kan" "kas_Arab" "kat" "kaz" "kea" "khm" "kir" "kmb" "kon" "kor" "kur" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mai" "mal" "mar" "mkd" "mlg" "mlt" "mon" "mri" "msa" "mya" "nld" "nob" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "quy" "ron" "run" "rus" "shn" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp_Cyrl" "ssw" "sun_Latn" "swe" "swh_Latn" "tam" "tel" "tgk" "tgl" "tha" "tir" "tpi" "tsn" "tur" "ukr" "umb" "urd" "uzb" "vie" "wol" "xho" "yid" "yor" "yue" "zho_Hans" "zho_Hant" "zul" "ace_Arab" "ace_Latn" "arz" "ban" "bjn_Arab" "bjn_Latn" "bak" "eus" "bem" "bho" "bug" "hne" "cjk" "crh_Latn" "prs" "dik" "dzo" "epo" "ewe" "fao" "fij" "fon" "fur" "grn" "kbp" "kab" "kau_Arab" "kau_Latn" "kas_Deva" "kik" "kin" "ltg" "lij" "lim" "lmo" "lua" "mag" "min_Latn" "lus" "mos" "nus" "nno" "pag" "pap" "smo" "sag" "san" "sat" "srd" "scn" "szl" "azb" "sot" "diq" "tmh_Latn" "tat_Cyrl" "bod" "ton" "tso" "tum" "tuk" "twi" "uig" "vec" "war" "wes" "abk" "ady" "bis" "che" "chv" "ewo" "gom" "kal" "krc" "arn" "xmf" "nav" "nia" "pcm" "oss" "roh" "bxr" "alt" "tah" "udm")

FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"


NLLB_DEC__NLLB_LANGS_GROUP_1=("jpn" "zho_Hans" "zho_Hant" "yue")


prepare_group_data_1 () {
    GROUP_NAME="1"

    ALL_DATA_FOR_GROUP="$TRAIN_DIR/all.group.$GROUP_NAME.txt"
    > $ALL_DATA_FOR_GROUP
    ALL_DATA_FOR_GROUP_UNSHUF="$TRAIN_DIR/all.group.$GROUP_NAME.unshuf.txt"
    > $ALL_DATA_FOR_GROUP_UNSHUF

    GROUP_LANG_NAME="NLLB_DEC__NLLB_LANGS_GROUP_${GROUP_NAME}";
    GROUP_LANG_NAME_ARR="$GROUP_LANG_NAME[@]";
    for NLLB_LANG in "${!GROUP_LANG_NAME_ARR}";
    do
        echo $NLLB_LANG
        INITIAL_TRAIN_DATA="$INITIAL_TRAIN_FOLDER/$NLLB_LANG.train.mono.txt"
        cat $INITIAL_TRAIN_DATA >> $ALL_DATA_FOR_GROUP_UNSHUF
    done

    shuf $ALL_DATA_FOR_GROUP_UNSHUF > $ALL_DATA_FOR_GROUP
    rm $ALL_DATA_FOR_GROUP_UNSHUF
}

prepare_flores_filled_data_1 () {
    GROUP_NAME="1"

    CONCAT_FILE="$TEST_DIR/concat.flores-filled.$GROUP_NAME.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-filled.$GROUP_NAME.txt"
    > $CONCAT_FILE


    # for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
    #     NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}
    #     FLORES_LANG=${NLLB_DEC__FLORES_LANGS[i]}
    #     FBSEED_LANG=${NLLB_DEC__FBSEED_LANGS[i]}
    #     JW300_LANG=${NLLB_DEC__JW300_LANGS[i]}
    #     LID_187_LANG=${NLLB_DEC__LID_187_LANGS[i]}

    GROUP_LANG_NAME="NLLB_DEC__NLLB_LANGS_GROUP_${GROUP_NAME}";
    GROUP_LANG_NAME_ARR="$GROUP_LANG_NAME[@]";
    for NLLB_LANG in "${!GROUP_LANG_NAME_ARR}";
    do
        echo $NLLB_LANG

        INITIAL_FLORES_FILLED="$INITIAL_TEST_FOLDER/flores-filled.txt"
        echo "$INITIAL_FLORES_FILLED"
        cat $INITIAL_FLORES_FILLED | grep "^__label__${NLLB_LANG}" >> $CONCAT_FILE


        # chosen_source=""
        # if [ ! -z "$FLORES_LANG" ]
        # then
        #     FILES_FOUND=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | head -n 1)
        #     FILES_FOUND_NUM=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | wc -l)

        #     # echo "LANG=$NLLB_LANG" "$FLORES_LANG" "$FILES_FOUND_NUM"
        #     # echo "$FILES_FOUND"
        #     chosen_source="${chosen_source}-flores"

        #     cat $FILES_FOUND | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
        # else


        #     # echo "Flores not available for $NLLB_LANG"
        #     VALID_FILE="$VALID_DIR/$NLLB_LANG.valid.mono.txt"
        #     if [ -f "$VALID_FILE" ]; then
        #         chosen_source="${chosen_source}-fromvalid"
        #         # echo "will use $VALID_FILE"
        #         shuf $VALID_FILE | head -n 1000 >> $CONCAT_FILE
        #     else
        #         echo "ERROR: File not found $VALID_FILE"
        #         return 1
        #     fi
        # fi

        # printf "%-20s %s \n" $NLLB_LANG $chosen_source
    done

    shuf $CONCAT_FILE > $TEST_FINAL_FILE
    rm $CONCAT_FILE
}

train_fasttext_group_1 () {
    GROUP_NAME="1"

    echo "Training fastText for group $GROUP_NAME"

    ALL_DATA_FOR_GROUP="$TRAIN_DIR/all.group.$GROUP_NAME.txt"
    RESULT_MODEL="$RESULT_FOLDER/model.group.$GROUP_NAME"

    $FASTTEXT_BIN supervised -input $ALL_DATA_FOR_GROUP -output $RESULT_MODEL \
        -lr 0.8 -minn 2 -maxn 5 -minCount 10 -epoch 10 -dim 256 -loss softmax -bucket 1000000 -thread 60
}

train_fasttext_manualclean_group_1 () {
    GROUP_NAME="1"

    echo "Training fastText for group $GROUP_NAME"

    ALL_DATA_FOR_GROUP="$TRAIN_DIR/all.group.$GROUP_NAME.manualclean.txt"
    RESULT_MODEL="$RESULT_FOLDER/model.group.$GROUP_NAME.manualclean"

    $FASTTEXT_BIN supervised -input $ALL_DATA_FOR_GROUP -output $RESULT_MODEL \
        -lr 0.8 -minn 1 -maxn 10 -minCount 1 -epoch 40 -dim 256 -loss softmax -bucket 10000000 -thread 60
}

# A
    # $FASTTEXT_BIN supervised -input $ALL_DATA_FOR_GROUP -output $RESULT_MODEL \
    #     -lr 0.8 -minn 2 -maxn 5 -minCount 10 -epoch 10 -dim 256 -loss softmax -bucket 1000000 -thread 60

# B
    # $FASTTEXT_BIN supervised -input $ALL_DATA_FOR_GROUP -output $RESULT_MODEL \
    #     -lr 0.8 -minn 1 -maxn 5 -minCount 1 -epoch 30 -dim 256 -loss softmax -bucket 1000000 -thread 60


eval_flores_filled_fasttext_group_1 () {
    GROUP_NAME="1"


    GOLD="$RESULT_FOLDER/flores-filled.$GROUP_NAME.gold"
    TEST_FILE="$TEST_DIR/flores-filled.$GROUP_NAME.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.group.$GROUP_NAME.txt"

    LID_MODEL="$RESULT_FOLDER/model.group.$GROUP_NAME.bin"

    if [ -s $LID_MODEL ]
    then
        echo "LID_MODEL $LID_MODEL"
        PREDICTIONS="$RESULT_FOLDER/flores-filled.fasttext.predictions.group.$GROUP_NAME"

        RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
        $RESULT_GATHER > $PREDICTIONS

        CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
        $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
    fi
}

eval_flores_filled_fasttext_manualclean_group_1 () {
    GROUP_NAME="1"

    GOLD="$RESULT_FOLDER/flores-filled.$GROUP_NAME.gold"
    TEST_FILE="$TEST_DIR/flores-filled.$GROUP_NAME.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.group.$GROUP_NAME.manualclean.txt"

    LID_MODEL="$RESULT_FOLDER/model.group.$GROUP_NAME.manualclean.bin"

    if [ -s $LID_MODEL ]
    then
        echo "LID_MODEL $LID_MODEL"
        PREDICTIONS="$RESULT_FOLDER/flores-filled.fasttext.predictions.group.$GROUP_NAME.manualclean"

        RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
        $RESULT_GATHER > $PREDICTIONS

        CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
        $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
    fi
}

eval_flores_filled_fasttext_group_autotune () {
    GROUP_NAME="1"


    GOLD="$RESULT_FOLDER/flores-filled.$GROUP_NAME.gold"
    TEST_FILE="$TEST_DIR/flores-filled.$GROUP_NAME.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.group.at.$GROUP_NAME.txt"

    LID_MODEL="$RESULT_FOLDER/model.at.bin"

    if [ -s $LID_MODEL ]
    then
        echo "LID_MODEL $LID_MODEL"
        PREDICTIONS="$RESULT_FOLDER/flores-filled.fasttext.predictions.group.at.$GROUP_NAME"

        RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
        $RESULT_GATHER > $PREDICTIONS

        CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
        $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
    fi
}




# prepare_group_data_1
#prepare_flores_filled_data_1

# train_fasttext_group_1

#eval_flores_filled_fasttext_group_1
#eval_flores_filled_fasttext_group_autotune

train_fasttext_manualclean_group_1
eval_flores_filled_fasttext_manualclean_group_1

