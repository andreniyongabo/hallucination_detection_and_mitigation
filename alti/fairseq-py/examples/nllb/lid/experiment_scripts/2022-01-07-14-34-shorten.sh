#!/bin/bash


EXPERIMENT_NAME="2022-01-07-14-34-shorten"
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

VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR

echo "DATA_FOLDER is $DATA_FOLDER"


ORIGINAL_VALID_FOLDER="/large_experiments/nllb/mmt/lidruns/2021-12-27-14-48-multifilter6-ndup/data/valid"
ORIGINAL_LID_MODEL="/large_experiments/nllb/mmt/lidruns/2021-12-27-14-48-multifilter6-ndup/result/model.8.1.bin"

FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"

NLLB_DEC__NLLB_LANGS=("abk" "ace_Arab" "ace_Latn" "ady" "afr" "alt" "amh" "ara" "arn" "arz" "asm" "ast" "aym" "azb" "azj" "bak" "ban" "bel" "bem" "ben" "bho" "bis" "bjn_Arab" "bjn_Latn" "bod" "bos" "bug" "bul" "bxr" "cat" "ceb" "ces" "che" "chv" "cjk" "ckb" "crh_Latn" "cym" "dan" "deu" "dik" "dyu" "dzo" "ell" "eng" "epo" "est" "eus" "ewe" "ewo" "fao" "fas" "fij" "fin" "fon" "fra" "ful" "fur" "gla" "gle" "glg" "gom" "grn" "guj" "hat" "hau" "heb" "hin" "hne" "hrv" "hun" "hye" "ibo" "ilo" "ind" "isl" "ita" "jav" "jpn" "kab" "kac" "kal" "kam" "kan" "kas_Arab" "kas_Deva" "kat" "kau_Arab" "kau_Latn" "kaz" "kbp" "kea" "khm" "kik" "kin" "kir" "kmb" "kon" "kor" "krc" "kur" "lao" "lav" "lij" "lim" "lin" "lit" "lmo" "ltg" "ltz" "lua" "lug" "luo" "lus" "mag" "mai" "mal" "mar" "min" "mkd" "mlg" "mlt" "mon" "mos" "mri" "msa" "mya" "nah" "nav" "nia" "nld" "nno" "nor" "npi" "nso" "nus" "nya" "oci" "orm" "ory" "oss" "pag" "pan" "pap" "pcm" "pol" "por" "pus" "que" "roh" "ron" "run" "rus" "sag" "san" "sat" "scn" "shn" "sin" "slk" "slv" "smo" "sna" "snd" "som" "sot" "spa" "sqi" "srd" "srp" "ssw" "sun" "swe" "swh" "tah" "tam" "tat" "tel" "tgk" "tgl" "tha" "tir" "tmh_Latn" "tog" "tpi" "tsn" "tso" "tuk" "tum" "tur" "twi" "udm" "uig" "ukr" "umb" "urd" "uzb" "vec" "vie" "war" "wes" "wol" "xho" "xmf" "yid" "yor" "yue" "zho" "zul" "zza")



create_short_sentences () {
    # LENGTHS=(100 50 40 30 20 10 5 )

    LENGTHS=(19 18 17 16 15 14 13 12 11 7 )

    for LENGTH in ${LENGTHS[@]}; do
        ALL_LENGTH_FILE="$TEST_DIR/all.$LENGTH.txt"
        ALL_LENGTH_FILE_UNSHUF="$TEST_DIR/all.$LENGTH.unshuf.txt"
        > $ALL_LENGTH_FILE_UNSHUF

        for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
            NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}

            ORIGINAL_FILE="$ORIGINAL_VALID_FOLDER/$NLLB_LANG.valid.mono.txt"
            NEW_FILE="$VALID_DIR/$NLLB_LANG.txt"

            cat $ORIGINAL_FILE | cut -f 2- -d" " | awk -v blength="$LENGTH" -v lang="$NLLB_LANG" '{print "__label__"lang " " substr($0,0,blength) }' > $NEW_FILE
            cat $NEW_FILE >> $ALL_LENGTH_FILE_UNSHUF
        done
        shuf $ALL_LENGTH_FILE_UNSHUF > $ALL_LENGTH_FILE
        rm $ALL_LENGTH_FILE_UNSHUF
        echo "Created $ALL_LENGTH_FILE"
    done
}


eval_short_sentences () {

    # LENGTHS=(100 50 40 30 20 10 5 )
    LENGTHS=(19 18 17 16 15 14 13 12 11 7 )

    for LENGTH in ${LENGTHS[@]}; do
        TEST_FILE="$TEST_DIR/all.$LENGTH.txt"
        GOLD="$RESULT_FOLDER/all.$LENGTH.gold"
        cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-shortened.$LENGTH.txt"

        LID_MODEL="$ORIGINAL_LID_MODEL"

        if [ -s $LID_MODEL ] && [ ! -s $RESULT_TXT ]
        then
            echo "LID_MODEL $LID_MODEL"
            PREDICTIONS="$RESULT_FOLDER/shortened.fasttext.predictions.$LENGTH"

            # RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
            # $RESULT_GATHER > $PREDICTIONS

            cat $TEST_FILE | parallel -j40 -k --pipe $FASTTEXT_BIN predict $LID_MODEL - > $PREDICTIONS

            CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
            $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
        fi


    done
}

display_results () {
    # LENGTHS=(100 50 40 30 20 10 5 ) ; for LENGTH in ${LENGTHS[@]}; do echo $LENGTH; cat "$RESULT_FOLDER/result.classifiermetrics-shortened.$LENGTH.txt" | grep "^Precision" | cut -f 3 -d" "; done | pr --columns 2  -a -t

    LENGTHS=(100 50 40 30 20 19 18 17 16 15 14 13 12 11 10 7 5 ) ; echo "META max_length precision"; for LENGTH in ${LENGTHS[@]}; do echo $LENGTH; cat "result.classifiermetrics-shortened.$LENGTH.txt" | grep "^Precision" | cut -f 3 -d" "; done | pr --columns 2  -a -t | grapher

}





create_short_sentences
eval_short_sentences

# display_results









