#!/bin/bash


EXPERIMENT_NAME="2021-10-05-11-53-english"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER



DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER

FLORES_DEV_DIR="/large_experiments/mmt/flores101/dev"
FLORES_DEVTEST_DIR="/large_experiments/mmt/flores101/devtest"
JW300_DETOK_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/detok"
LID187_DATA_DIR="/private/home/celebio/lid187_data"

TRAIN_DIR="$DATA_FOLDER/train"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TRAIN_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR

echo "DATA_FOLDER is $DATA_FOLDER"


GOAL124__ISO_639_3_LANGS=("aym" "ceb" "ckb" "cym" "gla" "gle" "hat" "jav" "kac" "kaz" "kea" "kir" "kur" "mlt" "mri" "mya" "que" "sun" "tgk" "uzb" "afr" "amh" "ara" "asm" "ast" "azj" "bel" "ben" "bos" "bul" "cat" "ces" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "ful" "glg" "guj" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ind" "isl" "ita" "jpn" "kam" "kan" "kat" "khm" "kmb" "kon" "kor" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mal" "mar" "mkd" "mlg" "mon" "msa" "nld" "nor" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "ron" "rus" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp" "ssw" "swe" "swh" "tam" "tel" "tgl" "tha" "tir" "tsn" "tur" "ukr" "umb" "urd" "vie" "wol" "xho" "yid" "yor" "yue" "zho" "zul" "eus" "wes" "epo" "kin" "tat" "war" "abk" "ady" "bak" "bem" "bho" "bis" "che" "chv" "ewe" "ewo" "fao" "fij" "fon" "gom" "kal" "grn" "kbp" "kab" "krc" "kik" "lim" "lmo" "lua" "mai" "arn" "min" "xmf" "lus" "mos" "nav" "nia" "pcm" "nno" "oss" "pag" "pap" "roh" "run" "bxr" "smo" "sag" "san" "sat" "srd" "scn" "azb" "alt" "sot" "tah" "bod" "tpi" "tog" "tso" "tum" "tuk" "twi" "udm" "uig" "vec" "zza" "" "cjk" "arz" "ilo")
GOAL124__JW300_LANGS=("ay" "ceb" "" "cy" "" "ga" "ht" "jv" "kac" "kk_Cyrl" "kea" "ky" "" "mt" "" "my" "que" "su" "tg" "uz_Latn" "af" "am" "ar" "as" "" "az" "" "bn" "" "bg" "cat" "cs" "da" "de" "dyu" "el" "en" "et" "fa" "fi" "fr" "" "gl" "gu" "ha" "he" "hi" "hr" "hu" "hy" "ig" "id" "is" "it" "ja" "kam" "kn" "ka" "km" "kmb" "kg" "ko" "lo" "lv" "ln" "lt" "" "lg" "luo" "ml" "mr" "mk" "mg" "mn" "zlm" "nl" "no" "ne" "nso" "nya" "" "om" "or" "pa" "pl" "pt" "" "ro" "ru" "si" "sk" "sl" "sn" "" "" "es" "sq" "sr_Cyrl" "ss" "sv" "sw" "ta" "te" "tl" "th" "ti" "tn" "tr" "uk" "umb" "ur" "vi" "" "xh" "" "yo" "" "" "zu" "eu" "wes" "" "rw" "tt" "war" "ab" "ady" "ba" "bem" "" "bi" "" "cv" "ee" "ewo" "fo" "fj" "fon" "gom" "kl" "gug" "kbp" "kab" "krc" "ki" "" "" "lua" "" "arn" "" "xmf" "lus" "mos" "nv" "nia" "pcm" "" "os" "pag" "pap" "" "run" "" "sm" "sg" "" "" "" "" "" "alt" "st" "ty" "" "tpi" "tog" "ts" "tum" "tk" "tw" "udm" "" "vec" "" "" "cjk" "" "ilo")
GOAL124__LID_187_LANGS=("" "ceb" "ckb" "cy" "gd" "ga" "ht" "jv" "" "kk" "" "ky" "ku" "mt" "" "my" "qu" "su" "tg" "uz" "af" "am" "ar" "as" "ast" "az" "be" "bn" "bs" "bg" "ca" "cs" "da" "de" "" "el" "en" "et" "fa" "fi" "fr" "" "gl" "gu" "ha" "he" "hi" "hr" "hu" "hy" "ig" "id" "is" "it" "ja" "" "kn" "ka" "km" "" "" "ko" "lo" "lv" "ln" "lt" "lb" "lg" "" "ml" "mr" "mk" "mg" "mn" "ms" "nl" "no" "ne" "" "" "oc" "om" "or" "pa" "pl" "pt" "ps" "ro" "ru" "si" "sk" "sl" "sn" "sd" "so" "es" "sq" "sr" "" "sv" "sw" "ta" "te" "tl" "th" "" "tn" "tr" "uk" "" "ur" "vi" "wo" "xh" "yi" "yo" "yue" "zh" "zu" "eu" "" "eo" "" "tt" "war" "" "" "ba" "" "bh" "" "ce" "cv" "" "" "" "" "" "gom" "" "gn" "" "kab" "krc" "" "li" "lmo" "" "mai" "" "min" "xmf" "" "" "" "" "" "nn" "os" "" "" "rm" "" "bxr" "" "" "sa" "sat" "sc" "scn" "azb" "" "" "" "bo" "" "" "" "" "tk" "" "" "ug" "vec" "diq" "nah" "" "arz" "ilo")
GOAL124__FLORES_LANGS=("aym" "ceb" "ckb" "cym" "gla" "gle" "hat" "jav" "kac" "kaz" "kea" "kir" "kur" "mlt" "mri" "mya" "que" "sun" "tgk" "uzb" "afr" "amh" "ara" "asm" "ast" "azj" "bel" "ben" "bos" "bul" "cat" "ces" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "ful" "glg" "guj" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ind" "isl" "ita" "jpn" "kam" "kan" "kat" "khm" "kmb" "kon" "kor" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mal" "mar" "mkd" "mlg" "mon" "msa" "nld" "nob" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "ron" "rus" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp" "ssw" "swe" "swh" "tam" "tel" "tgl" "tha" "tir" "tsn" "tur" "ukr" "umb" "urd" "vie" "wol" "xho" "yid" "yor" "yue" "zho_simpl" "zul" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "")


FILTER_CHAR_HISTOGRAM="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/filtering/filter_char_histogram.py"
HISTOGRAMS_VALID_FOLDER="/large_experiments/mmt/lidruns/2021-09-20-14-14-histogram-baseline/histograms/valid"


EXISTING_DATA_FOLDER="/large_experiments/nllb/mmt/lidruns/2021-09-20-23-26-goal124-filter-percentile/data/"

THRESH_SCORE="0.8"

prepare_data_from_existing () {
    OLD_TRAIN_FILE="$EXISTING_DATA_FOLDER/train/all.txt"
    # cat $OLD_TRAIN_FILE | grep "__label__eng" > "$TRAIN_DIR/eng.txt"
    NUMBER_OF_LINES=$(cat $TRAIN_DIR/eng.txt | wc -l)

    # cat $OLD_TRAIN_FILE | grep -v "__label__eng" | cut -f 2- -d" " | awk '{print "__label__neng " $0}' | head -n $NUMBER_OF_LINES > "$TRAIN_DIR/neng.txt"


    cat $TRAIN_DIR/eng.txt $TRAIN_DIR/neng.txt | shuf > "$TRAIN_DIR/all.txt"
}

prepare_data_from_existing_valid () {
    OLD_VALID_FILE="$EXISTING_DATA_FOLDER/valid/all.txt"
    cat $OLD_VALID_FILE | grep "__label__eng" > "$VALID_DIR/eng.txt"
    NUMBER_OF_LINES=$(cat $VALID_DIR/eng.txt | wc -l)

    cat $OLD_VALID_FILE | grep -v "__label__eng" | cut -f 2- -d" " | awk '{print "__label__neng " $0}' | head -n $NUMBER_OF_LINES > "$VALID_DIR/neng.txt"


    cat $VALID_DIR/eng.txt $VALID_DIR/neng.txt | shuf > "$VALID_DIR/all.txt"
}


FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"


train_fasttext_8_8 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.8.8 \
        -lr 0.8 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 256 -loss ova -bucket 10000000 -thread 40
}




# F1-Score : 0.995906  Precision : 0.992159  Recall : 0.999680  FPR : 0.007900   __label__eng
# F1-Score : 0.995874  Precision : 0.999678  Recall : 0.992100  FPR : 0.000320   __label__neng
# N       200000
# P@1     0.9958900000
# R@1     0.9958900000
# FPR@1   0.0041100000
# train_fasttext_2 () {
#     echo "Training fastText:"
#     $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.2 \
#         -lr 1.0 -minn 2 -maxn 5 -minCount 10 -epoch 10 -dim 16 -loss softmax -bucket 10000000 -thread 10
# }

train_fasttext_2 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.2 \
        -lr 1.0 -minn 1 -maxn 5 -minCount 5 -epoch 10 -dim 16 -loss softmax -bucket 10000000 -thread 10
}

eval_valid () {
    VALID_FILE="$VALID_DIR/all.txt"

    RESULT_TXT_FT="$RESULT_FOLDER/result.fasttext-valid.txt"

    $FASTTEXT_BIN test-label $RESULT_FOLDER/model.8.8.bin $VALID_FILE > $RESULT_TXT_FT
}

eval_valid_2 () {
    VALID_FILE="$VALID_DIR/all.txt"

    RESULT_TXT_FT="$RESULT_FOLDER/result.fasttext-valid.2.txt"

    $FASTTEXT_BIN test-label $RESULT_FOLDER/model.2.bin $VALID_FILE > $RESULT_TXT_FT
}

eval_valid_classifier_metrics () {
    VALID_FILE="$VALID_DIR/all.txt"
    VALID_GOLD="$RESULT_FOLDER/valid.gold"
    VALID_PREDICTIONS="$RESULT_FOLDER/valid.predictions"
    cat "$VALID_FILE" | cut -f 1 -d" " > $VALID_GOLD

    RESULT_TXT_CM="$RESULT_FOLDER/result.classifier_metrics-valid.txt"

    LID_MODEL="$RESULT_FOLDER/model.8.8.bin"

    if [ -s $LID_MODEL ]
    then
        RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $VALID_FILE"
        $RESULT_GATHER > $VALID_PREDICTIONS

        CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
        $CLASSIFIER_METRICS --prediction $VALID_PREDICTIONS --gold $VALID_GOLD > $RESULT_TXT_CM
    fi

}


prepare_data_from_existing

prepare_data_from_existing_valid

train_fasttext_8_8

train_fasttext_2
eval_valid_2

eval_valid
eval_valid_classifier_metrics





