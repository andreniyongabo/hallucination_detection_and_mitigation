#!/bin/bash


EXPERIMENT_NAME="2021-10-05-14-55-english-charhist-threshold"
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


# GOAL124__ISO_639_3_LANGS=("aym" "ceb" "ckb" "cym" "gla" "gle" "hat" "jav" "kac" "kaz" "kea" "kir" "kur" "mlt" "mri" "mya" "que" "sun" "tgk" "uzb" "afr" "amh" "ara" "asm" "ast" "azj" "bel" "ben" "bos" "bul" "cat" "ces" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "ful" "glg" "guj" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ind" "isl" "ita" "jpn" "kam" "kan" "kat" "khm" "kmb" "kon" "kor" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mal" "mar" "mkd" "mlg" "mon" "msa" "nld" "nor" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "ron" "rus" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp" "ssw" "swe" "swh" "tam" "tel" "tgl" "tha" "tir" "tsn" "tur" "ukr" "umb" "urd" "vie" "wol" "xho" "yid" "yor" "yue" "zho" "zul" "eus" "wes" "epo" "kin" "tat" "war" "abk" "ady" "bak" "bem" "bho" "bis" "che" "chv" "ewe" "ewo" "fao" "fij" "fon" "gom" "kal" "grn" "kbp" "kab" "krc" "kik" "lim" "lmo" "lua" "mai" "arn" "min" "xmf" "lus" "mos" "nav" "nia" "pcm" "nno" "oss" "pag" "pap" "roh" "run" "bxr" "smo" "sag" "san" "sat" "srd" "scn" "azb" "alt" "sot" "tah" "bod" "tpi" "tog" "tso" "tum" "tuk" "twi" "udm" "uig" "vec" "zza" "" "cjk" "arz" "ilo")
# GOAL124__JW300_LANGS=("ay" "ceb" "" "cy" "" "ga" "ht" "jv" "kac" "kk_Cyrl" "kea" "ky" "" "mt" "" "my" "que" "su" "tg" "uz_Latn" "af" "am" "ar" "as" "" "az" "" "bn" "" "bg" "cat" "cs" "da" "de" "dyu" "el" "en" "et" "fa" "fi" "fr" "" "gl" "gu" "ha" "he" "hi" "hr" "hu" "hy" "ig" "id" "is" "it" "ja" "kam" "kn" "ka" "km" "kmb" "kg" "ko" "lo" "lv" "ln" "lt" "" "lg" "luo" "ml" "mr" "mk" "mg" "mn" "zlm" "nl" "no" "ne" "nso" "nya" "" "om" "or" "pa" "pl" "pt" "" "ro" "ru" "si" "sk" "sl" "sn" "" "" "es" "sq" "sr_Cyrl" "ss" "sv" "sw" "ta" "te" "tl" "th" "ti" "tn" "tr" "uk" "umb" "ur" "vi" "" "xh" "" "yo" "" "" "zu" "eu" "wes" "" "rw" "tt" "war" "ab" "ady" "ba" "bem" "" "bi" "" "cv" "ee" "ewo" "fo" "fj" "fon" "gom" "kl" "gug" "kbp" "kab" "krc" "ki" "" "" "lua" "" "arn" "" "xmf" "lus" "mos" "nv" "nia" "pcm" "" "os" "pag" "pap" "" "run" "" "sm" "sg" "" "" "" "" "" "alt" "st" "ty" "" "tpi" "tog" "ts" "tum" "tk" "tw" "udm" "" "vec" "" "" "cjk" "" "ilo")
# GOAL124__LID_187_LANGS=("" "ceb" "ckb" "cy" "gd" "ga" "ht" "jv" "" "kk" "" "ky" "ku" "mt" "" "my" "qu" "su" "tg" "uz" "af" "am" "ar" "as" "ast" "az" "be" "bn" "bs" "bg" "ca" "cs" "da" "de" "" "el" "en" "et" "fa" "fi" "fr" "" "gl" "gu" "ha" "he" "hi" "hr" "hu" "hy" "ig" "id" "is" "it" "ja" "" "kn" "ka" "km" "" "" "ko" "lo" "lv" "ln" "lt" "lb" "lg" "" "ml" "mr" "mk" "mg" "mn" "ms" "nl" "no" "ne" "" "" "oc" "om" "or" "pa" "pl" "pt" "ps" "ro" "ru" "si" "sk" "sl" "sn" "sd" "so" "es" "sq" "sr" "" "sv" "sw" "ta" "te" "tl" "th" "" "tn" "tr" "uk" "" "ur" "vi" "wo" "xh" "yi" "yo" "yue" "zh" "zu" "eu" "" "eo" "" "tt" "war" "" "" "ba" "" "bh" "" "ce" "cv" "" "" "" "" "" "gom" "" "gn" "" "kab" "krc" "" "li" "lmo" "" "mai" "" "min" "xmf" "" "" "" "" "" "nn" "os" "" "" "rm" "" "bxr" "" "" "sa" "sat" "sc" "scn" "azb" "" "" "" "bo" "" "" "" "" "tk" "" "" "ug" "vec" "diq" "nah" "" "arz" "ilo")
# GOAL124__FLORES_LANGS=("aym" "ceb" "ckb" "cym" "gla" "gle" "hat" "jav" "kac" "kaz" "kea" "kir" "kur" "mlt" "mri" "mya" "que" "sun" "tgk" "uzb" "afr" "amh" "ara" "asm" "ast" "azj" "bel" "ben" "bos" "bul" "cat" "ces" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "ful" "glg" "guj" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ind" "isl" "ita" "jpn" "kam" "kan" "kat" "khm" "kmb" "kon" "kor" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mal" "mar" "mkd" "mlg" "mon" "msa" "nld" "nob" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "ron" "rus" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp" "ssw" "swe" "swh" "tam" "tel" "tgl" "tha" "tir" "tsn" "tur" "ukr" "umb" "urd" "vie" "wol" "xho" "yid" "yor" "yue" "zho_simpl" "zul" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "")


# GOAL124__ISO_639_3_LANGS=("srp" "tur")
# "khm")

# GOAL124__ISO_639_3_LANGS=("jpn" "zho")


FILTER_CHAR_HISTOGRAM="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/filtering/filter_char_histogram.py"
HISTOGRAMS_VALID_FOLDER="/large_experiments/mmt/lidruns/2021-09-20-14-14-histogram-baseline/histograms/valid"
FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"


OLD_DATA_TRAIN_FOLDER="/large_experiments/mmt/lidruns/2021-09-18-00-21-goal124-baseline/data/train"
LID_MODEL="/large_experiments/nllb/mmt/lidruns/2021-10-05-11-53-english/result/model.8.8.bin"

HIST_THRESH_SCORES=( 0.8 0.9 0.95 0.99 0.995 1.00 )

THRESH_SCORE="0.8"

prepare_data_2_filter () {

    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}

        MONO_CAT_FILE="$OLD_DATA_TRAIN_FOLDER/$ISO_639_3_LANG.cat.txt"

        for HIST_THRESH_SCORE in "${HIST_THRESH_SCORES[@]}"
        do
            CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"
            REJECTD_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.rejectd.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"

            echo "ISO_639_3_LANG = $ISO_639_3_LANG"
            cat $MONO_CAT_FILE | python $FILTER_CHAR_HISTOGRAM \
                --lang $ISO_639_3_LANG \
                --threshold $THRESH_SCORE \
                --histogram-threshold $HIST_THRESH_SCORE \
                --histograms $HISTOGRAMS_VALID_FOLDER \
                    2> $REJECTD_TRAIN_FILE \
                    1> $CLEANED_TRAIN_FILE &
        done

    done

    wait
}


run_lid_english () {
    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}

        MONO_CAT_FILE="$OLD_DATA_TRAIN_FOLDER/$ISO_639_3_LANG.cat.txt"

        for HIST_THRESH_SCORE in "${HIST_THRESH_SCORES[@]}"
        do
            CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"
            REJECTD_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.rejectd.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"

            RESULT_TXT_CLEANED="$RESULT_FOLDER/result.engparts.$ISO_639_3_LANG.cleaned.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"
            RESULT_TXT_REJECTD="$RESULT_FOLDER/result.engparts.$ISO_639_3_LANG.rejectd.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"

            RESULT_TXT_CLEANED_PREDS="$RESULT_FOLDER/engparts.predictions.$ISO_639_3_LANG.cleaned.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"
            RESULT_TXT_REJECTD_PREDS="$RESULT_FOLDER/engparts.predictions.$ISO_639_3_LANG.rejectd.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"

            RESULT_TXT_CLEANED_PARTS="$RESULT_FOLDER/engparts.$ISO_639_3_LANG.cleaned.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"
            RESULT_TXT_REJECTD_PARTS="$RESULT_FOLDER/engparts.$ISO_639_3_LANG.rejectd.$THRESH_SCORE-$HIST_THRESH_SCORE.txt"

            echo "$CLEANED_TRAIN_FILE"
            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $CLEANED_TRAIN_FILE"
            $RESULT_GATHER > $RESULT_TXT_CLEANED_PREDS
            cat $RESULT_TXT_CLEANED_PREDS | sort | uniq -c > $RESULT_TXT_CLEANED
            cat $RESULT_TXT_CLEANED
            paste $RESULT_TXT_CLEANED_PREDS $CLEANED_TRAIN_FILE | grep "__label__eng" > $RESULT_TXT_CLEANED_PARTS
            rm $RESULT_TXT_CLEANED_PREDS
            echo "   "


            echo "   rejected:"
            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL -"
            cat $REJECTD_TRAIN_FILE | cut -f 4- -d" " | $RESULT_GATHER > $RESULT_TXT_REJECTD_PREDS
            cat $RESULT_TXT_REJECTD_PREDS | sort | uniq -c > $RESULT_TXT_REJECTD
            cat $RESULT_TXT_REJECTD
            echo "   "
            paste $RESULT_TXT_REJECTD_PREDS $REJECTD_TRAIN_FILE | grep "__label__eng" > $RESULT_TXT_REJECTD_PARTS
            rm $RESULT_TXT_REJECTD_PREDS
            echo "   "
        done

    done

    RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $VALID_FILE"

}



prepare_data_2_filter
run_lid_english




