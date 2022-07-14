#!/bin/bash


EXPERIMENT_NAME="2021-09-20-14-14-histogram-baseline"
EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

RESULT_FOLDER="$EXPERIMENT_FOLDER/histograms"

mkdir -p $RESULT_FOLDER

RESULT_TRAIN_FOLDER="$RESULT_FOLDER/train"
RESULT_VALID_FOLDER="$RESULT_FOLDER/valid"

mkdir -p $RESULT_TRAIN_FOLDER
mkdir -p $RESULT_VALID_FOLDER


GOAL124__ISO_639_3_LANGS=("aym" "ceb" "ckb" "cym" "gla" "gle" "hat" "jav" "kac" "kaz" "kea" "kir" "kur" "mlt" "mri" "mya" "que" "sun" "tgk" "uzb" "afr" "amh" "ara" "asm" "ast" "azj" "bel" "ben" "bos" "bul" "cat" "ces" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "ful" "glg" "guj" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ind" "isl" "ita" "jpn" "kam" "kan" "kat" "khm" "kmb" "kon" "kor" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mal" "mar" "mkd" "mlg" "mon" "msa" "nld" "nor" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "ron" "rus" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp" "ssw" "swe" "swh" "tam" "tel" "tgl" "tha" "tir" "tsn" "tur" "ukr" "umb" "urd" "vie" "wol" "xho" "yid" "yor" "yue" "zho" "zul" "eus" "wes" "epo" "kin" "tat" "war" "abk" "ady" "bak" "bem" "bho" "bis" "che" "chv" "ewe" "ewo" "fao" "fij" "fon" "gom" "kal" "grn" "kbp" "kab" "krc" "kik" "lim" "lmo" "lua" "mai" "arn" "min" "xmf" "lus" "mos" "nav" "nia" "pcm" "nno" "oss" "pag" "pap" "roh" "run" "bxr" "smo" "sag" "san" "sat" "srd" "scn" "azb" "alt" "sot" "tah" "bod" "tpi" "tog" "tso" "tum" "tuk" "twi" "udm" "uig" "vec" "zza" "" "cjk" "arz" "ilo")
GOAL124__JW300_LANGS=("ay" "ceb" "" "cy" "" "ga" "ht" "jv" "kac" "kk_Cyrl" "kea" "ky" "" "mt" "" "my" "que" "su" "tg" "uz_Latn" "af" "am" "ar" "as" "" "az" "" "bn" "" "bg" "cat" "cs" "da" "de" "dyu" "el" "en" "et" "fa" "fi" "fr" "" "gl" "gu" "ha" "he" "hi" "hr" "hu" "hy" "ig" "id" "is" "it" "ja" "kam" "kn" "ka" "km" "kmb" "kg" "ko" "lo" "lv" "ln" "lt" "" "lg" "luo" "ml" "mr" "mk" "mg" "mn" "zlm" "nl" "no" "ne" "nso" "nya" "" "om" "or" "pa" "pl" "pt" "" "ro" "ru" "si" "sk" "sl" "sn" "" "" "es" "sq" "sr_Cyrl" "ss" "sv" "sw" "ta" "te" "tl" "th" "ti" "tn" "tr" "uk" "umb" "ur" "vi" "" "xh" "" "yo" "" "" "zu" "eu" "wes" "" "rw" "tt" "war" "ab" "ady" "ba" "bem" "" "bi" "" "cv" "ee" "ewo" "fo" "fj" "fon" "gom" "kl" "gug" "kbp" "kab" "krc" "ki" "" "" "lua" "" "arn" "" "xmf" "lus" "mos" "nv" "nia" "pcm" "" "os" "pag" "pap" "" "run" "" "sm" "sg" "" "" "" "" "" "alt" "st" "ty" "" "tpi" "tog" "ts" "tum" "tk" "tw" "udm" "" "vec" "" "" "cjk" "" "ilo")
GOAL124__LID_187_LANGS=("" "ceb" "ckb" "cy" "gd" "ga" "ht" "jv" "" "kk" "" "ky" "ku" "mt" "" "my" "qu" "su" "tg" "uz" "af" "am" "ar" "as" "ast" "az" "be" "bn" "bs" "bg" "ca" "cs" "da" "de" "" "el" "en" "et" "fa" "fi" "fr" "" "gl" "gu" "ha" "he" "hi" "hr" "hu" "hy" "ig" "id" "is" "it" "ja" "" "kn" "ka" "km" "" "" "ko" "lo" "lv" "ln" "lt" "lb" "lg" "" "ml" "mr" "mk" "mg" "mn" "ms" "nl" "no" "ne" "" "" "oc" "om" "or" "pa" "pl" "pt" "ps" "ro" "ru" "si" "sk" "sl" "sn" "sd" "so" "es" "sq" "sr" "" "sv" "sw" "ta" "te" "tl" "th" "" "tn" "tr" "uk" "" "ur" "vi" "wo" "xh" "yi" "yo" "yue" "zh" "zu" "eu" "" "eo" "" "tt" "war" "" "" "ba" "" "bh" "" "ce" "cv" "" "" "" "" "" "gom" "" "gn" "" "kab" "krc" "" "li" "lmo" "" "mai" "" "min" "xmf" "" "" "" "" "" "nn" "os" "" "" "rm" "" "bxr" "" "" "sa" "sat" "sc" "scn" "azb" "" "" "" "bo" "" "" "" "" "tk" "" "" "ug" "vec" "diq" "nah" "" "arz" "ilo")
GOAL124__FLORES_LANGS=("aym" "ceb" "ckb" "cym" "gla" "gle" "hat" "jav" "kac" "kaz" "kea" "kir" "kur" "mlt" "mri" "mya" "que" "sun" "tgk" "uzb" "afr" "amh" "ara" "asm" "ast" "azj" "bel" "ben" "bos" "bul" "cat" "ces" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "ful" "glg" "guj" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ind" "isl" "ita" "jpn" "kam" "kan" "kat" "khm" "kmb" "kon" "kor" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mal" "mar" "mkd" "mlg" "mon" "msa" "nld" "nob" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "ron" "rus" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp" "ssw" "swe" "swh" "tam" "tel" "tgl" "tha" "tir" "tsn" "tur" "ukr" "umb" "urd" "vie" "wol" "xho" "yid" "yor" "yue" "zho_simpl" "zul" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "")



FLORES_DEV_DIR="/large_experiments/mmt/flores101/dev"
CREATE_CHAR_HISTOGRAM_SCRIPT="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/filtering/create_char_histogram.py"
DATA_TRAIN_DIR="/large_experiments/mmt/lidruns/2021-09-18-00-21-goal124-baseline/data/train/"

create_train_histogram() {

    HISTOGRAM_OUT_DIR=""
    HISTOGRAM_LANG=$1
    echo "Lang = $HISTOGRAM_LANG"

    $CREATE_CHAR_HISTOGRAM_SCRIPT $DATA_TRAIN_DIR/$HISTOGRAM_LANG.train.mono.txt $RESULT_TRAIN_FOLDER/$HISTOGRAM_LANG
}


create_train_histograms(){
    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        JW300_LANG=${GOAL124__JW300_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}
        FLORES_LANG=${GOAL124__FLORES_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
        fi

        create_train_histogram $ISO_639_3_LANG &
    done

    wait
}

create_valid_histograms(){
    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        JW300_LANG=${GOAL124__JW300_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}
        FLORES_LANG=${GOAL124__FLORES_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
        fi

        HISTOGRAM_LANG="$ISO_639_3_LANG"

        if [ ! -z "$FLORES_LANG" ]
        then
            FILES_FOUND=$(ls -1 $FLORES_DEV_DIR/${FLORES_LANG}.dev | head -n 1)
            FILES_FOUND_NUM=$(ls -1 $FLORES_DEV_DIR/${FLORES_LANG}.dev | wc -l)

            $CREATE_CHAR_HISTOGRAM_SCRIPT $FILES_FOUND $RESULT_VALID_FOLDER/$HISTOGRAM_LANG
        else
            echo "Flores not available for $ISO_639_3_LANG"
            TRAIN_FILE="$DATA_TRAIN_DIR/$ISO_639_3_LANG.train.mono.txt"
            if [ -f "$TRAIN_FILE" ]; then
                echo "will use $TRAIN_FILE"

                $CREATE_CHAR_HISTOGRAM_SCRIPT $TRAIN_FILE $RESULT_VALID_FOLDER/$HISTOGRAM_LANG
            else
                echo "ERROR: File not found $TRAIN_FILE"
                return 1
            fi
        fi
    done

    wait
}

create_train_histograms
create_valid_histograms

