#!/bin/bash


EXPERIMENT_NAME="2022-02-17-23-56-msrc-2"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER


DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"
LOGS_FOLDER="$EXPERIMENT_FOLDER/logs"

mkdir -p $RESULT_FOLDER
mkdir -p $LOGS_FOLDER

FLORES_DEV_DIR="/large_experiments/nllb/mmt/flores101/dev"
FLORES_DEVTEST_DIR="/large_experiments/nllb/mmt/flores101/devtest"
FLORES_BETA_DEV_DIR="/large_experiments/nllb/mmt/flores101_beta/dev"
FLORES_BETA_DEVTEST_DIR="/large_experiments/nllb/mmt/flores101_beta/devtest"
JW300_DETOK_DIR="/large_experiments/nllb/mmt/data/monolingual/lid/jw300/detok"
LID187_DATA_DIR="/private/home/celebio/lid187_data_2"
FBSEED_DATA_DIR="/private/home/celebio/fbseed20220214_data_2"

TRAIN_DIR="$DATA_FOLDER/train"
TRAIN_FILTER_DIR="$DATA_FOLDER/train_filter"
TRAIN_HIST_DIR="$DATA_FOLDER/histograms"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TRAIN_DIR
mkdir -p $TRAIN_FILTER_DIR
mkdir -p $TRAIN_HIST_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR

NLLB_2022__ISO3=("abk" "ace_Arab" "ace_Latn" "acm" "acq" "ady" "aeb" "afr" "aka" "alt" "amh" "apc" "ara_Arab" "ara_Latn" "arn" "ars" "ary" "arz" "asm" "ast" "awa" "ayr" "azb" "azj" "bak" "bam" "ban" "bel" "bem" "ben" "bho" "bis" "bjn_Arab" "bjn_Latn" "bod" "bos" "bug" "bul" "bxr" "cat" "ceb" "ces" "che" "chv" "cjk" "ckb" "crh_Latn" "cym" "dan" "deu" "dik" "diq" "dyu" "dzo" "ell" "eng" "epo" "est" "eus" "ewe" "ewo" "fao" "fas" "fij" "fin" "fon" "fra" "fur" "fuv" "gla" "gle" "glg" "gom" "grn" "guj" "hat" "hau" "heb" "hin" "hne" "hrv" "hun" "hye" "ibo" "ilo" "ind" "isl" "ita" "jav" "jpn" "kab" "kac" "kal" "kam" "kan" "kas_Arab" "kas_Deva" "kat" "kau_Arab" "kau_Latn" "kaz" "kbp" "kea" "khm" "kik" "kin" "kir" "kmb" "kon" "kor" "krc" "kur" "lao" "lav" "lij" "lim" "lin" "lit" "lmo" "ltg" "ltz" "lua" "lug" "luo" "lus" "mag" "mai" "mal" "mar" "min_Latn" "mkd" "mlg" "mlt" "mni_Mtei" "mon" "mos" "mri" "msa" "mya" "nav" "nia" "nld" "nno" "nob" "npi" "nso" "nus" "nya" "oci" "orm" "ory" "oss" "pag" "pan" "pap" "pcm" "pol" "por" "prs" "pus" "que" "roh" "ron" "run" "rus" "sag" "san" "sat" "scn" "shn" "sin" "slk" "slv" "smo" "sna" "snd" "som" "sot" "spa" "sqi" "srd" "srp_Cyrl" "ssw" "sun" "swe" "swh" "szl" "tah" "tam" "tat_Cyrl" "tel" "tgk" "tgl" "tha" "tir" "tmh_Latn" "tmh_Tfng" "ton" "tpi" "tsn" "tso" "tuk" "tum" "tur" "twi" "tzm" "udm" "uig" "ukr" "umb" "urd" "uzb" "vec" "vie" "war" "wes" "wol" "xho" "xmf" "yid" "yor" "yue" "zho_Hans" "zho_Hant" "zul")
NLLB_2022__FLORES_CODE=("" "" "" "ara-IQ" "ara-YE" "" "ara-TN" "afr" "aka" "" "amh" "ara-LB" "ara" "ara_Latn" "" "ara-SA" "ara-MA" "arz" "asm" "ast" "awa" "aym" "" "azj" "bak" "bam" "" "bel" "bem" "ben" "bho" "" "" "" "bod" "bos" "" "bul" "" "cat" "ceb" "ces" "" "" "cjk" "ckb" "" "cym" "dan" "deu" "" "" "dyu" "dzo" "ell" "eng" "epo" "est" "eus" "ewe" "" "fao" "fas" "fij" "fin" "fon" "fra" "" "ful" "gla" "gle" "glg" "" "grn" "guj" "hat" "hau" "heb" "hin" "hne" "hrv" "hun" "hye" "ibo" "ilo" "ind" "isl" "ita" "jav" "jpn" "kab-Latn" "kac" "" "kam" "kan" "kas-Arab" "kas-Deva" "kat" "kau-Arab" "kau-Latn" "kaz" "" "kea" "khm" "kik" "kin" "kir" "kmb" "kon" "kor" "" "kur" "lao" "lav" "" "" "lin" "lit" "" "" "ltz" "lua" "lug" "luo" "" "mag" "mai" "mal" "mar" "" "mkd" "mlg" "mlt" "mni-Mtei" "mon" "" "mri" "msa" "mya" "" "" "nld" "nno" "nob" "npi" "nso" "" "nya" "oci" "orm" "ory" "" "" "pan" "" "" "pol" "por" "prs" "pus" "que" "" "ron" "run" "rus" "sag" "san" "sat" "scn" "shn" "sin" "slk" "slv" "smo" "sna" "snd" "som" "sot" "spa" "sqi" "srd" "srp" "ssw" "sun" "swe" "swh" "szl" "" "tam" "tat-Cyrl" "tel" "tgk" "tgl" "tha" "tir" "" "" "" "tpi" "tsn" "" "tuk-Latn" "" "tur" "twi" "" "" "uig" "ukr" "umb" "urd" "uzb" "" "vie" "" "" "wol" "xho" "" "yid" "yor" "yue" "zho_simpl" "zho_trad" "zul")
NLLB_2022__SEEDDATA_CODE=("" "ace_Arab" "ace_Latn" "" "" "" "" "" "" "" "" "" "" "" "" "" "ary" "arz" "" "" "" "" "" "" "" "bam" "ban" "" "" "" "bho" "" "bjn_Arab" "bjn_Latn" "" "" "bug" "" "" "" "" "" "" "" "" "" "crh_Latn" "" "" "" "dik" "" "" "dzo" "" "" "" "" "" "" "" "" "" "" "" "" "" "fur" "fuv" "" "" "" "" "grn" "" "" "" "" "" "hne" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "kas_Arab" "kas_Deva" "" "kau_Arab" "kau_Latn" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "lij" "" "" "" "" "ltg" "" "" "" "" "" "mag" "" "" "" "" "" "" "" "mni_Mtei" "" "" "mri" "" "" "" "" "" "" "" "" "" "nus" "" "" "" "" "" "" "" "" "" "" "" "prs" "pus" "" "" "" "" "" "" "" "" "" "shn" "" "" "" "" "" "" "" "" "" "" "srd" "" "" "" "" "" "szl" "" "" "" "" "" "" "" "" "tmh_Latn" "tmh_Tfng" "" "" "" "" "" "" "" "" "tzm_Tfng" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "")
NLLB_2022__ISO15924=("Cyrl" "Arab" "Latn" "Arab" "Arab" "Cyrl" "Arab" "Latn" "Latn" "Cyrl" "Ethi" "Arab" "Arab" "Latn" "Latn" "Arab" "Arab" "Arab" "Beng" "Latn" "Deva" "Latn" "Arab" "Arab" "Cyrl" "Latn" "Latn" "Cyrl" "Latn" "Latn" "Deva" "" "Arab" "Latn" "Tibt" "Latn" "Latn" "Cyrl" "Cyrl" "Latn" "Latn" "Latn" "Cyrl" "Cyrl" "Latn" "Arab" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Tibt" "Grek" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Arab" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Gujr" "Latn" "Latn" "Hebr" "Deva" "Deva" "Latn" "Latn" "Armn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "" "Latn" "Latn" "Latn" "Latn" "Knda" "Arab" "Deva" "Geor" "Arab" "Latn" "Cyrl" "Latn" "Latn" "Khmr" "Latn" "Latn" "Cyrl" "Latn" "Latn" "Hang" "" "Latn" "Laoo" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Deva" "Deva" "Mlym" "Deva" "Latn" "Cyrl" "Latn" "Latn" "Beng" "Cyrl" "Latn" "Latn" "Latn" "Mymr" "Latn" "Latn" "Latn" "Latn" "Latn" "Deva" "Latn" "Latn" "Latn" "Latn" "Latn" "Orya" "Cyrl" "Latn" "Guru" "Latn" "Latn" "Latn" "Latn" "Arab" "Arab" "Latn" "Latn" "Latn" "Latn" "Cyrl" "Latn" "Deva" "Olck" "Latn" "Mymr" "Sinh" "Latn" "Latn" "Latn" "Latn" "Arab" "Latn" "Latn" "Latn" "Latn" "Latn" "Cyrl" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Taml" "Cyrl" "Telu" "Cyrl" "Latn" "Thai" "Ethi" "Latn" "Tfng" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Tfng" "Cyrl" "" "Cyrl" "Latn" "Arab" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Geor" "Hebr" "Latn" "" "" "" "Latn")
NLLB_2022__OTHER_JW300_CODE=("ab" "" "" "" "" "ady" "" "af" "" "alt" "am" "" "ar" "" "arn" "" "" "" "as" "" "" "ay" "" "az" "ba" "" "" "" "bem" "bn" "" "bi" "" "" "" "" "" "bg" "" "cat" "ceb" "cs" "" "cv" "cjk" "" "" "cy" "da" "de" "" "" "dyu" "" "el" "en" "" "et" "eu" "ee" "ewo" "fo" "fa" "fj" "fi" "fon" "fr" "" "" "" "ga" "gl" "gom" "gug" "gu" "ht" "ha" "he" "hi" "" "hr" "hu" "hy" "ig" "ilo" "id" "is" "it" "jv" "" "kab" "kac" "kl" "kam" "kn" "" "" "ka" "" "" "kk_Cyrl" "kbp" "kea" "km" "ki" "rw" "ky" "kmb" "kg" "ko" "krc" "" "lo" "lv" "" "" "ln" "lt" "" "" "" "lua" "lg" "luo" "lus" "" "" "ml" "mr" "" "mk" "mg" "mt" "" "mn" "mos" "" "zlm" "my" "nv" "nia" "nl" "" "no" "ne" "nso" "" "nya" "" "om" "or" "os" "pag" "pa" "pap" "pcm" "pl" "pt" "" "" "que" "" "ro" "run" "ru" "sg" "" "" "" "" "si" "sk" "sl" "sm" "sn" "" "" "st" "es" "sq" "" "sr_Cyrl" "ss" "su" "sv" "sw" "" "ty" "ta" "tt" "te" "tg" "tl" "th" "ti" "" "" "tog" "tpi" "tn" "ts" "tk" "tum" "tr" "tw" "" "udm" "" "uk" "umb" "ur" "uz_Latn" "vec" "vi" "war" "wes" "" "xh" "xmf" "" "yo" "" "" "" "zu")
NLLB_2022__OTHER_LID187_CODE=("" "" "" "" "" "" "" "af" "" "" "am" "" "ar" "" "" "" "" "arz" "as" "ast" "" "" "azb" "az" "ba" "" "" "be" "" "bn" "bh" "" "" "" "bo" "bs" "" "bg" "bxr" "ca" "ceb" "cs" "ce" "cv" "" "ckb" "" "cy" "da" "de" "" "diq" "" "" "el" "en" "eo" "et" "eu" "" "" "" "fa" "" "fi" "" "fr" "" "" "gd" "ga" "gl" "gom" "gn" "gu" "ht" "ha" "he" "hi" "" "hr" "hu" "hy" "ig" "ilo" "id" "is" "it" "jv" "ja" "kab" "" "" "" "kn" "" "" "ka" "" "" "kk" "" "" "km" "" "" "ky" "" "" "ko" "krc" "ku" "lo" "lv" "" "li" "ln" "lt" "lmo" "" "lb" "" "lg" "" "" "" "mai" "ml" "mr" "min" "mk" "mg" "mt" "" "mn" "" "" "ms" "my" "" "" "nl" "nn" "no" "ne" "" "" "" "oc" "om" "or" "os" "" "pa" "" "" "pl" "pt" "" "ps" "qu" "rm" "ro" "" "ru" "" "sa" "sat" "scn" "" "si" "sk" "sl" "" "sn" "sd" "so" "" "es" "sq" "sc" "sr" "" "su" "sv" "sw" "" "" "ta" "tt" "te" "tg" "tl" "th" "" "" "" "" "" "tn" "" "tk" "" "tr" "" "" "" "ug" "uk" "" "ur" "uz" "vec" "vi" "war" "" "wo" "xh" "xmf" "yi" "yo" "yue" "" "" "zu")


CREATE_CHAR_HISTOGRAM_SCRIPT="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/filtering/create_char_histogram.py"
FILTER_CHAR_HISTOGRAM="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/filtering/filter_char_histogram.py"
SCRIPT_DETECTOR="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/classifier/script_detector.py"
FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"
LID_ENG_MODEL="/large_experiments/nllb/mmt/lidruns/2021-10-05-11-53-english/result/model.8.8.bin"
UPSAMPLING_COMPUTE="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/utils/compute_upsample.py"
CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"


remove_punc_numbers () {
  sed "s/[0-9:.\'(),\"]*//g" | sed -e "s/[[:punct:]]\+//g"
}


prepare_valid_for_histograms () {
    for i in "${!NLLB_2022__ISO3[@]}"; do
        NLLB_LANG=${NLLB_2022__ISO3[i]}
        FLORES_LANG=${NLLB_2022__FLORES_CODE[i]}
        FBSEED_LANG=${NLLB_2022__SEEDDATA_CODE[i]}
        LANG_SCRIPT=${NLLB_2022__ISO15924[i]}
        JW300_LANG=${NLLB_2022__OTHER_JW300_CODE[i]}
        LID_187_LANG=${NLLB_2022__OTHER_LID187_CODE[i]}

        MONO_CAT_FILE="$TRAIN_FILTER_DIR/$NLLB_LANG.cat.forhist.txt"
        > $MONO_CAT_FILE

        chosen_source=""

        if [ -z "$chosen_source" ]
        then
            FLORES_DEV_FILE="$FLORES_DEV_DIR/$FLORES_LANG.dev"
            if [ -f "$FLORES_DEV_FILE" ]; then
                chosen_source="${chosen_source}-flores"

                cat $FLORES_DEV_FILE | remove_punc_numbers >> $MONO_CAT_FILE
            fi
        fi

        if [ -z "$chosen_source" ]
        then
            FLORES_DEV_FILE="$FLORES_BETA_DEV_DIR/$FLORES_LANG.dev"
            if [ -f "$FLORES_DEV_FILE" ]; then
                chosen_source="${chosen_source}-flores_beta"

                cat $FLORES_DEV_FILE | remove_punc_numbers >> $MONO_CAT_FILE
            fi
        fi


        if [ -z "$chosen_source" ]
        then
            if [ ! -z "$FBSEED_LANG" ]
            then
                FBSEED_FILE="$FBSEED_DATA_DIR/fbseed20220214.$FBSEED_LANG.txt"
                if [ -f "$FBSEED_FILE" ]
                then
                    chosen_source="${chosen_source}-fbseed"

                    cat $FBSEED_FILE | remove_punc_numbers >> $MONO_CAT_FILE
                fi
            fi
            if [ ! -z "$JW300_LANG" ]
            then
                JW300_FILE="$JW300_DETOK_DIR/$JW300_LANG.detok.txt"
                if [ -f "$JW300_FILE" ]; then
                    chosen_source="${chosen_source}-jw300"

                    cat $JW300_FILE | remove_punc_numbers >> $MONO_CAT_FILE
                fi
            fi

            if [ ! -z "$LID_187_LANG" ]
            then
                LID187_FILE="$LID187_DATA_DIR/$LID_187_LANG.txt"
                if [ -f "$LID187_FILE" ]; then
                    chosen_source="${chosen_source}-lid187"

                    cat $LID187_FILE | remove_punc_numbers >> $MONO_CAT_FILE
                fi
            fi
        fi



        if [ -z "$chosen_source" ]
        then
            chosen_source="no-source"
        fi

        printf "%-20s %s \n" $NLLB_LANG $chosen_source
    done

}

create_valid_histograms(){
    for i in "${!NLLB_2022__ISO3[@]}"; do
        NLLB_LANG=${NLLB_2022__ISO3[i]}
        FLORES_LANG=${NLLB_2022__FLORES_CODE[i]}
        FBSEED_LANG=${NLLB_2022__SEEDDATA_CODE[i]}
        LANG_SCRIPT=${NLLB_2022__ISO15924[i]}
        JW300_LANG=${NLLB_2022__OTHER_JW300_CODE[i]}
        LID_187_LANG=${NLLB_2022__OTHER_LID187_CODE[i]}

        MONO_CAT_FILE="$TRAIN_FILTER_DIR/$NLLB_LANG.cat.forhist.txt"

        printf "histogram creation: %-10s \n" $NLLB_LANG
        $CREATE_CHAR_HISTOGRAM_SCRIPT $MONO_CAT_FILE $TRAIN_HIST_DIR/$NLLB_LANG

    done
}

prepare_data_1 () {
    for i in "${!NLLB_2022__ISO3[@]}"; do
        NLLB_LANG=${NLLB_2022__ISO3[i]}
        FLORES_LANG=${NLLB_2022__FLORES_CODE[i]}
        FBSEED_LANG=${NLLB_2022__SEEDDATA_CODE[i]}
        LANG_SCRIPT=${NLLB_2022__ISO15924[i]}
        JW300_LANG=${NLLB_2022__OTHER_JW300_CODE[i]}
        LID_187_LANG=${NLLB_2022__OTHER_LID187_CODE[i]}

        MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"
        > $MONO_CAT_FILE

        chosen_source=""

        STAT_FBSEED="0"
        STAT_JW300="0"
        STAT_LID187="0"
        STAT_FLORES="0"

        if [ ! -z "$FBSEED_LANG" ]
        then
            FBSEED_FILE="$FBSEED_DATA_DIR/fbseed20220214.$FBSEED_LANG.txt"
            if [ -f "$FBSEED_FILE" ]
            then
                chosen_source="${chosen_source}-fbseed"

                STAT_FBSEED=$(wc -l $FBSEED_FILE | cut -f 1 -d" ")
                cat $FBSEED_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
            fi
        fi

        if [ -z "$chosen_source" ]
        then

            if [ ! -z "$JW300_LANG" ]
            then
                JW300_FILE="$JW300_DETOK_DIR/$JW300_LANG.detok.txt"
                if [ -f "$JW300_FILE" ]; then
                    chosen_source="${chosen_source}-jw300"

                    STAT_JW300=$(wc -l $JW300_FILE | cut -f 1 -d" ")
                    cat $JW300_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
                fi
            fi

            if [ ! -z "$LID_187_LANG" ]
            then
                LID187_FILE="$LID187_DATA_DIR/$LID_187_LANG.txt"
                if [ -f "$LID187_FILE" ]; then
                    chosen_source="${chosen_source}-lid187"

                    STAT_LID187=$(wc -l $LID187_FILE | cut -f 1 -d" ")
                    cat $LID187_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
                fi
            fi
        fi

        if [ -z "$chosen_source" ]
        then
            FLORES_DEV_FILE="$FLORES_DEV_DIR/$FLORES_LANG.dev"
            if [ -f "$FLORES_DEV_FILE" ]; then
                chosen_source="${chosen_source}-flores"

                # chosen_source="flores-only"

                STAT_FLORES=$(wc -l $FLORES_DEV_FILE | cut -f 1 -d" ")
                cat $FLORES_DEV_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
            fi
        fi

        if [ -z "$chosen_source" ]
        then
            FLORES_DEV_FILE="$FLORES_BETA_DEV_DIR/$FLORES_LANG.dev"
            if [ -f "$FLORES_DEV_FILE" ]; then
                chosen_source="${chosen_source}-flores_beta"
                # chosen_source="flores-only"

                STAT_FLORES=$(wc -l $FLORES_DEV_FILE | cut -f 1 -d" ")
                cat $FLORES_DEV_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
            fi
        fi

        if [ -z "$chosen_source" ]
        then
            chosen_source="no-source"
        fi

        # printf "%-20s %s \n" $NLLB_LANG $chosen_source
        printf "%-20s %-30s flores:%-10s  fbseed:%-10s jw300:%-10s lid187:%-10s \n" $NLLB_LANG $chosen_source $STAT_FLORES $STAT_FBSEED $STAT_JW300 $STAT_LID187

        # printf "%-20s %-10s %-10s %-10s %-10s %-10s \n" $NLLB_LANG $FLORES_LANG $FBSEED_LANG $LANG_SCRIPT $JW300_LANG $LID_187_LANG
    done
}

prepare_data_filter_step_1 () {
    for i in "${!NLLB_2022__ISO3[@]}"; do
        NLLB_LANG=${NLLB_2022__ISO3[i]}

        if [ "$NLLB_LANG" == "zho_Hans" ] || [ "$NLLB_LANG" == "zho_Hant" ] || [ "$NLLB_LANG" == "jpn" ]
        then
            echo "Skipping $NLLB_LANG"
            continue
        fi

        MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"

        THRESH_SCORE="0.8"
        CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step1.txt"
        REJECTD_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.rejectd.step1.txt"

        echo "NLLB_LANG = $NLLB_LANG"
        cat $MONO_CAT_FILE | python $FILTER_CHAR_HISTOGRAM \
            --lang $NLLB_LANG \
            --threshold $THRESH_SCORE \
            --histogram-threshold 0.95 \
            --histograms $TRAIN_HIST_DIR \
                2> $REJECTD_TRAIN_FILE \
                1> $CLEANED_TRAIN_FILE &
    done

    wait

    # for i in "${!NLLB_2022__ISO3[@]}"; do
    #     NLLB_LANG=${NLLB_2022__ISO3[i]}

    #     if [ "$NLLB_LANG" == "zho_Hans" ] || [ "$NLLB_LANG" == "zho_Hant" ] || [ "$NLLB_LANG" == "jpn" ]
    #     then
    #         echo "Skipping $NLLB_LANG"
    #         continue
    #     fi

    #     STAT_CLEANED=$(wc -l $CLEANED_TRAIN_FILE | cut -f 1 -d" ")
    #     STAT_REJECTD=$(wc -l $REJECTD_TRAIN_FILE | cut -f 1 -d" ")

    #     STAT_SUM=$(expr $STAT_CLEANED \+ $STAT_REJECTD)
    #     STAT_REJECTED_PERCENTAGE=$(echo "scale = 4; $STAT_REJECTD * 1.0 / $STAT_SUM" | bc -l)
    #     printf "%-20s sum:%-10s cleaned:%-10s rejected:%-10s rejected-percentage:%-10s \n" $NLLB_LANG $STAT_SUM $STAT_CLEANED $STAT_REJECTD $STAT_REJECTED_PERCENTAGE
    # done
}

prepare_data_filter_step_1_jpn () {
    NLLB_LANG="jpn"
    MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"

    rm $MONO_CAT_FILE

    LANG_SCRIPT_PRED_FILE="$TRAIN_DIR/$NLLB_LANG.script.step1.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step1.txt"
    REJECTD_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.rejectd.step1.txt"

    cat /large_experiments/nllb/mmt/data/monolingual/ccmatrix.v1/jpn1.txt | head -n 1000000 | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' > $CLEANED_TRAIN_FILE
    > $REJECTD_TRAIN_FILE
}

prepare_data_filter_step_1_zho_Hant () {
    NLLB_LANG="zho_Hant"
    MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"

    LANG_SCRIPT_PRED_FILE="$TRAIN_DIR/$NLLB_LANG.script.step1.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step1.txt"
    REJECTD_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.rejectd.step1.txt"

    cp $MONO_CAT_FILE $CLEANED_TRAIN_FILE
    > $REJECTD_TRAIN_FILE
}
prepare_data_filter_step_1_zho_Hans () {
    NLLB_LANG="zho_Hans"
    MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"

    LANG_SCRIPT_PRED_FILE="$TRAIN_DIR/$NLLB_LANG.script.step1.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step1.txt"
    REJECTD_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.rejectd.step1.txt"

    cp $MONO_CAT_FILE $CLEANED_TRAIN_FILE
    > $REJECTD_TRAIN_FILE
}



prepare_data_filter_step_3_lang() {
    NLLB_LANG=$1

    PREV_CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step1.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step3.txt"
    REJECTD_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.rejectd.step3.txt"

    RESULT_TXT_CLEANED_PREDS="$TRAIN_DIR/engparts.predictions.$NLLB_LANG.cleaned.step3.txt"

    RESULT_GATHER="$FASTTEXT_BIN predict $LID_ENG_MODEL -"
    cat $PREV_CLEANED_TRAIN_FILE | cut -f 2- -d" " | $RESULT_GATHER > $RESULT_TXT_CLEANED_PREDS

    paste $RESULT_TXT_CLEANED_PREDS $PREV_CLEANED_TRAIN_FILE | grep "__label__neng" | cut -f 2- > $CLEANED_TRAIN_FILE
    paste $RESULT_TXT_CLEANED_PREDS $PREV_CLEANED_TRAIN_FILE | grep "__label__eng" | cut -f 2- > $REJECTD_TRAIN_FILE

    rm $RESULT_TXT_CLEANED_PREDS
    echo "    FINISHED eng lang filtering for $NLLB_LANG"
}

prepare_data_filter_step_3 () {
    LENGTH=${#NLLB_2022__ISO3[@]}
    BATCH_SIZE=40
    i=0

    while [ $i -lt $LENGTH ]
    do
        for (( j=0; j<${BATCH_SIZE}; j++ ));
        do
            if [ $i -ge $LENGTH ]; then
                break
            fi

            NLLB_LANG=${NLLB_2022__ISO3[i]}

            echo "eng lang filtering for $NLLB_LANG"
            prepare_data_filter_step_3_lang $NLLB_LANG &

            let "i+=1"
        done

        wait
    done

    mv "$TRAIN_DIR/eng.rejectd.step3.txt" "$TRAIN_DIR/eng.rejectd.step3.txt.aux"
    mv "$TRAIN_DIR/eng.cleaned.step3.txt" "$TRAIN_DIR/eng.rejectd.step3.txt"
    mv "$TRAIN_DIR/eng.rejectd.step3.txt.aux" "$TRAIN_DIR/eng.cleaned.step3.txt"
}

modify_lang_label () {
    ORIGINAL_FILE=$1
    RESULT_FILE=$2
    cat $ORIGINAL_FILE \
        | sed 's/__label__ary/__label__ara_Arab/g' \
        | sed 's/__label__ars/__label__ara_Arab/g' \
        | sed 's/__label__acq/__label__ara_Arab/g' \
        | sed 's/__label__acm/__label__ara_Arab/g' \
        | sed 's/__label__aeb/__label__ara_Arab/g' \
        | sed 's/__label__apc/__label__ara_Arab/g' \
        | sed 's/__label__arz/__label__ara_Arab/g' \
     > $RESULT_FILE
}


group_train_valid_data_langs() {
    for NLLB_LANG in ary ars acq acm aeb apc arz
    do
        cp "$TRAIN_DIR/$NLLB_LANG.cleaned.step3.txt" "$TRAIN_DIR/$NLLB_LANG.cleaned.step3.old.txt"
        modify_lang_label "$TRAIN_DIR/$NLLB_LANG.cleaned.step3.txt" "$TRAIN_DIR/$NLLB_LANG.cleaned.step3.mod.txt"
        cp "$TRAIN_DIR/$NLLB_LANG.cleaned.step3.mod.txt" "$TRAIN_DIR/$NLLB_LANG.cleaned.step3.txt"
    done
}

merge_grouped_files() {
    cat \
        "$TRAIN_DIR/ary.cleaned.step3.txt" \
        "$TRAIN_DIR/ars.cleaned.step3.txt" \
        "$TRAIN_DIR/acq.cleaned.step3.txt" \
        "$TRAIN_DIR/acm.cleaned.step3.txt" \
        "$TRAIN_DIR/aeb.cleaned.step3.txt" \
        "$TRAIN_DIR/apc.cleaned.step3.txt" \
        "$TRAIN_DIR/arz.cleaned.step3.txt" \
        "$TRAIN_DIR/ara_Arab.cleaned.step3.txt" \
        > "$TRAIN_DIR/all.ara_Arab.cleaned.step3.txt"
}


prepare_data_combine() {
    UNSHUFFLED_TRAIN_FILE="$TRAIN_DIR/all.unshuf.txt"
    > $UNSHUFFLED_TRAIN_FILE
    UNSHUFFLED_VALID_FILE="$VALID_DIR/all.unshuf.txt"
    > $UNSHUFFLED_VALID_FILE
    ALL_TRAIN_FILE="$TRAIN_DIR/all.txt"
    > $ALL_TRAIN_FILE
    ALL_VALID_FILE="$VALID_DIR/all.txt"
    > $ALL_VALID_FILE

    for i in "${!NLLB_2022__ISO3[@]}"; do
        NLLB_LANG=${NLLB_2022__ISO3[i]}

        if [ "$NLLB_LANG" == "ary" ] || [ "$NLLB_LANG" == "ars" ] || [ "$NLLB_LANG" == "acq" ] || [ "$NLLB_LANG" == "acm" ] || [ "$NLLB_LANG" == "aeb" ] || [ "$NLLB_LANG" == "apc" ] || [ "$NLLB_LANG" == "arz" ]
        then
            echo "Skipping $NLLB_LANG"
            continue
        fi

        VALID_SIZE=100000
        VALID_COMPARE_SIZE=$(expr $VALID_SIZE \* 4)

        MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step3.txt"
        MONO_SHUF_FILE="$TRAIN_DIR/$NLLB_LANG.shuf.txt"

        if [ "$NLLB_LANG" == "ara_Arab" ]
        then
            MONO_CAT_FILE="$TRAIN_DIR/all.ara_Arab.cleaned.step3.txt"
        fi

        shuf $MONO_CAT_FILE > $MONO_SHUF_FILE

        NUMBER_OF_LINES=$(cat $MONO_CAT_FILE | wc -l)


        if [ "$NUMBER_OF_LINES" -lt "$VALID_COMPARE_SIZE" ]; then
            VALID_SIZE=$(expr $NUMBER_OF_LINES / 10)
        fi
        TRAIN_NB_LINES=$(expr $NUMBER_OF_LINES - $VALID_SIZE)

        TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.punct.txt"
        VALID_FILE="$VALID_DIR/$NLLB_LANG.valid.mono.txt"

        # echo "head -n $TRAIN_NB_LINES $MONO_CAT_FILE > $TRAIN_FILE"
        # shuf $MONO_CAT_FILE | head -n $TRAIN_NB_LINES > $TRAIN_FILE
        head -n $TRAIN_NB_LINES $MONO_SHUF_FILE > $TRAIN_FILE
        tail -n $VALID_SIZE $MONO_SHUF_FILE > $VALID_FILE

        rm $MONO_SHUF_FILE

        printf "%-20s train size: %-10s    valid size: %-10s \n" $NLLB_LANG $TRAIN_NB_LINES $VALID_SIZE

        cat $VALID_FILE >> $UNSHUFFLED_VALID_FILE
    done
    shuf $UNSHUFFLED_VALID_FILE > $ALL_VALID_FILE
}


prepare_data_remove_punct_train_lang() {
    NLLB_LANG=$1

    if [ "$NLLB_LANG" == "ary" ] || [ "$NLLB_LANG" == "ars" ] || [ "$NLLB_LANG" == "acq" ] || [ "$NLLB_LANG" == "acm" ] || [ "$NLLB_LANG" == "aeb" ] || [ "$NLLB_LANG" == "apc" ] || [ "$NLLB_LANG" == "arz" ]
    then
        echo "    Skipping $NLLB_LANG"
    else
        PREV_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.punct.txt"
        NEW_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.no-punct.txt"
        cat $PREV_TRAIN_FILE | cut -f 2- -d" " | remove_punc_numbers | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' > $NEW_TRAIN_FILE
    fi

    echo "    FINISHED remove punct. for $NLLB_LANG"
}

prepare_data_remove_punct_train () {
    LENGTH=${#NLLB_2022__ISO3[@]}
    BATCH_SIZE=120
    i=0

    while [ $i -lt $LENGTH ]
    do
        for (( j=0; j<${BATCH_SIZE}; j++ ));
        do
            if [ $i -ge $LENGTH ]; then
                break
            fi

            NLLB_LANG=${NLLB_2022__ISO3[i]}

            echo "remove punct. for $NLLB_LANG"
            prepare_data_remove_punct_train_lang $NLLB_LANG &

            let "i+=1"
        done

        wait
    done
}

prepare_data_combine_train () {
    UNSHUFFLED_TRAIN_FILE="$TRAIN_DIR/all.unshuf.txt"
    > $UNSHUFFLED_TRAIN_FILE
    UNSHUFFLED_VALID_FILE="$VALID_DIR/all.unshuf.txt"
    > $UNSHUFFLED_VALID_FILE
    ALL_TRAIN_FILE="$TRAIN_DIR/all.txt"
    > $ALL_TRAIN_FILE
    ALL_VALID_FILE="$VALID_DIR/all.txt"
    > $ALL_VALID_FILE

    for i in "${!NLLB_2022__ISO3[@]}"; do
        NLLB_LANG=${NLLB_2022__ISO3[i]}

        if [ "$NLLB_LANG" == "ary" ] || [ "$NLLB_LANG" == "ars" ] || [ "$NLLB_LANG" == "acq" ] || [ "$NLLB_LANG" == "acm" ] || [ "$NLLB_LANG" == "aeb" ] || [ "$NLLB_LANG" == "apc" ] || [ "$NLLB_LANG" == "arz" ]
        then
            echo "Skipping $NLLB_LANG"
            continue
        fi

        TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.no-punct.txt"
        cat $TRAIN_FILE >> $UNSHUFFLED_TRAIN_FILE
    done
    shuf $UNSHUFFLED_TRAIN_FILE > $ALL_TRAIN_FILE

}

prepare_data_combine_upsampl () {
    UNSHUFFLED_TRAIN_UPSAMPLED_FILE="$TRAIN_DIR/all.unshuf.upsampl.txt"
    > $UNSHUFFLED_TRAIN_UPSAMPLED_FILE
    ALL_TRAIN_UPSAMPLED_FILE="$TRAIN_DIR/all.upsampl.txt"
    > $ALL_TRAIN_UPSAMPLED_FILE

    UNSHUFFLED_TRAIN_SAMELIMIT_FILE="$TRAIN_DIR/all.unshuf.samelimit.txt"
    > $UNSHUFFLED_TRAIN_SAMELIMIT_FILE
    ALL_TRAIN_SAMELIMIT_FILE="$TRAIN_DIR/all.samelimit.txt"
    > $ALL_TRAIN_SAMELIMIT_FILE


    NB_LINES_FILE="$TRAIN_DIR/nblines.txt"
    NB_LINES_UPSAMPLED_FILE="$TRAIN_DIR/nblines_upsampled.txt"
    > $NB_LINES_FILE

    TAKE_NB="1000000"


    for i in "${!NLLB_2022__ISO3[@]}"; do
        NLLB_LANG=${NLLB_2022__ISO3[i]}

        if [ "$NLLB_LANG" == "ary" ] || [ "$NLLB_LANG" == "ars" ] || [ "$NLLB_LANG" == "acq" ] || [ "$NLLB_LANG" == "acm" ] || [ "$NLLB_LANG" == "aeb" ] || [ "$NLLB_LANG" == "apc" ] || [ "$NLLB_LANG" == "arz" ]
        then
            echo "Skipping $NLLB_LANG"
            continue
        fi

        TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.no-punct.txt"

        TRAIN_NB_LINES=$(cat $TRAIN_FILE | head -n $TAKE_NB | wc -l)
        printf "%-10s %d \n" $NLLB_LANG $TRAIN_NB_LINES >> $NB_LINES_FILE

    done

    cat $NB_LINES_FILE | $UPSAMPLING_COMPUTE --alpha 0.6 > $NB_LINES_UPSAMPLED_FILE


    for i in "${!NLLB_2022__ISO3[@]}"; do
        NLLB_LANG=${NLLB_2022__ISO3[i]}

        if [ "$NLLB_LANG" == "ary" ] || [ "$NLLB_LANG" == "ars" ] || [ "$NLLB_LANG" == "acq" ] || [ "$NLLB_LANG" == "acm" ] || [ "$NLLB_LANG" == "aeb" ] || [ "$NLLB_LANG" == "apc" ] || [ "$NLLB_LANG" == "arz" ]
        then
            echo "Skipping $NLLB_LANG"
            continue
        fi

        TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.no-punct.txt"

        TRAIN_UPSAMPLED_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.upsampl.txt"
        TRAIN_SAME_LIMIT_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.samelimit.txt"

        cat $NB_LINES_UPSAMPLED_FILE | grep -P "^$NLLB_LANG\t"

        NB_REPEAT=$(cat $NB_LINES_UPSAMPLED_FILE | grep -P "^$NLLB_LANG\t" | awk '{ print $3 }')
        NB_LIMIT=$(cat $NB_LINES_UPSAMPLED_FILE | grep -P "^$NLLB_LANG\t" | awk '{ print $4 }')
        # echo "   $NB_REPEAT"
        # echo "   $NB_LIMIT"

        yes $TRAIN_FILE | head -n $NB_REPEAT | xargs cat | head -n $NB_LIMIT >> $UNSHUFFLED_TRAIN_UPSAMPLED_FILE
        # cat $TRAIN_FILE | head -n $TAKE_NB >> $UNSHUFFLED_TRAIN_SAMELIMIT_FILE

    done
    shuf $UNSHUFFLED_TRAIN_UPSAMPLED_FILE > $ALL_TRAIN_UPSAMPLED_FILE
    # shuf $UNSHUFFLED_TRAIN_SAMELIMIT_FILE > $ALL_TRAIN_SAMELIMIT_FILE
}


# flores_filled is lang.flores when flores available, lang.valid when flores not available
prepare_flores_filled_dev_data () {
    CONCAT_FILE="$VALID_DIR/concat.flores-filled.txt"
    TEST_FINAL_FILE="$VALID_DIR/flores-filled.txt"
    > $CONCAT_FILE

    for i in "${!NLLB_2022__ISO3[@]}"; do
        NLLB_LANG=${NLLB_2022__ISO3[i]}
        FLORES_LANG=${NLLB_2022__FLORES_CODE[i]}

        LABEL_LANG="$NLLB_LANG"
        if [ "$NLLB_LANG" == "ary" ] || [ "$NLLB_LANG" == "ars" ] || [ "$NLLB_LANG" == "acq" ] || [ "$NLLB_LANG" == "acm" ] || [ "$NLLB_LANG" == "aeb" ] || [ "$NLLB_LANG" == "apc" ] || [ "$NLLB_LANG" == "arz" ]
        then
            LABEL_LANG="ara_Arab"
        fi

        chosen_source=""

        if [ -z "$chosen_source" ]
        then
            FLORES_DEV_FILE="$FLORES_DEV_DIR/$FLORES_LANG.dev"
            if [ -f "$FLORES_DEV_FILE" ]; then
                chosen_source="${chosen_source}-flores"
                cat $FLORES_DEV_FILE | awk -v lang="$LABEL_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
            fi
        fi

        if [ -z "$chosen_source" ]
        then
            FLORES_DEV_FILE="$FLORES_BETA_DEV_DIR/$FLORES_LANG.dev"
            if [ -f "$FLORES_DEV_FILE" ]; then
                chosen_source="${chosen_source}-flores_beta"
                cat $FLORES_DEV_FILE | awk -v lang="$LABEL_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
            fi
        fi

        if [ -z "$chosen_source" ]
        then
            if [ "$NLLB_LANG" == "ary" ] || [ "$NLLB_LANG" == "ars" ] || [ "$NLLB_LANG" == "acq" ] || [ "$NLLB_LANG" == "acm" ] || [ "$NLLB_LANG" == "aeb" ] || [ "$NLLB_LANG" == "apc" ] || [ "$NLLB_LANG" == "arz" ]
            then
                :
            else
                VALID_FILE="$VALID_DIR/$NLLB_LANG.valid.mono.txt"
                if [ -f "$VALID_FILE" ]; then
                    chosen_source="${chosen_source}-fromvalid"
                    shuf $VALID_FILE | head -n 1000 >> $CONCAT_FILE
                else
                    echo "ERROR: File not found $VALID_FILE"
                    return 1
                fi
            fi

        fi

        printf "%-20s %s \n" $NLLB_LANG $chosen_source
    done

    shuf $CONCAT_FILE > $TEST_FINAL_FILE
    rm $CONCAT_FILE
}

train_fasttext_8_1 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.8.1 \
        -lr 0.8 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 256 -loss softmax -bucket 1000000 -thread 40
}

train_fasttext_8_2 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.upsampl.txt -output $RESULT_FOLDER/model.8.2 \
        -lr 0.8 -minn 2 -maxn 5 -minCount 1 -epoch 5 -dim 256 -loss softmax -bucket 5000000 -thread 40
}


eval_flores_filled_dev_fasttext_variants () {
    GOLD="$RESULT_FOLDER/flores-filled-dev.gold"
    TEST_FILE="$VALID_DIR/flores-filled.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    for MODEL_MAJOR in 8 9
    do
        for i in `seq 1 20`
        do
            RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled-dev.fasttext.$MODEL_MAJOR.$i.txt"

            LID_MODEL="$RESULT_FOLDER/model.$MODEL_MAJOR.$i.bin"

            if [ -s $LID_MODEL ] && [ ! -s $RESULT_TXT ]
            then
                echo "LID_MODEL $LID_MODEL"
                PREDICTIONS="$RESULT_FOLDER/flores-filled.dev.fasttext.predictions.$MODEL_MAJOR.$i"

                RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
                $RESULT_GATHER > $PREDICTIONS

                $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
            fi
        done
    done
}


prepare_valid_for_histograms
create_valid_histograms
prepare_data_1
prepare_data_filter_step_1
prepare_data_filter_step_1_zho_Hant
prepare_data_filter_step_1_zho_Hans
prepare_data_filter_step_1_jpn

prepare_data_filter_step_3


group_train_valid_data_langs
merge_grouped_files
prepare_data_combine

prepare_data_remove_punct_train

prepare_data_combine_train
prepare_data_combine_upsampl

train_fasttext_8_2
train_fasttext_8_1

prepare_flores_filled_dev_data

eval_flores_filled_dev_fasttext_variants

