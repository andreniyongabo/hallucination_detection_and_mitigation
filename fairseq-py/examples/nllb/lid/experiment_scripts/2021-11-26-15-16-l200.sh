#!/bin/bash


EXPERIMENT_NAME="2021-11-26-15-16-l200"
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


NLLB_DEC__NLLB_LANGS=("afr" "aka" "amh" "ara_Arab" "acm" "apc" "asm" "ast" "ayr" "azj" "bam" "bel" "ben" "bos" "bul" "cat" "ceb" "ces" "ckb" "cym" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "fuv" "gla" "gle" "glg" "guj" "hat" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ilo" "ind" "isl" "ita" "jav" "jpn" "kac" "kam" "kan" "kas_Arab" "kat" "kaz" "kea" "khm" "kir" "kmb" "kon" "kor" "kur" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mai" "mal" "mar" "mkd" "mlg" "mlt" "mon" "mri" "msa" "mya" "nld" "nob" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "quy" "ron" "run" "rus" "shn" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp_Cyrl" "ssw" "sun_Latn" "swe" "swh_Latn" "tam" "tel" "tgk" "tgl" "tha" "tir" "tpi" "tsn" "tur" "ukr" "umb" "urd" "uzb" "vie" "wol" "xho" "yid" "yor" "yue" "zho_Hans" "zho_Hant" "zul" "ace_Arab" "ace_Latn" "arz" "ban" "bjn_Arab" "bjn_Latn" "bak" "eus" "bem" "bho" "bug" "hne" "cjk" "crh_Latn" "prs" "dik" "dzo" "epo" "ewe" "fao" "fij" "fon" "fur" "grn" "kbp" "kab" "kau_Arab" "kau_Latn" "kas_Deva" "kik" "kin" "ltg" "lij" "lim" "lmo" "lua" "mag" "min_Latn" "lus" "mos" "nus" "nno" "pag" "pap" "smo" "sag" "san" "sat" "srd" "scn" "szl" "azb" "sot" "diq" "tmh_Latn" "tat_Cyrl" "bod" "ton" "tso" "tum" "tuk" "twi" "uig" "vec" "war" "wes" "abk" "ady" "bis" "che" "chv" "ewo" "gom" "kal" "krc" "arn" "xmf" "nav" "nia" "pcm" "oss" "roh" "bxr" "alt" "tah" "udm")
NLLB_DEC__FLORES_LANGS=("afr" "aka" "amh" "ara" "ara-IQ" "ara-LB" "asm" "ast" "aym" "azj" "bam" "bel" "ben" "bos" "bul" "cat" "ceb" "ces" "ckb" "cym" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "ful" "gla" "gle" "glg" "guj" "hat" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ilo" "ind" "isl" "ita" "jav" "jpn" "kac" "kam" "kan" "kas-Arab" "kat" "kaz" "kea" "khm" "kir" "kmb" "kon" "kor" "kur" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mai" "mal" "mar" "mkd" "mlg" "mlt" "mon" "mri" "msa" "mya" "nld" "nob" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "que" "ron" "run" "rus" "shn" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp" "ssw" "sun" "swe" "swh" "tam" "tel" "tgk" "tgl" "tha" "tir" "tpi" "tsn" "tur" "ukr" "umb" "urd" "uzb" "vie" "wol" "xho" "yid" "yor" "yue" "zho_simpl" "zho_trad" "zul" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "")
NLLB_DEC__FBSEED_LANGS=("" "" "" "" "" "" "" "" "" "" "bam" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "fuv" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "kas_Arab" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "mri" "" "" "" "" "" "" "" "" "" "" "" "" "" "pus" "" "" "" "" "shn" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "ace_Arab" "ace_Latn" "" "ban" "bjn_Arab" "bjn_Latn" "" "" "" "" "bug" "hne" "" "crh_Latn" "prs" "dik" "dzo" "" "" "" "" "" "fur" "" "" "" "kau_Arab" "kau_Latn" "kas_Deva" "" "" "ltg" "lij" "" "" "" "mag" "" "" "" "nus" "" "" "" "" "" "" "" "" "" "szl" "" "" "" "tmh_Latn" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "")
NLLB_DEC__JW300_LANGS=("af" "" "am" "ar" "" "" "as" "" "ay" "az" "" "" "bn" "" "bg" "cat" "ceb" "cs" "" "cy" "da" "de" "dyu" "el" "en" "et" "fa" "fi" "fr" "" "" "ga" "gl" "gu" "ht" "ha" "he" "hi" "hr" "hu" "hy" "ig" "ilo" "id" "is" "it" "jv" "" "kac" "kam" "kn" "" "ka" "kk_Cyrl" "kea" "km" "ky" "kmb" "kg" "ko" "" "lo" "lv" "ln" "lt" "" "lg" "luo" "" "ml" "mr" "mk" "mg" "mt" "mn" "" "zlm" "my" "nl" "no" "ne" "nso" "nya" "" "om" "or" "pa" "pl" "pt" "" "que" "ro" "run" "ru" "" "si" "sk" "sl" "sn" "" "" "es" "sq" "sr_Cyrl" "ss" "su" "sv" "sw" "ta" "te" "tg" "tl" "th" "ti" "tpi" "tn" "tr" "uk" "umb" "ur" "uz_Latn" "vi" "" "xh" "" "yo" "" "" "" "zu" "" "" "" "" "" "" "ba" "eu" "bem" "" "" "" "cjk" "" "" "" "" "" "ee" "fo" "fj" "fon" "" "gug" "kbp" "kab" "" "" "" "ki" "rw" "" "" "" "" "lua" "" "" "lus" "mos" "" "" "pag" "pap" "sm" "sg" "" "" "" "" "" "" "st" "" "" "tt" "" "tog" "ts" "tum" "tk" "tw" "" "vec" "war" "wes" "ab" "ady" "bi" "" "cv" "ewo" "gom" "kl" "krc" "arn" "xmf" "nv" "nia" "pcm" "os" "" "" "alt" "ty" "udm")
NLLB_DEC__LID_187_LANGS=("af" "" "am" "ar" "" "" "as" "ast" "" "az" "" "be" "bn" "bs" "bg" "ca" "ceb" "cs" "ckb" "cy" "da" "de" "" "el" "en" "et" "fa" "fi" "fr" "" "gd" "ga" "gl" "gu" "ht" "ha" "he" "hi" "hr" "hu" "hy" "ig" "ilo" "id" "is" "it" "jv" "ja" "" "" "kn" "" "ka" "kk" "" "km" "ky" "" "" "ko" "ku" "lo" "lv" "ln" "lt" "lb" "lg" "" "mai" "ml" "mr" "mk" "mg" "mt" "mn" "" "ms" "my" "nl" "no" "ne" "" "" "oc" "om" "or" "pa" "pl" "pt" "ps" "qu" "ro" "" "ru" "" "si" "sk" "sl" "sn" "sd" "so" "es" "sq" "sr" "" "su" "sv" "sw" "ta" "te" "tg" "tl" "th" "" "" "tn" "tr" "uk" "" "ur" "uz" "vi" "wo" "xh" "yi" "yo" "yue" "" "" "zu" "" "" "arz" "" "" "" "ba" "eu" "" "bh" "" "" "" "" "" "" "" "eo" "" "" "" "" "" "gn" "" "kab" "" "" "" "" "" "" "" "li" "lmo" "" "" "min" "" "" "" "nn" "" "" "" "" "sa" "sat" "sc" "scn" "" "azb" "" "diq" "" "tt" "bo" "" "" "" "tk" "" "ug" "vec" "war" "" "" "" "" "ce" "cv" "" "gom" "" "krc" "" "xmf" "" "" "" "os" "rm" "bxr" "" "" "")
NLLB_DEC__NLLB_LANG_SCRIPTS=("Latn" "Latn" "Ethi" "Arab" "Arab" "Arab" "Beng" "Latn" "Latn" "Arab" "Latn" "Cyrl" "Latn" "Latn" "Cyrl" "Latn" "Latn" "Latn" "Arab" "Latn" "Latn" "Latn" "Latn" "Grek" "Latn" "Latn" "Arab" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Gujr" "Latn" "Latn" "Hebr" "Deva" "Latn" "Latn" "Armn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "" "Latn" "Latn" "Knda" "Arab" "Geor" "Cyrl" "Latn" "Khmr" "Cyrl" "Latn" "Latn" "Hang" "Latn" "Laoo" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Deva" "Mlym" "Deva" "Cyrl" "Latn" "Latn" "Cyrl" "Latn" "Latn" "Mymr" "Latn" "Latn" "Deva" "Latn" "Latn" "Latn" "Latn" "Orya" "Guru" "Latn" "Latn" "Arab" "Latn" "Latn" "Latn" "Cyrl" "Mymr" "Sinh" "Latn" "Latn" "Latn" "Arab" "Latn" "Latn" "Latn" "Cyrl" "Latn" "Latn" "Latn" "Latn" "Taml" "Telu" "Cyrl" "Latn" "Thai" "Ethi" "Latn" "Latn" "Latn" "Cyrl" "Latn" "Arab" "Latn" "Latn" "Latn" "Latn" "Hebr" "Latn" "" "" "" "Latn" "Arab" "Latn" "Arab" "Latn" "Arab" "Latn" "Cyrl" "Latn" "Latn" "Deva" "Latn" "Deva" "Latn" "Latn" "Arab" "Latn" "Tibt" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Arab" "Latn" "Deva" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Deva" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Latn" "Deva" "Olck" "Latn" "Latn" "Latn" "Arab" "Latn" "Latn" "Latn" "Cyrl" "" "Latn" "Latn" "Latn" "Latn" "Latn" "" "Latn" "Latn" "Latn" "Cyrl" "Cyrl" "" "Cyrl" "Cyrl" "Latn" "Latn" "Latn" "" "Latn" "Geor" "Latn" "Latn" "Latn" "Cyrl" "Latn" "Cyrl" "Cyrl" "Latn" "Cyrl")

# NLLB_DEC__NLLB_LANGS=("afr" "aka" "amh" "ara_Arab")
# NLLB_DEC__FLORES_LANGS=("afr" "aka" "amh" "ara")
# NLLB_DEC__FBSEED_LANGS=("" "" "" "")
# NLLB_DEC__JW300_LANGS=("af" "" "am" "ar")
# NLLB_DEC__LID_187_LANGS=("af" "" "am" "ar")
# NLLB_DEC__NLLB_LANG_SCRIPTS=("Latn" "Latn" "Ethi" "Arab")



FILTER_CHAR_HISTOGRAM="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/filtering/filter_char_histogram.py"
HISTOGRAMS_VALID_FOLDER="/large_experiments/mmt/lidruns/2021-09-20-14-14-histogram-baseline/histograms/valid"

FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"
# OLD_DATA_TRAIN_FOLDER="/large_experiments/mmt/lidruns/2021-09-18-00-21-goal124-baseline/data/train"
LID_MODEL="/large_experiments/nllb/mmt/lidruns/2021-10-05-11-53-english/result/model.8.8.bin"


THRESH_SCORE="0.8"




prepare_data_1 () {
    for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
        NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}
        FLORES_LANG=${NLLB_DEC__FLORES_LANGS[i]}
        FBSEED_LANG=${NLLB_DEC__FBSEED_LANGS[i]}
        JW300_LANG=${NLLB_DEC__JW300_LANGS[i]}
        LID_187_LANG=${NLLB_DEC__LID_187_LANGS[i]}

        if [ -z "$NLLB_LANG" ]
        then
            echo "Empty lang code"
            exit 1
        fi

        MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"
        > $MONO_CAT_FILE

        chosen_source=""

        if [ ! -z "$FBSEED_LANG" ]
        then
            FBSEED_FILE="$FBSEED_DATA_DIR/fbseed20211130.$FBSEED_LANG.txt"
            if [ -f "$FBSEED_FILE" ]
            then
                chosen_source="${chosen_source}-fbseed"

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

                    cat $JW300_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
                fi
            fi

            if [ ! -z "$LID_187_LANG" ]
            then
                LID187_FILE="$LID187_DATA_DIR/$LID_187_LANG.txt"
                if [ -f "$LID187_FILE" ]; then
                    chosen_source="${chosen_source}-lid187"

                    cat $LID187_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
                fi
            fi
        fi

        if [ -z "$chosen_source" ]
        then
            FLORES_DEV_FILE="$FLORES_DEV_DIR/$FLORES_LANG.dev"
            if [ -f "$FLORES_DEV_FILE" ]; then
                chosen_source="${chosen_source}-floresdev"

                cat $FLORES_DEV_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
            fi
        fi
        printf "%-20s %s \n" $NLLB_LANG $chosen_source

    done
}


prepare_data_filter_step_1_lang () {
    NLLB_LANG=$1
    NLLB_SCRIPT=$2

    SCRIPT_DETECTOR="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/classifier/script_detector.py"

    MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step1.txt"

    if [ -z $NLLB_SCRIPT ]
    then

        cat $MONO_CAT_FILE > $CLEANED_TRAIN_FILE
        printf "%-20s skipping \n" $NLLB_LANG
        continue
    fi

    cat $MONO_CAT_FILE \
        | cut -f 2- -d" " \
        | parallel -j6 -k --pipe $SCRIPT_DETECTOR --filter-mode 2> /dev/null \
        | grep "^${NLLB_SCRIPT}" \
        | cut -f 2- \
        | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' \
        > $CLEANED_TRAIN_FILE

    NUMBER_OF_LINES_BEFORE=$(cat $MONO_CAT_FILE | wc -l)
    NUMBER_OF_LINES_AFTER=$(cat $CLEANED_TRAIN_FILE | wc -l)

    printf "%-20s script: %-10s : from %-10s to %-10s \n" $NLLB_LANG $NLLB_SCRIPT $NUMBER_OF_LINES_BEFORE $NUMBER_OF_LINES_AFTER
}


prepare_data_filter_step_1_lang_dist () {
    NLLB_LANG=$1
    NLLB_SCRIPT=$2

    SCRIPT_DETECTOR="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/classifier/script_detector.py"

    MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step1.txt"

    if [ -z $NLLB_SCRIPT ]
    then

        cat $MONO_CAT_FILE > $CLEANED_TRAIN_FILE
        printf "%-20s skipping \n" $NLLB_LANG
        continue
    fi

    LAUNCHABLE="$RESULT_FOLDER/launchable.$NLLB_LANG.sh"


    cat > ${LAUNCHABLE} <<- EOM
#!/bin/bash

srun --unbuffered cat ${MONO_CAT_FILE} \\
    | cut -f 2- -d" " \\
    | parallel -j35 -k --pipe ${SCRIPT_DETECTOR} --filter-mode \\
    | grep "^${NLLB_SCRIPT}" \\
    | cut -f 2- \\
    | awk -v lang="${NLLB_LANG}" '{print "__label__"lang " " \$0}' \\
    > ${CLEANED_TRAIN_FILE}
EOM

    chmod +x $LAUNCHABLE

    # --partition nllb
    sbatch --job-name "datafilter_$NLLB_LANG" \
        --partition devaccel --comment "datafilter_$NLLB_LANG" \
        --nodes=1 --gpus-per-node=0 --cpus-per-task=35 --ntasks-per-node=1 --time=10 \
        --output "$RESULT_FOLDER/datafilter_$NLLB_LANG.log" \
        --error "$RESULT_FOLDER/datafilter_$NLLB_LANG.err" \
        --mem=100G \
        $LAUNCHABLE

    printf "launched %-20s script: %-10s : from %-10s to %-10s \n" $NLLB_LANG $NLLB_SCRIPT $NUMBER_OF_LINES_BEFORE $NUMBER_OF_LINES_AFTER
}

prepare_data_filter_step_1 () {
    LENGTH=${#NLLB_DEC__NLLB_LANGS[@]}
    BATCH_SIZE=10
    i=0

    while [ $i -lt $LENGTH ]
    do
        for (( j=0; j<${BATCH_SIZE}; j++ ));
        do
            if [ $i -ge $LENGTH ]; then
                break
            fi

            NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}
            NLLB_SCRIPT=${NLLB_DEC__NLLB_LANG_SCRIPTS[i]}

            prepare_data_filter_step_1_lang $NLLB_LANG $NLLB_SCRIPT &

            let "i+=1"
        done

        wait
    done

}

prepare_data_filter_step_1_exceptions () {
    OTHER_TRAIN_DATA="/large_experiments/nllb/mmt/lidruns/2021-10-25-22-21-multifilter5-softmax/data/train/"

    for NLLB_LANG in azj kor
    do
        echo $NLLB_LANG
        CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step1.txt"

        cat "$OTHER_TRAIN_DATA/$NLLB_LANG.cleaned.step1.txt" > $CLEANED_TRAIN_FILE

    done
}


prepare_data_filter_step_2 () {
    for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
        NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}

        PREV_CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step1.txt"
        CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step2.txt"

        cat $PREV_CLEANED_TRAIN_FILE | awk 'length($0) > 30 { print }' > $CLEANED_TRAIN_FILE &
    done

    wait
}


prepare_data_filter_step_3_lang() {
    NLLB_LANG=$1

    PREV_CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step2.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step3.txt"
    REJECTD_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.rejectd.step3.txt"

    RESULT_TXT_CLEANED_PREDS="$TRAIN_DIR/engparts.predictions.$NLLB_LANG.cleaned.step3.txt"

    RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL -"
    cat $PREV_CLEANED_TRAIN_FILE | cut -f 2- -d" " | $RESULT_GATHER > $RESULT_TXT_CLEANED_PREDS

    paste $RESULT_TXT_CLEANED_PREDS $PREV_CLEANED_TRAIN_FILE | grep "__label__neng" | cut -f 2- > $CLEANED_TRAIN_FILE
    paste $RESULT_TXT_CLEANED_PREDS $PREV_CLEANED_TRAIN_FILE | grep "__label__eng" | cut -f 2- > $REJECTD_TRAIN_FILE

    rm $RESULT_TXT_CLEANED_PREDS
    echo "    FINISHED eng lang filtering for $NLLB_LANG"
}

prepare_data_filter_step_3 () {
    LENGTH=${#NLLB_DEC__NLLB_LANGS[@]}
    BATCH_SIZE=40
    i=0

    while [ $i -lt $LENGTH ]
    do
        for (( j=0; j<${BATCH_SIZE}; j++ ));
        do
            if [ $i -ge $LENGTH ]; then
                break
            fi

            NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}

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

prepare_data_filter_step_4 () {
    for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
        NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}

        PREV_CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step3.txt"
        CLEANED_TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step4.txt"

        cat $PREV_CLEANED_TRAIN_FILE | sed 's/[0-9]*//g' > $CLEANED_TRAIN_FILE &
        # cat $PREV_CLEANED_TRAIN_FILE | awk 'length($0) > 30 { print }' > $CLEANED_TRAIN_FILE &
    done

    wait
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

    for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
        NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}

        VALID_SIZE=100000
        VALID_COMPARE_SIZE=$(expr $VALID_SIZE \* 4)


        MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step4.txt"
        # MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cat.txt"
        MONO_SHUF_FILE="$TRAIN_DIR/$NLLB_LANG.shuf.txt"

        shuf $MONO_CAT_FILE > $MONO_SHUF_FILE

        NUMBER_OF_LINES=$(cat $MONO_CAT_FILE | wc -l)


        if [ "$NUMBER_OF_LINES" -lt "$VALID_COMPARE_SIZE" ]; then
            VALID_SIZE=$(expr $NUMBER_OF_LINES / 10)
        fi
        TRAIN_NB_LINES=$(expr $NUMBER_OF_LINES - $VALID_SIZE)

        # if [ "$TRAIN_NB_LINES" -gt "400000" ]; then
        #     TRAIN_NB_LINES="400000"
        # fi

        TRAIN_FILE="$TRAIN_DIR/$NLLB_LANG.train.mono.txt"
        VALID_FILE="$VALID_DIR/$NLLB_LANG.valid.mono.txt"

        # echo "head -n $TRAIN_NB_LINES $MONO_CAT_FILE > $TRAIN_FILE"
        # shuf $MONO_CAT_FILE | head -n $TRAIN_NB_LINES > $TRAIN_FILE
        head -n $TRAIN_NB_LINES $MONO_SHUF_FILE > $TRAIN_FILE
        tail -n $VALID_SIZE $MONO_SHUF_FILE > $VALID_FILE

        rm $MONO_SHUF_FILE

        printf "%-20s train size: %-10s    valid size: %-10s \n" $NLLB_LANG $TRAIN_NB_LINES $VALID_SIZE

        cat $TRAIN_FILE >> $UNSHUFFLED_TRAIN_FILE
        cat $VALID_FILE >> $UNSHUFFLED_VALID_FILE
    done
    shuf $UNSHUFFLED_TRAIN_FILE > $ALL_TRAIN_FILE
    shuf $UNSHUFFLED_VALID_FILE > $ALL_VALID_FILE
}

# flores_filled is lang.flores when flores available, lang.valid when flores not available
prepare_flores_filled_data () {
    CONCAT_FILE="$TEST_DIR/concat.flores-filled.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-filled.txt"
    > $CONCAT_FILE


    for i in "${!NLLB_DEC__NLLB_LANGS[@]}"; do
        NLLB_LANG=${NLLB_DEC__NLLB_LANGS[i]}
        FLORES_LANG=${NLLB_DEC__FLORES_LANGS[i]}
        FBSEED_LANG=${NLLB_DEC__FBSEED_LANGS[i]}
        JW300_LANG=${NLLB_DEC__JW300_LANGS[i]}
        LID_187_LANG=${NLLB_DEC__LID_187_LANGS[i]}

        chosen_source=""
        if [ ! -z "$FLORES_LANG" ]
        then
            FILES_FOUND=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | head -n 1)
            FILES_FOUND_NUM=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | wc -l)

            # echo "LANG=$NLLB_LANG" "$FLORES_LANG" "$FILES_FOUND_NUM"
            # echo "$FILES_FOUND"
            chosen_source="${chosen_source}-flores"

            cat $FILES_FOUND | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
        else


            # echo "Flores not available for $NLLB_LANG"
            VALID_FILE="$VALID_DIR/$NLLB_LANG.valid.mono.txt"
            if [ -f "$VALID_FILE" ]; then
                chosen_source="${chosen_source}-fromvalid"
                # echo "will use $VALID_FILE"
                shuf $VALID_FILE | head -n 1000 >> $CONCAT_FILE
            else
                echo "ERROR: File not found $VALID_FILE"
                return 1
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
        -lr 0.8 -minn 2 -maxn 5 -minCount 10 -epoch 2 -dim 256 -loss hs -bucket 1000000 -thread 60
}

train_fasttext_8_1_aclean () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.asianclean.txt -output $RESULT_FOLDER/model.8.1.aclean \
        -lr 0.8 -minn 2 -maxn 5 -minCount 10 -epoch 2 -dim 256 -loss hs -bucket 1000000 -thread 60
}

eval_flores_filled_fasttext_variants_8 () {
    GOLD="$RESULT_FOLDER/flores-filled.gold"
    TEST_FILE="$TEST_DIR/flores-filled.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 9`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.8.$i.txt"

        LID_MODEL="$RESULT_FOLDER/model.8.$i.bin"

        if [ -s $LID_MODEL ]
        then
            echo "LID_MODEL $LID_MODEL"
            PREDICTIONS="$RESULT_FOLDER/flores-filled.fasttext.predictions.8.$i"

            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
            $RESULT_GATHER > $PREDICTIONS

            CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
            $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
        fi
    done
}


eval_flores_filled_fasttext_variants_8_aclean () {
    GOLD="$RESULT_FOLDER/flores-filled.gold"
    TEST_FILE="$TEST_DIR/flores-filled.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 9`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.8.$i.aclean.txt"

        LID_MODEL="$RESULT_FOLDER/model.8.$i.aclean.bin"

        if [ -s $LID_MODEL ]
        then
            echo "LID_MODEL $LID_MODEL"
            PREDICTIONS="$RESULT_FOLDER/flores-filled.fasttext.predictions.8.$i.aclean"

            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
            $RESULT_GATHER > $PREDICTIONS

            CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
            $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
        fi
    done
}



# prepare_data_1

# prepare_data_filter_step_1
# prepare_data_filter_step_1_exceptions

# prepare_data_filter_step_2

# prepare_data_filter_step_3
# prepare_data_filter_step_4

# prepare_data_combine


# prepare_flores_filled_data

# train_fasttext_8_1
# eval_flores_filled_fasttext_variants_8


train_fasttext_8_1_aclean
eval_flores_filled_fasttext_variants_8_aclean




