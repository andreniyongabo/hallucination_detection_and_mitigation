#!/bin/bash


EXPERIMENT_NAME="2021-10-25-22-21-multifilter5-softmax"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER




DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER

FLORES_DEV_DIR="/large_experiments/mmt/flores101/dev"
FLORES_DEVTEST_DIR="/large_experiments/mmt/flores101/devtest"
JW300_DETOK_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/detok"
LID187_DATA_DIR="/private/home/celebio/lid187_data_2"

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

FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"
# OLD_DATA_TRAIN_FOLDER="/large_experiments/mmt/lidruns/2021-09-18-00-21-goal124-baseline/data/train"
LID_MODEL="/large_experiments/nllb/mmt/lidruns/2021-10-05-11-53-english/result/model.8.8.bin"


THRESH_SCORE="0.8"


prepare_data_1 () {
    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        JW300_LANG=${GOAL124__JW300_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}
        FLORES_LANG=${GOAL124__FLORES_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
        fi

        MONO_CAT_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cat.txt"
        > $MONO_CAT_FILE

        has_jw300=false
        if [ ! -z "$JW300_LANG" ]
        then
            if [ "$ISO_639_3_LANG" != "jpn" ]
            then
                JW300_FILE="$JW300_DETOK_DIR/$JW300_LANG.detok.txt"
                if [ -f "$JW300_FILE" ]; then
                    has_jw300=true

                    cat $JW300_FILE | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
                fi
            fi
        fi

        has_lid187=false
        if [ ! -z "$LID_187_LANG" ]
        then
            LID187_FILE="$LID187_DATA_DIR/$LID_187_LANG.txt"
            if [ -f "$LID187_FILE" ]; then
                has_lid187=true

                cat $LID187_FILE | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
            fi
        fi

        if [[ "$has_lid187" = false && "$has_jw300" = false ]]
        then
            printf "%s \n" $ISO_639_3_LANG
            echo "\\t ooops"
            # in this case: use flores dev
            FLORES_DEV_FILE="$FLORES_DEV_DIR/$FLORES_LANG.dev"
            if [ -f "$FLORES_DEV_FILE" ]; then
                echo "    $FLORES_DEV_FILE"

                cat $FLORES_DEV_FILE | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' >> $MONO_CAT_FILE
            fi
        fi
    done

}

prepare_data_filter_step_1 () {
    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
        fi

        if [ "$ISO_639_3_LANG" == "zho" ] || [ "$ISO_639_3_LANG" == "jpn" ]
        then
            echo "Skipping $ISO_639_3_LANG"
            continue
        fi

        MONO_CAT_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cat.txt"

        THRESH_SCORE="0.8"
        CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step1.txt"
        REJECTD_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.rejectd.step1.txt"

        echo "ISO_639_3_LANG = $ISO_639_3_LANG"
        cat $MONO_CAT_FILE | python $FILTER_CHAR_HISTOGRAM \
            --lang $ISO_639_3_LANG \
            --threshold $THRESH_SCORE \
            --histogram-threshold 0.95 \
            --histograms $HISTOGRAMS_VALID_FOLDER \
                2> $REJECTD_TRAIN_FILE \
                1> $CLEANED_TRAIN_FILE &

    done

    wait
}

prepare_data_filter_step_1_zho () {
    SCRIPT_DETECTOR="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/classifier/script_detector.py"

    ISO_639_3_LANG="zho"
    MONO_CAT_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cat.txt"

    LANG_SCRIPT_PRED_FILE="$TRAIN_DIR/$ISO_639_3_LANG.script.step1.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step1.txt"
    REJECTD_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.rejectd.step1.txt"

    # mv $CLEANED_TRAIN_FILE "$CLEANED_TRAIN_FILE.bck"
    # mv $REJECTD_TRAIN_FILE "$REJECTD_TRAIN_FILE.bck"

    cat $MONO_CAT_FILE | $SCRIPT_DETECTOR > $LANG_SCRIPT_PRED_FILE

    paste $LANG_SCRIPT_PRED_FILE $MONO_CAT_FILE | grep "^Han" | cut -f 2 > $CLEANED_TRAIN_FILE
}

prepare_data_filter_step_1_jpn () {
    SCRIPT_DETECTOR="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/classifier/script_detector.py"

    ISO_639_3_LANG="jpn"
    MONO_CAT_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cat.txt"

    LANG_SCRIPT_PRED_FILE="$TRAIN_DIR/$ISO_639_3_LANG.script.step1.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step1.txt"
    REJECTD_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.rejectd.step1.txt"

    cat $MONO_CAT_FILE | $SCRIPT_DETECTOR > $LANG_SCRIPT_PRED_FILE

    paste $LANG_SCRIPT_PRED_FILE $MONO_CAT_FILE | grep "^Han" | cut -f 2 > $CLEANED_TRAIN_FILE
}

prepare_data_filter_step_2 () {
    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
        fi

        PREV_CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step1.txt"
        CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step2.txt"

        cat $PREV_CLEANED_TRAIN_FILE | awk 'length($0) > 30 { print }' > $CLEANED_TRAIN_FILE &
    done

    wait
}

prepare_data_filter_step_3_lang() {
    ISO_639_3_LANG=$1

    PREV_CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step2.txt"
    CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step3.txt"
    REJECTD_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.rejectd.step3.txt"

    RESULT_TXT_CLEANED_PREDS="$TRAIN_DIR/engparts.predictions.$ISO_639_3_LANG.cleaned.step3.txt"

    RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL -"
    cat $PREV_CLEANED_TRAIN_FILE | cut -f 2- -d" " | $RESULT_GATHER > $RESULT_TXT_CLEANED_PREDS

    paste $RESULT_TXT_CLEANED_PREDS $PREV_CLEANED_TRAIN_FILE | grep "__label__neng" | cut -f 2- > $CLEANED_TRAIN_FILE
    paste $RESULT_TXT_CLEANED_PREDS $PREV_CLEANED_TRAIN_FILE | grep "__label__eng" | cut -f 2- > $REJECTD_TRAIN_FILE

    rm $RESULT_TXT_CLEANED_PREDS
    echo "    FINISHED eng lang filtering for $ISO_639_3_LANG"
}

prepare_data_filter_step_3 () {
    LENGTH=${#GOAL124__ISO_639_3_LANGS[@]}
    BATCH_SIZE=40
    i=0

    while [ $i -lt $LENGTH ]
    do
        for (( j=0; j<${BATCH_SIZE}; j++ ));
        do
            if [ $i -ge $LENGTH ]; then
                break
            fi

            ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
            LID_187_LANG=${GOAL124__LID_187_LANGS[i]}

            if [ -z "$ISO_639_3_LANG" ]
            then
                echo "Vide $LID_187_LANG"
                ISO_639_3_LANG=$LID_187_LANG
            fi

            echo "eng lang filtering for $ISO_639_3_LANG"
            prepare_data_filter_step_3_lang $ISO_639_3_LANG &

            let "i+=1"
        done

        wait
    done

    mv "$TRAIN_DIR/eng.rejectd.step3.txt" "$TRAIN_DIR/eng.rejectd.step3.txt.aux"
    mv "$TRAIN_DIR/eng.cleaned.step3.txt" "$TRAIN_DIR/eng.rejectd.step3.txt"
    mv "$TRAIN_DIR/eng.rejectd.step3.txt.aux" "$TRAIN_DIR/eng.cleaned.step3.txt"
}

prepare_data_filter_step_4 () {
    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
        fi

        PREV_CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step3.txt"
        CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step4.txt"

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


    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}

        VALID_SIZE=100000
        VALID_COMPARE_SIZE=$(expr $VALID_SIZE \* 4)

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
            echo "ISO_639_3_LANG set to $ISO_639_3_LANG"
        fi

        MONO_CAT_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.step4.txt"

        echo "MONO_CAT_FILE=$MONO_CAT_FILE"
        NUMBER_OF_LINES=$(cat $MONO_CAT_FILE | wc -l)

        echo "    $ISO_639_3_LANG NUMBER_OF_LINES = $NUMBER_OF_LINES"

        if [ "$NUMBER_OF_LINES" -lt "$VALID_COMPARE_SIZE" ]; then
            VALID_SIZE=$(expr $NUMBER_OF_LINES / 10)
            echo "        /!\ specific valid size for this language ($ISO_639_3_LANG) = $VALID_SIZE"
        fi
        TRAIN_NB_LINES=$(expr $NUMBER_OF_LINES - $VALID_SIZE)
        echo "VALID_SIZE = $VALID_SIZE"
        echo "    $ISO_639_3_LANG TRAIN_NB_LINES = $TRAIN_NB_LINES"


        TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.train.mono.txt"
        VALID_FILE="$VALID_DIR/$ISO_639_3_LANG.valid.mono.txt"

        echo "head -n $TRAIN_NB_LINES $MONO_CAT_FILE > $TRAIN_FILE"
        head -n $TRAIN_NB_LINES $MONO_CAT_FILE > $TRAIN_FILE
        tail -n $VALID_SIZE $MONO_CAT_FILE > $VALID_FILE

        cat $TRAIN_FILE >> $UNSHUFFLED_TRAIN_FILE
        cat $VALID_FILE >> $UNSHUFFLED_VALID_FILE
    done
    shuf $UNSHUFFLED_TRAIN_FILE > $ALL_TRAIN_FILE
    shuf $UNSHUFFLED_VALID_FILE > $ALL_VALID_FILE
}

prepare_flores_devtest_data () {
    CONCAT_FILE="$TEST_DIR/concat.all.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-devtest.txt"
    > $CONCAT_FILE

    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        JW300_LANG=${GOAL124__JW300_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}
        FLORES_LANG=${GOAL124__FLORES_LANGS[i]}


        if [ ! -z "$FLORES_LANG" ]
        then
            FILES_FOUND=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | head -n 1)
            FILES_FOUND_NUM=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | wc -l)

            # echo "LANG=$ISO_639_3_LANG" "$FLORES_LANG" "$FILES_FOUND_NUM"
            # echo "$FILES_FOUND"
            cat $FILES_FOUND | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
        fi


    done

    shuf $CONCAT_FILE > $TEST_FINAL_FILE
    rm $CONCAT_FILE
}


# flores_filled is lang.flores when flores available, lang.valid when flores not available
prepare_flores_filled_data () {
    CONCAT_FILE="$TEST_DIR/concat.flores-filled.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-filled.txt"
    > $CONCAT_FILE

    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        JW300_LANG=${GOAL124__JW300_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}
        FLORES_LANG=${GOAL124__FLORES_LANGS[i]}
        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
        fi


        if [ ! -z "$FLORES_LANG" ]
        then
            FILES_FOUND=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | head -n 1)
            FILES_FOUND_NUM=$(ls -1 $FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest | wc -l)

            # echo "LANG=$ISO_639_3_LANG" "$FLORES_LANG" "$FILES_FOUND_NUM"
            # echo "$FILES_FOUND"

            cat $FILES_FOUND | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
        else
            echo "Flores not available for $ISO_639_3_LANG"
            VALID_FILE="$VALID_DIR/$ISO_639_3_LANG.valid.mono.txt"
            if [ -f "$VALID_FILE" ]; then
                echo "will use $VALID_FILE"
                shuf $VALID_FILE | head -n 1000 >> $CONCAT_FILE
            else
                echo "ERROR: File not found $VALID_FILE"
                return 1
            fi
        fi


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
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.8.2 \
        -lr 0.8 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 256 -loss softmax -bucket 2000000 -thread 40
}

train_fasttext_8_3 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.8.3 \
        -lr 0.8 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 256 -loss softmax -bucket 5000000 -thread 40
}

train_fasttext_8_4 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.8.4 \
        -lr 0.8 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 128 -loss softmax -bucket 2000000 -thread 40
}

train_fasttext_8_8 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.8.8 \
        -lr 0.8 -minn 2 -maxn 5 -minCount 1000 -epoch 2 -dim 256 -loss softmax -bucket 10000000 -thread 40
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


eval_flores_filled_fasttext_variants_8_noisy () {
    GOLD_NOISY="$RESULT_FOLDER/flores-filled.noisy.gold"
    TEST_FILE_NOISY="/large_experiments/nllb/mmt/lidruns/2021-09-20-23-26-goal124-filter-percentile/data/test/flores-filled.txt"
    cat "$TEST_FILE_NOISY" | cut -f 1 -d" " > $GOLD_NOISY

    for i in `seq 1 9`
    do
        RESULT_TXT_NOISY="$RESULT_FOLDER/result.classifiermetrics-flores-filled.noisy.fasttext.8.$i.txt"

        LID_MODEL="$RESULT_FOLDER/model.8.$i.bin"

        if [ -s $LID_MODEL ]
        then
            echo "LID_MODEL $LID_MODEL"
            PREDICTIONS_NOISY="$RESULT_FOLDER/flores-filled.noisy.fasttext.predictions.8.$i"

            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE_NOISY"
            $RESULT_GATHER > $PREDICTIONS_NOISY

            CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
            $CLASSIFIER_METRICS --prediction $PREDICTIONS_NOISY --gold $GOLD_NOISY > $RESULT_TXT_NOISY
        fi
    done
}

eval_flores_devtest_fasttext_variants () {
    GOLD="$RESULT_FOLDER/flores-devtest.gold"
    TEST_FILE="$TEST_DIR/flores-devtest.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-devtest.fasttext.8.8.txt"

    LID_MODEL="$RESULT_FOLDER/model.8.8.bin"
    PREDICTIONS="$RESULT_FOLDER/flores-devtest.fasttext.predictions.8.8"

    if [ -s $LID_MODEL ]
    then
        RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
        $RESULT_GATHER > $PREDICTIONS

        CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
        $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
    fi
}



prepare_data_1

prepare_data_filter_step_1
prepare_data_filter_step_1_zho
prepare_data_filter_step_1_jpn

prepare_data_filter_step_2
prepare_data_filter_step_3
prepare_data_filter_step_4

prepare_data_combine

prepare_flores_devtest_data
prepare_flores_filled_data

train_fasttext_8_1

train_fasttext_8_2
train_fasttext_8_3
train_fasttext_8_4

train_fasttext_8_8

eval_flores_filled_fasttext_variants_8
eval_flores_filled_fasttext_variants_8_noisy
eval_flores_devtest_fasttext_variants
