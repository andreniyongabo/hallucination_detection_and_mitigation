#!/bin/bash


EXPERIMENT_NAME="2021-06-16-09-44-compare-on-earl"
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

# Languages that exist in flores99
ISO_639_3_LANGS_FLORES99=("nya" "afr" "amh" "ara" "hye" "asm" "ast" "aze" "ben" "bos" "bul" "mya" "cat" "ceb" "ckb" "zho" "hrv" "ces" "dan" "nld" "eng" "est" "fin" "fra" "ful" "glg" "lug" "kat" "deu" "guj" "hau" "heb" "hin" "hun" "isl" "ibo" "ind" "ita" "jpn" "jav" "kea" "kan" "kaz" "khm" "kor" "kur" "kir" "lao" "lav" "lin" "lit" "luo" "ltz" "mkd" "mlg" "msa" "mal" "mlt" "mar" "ell" "mon" "npi" "nso" "ory" "orm" "pus" "fas" "pol" "por" "pan" "ron" "rus" "srp" "sna" "snd" "sin" "slk" "slv" "som" "spa" "sun" "swh" "swe" "tgl" "tam" "tel" "tha" "tsn" "tur" "ukr" "urd" "uzb" "vie" "cym" "wol" "xho" "yor" "zul" "bel")
JW300_LANGS_FLORES99=("nya" "af" "am" "ar" "hy" "as" "" "az" "bn" "" "bg" "my" "cat" "ceb" "" "" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "kea" "kn" "kk_Cyrl" "km" "ko" "" "ky" "lo" "lv" "ln" "lt" "luo" "" "mk" "mg" "zlm" "ml" "mt" "mr" "el" "mn" "ne" "nso" "or" "om" "" "fa" "pl" "pt" "pa" "ro" "ru" "sr_Cyrl" "sn" "" "si" "sk" "sl" "" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz_Latn" "vi" "cy" "" "xh" "yo" "zu" "")
FLORES_101_LANGS_FLORES99=("ny" "af" "am" "ar" "hy" "as" "ast" "az" "bn" "bs" "bg" "my" "ca" "cx" "cb" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "ff" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "q3" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "qy" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "ns" "or" "om" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz" "vi" "cy" "wo" "xh" "yo" "zu" "be")
LID_187_LANGS_FLORES99=("" "af" "am" "ar" "hy" "as" "ast" "az" "bn" "bs" "bg" "my" "ca" "ceb" "ckb" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ha" "he" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "" "or" "om" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "tl" "ta" "te" "th" "tn" "tr" "uk" "ur" "uz" "vi" "cy" "wo" "xh" "yo" "zu" "be")
CLD3_LANGS_FLORES99=("ny" "af" "am" "ar" "hy" "" "" "az" "bn" "bs" "bg" "my" "ca" "ceb" "" "zh" "hr" "cs" "da" "nl" "en" "et" "fi" "fr" "" "gl" "" "ka" "de" "gu" "ha" "iw" "hi" "hu" "is" "ig" "id" "it" "ja" "jv" "" "kn" "kk" "km" "ko" "ku" "ky" "lo" "lv" "" "lt" "" "lb" "mk" "mg" "ms" "ml" "mt" "mr" "el" "mn" "ne" "" "" "" "ps" "fa" "pl" "pt" "pa" "ro" "ru" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "sv" "fil" "ta" "te" "th" "" "tr" "uk" "ur" "uz" "vi" "cy" "" "xh" "yo" "zu" "be")

ISO_639_3_LANGS_EARL=("eus" "wes" "epo" "kin" "tat" "war" "abk" "ady" "bak" "bem" "bho" "bis" "che" "nya" "chv" "ewe" "ewo" "fao" "fij" "fon" "gom" "kal" "grn" "haw" "kbp" "kab" "krc" "kik" "kon" "lim" "lmo" "lua" "mai" "arn" "min" "xmf" "lus" "mos" "nav" "nia" "pcm" "nno" "oss" "pag" "pap" "roh" "run" "bxr" "smo" "sag" "san" "sat" "srd" "scn" "azb" "alt" "sot" "tah" "bod" "tpi" "tog" "tso" "tum" "tuk" "twi" "udm" "uig" "vec" "zza" "dyu" "nan" "afr" "sqi" "amh" "ara" "hye" "asm" "ast" "aym" "aze" "ben" "nor" "bos" "bul" "mya" "cat" "ceb" "ckb" "zho" "cjk" "hrv" "ces" "dan" "nld" "arz" "eng" "est" "fin" "fra" "ful" "glg" "lug" "kat" "deu" "guj" "hat" "hau" "heb" "hin" "hun" "isl" "ibo" "ilo" "ind" "gle" "ita" "jpn" "jav" "kea" "kac" "kam" "kan" "kaz" "khm" "kmb" "kor" "kur" "kir" "lao" "lav" "lin" "lit" "luo" "ltz" "mkd" "mlg" "msa" "mal" "mlt" "mri" "mar" "ell" "mon" "npi" "nso" "oci" "ory" "orm" "pus" "fas" "pol" "por" "pan" "que" "ron" "rus" "gla" "srp" "sna" "snd" "sin" "slk" "slv" "som" "spa" "sun" "swh" "ssw" "swe" "tgl" "tgk" "tam" "tel" "tha" "tir" "tsn" "tur" "ukr" "umb" "urd" "uzb" "vie" "cym" "wol" "xho" "yid" "yor" "zul" "bel" "yue")
JW300_LANGS_EARL=("eu" "wes" "" "rw" "tt" "war" "ab" "ady" "ba" "bem" "" "bi" "" "nya" "cv" "ee" "ewo" "fo" "fj" "fon" "gom" "kl" "gug" "" "kbp" "kab" "krc" "ki" "kg" "" "" "lua" "" "arn" "" "xmf" "lus" "mos" "nv" "nia" "pcm" "" "os" "pag" "pap" "" "run" "" "sm" "sg" "" "" "" "" "" "alt" "st" "ty" "" "tpi" "tog" "ts" "tum" "tk" "tw" "udm" "" "vec" "" "dyu" "" "af" "sq" "am" "ar" "hy" "as" "" "ay" "az" "bn" "no" "" "bg" "my" "cat" "ceb" "" "" "cjk" "hr" "cs" "da" "nl" "" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ht" "ha" "he" "hi" "hu" "is" "ig" "ilo" "id" "ga" "it" "ja" "jv" "kea" "kac" "kam" "kn" "kk_Cyrl" "km" "kmb" "ko" "" "ky" "lo" "lv" "ln" "lt" "luo" "" "mk" "mg" "zlm" "ml" "mt" "" "mr" "el" "mn" "ne" "nso" "" "or" "om" "" "fa" "pl" "pt" "pa" "que" "ro" "ru" "" "sr_Cyrl" "sn" "" "si" "sk" "sl" "" "es" "su" "sw" "ss" "sv" "tl" "tg" "ta" "te" "th" "ti" "tn" "tr" "uk" "umb" "ur" "uz_Latn" "vi" "cy" "" "xh" "" "yo" "zu" "" "")
FLORES_101_LANGS_EARL=("" "" "" "" "" "" "" "" "" "" "" "" "" "ny" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "af" "" "am" "ar" "hy" "as" "ast" "" "az" "bn" "" "bs" "bg" "my" "ca" "cx" "cb" "zh" "" "hr" "cs" "da" "nl" "" "en" "et" "fi" "fr" "ff" "gl" "lg" "ka" "de" "gu" "" "ha" "he" "hi" "hu" "is" "ig" "" "id" "" "it" "ja" "jv" "q3" "" "" "kn" "kk" "km" "" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "qy" "lb" "mk" "mg" "ms" "ml" "mt" "" "mr" "el" "mn" "ne" "ns" "" "or" "om" "ps" "fa" "pl" "pt" "pa" "" "ro" "ru" "" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "" "sv" "tl" "" "ta" "te" "th" "" "tn" "tr" "uk" "" "ur" "uz" "vi" "cy" "wo" "xh" "" "yo" "zu" "be" "")
LID_187_LANGS_EARL=("eu" "" "eo" "" "tt" "war" "" "" "ba" "" "bh" "" "ce" "" "cv" "" "" "" "" "" "gom" "" "gn" "" "" "kab" "krc" "" "" "li" "lmo" "" "mai" "" "min" "xmf" "" "" "" "" "" "nn" "os" "" "" "rm" "" "bxr" "" "" "sa" "sat" "sc" "scn" "azb" "" "" "" "bo" "" "" "" "" "tk" "" "" "ug" "vec" "diq" "" "nah" "af" "sq" "am" "ar" "hy" "as" "ast" "" "az" "bn" "no" "bs" "bg" "my" "ca" "ceb" "ckb" "zh" "" "hr" "cs" "da" "nl" "arz" "en" "et" "fi" "fr" "" "gl" "lg" "ka" "de" "gu" "ht" "ha" "he" "hi" "hu" "is" "ig" "ilo" "id" "ga" "it" "ja" "jv" "" "" "" "kn" "kk" "km" "" "ko" "ku" "ky" "lo" "lv" "ln" "lt" "" "lb" "mk" "mg" "ms" "ml" "mt" "" "mr" "el" "mn" "ne" "" "oc" "or" "om" "ps" "fa" "pl" "pt" "pa" "qu" "ro" "ru" "gd" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "" "sv" "tl" "tg" "ta" "te" "th" "" "tn" "tr" "uk" "" "ur" "uz" "vi" "cy" "wo" "xh" "yi" "yo" "zu" "be" "yue")
CLD3_LANGS_EARL=("eu" "" "eo" "" "" "" "" "" "" "" "" "" "" "ny" "" "" "" "" "" "" "" "" "" "haw" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "sm" "" "" "" "" "" "" "" "st" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "af" "sq" "am" "ar" "hy" "" "" "" "az" "bn" "no" "bs" "bg" "my" "ca" "ceb" "" "zh" "" "hr" "cs" "da" "nl" "" "en" "et" "fi" "fr" "" "gl" "" "ka" "de" "gu" "ht" "ha" "iw" "hi" "hu" "is" "ig" "" "id" "ga" "it" "ja" "jv" "" "" "" "kn" "kk" "km" "" "ko" "ku" "ky" "lo" "lv" "" "lt" "" "lb" "mk" "mg" "ms" "ml" "mt" "mi" "mr" "el" "mn" "ne" "" "" "" "" "ps" "fa" "pl" "pt" "pa" "" "ro" "ru" "gd" "sr" "sn" "sd" "si" "sk" "sl" "so" "es" "su" "sw" "" "sv" "fil" "tg" "ta" "te" "th" "" "" "tr" "uk" "" "ur" "uz" "vi" "cy" "" "xh" "yi" "yo" "zu" "be" "")



prepare_data_1 () {
    for i in "${!ISO_639_3_LANGS_EARL[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS_EARL[i]}
        JW300_LANG=${JW300_LANGS_EARL[i]}
        LID_187_LANG=${LID_187_LANGS_EARL[i]}

        # JW300
        # -----------

        JW300_DETOK_FILE="$DETOK_DIR/$JW300_LANG.detok.txt"
        if [ -f "$JW300_DETOK_FILE" ]; then
            NUMBER_OF_LINES=$(cat $JW300_DETOK_FILE | wc -l)
            echo -e "$ISO_639_3_LANG \t jw300 lines : $NUMBER_OF_LINES"


            MONO_JW300_FILE="$TRAIN_DIR/$ISO_639_3_LANG.jw300.mono.txt"
            cat $JW300_DETOK_FILE | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' > $MONO_JW300_FILE
        else
            echo -e "lang:$ISO_639_3_LANG \t $JW300_LANG does not exist: $JW300_DETOK_FILE"
        fi


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

    for i in "${!ISO_639_3_LANGS_EARL[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS_EARL[i]}
        JW300_LANG=${JW300_LANGS_EARL[i]}
        LID_187_LANG=${LID_187_LANGS_EARL[i]}

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
    cat $VALID_DIR/*.valid.mono.txt | shuf > "$VALID_DIR/all.fromtrain.txt"

    rm $TRAIN_DIR/*.cat.txt
    rm $TRAIN_DIR/*.train.mono.txt
    rm $VALID_DIR/*.valid.mono.txt
}


prepare_flores_dev_data () {
    CONCAT_FILE="$VALID_DIR/concat.all.txt"
    rm $CONCAT_FILE

    for i in "${!ISO_639_3_LANGS_FLORES99[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS_FLORES99[i]}
        FLORES_101_LANG=${FLORES_101_LANGS_FLORES99[i]}

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

    cat $CONCAT_FILE "$VALID_DIR/all.fromtrain.txt" | shuf > "$VALID_DIR/all.txt"
    rm $CONCAT_FILE

}



FASTTEXT_BIN="/private/home/celebio/nlp/nllb_lid/fastText/fasttext"

train_fasttext () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1 -epoch 1 -dim 16 -loss softmax -bucket 5000000 -thread 10
}

train_fasttext_1 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.1 \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1 -epoch 1 -dim 16 -loss ova -bucket 5000000 -thread 10
}

train_fasttext_2 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.2 \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1 -epoch 1 -dim 64 -loss ova -bucket 5000000 -thread 10
}

train_fasttext_3 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.3 \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 64 -loss ova -bucket 5000000 -thread 10
}

train_fasttext_4 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.4 \
        -lr 1.0 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 64 -loss ova -bucket 5000000 -thread 10
}

train_fasttext_5 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.5 \
        -lr 1.0 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 64 -loss ova -bucket 10000000 -thread 10
}

train_fasttext_6 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.6 \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 128 -loss ova -bucket 10000000 -thread 10
}

train_fasttext_7 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.7 \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 128 -loss softmax -bucket 10000000 -thread 10
}

eval_dev_fasttext () {
    PREDICTIONS="$RESULT_FOLDER/earl.fasttext.predictions"
    GOLD="$RESULT_FOLDER/earl.gold"
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-earl.fasttext.txt"

    TEST_FILE="$VALID_DIR/all.txt"
    RESULT_GATHER="$FASTTEXT_BIN predict $RESULT_FOLDER/model.bin $TEST_FILE"
    $RESULT_GATHER > $PREDICTIONS

    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb_lid/misc/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT

    cat $RESULT_TXT
}

eval_dev_fasttext_variants () {
    GOLD="$RESULT_FOLDER/earl.gold"
    TEST_FILE="$VALID_DIR/all.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb_lid/misc/classifier_metrics.py"

    for i in `seq 1 7`;
    do
        PREDICTIONS="$RESULT_FOLDER/earl.fasttext.predictions.$i"
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-earl.fasttext.$i.txt"
        echo $i
        RESULT_GATHER="$FASTTEXT_BIN predict $RESULT_FOLDER/model.$i.bin $TEST_FILE"
        $RESULT_GATHER > $PREDICTIONS

        $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT

        cat $RESULT_TXT
    done
}


CLD3_BIN="/private/home/celebio/nlp/nllb_lid/cld3_scripts/run_cld3.py"

eval_dev_cld3 () {
    PREDICTIONS="$RESULT_FOLDER/earl.cld3.predictions"
    GOLD="$RESULT_FOLDER/earl.gold"

    TEST_FILE="$VALID_DIR/all.txt"
    cat "$TEST_FILE" | $CLD3_BIN > $PREDICTIONS
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD

    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-earl.cld3.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb_lid/misc/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT

    cat $RESULT_TXT
}



prepare_data_1
prepare_data_2
prepare_flores_dev_data
prepare_flores_devtest_data
train_fasttext

train_fasttext_1
train_fasttext_2
train_fasttext_3
train_fasttext_4
train_fasttext_5
train_fasttext_6
train_fasttext_7

eval_dev_fasttext

eval_dev_fasttext_variants
eval_dev_cld3






