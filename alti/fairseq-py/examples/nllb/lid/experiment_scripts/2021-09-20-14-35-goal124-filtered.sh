#!/bin/bash


EXPERIMENT_NAME="2021-09-20-14-35-goal124-filtered"
EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"

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
HISTOGRAMS_TRAIN_FOLDER="/large_experiments/mmt/lidruns/2021-09-20-14-14-histogram-baseline/histograms/train"

prepare_data_1_score () {

    OLD_DATA_TRAIN_FOLDER="/large_experiments/mmt/lidruns/2021-09-18-00-21-goal124-baseline/data/train"

    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
        fi

        MONO_CAT_FILE="$OLD_DATA_TRAIN_FOLDER/$ISO_639_3_LANG.cat.txt"

        FILTER_SCORED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.filterscore.txt"

        cat $MONO_CAT_FILE | python $FILTER_CHAR_HISTOGRAM \
            --lang $ISO_639_3_LANG \
            --threshold 0.5 \
            --histograms $HISTOGRAMS_TRAIN_FOLDER \
            --show-score | sort -n -k1,1 > $FILTER_SCORED_TRAIN_FILE &

    done

    wait
}


prepare_data_2_filter () {
    OLD_DATA_TRAIN_FOLDER="/large_experiments/mmt/lidruns/2021-09-18-00-21-goal124-baseline/data/train"

    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
        fi

        MONO_CAT_FILE="$OLD_DATA_TRAIN_FOLDER/$ISO_639_3_LANG.cat.txt"

        THRESH_SCORE="0.8"
        CLEANED_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.$THRESH_SCORE.txt"
        REJECTD_TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.rejectd.$THRESH_SCORE.txt"

        cat $MONO_CAT_FILE | python $FILTER_CHAR_HISTOGRAM \
            --lang $ISO_639_3_LANG \
            --threshold $THRESH_SCORE \
            --histograms $HISTOGRAMS_TRAIN_FOLDER \
                2> $REJECTD_TRAIN_FILE \
                1> $CLEANED_TRAIN_FILE &

    done

    wait
}

prepare_data_3_combine() {
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

        MONO_CAT_FILE="$TRAIN_DIR/$ISO_639_3_LANG.cleaned.0.8.txt"

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

FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"



train_fasttext_1 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.1 \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1 -epoch 1 -dim 16 -loss ova -bucket 5000000 -thread 10
}

train_fasttext_7 () {
    echo "Training fastText:"
    $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.7 \
        -lr 0.5 -minn 2 -maxn 5 -minCount 1000 -epoch 1 -dim 128 -loss softmax -bucket 10000000 -thread 40
}


train_fasttext_cluster() {
    FT_LOGS="$RESULT_FOLDER/ft_logs"
    mkdir -p $FT_LOGS

    # sbatch --job-name lid_10 --partition nllb --comment lid_10 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_10.log --error $FT_LOGS/train_10.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.10 -lr 0.5 -minn 1 -maxn 4 -minCount 1 -epoch 3 -dim 128 -loss softmax -bucket 100000 -thread 10"
    sbatch --job-name lid_11 --partition nllb --comment lid_11 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_11.log --error $FT_LOGS/train_11.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.11 -lr 0.5 -minn 2 -maxn 5 -minCount 10 -epoch 1 -dim 32 -loss softmax -bucket 1000 -thread 10"
    # sbatch --job-name lid_12 --partition nllb --comment lid_12 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_12.log --error $FT_LOGS/train_12.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.12 -lr 1.0 -minn 0 -maxn 0 -minCount 1 -epoch 1 -dim 64 -loss softmax -bucket 100000 -thread 10"
    # sbatch --job-name lid_13 --partition nllb --comment lid_13 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_13.log --error $FT_LOGS/train_13.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.13 -lr 0.5 -minn 2 -maxn 4 -minCount 100 -epoch 4 -dim 512 -loss softmax -bucket 1000000 -thread 10"
    # sbatch --job-name lid_14 --partition nllb --comment lid_14 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_14.log --error $FT_LOGS/train_14.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.14 -lr 1.0 -minn 0 -maxn 0 -minCount 10 -epoch 2 -dim 16 -loss softmax -bucket 100000 -thread 10"
    # sbatch --job-name lid_15 --partition nllb --comment lid_15 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_15.log --error $FT_LOGS/train_15.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.15 -lr 1.0 -minn 3 -maxn 5 -minCount 100000 -epoch 2 -dim 512 -loss softmax -bucket 10000 -thread 10"
    sbatch --job-name lid_16 --partition nllb --comment lid_16 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_16.log --error $FT_LOGS/train_16.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.16 -lr 1.0 -minn 3 -maxn 5 -minCount 1000 -epoch 1 -dim 64 -loss softmax -bucket 100000 -thread 10"
    sbatch --job-name lid_17 --partition nllb --comment lid_17 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_17.log --error $FT_LOGS/train_17.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.17 -lr 0.5 -minn 3 -maxn 5 -minCount 100000 -epoch 3 -dim 32 -loss softmax -bucket 100000 -thread 10"
    # sbatch --job-name lid_18 --partition nllb --comment lid_18 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_18.log --error $FT_LOGS/train_18.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.18 -lr 0.5 -minn 2 -maxn 5 -minCount 1 -epoch 4 -dim 64 -loss softmax -bucket 10000 -thread 10"
    sbatch --job-name lid_19 --partition nllb --comment lid_19 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_19.log --error $FT_LOGS/train_19.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.19 -lr 1.0 -minn 3 -maxn 6 -minCount 100 -epoch 3 -dim 32 -loss softmax -bucket 1000 -thread 10"
    # sbatch --job-name lid_20 --partition nllb --comment lid_20 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_20.log --error $FT_LOGS/train_20.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.20 -lr 1.0 -minn 1 -maxn 4 -minCount 10 -epoch 4 -dim 128 -loss softmax -bucket 10000 -thread 10"
    # sbatch --job-name lid_21 --partition nllb --comment lid_21 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_21.log --error $FT_LOGS/train_21.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.21 -lr 1.0 -minn 2 -maxn 4 -minCount 10000 -epoch 1 -dim 256 -loss softmax -bucket 1000000 -thread 10"
    sbatch --job-name lid_22 --partition nllb --comment lid_22 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_22.log --error $FT_LOGS/train_22.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.22 -lr 1.0 -minn 2 -maxn 4 -minCount 100 -epoch 1 -dim 16 -loss softmax -bucket 100000 -thread 10"
    # sbatch --job-name lid_23 --partition nllb --comment lid_23 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_23.log --error $FT_LOGS/train_23.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.23 -lr 0.5 -minn 0 -maxn 0 -minCount 1 -epoch 5 -dim 128 -loss softmax -bucket 100000 -thread 10"
    # sbatch --job-name lid_24 --partition nllb --comment lid_24 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_24.log --error $FT_LOGS/train_24.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.24 -lr 0.5 -minn 0 -maxn 0 -minCount 10000 -epoch 4 -dim 256 -loss softmax -bucket 10000 -thread 10"
    # sbatch --job-name lid_25 --partition nllb --comment lid_25 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_25.log --error $FT_LOGS/train_25.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.25 -lr 0.5 -minn 2 -maxn 4 -minCount 10000 -epoch 4 -dim 64 -loss softmax -bucket 10000000 -thread 10"
    sbatch --job-name lid_26 --partition nllb --comment lid_26 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_26.log --error $FT_LOGS/train_26.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.26 -lr 0.5 -minn 0 -maxn 0 -minCount 1 -epoch 1 -dim 32 -loss softmax -bucket 100000 -thread 10"
    sbatch --job-name lid_27 --partition nllb --comment lid_27 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_27.log --error $FT_LOGS/train_27.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.27 -lr 0.5 -minn 3 -maxn 6 -minCount 100000 -epoch 1 -dim 128 -loss softmax -bucket 10000 -thread 10"
    # sbatch --job-name lid_28 --partition nllb --comment lid_28 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_28.log --error $FT_LOGS/train_28.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.28 -lr 1.0 -minn 0 -maxn 0 -minCount 10000 -epoch 5 -dim 128 -loss softmax -bucket 1000000 -thread 10"
    # sbatch --job-name lid_29 --partition nllb --comment lid_29 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_29.log --error $FT_LOGS/train_29.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.29 -lr 0.5 -minn 0 -maxn 0 -minCount 100 -epoch 4 -dim 512 -loss softmax -bucket 1000000 -thread 10"
    # sbatch --job-name lid_30 --partition nllb --comment lid_30 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_30.log --error $FT_LOGS/train_30.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.30 -lr 0.5 -minn 1 -maxn 4 -minCount 100 -epoch 5 -dim 64 -loss softmax -bucket 1000000 -thread 10"
    # sbatch --job-name lid_31 --partition nllb --comment lid_31 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_31.log --error $FT_LOGS/train_31.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.31 -lr 1.0 -minn 0 -maxn 0 -minCount 10000 -epoch 2 -dim 32 -loss softmax -bucket 10000 -thread 10"
    # sbatch --job-name lid_32 --partition nllb --comment lid_32 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_32.log --error $FT_LOGS/train_32.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.32 -lr 1.0 -minn 3 -maxn 6 -minCount 1 -epoch 5 -dim 256 -loss softmax -bucket 10000 -thread 10"
    # sbatch --job-name lid_33 --partition nllb --comment lid_33 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_33.log --error $FT_LOGS/train_33.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.33 -lr 0.5 -minn 1 -maxn 4 -minCount 10000 -epoch 5 -dim 128 -loss softmax -bucket 1000 -thread 10"
    # sbatch --job-name lid_34 --partition nllb --comment lid_34 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_34.log --error $FT_LOGS/train_34.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.34 -lr 1.0 -minn 1 -maxn 6 -minCount 100 -epoch 5 -dim 32 -loss softmax -bucket 1000000 -thread 10"
    # sbatch --job-name lid_35 --partition nllb --comment lid_35 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_35.log --error $FT_LOGS/train_35.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.35 -lr 1.0 -minn 0 -maxn 0 -minCount 1 -epoch 3 -dim 32 -loss softmax -bucket 1000 -thread 10"
    sbatch --job-name lid_36 --partition nllb --comment lid_36 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_36.log --error $FT_LOGS/train_36.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.36 -lr 1.0 -minn 1 -maxn 4 -minCount 100000 -epoch 5 -dim 16 -loss softmax -bucket 100000 -thread 10"
    # sbatch --job-name lid_37 --partition nllb --comment lid_37 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_37.log --error $FT_LOGS/train_37.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.37 -lr 1.0 -minn 0 -maxn 0 -minCount 10000 -epoch 4 -dim 128 -loss softmax -bucket 100000 -thread 10"
    # sbatch --job-name lid_38 --partition nllb --comment lid_38 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_38.log --error $FT_LOGS/train_38.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.38 -lr 1.0 -minn 0 -maxn 0 -minCount 100 -epoch 1 -dim 256 -loss softmax -bucket 10000000 -thread 10"
    # sbatch --job-name lid_39 --partition nllb --comment lid_39 --nodes=1 --gpus-per-node=0 --cpus-per-task=1 --ntasks-per-node=10 --time=4320 --output $FT_LOGS/train_39.log --error $FT_LOGS/train_39.err --open-mode append --no-requeue --wrap "srun --unbuffered $FASTTEXT_BIN supervised -input $TRAIN_DIR/all.txt -output $RESULT_FOLDER/model.39 -lr 0.5 -minn 2 -maxn 5 -minCount 10 -epoch 2 -dim 256 -loss softmax -bucket 100000 -thread 10"

}

eval_flores_devtest_fasttext_variants () {
    GOLD="$RESULT_FOLDER/flores-devtest.gold"
    TEST_FILE="$TEST_DIR/flores-devtest.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 39`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-devtest.fasttext.$i.txt"

        LID_MODEL="$RESULT_FOLDER/model.$i.bin"
        PREDICTIONS="$RESULT_FOLDER/flores-devtest.fasttext.predictions.$i"

        if [ -s $LID_MODEL ]
        then
            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
            $RESULT_GATHER > $PREDICTIONS

            CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
            $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
        fi
    done
}




eval_flores_filled_fasttext_variants () {
    GOLD="$RESULT_FOLDER/flores-filled.gold"
    TEST_FILE="$TEST_DIR/flores-filled.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 39`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.$i.txt"

        LID_MODEL="$RESULT_FOLDER/model.$i.bin"

        if [ -s $LID_MODEL ]
        then
            echo "LID_MODEL $LID_MODEL"
            PREDICTIONS="$RESULT_FOLDER/flores-filled.fasttext.predictions.$i"

            RESULT_GATHER="$FASTTEXT_BIN predict $LID_MODEL $TEST_FILE"
            $RESULT_GATHER > $PREDICTIONS

            CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
            $CLASSIFIER_METRICS --prediction $PREDICTIONS --gold $GOLD > $RESULT_TXT
        fi
    done
}




prepare_data_1_score
prepare_data_2_filter

prepare_data_3_combine

prepare_flores_devtest_data
prepare_flores_filled_data


train_fasttext_cluster
train_fasttext_1
train_fasttext_7

eval_flores_devtest_fasttext_variants

eval_flores_filled_fasttext_variants

