#!/bin/bash


EXPERIMENT_NAME="2021-06-23-11-21-fairseq-lid"
EXPERIMENT_FOLDER="/large_experiments/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER

FLORES_DIR="/checkpoint/angelafan/flores_preliminary_data"
DETOK_DIR="/large_experiments/mmt/data/monolingual/lid/jw300/detok"
SPM_DATA_DIR="$DATA_FOLDER/spm"
TRAIN_DIR="$DATA_FOLDER/train"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
DATA_BIN_DIR="$DATA_FOLDER/data-bin"
mkdir -p $SPM_DATA_DIR
mkdir -p $TRAIN_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR

echo "DATA_FOLDER is $DATA_FOLDER"


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
    rm $TRAIN_DIR/*.mono.txt
    rm $TRAIN_DIR/*.train.mono.txt
    rm $VALID_DIR/*.valid.mono.txt
}


prepare_spm_data_1 () {
    SPM_LIMIT_SENTENCE_NB=10000
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


            MONO_JW300_FILE="$SPM_DATA_DIR/$ISO_639_3_LANG.jw300.mono.txt"
            cat $JW300_DETOK_FILE | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}' | head -n $SPM_LIMIT_SENTENCE_NB > $MONO_JW300_FILE
        else
            echo -e "lang:$ISO_639_3_LANG \t $JW300_LANG does not exist: $JW300_DETOK_FILE"
        fi


        # LID187
        # -----------

        echo -e "$ISO_639_3_LANG \t $LID_187_LANG "

        if [ "$LID_187_LANG" != "" ]; then
            MONO_LID187_FILE="$SPM_DATA_DIR/$ISO_639_3_LANG.lid187.mono.txt"

            OLD_LID_TRAIN="/private/home/egrave/lid/train.txt"
            cat $OLD_LID_TRAIN | grep "__label__${LID_187_LANG} " | cut -f 2- -d" " | awk -v lang="$ISO_639_3_LANG" '{print "__label__"lang " " $0}'  | head -n $SPM_LIMIT_SENTENCE_NB > $MONO_LID187_FILE

            MONO_LID187_FILE_NB_LINES=$(wc -l $MONO_LID187_FILE)

            echo -e "$ISO_639_3_LANG \t $LID_187_LANG \t : $MONO_LID187_FILE_NB_LINES"
        fi

    done
}

prepare_spm_data_2 () {
    for i in "${!ISO_639_3_LANGS_EARL[@]}"; do
        ISO_639_3_LANG=${ISO_639_3_LANGS_EARL[i]}
        JW300_LANG=${JW300_LANGS_EARL[i]}
        LID_187_LANG=${LID_187_LANGS_EARL[i]}

        MONO_JW300_FILE="$SPM_DATA_DIR/$ISO_639_3_LANG.jw300.mono.txt"
        MONO_LID187_FILE="$SPM_DATA_DIR/$ISO_639_3_LANG.lid187.mono.txt"

        MONO_CAT_FILE="$SPM_DATA_DIR/$ISO_639_3_LANG.cat.txt"

        cat $MONO_JW300_FILE $MONO_LID187_FILE | shuf > $MONO_CAT_FILE


        TRAIN_FILE="$SPM_DATA_DIR/$ISO_639_3_LANG.train.mono.txt"

        cat $MONO_CAT_FILE > $TRAIN_FILE

    done

    cat $SPM_DATA_DIR/*.train.mono.txt | shuf > "$SPM_DATA_DIR/all.txt"

    rm $SPM_DATA_DIR/*.cat.txt
    rm $SPM_DATA_DIR/*.train.mono.txt
    rm $SPM_DATA_DIR/*.mono.txt
}


split_data_labels() {
    FILE_TO_SPLIT=$1
    echo "FILE_TO_SPLIT = $FILE_TO_SPLIT"
    filename="$(basename -- $FILE_TO_SPLIT)"
    dirname="$(dirname -- $FILE_TO_SPLIT)"
    echo "filename = $filename"
    echo "dirname = $dirname"

    SPLIT_0_FILE="$dirname/$filename.0"
    SPLIT_1_FILE="$dirname/$filename.1"
    cat $FILE_TO_SPLIT | cut -f 1 -d" " > $SPLIT_0_FILE
    cat $FILE_TO_SPLIT | cut -f 2- -d" " > $SPLIT_1_FILE

}

split_train_data_labels() {
    split_data_labels $SPM_DATA_DIR/all.txt
    split_data_labels $TRAIN_DIR/all.txt
    split_data_labels $VALID_DIR/all.txt
}

train_spm() {
    SPM_BIN_DIR="/private/home/celebio/thirdparty/sentencepiece/build/src"
    SPM_TRAIN="$SPM_BIN_DIR/spm_train"
    SPM_ENCODE="$SPM_BIN_DIR/spm_encode"

    SPM_TRAIN_INPUT="$SPM_DATA_DIR/all.txt.1"
    SPM_MODEL="$SPM_DATA_DIR/spmmodel"

    $SPM_TRAIN \
        --input $SPM_TRAIN_INPUT \
        --vocab_size 128000 \
        --model_type unigram \
        --model_prefix $SPM_MODEL \
        --input_sentence_size=10000000 --eos_id=-1 --pad_id=-1 --bos_id=-1 --character_coverage=1
}

encode_spm() {
    SPM_BIN_DIR="/private/home/celebio/thirdparty/sentencepiece/build/src"
    SPM_TRAIN="$SPM_BIN_DIR/spm_train"
    SPM_ENCODE="$SPM_BIN_DIR/spm_encode"
    SPM_MODEL="$SPM_DATA_DIR/spmmodel.model"
    SPM_VOCAB="$SPM_DATA_DIR/spmmodel.vocab"
    SPM_TRAIN_DATA_IN="$TRAIN_DIR/all.txt.1"
    SPM_TRAIN_DATA_OUT="$TRAIN_DIR/all.txt.1.spm"
    SPM_VALID_DATA_IN="$VALID_DIR/all.txt.1"
    SPM_VALID_DATA_OUT="$VALID_DIR/all.txt.1.spm"
    SPM_DATA_DICT="$SPM_DATA_DIR/dict.1.txt"
    SPM_LABEL_DATA_DICT="$SPM_DATA_DIR/dict.0.txt"

    $SPM_ENCODE --model $SPM_MODEL < $SPM_TRAIN_DATA_IN > $SPM_TRAIN_DATA_OUT

    cat $SPM_VOCAB | tr ' ' '\n' | uniq -c | awk '{print $2 " " $1}'  \
                   | grep -v "<unk>" | grep -v "<s>" | grep -v "</s>" > $SPM_DATA_DICT

    cat "$TRAIN_DIR/all.txt.0" "$VALID_DIR/all.txt.0" | sort | uniq | awk '{print $1 " 1"}' > $SPM_LABEL_DATA_DICT

    $SPM_ENCODE --model $SPM_MODEL < $SPM_VALID_DATA_IN > $SPM_VALID_DATA_OUT
}

create_shards() {
    split --lines 10000000 --numeric-suffixes \
        "$TRAIN_DIR/all.txt.1.spm" \
        "$TRAIN_DIR/shard-all-sentence-"

    split --lines 10000000 --numeric-suffixes \
        "$TRAIN_DIR/all.txt.0" \
        "$TRAIN_DIR/shard-all-label-"

}

binarize_shards_data() {
    for i in $(seq -f "%02g" 0 20)
    do
        SENTENCE_FILE="$TRAIN_DIR/shard-all-sentence-${i}"
        LABEL_FILE="$TRAIN_DIR/shard-all-label-${i}"

        DATA_BIN_DIR_SHARD="$DATA_BIN_DIR/shard-${i}"
        mkdir -p "$DATA_BIN_DIR_SHARD"

        JOB_NAME="lid-shard-binarization-${i}"
        sbatch --job-name $JOB_NAME --partition No_Language_Left_Behind \
            --comment $JOB_NAME \
            --time 60 \
            --nodes 1 \
            --ntasks-per-node 1 --cpus-per-task 60 \
            --output "$DATA_BIN_DIR_SHARD/job.log" \
            --error "$DATA_BIN_DIR_SHARD/job.err" \
            --open-mode append --no-requeue \
            --wrap "srun --unbuffered fairseq-preprocess \
                    --only-source \
                    --srcdict "$SPM_DATA_DIR/dict.0.txt" \
                    --trainpref "$LABEL_FILE" \
                    --destdir "$DATA_BIN_DIR_SHARD" \
                    --workers 60 \
                    && mv $DATA_BIN_DIR_SHARD/train.bin $DATA_BIN_DIR_SHARD/train.label.bin \
                    && mv $DATA_BIN_DIR_SHARD/train.idx $DATA_BIN_DIR_SHARD/train.label.idx \
                    && mv $DATA_BIN_DIR_SHARD/preprocess.log $DATA_BIN_DIR_SHARD/preprocess.label.log \
                    && mv $DATA_BIN_DIR_SHARD/dict.txt $DATA_BIN_DIR_SHARD/dict.label.txt \
                    && fairseq-preprocess \
                        --only-source \
                        --srcdict "$SPM_DATA_DIR/dict.1.txt" \
                        --trainpref "$SENTENCE_FILE" \
                        --destdir "$DATA_BIN_DIR_SHARD" \
                        --workers 60 \
                    && mv $DATA_BIN_DIR_SHARD/train.bin $DATA_BIN_DIR_SHARD/train.sentence.bin \
                    && mv $DATA_BIN_DIR_SHARD/train.idx $DATA_BIN_DIR_SHARD/train.sentence.idx \
                    && mv $DATA_BIN_DIR_SHARD/preprocess.log $DATA_BIN_DIR_SHARD/preprocess.sentence.log \
                    && mv $DATA_BIN_DIR_SHARD/dict.txt $DATA_BIN_DIR_SHARD/dict.sentence.txt"
    done
}

binarize_valid_data() {
    DATA_BIN_DIR_SHARD="$DATA_BIN_DIR/shard-00"

    mkdir -p "$DATA_BIN_DIR_SHARD"

    fairseq-preprocess \
        --only-source \
        --srcdict "$SPM_DATA_DIR/dict.0.txt" \
        --validpref "$VALID_DIR/all.txt.0" \
        --destdir "$DATA_BIN_DIR_SHARD" \
        --workers 60

    mv "$DATA_BIN_DIR_SHARD/valid.bin" "$DATA_BIN_DIR_SHARD/valid.label.bin"
    mv "$DATA_BIN_DIR_SHARD/valid.idx" "$DATA_BIN_DIR_SHARD/valid.label.idx"
    mv "$DATA_BIN_DIR_SHARD/preprocess.log" "$DATA_BIN_DIR_SHARD/preprocess.label.log"
    mv "$DATA_BIN_DIR_SHARD/dict.txt" "$DATA_BIN_DIR_SHARD/dict.label.txt"

    fairseq-preprocess \
        --only-source \
        --srcdict "$SPM_DATA_DIR/dict.1.txt" \
        --validpref "$VALID_DIR/all.txt.1.spm" \
        --destdir "$DATA_BIN_DIR_SHARD" \
        --workers 60 \

    mv "$DATA_BIN_DIR_SHARD/valid.bin" "$DATA_BIN_DIR_SHARD/valid.sentence.bin"
    mv "$DATA_BIN_DIR_SHARD/valid.idx" "$DATA_BIN_DIR_SHARD/valid.sentence.idx"
    mv "$DATA_BIN_DIR_SHARD/preprocess.log" "$DATA_BIN_DIR_SHARD/preprocess.sentence.log"
    mv "$DATA_BIN_DIR_SHARD/dict.txt" "$DATA_BIN_DIR_SHARD/dict.sentence.txt"

}


launch_train() {
  TRAIN_SH="$RESULT_FOLDER/train.sh"

  cat > ${TRAIN_SH} <<- EOM
#!/bin/bash

cd ${RESULT_FOLDER}

srun fairseq-train \\
    ${DATA_BIN_DIR}/shard-00:${DATA_BIN_DIR}/shard-01:${DATA_BIN_DIR}/shard-02:${DATA_BIN_DIR}/shard-03:${DATA_BIN_DIR}/shard-04:${DATA_BIN_DIR}/shard-05:${DATA_BIN_DIR}/shard-06:${DATA_BIN_DIR}/shard-07:${DATA_BIN_DIR}/shard-08:${DATA_BIN_DIR}/shard-09:${DATA_BIN_DIR}/shard-10:${DATA_BIN_DIR}/shard-11:${DATA_BIN_DIR}/shard-12:${DATA_BIN_DIR}/shard-13:${DATA_BIN_DIR}/shard-14:${DATA_BIN_DIR}/shard-15:${DATA_BIN_DIR}/shard-16:${DATA_BIN_DIR}/shard-17:${DATA_BIN_DIR}/shard-18:${DATA_BIN_DIR}/shard-19:${DATA_BIN_DIR}/shard-20 \\
    --arch roberta_base \\
    --task lid \\
    --criterion sentence_prediction \\
    --max-tokens 32768 \\
    --max-positions 256 \\
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \\
    --lr-scheduler inverse_sqrt \\
    --lr 0.00001 \\
    --no-shuffle \\
    --log-interval 200 \\
    --log-format simple \\
    --save-dir ${RESULT_FOLDER} \\
    --find-unused-parameters \\
    --update-freq 1 \\
    --encoder-embed-dim 256 \\
    --encoder-ffn-embed-dim 256 \\
    --encoder-attention-heads 4 \\
    --encoder-layers 2 \\
    --keep-last-epochs 2 \\
    --keep-best-checkpoints 2 \\
    --keep-interval-updates 2 \\
    --save-interval-updates 4000 \\
    --skip-invalid-size-inputs-valid-test \\
    --distributed-world-size 32 \\
    --distributed-port 9218 \\
    --fp16 \\
    --tensorboard-logdir ${RESULT_FOLDER}/tfboard_log \\
    >> ${RESULT_FOLDER}/train.log 2>&1

EOM

  chmod +x $TRAIN_SH

  sbatch -J "LID" \
      --partition=No_Language_Left_Behind --comment "LID" \
      --nodes=4 --gpus-per-node=8 --cpus-per-task=10 \
      --ntasks-per-node=8 --time=4320 \
      $TRAIN_SH

  echo "launched $TRAIN_SH"
}



prepare_data_1
prepare_data_2
prepare_spm_data_1
prepare_spm_data_2
split_train_data_labels
train_spm
encode_spm
create_shards
binarize_shards_data
binarize_valid_data
launch_train



