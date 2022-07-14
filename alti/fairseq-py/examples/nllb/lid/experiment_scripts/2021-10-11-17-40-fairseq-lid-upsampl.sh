#!/bin/bash


EXPERIMENT_NAME="2021-10-11-17-40-fairseq-lid-upsampl"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER


DATA_FOLDER="$EXPERIMENT_FOLDER/data"
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"
RESULT_FOLDER_2="$EXPERIMENT_FOLDER/result2"

mkdir -p $RESULT_FOLDER
mkdir -p $RESULT_FOLDER_2

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


GOAL124__ISO_639_3_LANGS=("aym" "ceb" "ckb" "cym" "gla" "gle" "hat" "jav" "kac" "kaz" "kea" "kir" "kur" "mlt" "mri" "mya" "que" "sun" "tgk" "uzb" "afr" "amh" "ara" "asm" "ast" "azj" "bel" "ben" "bos" "bul" "cat" "ces" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "ful" "glg" "guj" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ind" "isl" "ita" "jpn" "kam" "kan" "kat" "khm" "kmb" "kon" "kor" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mal" "mar" "mkd" "mlg" "mon" "msa" "nld" "nor" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "ron" "rus" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp" "ssw" "swe" "swh" "tam" "tel" "tgl" "tha" "tir" "tsn" "tur" "ukr" "umb" "urd" "vie" "wol" "xho" "yid" "yor" "yue" "zho" "zul" "eus" "wes" "epo" "kin" "tat" "war" "abk" "ady" "bak" "bem" "bho" "bis" "che" "chv" "ewe" "ewo" "fao" "fij" "fon" "gom" "kal" "grn" "kbp" "kab" "krc" "kik" "lim" "lmo" "lua" "mai" "arn" "min" "xmf" "lus" "mos" "nav" "nia" "pcm" "nno" "oss" "pag" "pap" "roh" "run" "bxr" "smo" "sag" "san" "sat" "srd" "scn" "azb" "alt" "sot" "tah" "bod" "tpi" "tog" "tso" "tum" "tuk" "twi" "udm" "uig" "vec" "zza" "" "cjk" "arz" "ilo")
GOAL124__JW300_LANGS=("ay" "ceb" "" "cy" "" "ga" "ht" "jv" "kac" "kk_Cyrl" "kea" "ky" "" "mt" "" "my" "que" "su" "tg" "uz_Latn" "af" "am" "ar" "as" "" "az" "" "bn" "" "bg" "cat" "cs" "da" "de" "dyu" "el" "en" "et" "fa" "fi" "fr" "" "gl" "gu" "ha" "he" "hi" "hr" "hu" "hy" "ig" "id" "is" "it" "ja" "kam" "kn" "ka" "km" "kmb" "kg" "ko" "lo" "lv" "ln" "lt" "" "lg" "luo" "ml" "mr" "mk" "mg" "mn" "zlm" "nl" "no" "ne" "nso" "nya" "" "om" "or" "pa" "pl" "pt" "" "ro" "ru" "si" "sk" "sl" "sn" "" "" "es" "sq" "sr_Cyrl" "ss" "sv" "sw" "ta" "te" "tl" "th" "ti" "tn" "tr" "uk" "umb" "ur" "vi" "" "xh" "" "yo" "" "" "zu" "eu" "wes" "" "rw" "tt" "war" "ab" "ady" "ba" "bem" "" "bi" "" "cv" "ee" "ewo" "fo" "fj" "fon" "gom" "kl" "gug" "kbp" "kab" "krc" "ki" "" "" "lua" "" "arn" "" "xmf" "lus" "mos" "nv" "nia" "pcm" "" "os" "pag" "pap" "" "run" "" "sm" "sg" "" "" "" "" "" "alt" "st" "ty" "" "tpi" "tog" "ts" "tum" "tk" "tw" "udm" "" "vec" "" "" "cjk" "" "ilo")
GOAL124__LID_187_LANGS=("" "ceb" "ckb" "cy" "gd" "ga" "ht" "jv" "" "kk" "" "ky" "ku" "mt" "" "my" "qu" "su" "tg" "uz" "af" "am" "ar" "as" "ast" "az" "be" "bn" "bs" "bg" "ca" "cs" "da" "de" "" "el" "en" "et" "fa" "fi" "fr" "" "gl" "gu" "ha" "he" "hi" "hr" "hu" "hy" "ig" "id" "is" "it" "ja" "" "kn" "ka" "km" "" "" "ko" "lo" "lv" "ln" "lt" "lb" "lg" "" "ml" "mr" "mk" "mg" "mn" "ms" "nl" "no" "ne" "" "" "oc" "om" "or" "pa" "pl" "pt" "ps" "ro" "ru" "si" "sk" "sl" "sn" "sd" "so" "es" "sq" "sr" "" "sv" "sw" "ta" "te" "tl" "th" "" "tn" "tr" "uk" "" "ur" "vi" "wo" "xh" "yi" "yo" "yue" "zh" "zu" "eu" "" "eo" "" "tt" "war" "" "" "ba" "" "bh" "" "ce" "cv" "" "" "" "" "" "gom" "" "gn" "" "kab" "krc" "" "li" "lmo" "" "mai" "" "min" "xmf" "" "" "" "" "" "nn" "os" "" "" "rm" "" "bxr" "" "" "sa" "sat" "sc" "scn" "azb" "" "" "" "bo" "" "" "" "" "tk" "" "" "ug" "vec" "diq" "nah" "" "arz" "ilo")
GOAL124__FLORES_LANGS=("aym" "ceb" "ckb" "cym" "gla" "gle" "hat" "jav" "kac" "kaz" "kea" "kir" "kur" "mlt" "mri" "mya" "que" "sun" "tgk" "uzb" "afr" "amh" "ara" "asm" "ast" "azj" "bel" "ben" "bos" "bul" "cat" "ces" "dan" "deu" "dyu" "ell" "eng" "est" "fas" "fin" "fra" "ful" "glg" "guj" "hau" "heb" "hin" "hrv" "hun" "hye" "ibo" "ind" "isl" "ita" "jpn" "kam" "kan" "kat" "khm" "kmb" "kon" "kor" "lao" "lav" "lin" "lit" "ltz" "lug" "luo" "mal" "mar" "mkd" "mlg" "mon" "msa" "nld" "nob" "npi" "nso" "nya" "oci" "orm" "ory" "pan" "pol" "por" "pus" "ron" "rus" "sin" "slk" "slv" "sna" "snd" "som" "spa" "sqi" "srp" "ssw" "swe" "swh" "tam" "tel" "tgl" "tha" "tir" "tsn" "tur" "ukr" "umb" "urd" "vie" "wol" "xho" "yid" "yor" "yue" "zho_simpl" "zul" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "" "")

UPSAMPLING_COMPUTE="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/utils/compute_upsample.py"

prepare_data_3_combine() {
    OLD_DATA_TRAIN_FOLDER="/large_experiments/nllb/mmt/lidruns/2021-10-08-15-09-multifilter2/data/train"

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

        MONO_CAT_FILE="$OLD_DATA_TRAIN_FOLDER/$ISO_639_3_LANG.cleaned.step3.txt"

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


prepare_spm_data_1 () {
    NB_LINES_FILE="$SPM_DATA_DIR/nblines.txt"
    NB_LINES_UPSAMPLED_FILE="$SPM_DATA_DIR/nblines_upsampled.txt"
    > $NB_LINES_FILE
    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
            echo "ISO_639_3_LANG set to $ISO_639_3_LANG"
        fi

        TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.train.mono.txt"
        SPM_TRAIN_FILE="$SPM_DATA_DIR/$ISO_639_3_LANG.train.mono.txt"

        TRAIN_NB_LINES=$(cat $TRAIN_FILE | head -n 2000000 | wc -l)
        printf "%s %d \n" $ISO_639_3_LANG $TRAIN_NB_LINES >> $NB_LINES_FILE

    done

    cat $NB_LINES_FILE | $UPSAMPLING_COMPUTE --alpha 0.4 > $NB_LINES_UPSAMPLED_FILE

    for i in "${!GOAL124__ISO_639_3_LANGS[@]}"; do
        ISO_639_3_LANG=${GOAL124__ISO_639_3_LANGS[i]}
        LID_187_LANG=${GOAL124__LID_187_LANGS[i]}

        if [ -z "$ISO_639_3_LANG" ]
        then
            ISO_639_3_LANG=$LID_187_LANG
            echo "ISO_639_3_LANG set to $ISO_639_3_LANG"
        fi

        TRAIN_FILE="$TRAIN_DIR/$ISO_639_3_LANG.train.mono.txt"
        SPM_TRAIN_FILE="$SPM_DATA_DIR/$ISO_639_3_LANG.train.mono.txt"
        > $SPM_TRAIN_FILE

        cat $NB_LINES_UPSAMPLED_FILE | grep -P "^$ISO_639_3_LANG\t"

        NB_REPEAT=$(cat $NB_LINES_UPSAMPLED_FILE | grep -P "^$ISO_639_3_LANG\t" | awk '{ print $3 }')
        NB_LIMIT=$(cat $NB_LINES_UPSAMPLED_FILE | grep -P "^$ISO_639_3_LANG\t" | awk '{ print $4 }')
        # echo "   $NB_REPEAT"
        # echo "   $NB_LIMIT"

        yes $TRAIN_FILE | head -n $NB_REPEAT | xargs cat | head -n $NB_LIMIT | cut -f2- -d" " >$SPM_TRAIN_FILE

    done

    cat $SPM_DATA_DIR/*.train.mono.txt | shuf > "$SPM_DATA_DIR/all.txt"
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
    split_data_labels $TRAIN_DIR/all.txt
    split_data_labels $VALID_DIR/all.txt
}

train_spm() {
    SPM_BIN_DIR="/private/home/celebio/thirdparty/sentencepiece/build/src"
    SPM_TRAIN="$SPM_BIN_DIR/spm_train"
    SPM_ENCODE="$SPM_BIN_DIR/spm_encode"

    SPM_TRAIN_INPUT="$SPM_DATA_DIR/all.txt"
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
        sbatch --job-name $JOB_NAME --partition nllb \
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
    ${DATA_BIN_DIR}/shard-00:${DATA_BIN_DIR}/shard-01:${DATA_BIN_DIR}/shard-02:${DATA_BIN_DIR}/shard-03:${DATA_BIN_DIR}/shard-04:${DATA_BIN_DIR}/shard-05:${DATA_BIN_DIR}/shard-06:${DATA_BIN_DIR}/shard-07:${DATA_BIN_DIR}/shard-08:${DATA_BIN_DIR}/shard-09:${DATA_BIN_DIR}/shard-10:${DATA_BIN_DIR}/shard-11:${DATA_BIN_DIR}/shard-12:${DATA_BIN_DIR}/shard-13:${DATA_BIN_DIR}/shard-14:${DATA_BIN_DIR}/shard-15 \\
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

  sbatch -J "LID-spm-upsampl" \
      --partition=learnaccel --comment "LID-spm-upsampl" \
      --nodes=4 --gpus-per-node=8 --cpus-per-task=10 \
      --ntasks-per-node=8 --time=4320 \
      $TRAIN_SH

  echo "launched $TRAIN_SH"
}

launch_train_2() {
  TRAIN_SH="$RESULT_FOLDER_2/train.sh"

  cat > ${TRAIN_SH} <<- EOM
#!/bin/bash

cd ${RESULT_FOLDER_2}

srun fairseq-train \\
    ${DATA_BIN_DIR}/shard-00:${DATA_BIN_DIR}/shard-01:${DATA_BIN_DIR}/shard-02:${DATA_BIN_DIR}/shard-03:${DATA_BIN_DIR}/shard-04:${DATA_BIN_DIR}/shard-05:${DATA_BIN_DIR}/shard-06:${DATA_BIN_DIR}/shard-07:${DATA_BIN_DIR}/shard-08:${DATA_BIN_DIR}/shard-09:${DATA_BIN_DIR}/shard-10:${DATA_BIN_DIR}/shard-11:${DATA_BIN_DIR}/shard-12:${DATA_BIN_DIR}/shard-13:${DATA_BIN_DIR}/shard-14:${DATA_BIN_DIR}/shard-15 \\
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
    --save-dir ${RESULT_FOLDER_2} \\
    --find-unused-parameters \\
    --update-freq 1 \\
    --encoder-embed-dim 256 \\
    --encoder-ffn-embed-dim 256 \\
    --encoder-attention-heads 4 \\
    --encoder-layers 6 \\
    --keep-last-epochs 2 \\
    --keep-best-checkpoints 2 \\
    --keep-interval-updates 2 \\
    --save-interval-updates 4000 \\
    --skip-invalid-size-inputs-valid-test \\
    --distributed-world-size 32 \\
    --distributed-port 9218 \\
    --fp16 \\
    --tensorboard-logdir ${RESULT_FOLDER_2}/tfboard_log \\
    >> ${RESULT_FOLDER_2}/train.log 2>&1

EOM

  chmod +x $TRAIN_SH

  sbatch -J "LID-spm-upsampl" \
      --partition=learnaccel --comment "LID-spm-upsampl" \
      --nodes=4 --gpus-per-node=8 --cpus-per-task=10 \
      --ntasks-per-node=8 --time=4320 \
      $TRAIN_SH

  echo "launched $TRAIN_SH"
}


prepare_data_3_combine
prepare_spm_data_1

split_train_data_labels
train_spm
encode_spm
create_shards
binarize_shards_data
binarize_valid_data


launch_train
launch_train_2






