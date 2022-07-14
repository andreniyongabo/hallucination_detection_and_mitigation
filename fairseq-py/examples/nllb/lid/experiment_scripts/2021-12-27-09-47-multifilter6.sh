#!/bin/bash


EXPERIMENT_NAME="2021-12-27-09-47-multifilter6"
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
TRAIN_FILTER_DIR="$DATA_FOLDER/train_filter"
VALID_DIR="$DATA_FOLDER/valid"
TEST_DIR="$DATA_FOLDER/test"
mkdir -p $TRAIN_DIR
mkdir -p $TRAIN_FILTER_DIR
mkdir -p $VALID_DIR
mkdir -p $TEST_DIR

echo "DATA_FOLDER is $DATA_FOLDER"

ALL_LANGS=( abk ace_Arab ace_Latn ady afr alt amh ara arn arz asm ast aym azb azj bak ban bel bem ben bho bis bjn_Arab bjn_Latn bod bos bug bul bxr cat ceb ces che chv cjk ckb crh_Latn cym dan deu dik dyu dzo ell eng epo est eus ewe ewo fao fas fij fin fon fra ful fur gla gle glg gom grn guj hat hau heb hin hne hrv hun hye ibo ilo ind isl ita jav jpn kab kac kal kam kan kas_Arab kas_Deva kat kau_Arab kau_Latn kaz kbp kea khm kik kin kir kmb kon kor krc kur lao lav lij lim lin lit lmo ltg ltz lua lug luo lus mag mai mal mar min mkd mlg mlt mon mos mri msa mya nah nav nia nld nno nor npi nso nus nya oci orm ory oss pag pan pap pcm pol por pus pus que roh ron run rus sag san sat scn shn sin slk slv smo sna snd som sot spa sqi srd srp ssw sun swe swh tah tam tat tel tgk tgl tha tir tmh_Latn tog tpi tsn tso tuk tum tur twi udm uig ukr umb urd uzb vec vie war wes wol xho xmf yid yor yue zho zul zza )
FBSEED_LANGS=( bjn_Latn tmh_Latn fur ace_Latn crh_Latn mag dik bug ban kau_Latn lij bjn_Arab ltg kas_Arab ace_Arab hne kas_Deva nus pus mri dzo kau_Arab shn )

FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"



gather_from_others () {
    TRAIN_FOLDER_1="/large_experiments/nllb/mmt/lidruns/2021-10-25-22-21-multifilter5-softmax/data/train"
    TRAIN_FOLDER_2="/large_experiments/nllb/mmt/lidruns/2021-12-24-10-57-semisuper-ndup/data/train"

    # cp "$TRAIN_FOLDER_1/"*.cleaned.step4.txt "$TRAIN_DIR/"

    for FBSEED_LANG in "${FBSEED_LANGS[@]}"
    do
        echo "$FBSEED_LANG"
        cp "$TRAIN_FOLDER_2/$FBSEED_LANG.cleaned.step6.txt" "$TRAIN_DIR/"

    done

    rm "$TRAIN_DIR/mri.cleaned.step4.txt"
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



    for NLLB_LANG in "${ALL_LANGS[@]}"
    do

        VALID_SIZE=100000
        VALID_COMPARE_SIZE=$(expr $VALID_SIZE \* 4)


        MONO_CAT_FILE=""
        if [ -s "$TRAIN_DIR/$NLLB_LANG.cleaned.step4.txt" ]
        then
            MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step4.txt"
        fi
        if [ -s "$TRAIN_DIR/$NLLB_LANG.cleaned.step6.txt" ]
        then
            MONO_CAT_FILE="$TRAIN_DIR/$NLLB_LANG.cleaned.step6.txt"
        fi

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

prepare_flores_filled_data () {
    CONCAT_FILE="$TEST_DIR/concat.flores-filled.txt"
    TEST_FINAL_FILE="$TEST_DIR/flores-filled.txt"
    > $CONCAT_FILE


    for NLLB_LANG in "${ALL_LANGS[@]}"
    do
        FLORES_LANG="$NLLB_LANG"


        if [ $NLLB_LANG == "kas_Arab" ]; then
            FLORES_LANG="kas-Arab"
        fi

        FLORES_FILE="$FLORES_DEVTEST_DIR/${FLORES_LANG}.devtest"

        chosen_source=""
        if [ -s $FLORES_FILE ]
        then
            chosen_source="${chosen_source}-flores"
            printf "%-20s %s \n" $NLLB_LANG $chosen_source
            cat $FLORES_FILE | awk -v lang="$NLLB_LANG" '{print "__label__"lang " " $0}' >> $CONCAT_FILE
        else
            VALID_FILE="$VALID_DIR/$NLLB_LANG.valid.mono.txt"
            if [ -f "$VALID_FILE" ]; then
                chosen_source="${chosen_source}-fromvalid"
                printf "%-20s %s \n" $NLLB_LANG $chosen_source

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
        -lr 0.8 -minn 2 -maxn 5 -minCount 10 -epoch 5 -dim 256 -loss hs -bucket 1000000 -thread 60
}


eval_flores_filled_fasttext_variants_8 () {
    GOLD="$RESULT_FOLDER/flores-filled.gold"
    TEST_FILE="$TEST_DIR/flores-filled.txt"
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD


    for i in `seq 1 20`
    do
        RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-filled.fasttext.8.$i.txt"

        LID_MODEL="$RESULT_FOLDER/model.8.$i.bin"

        if [ -s $LID_MODEL ] && [ ! -s $RESULT_TXT ]
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


# gather_from_others
# prepare_data_combine

# prepare_flores_filled_data

# train_fasttext_8_1
eval_flores_filled_fasttext_variants_8

train_fasttext_8_2
eval_flores_filled_fasttext_variants_8








