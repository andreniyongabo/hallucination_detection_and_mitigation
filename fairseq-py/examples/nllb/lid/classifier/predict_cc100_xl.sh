#!/bin/bash


SCRIPTS_PATH="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/classifier"
PREDICT_SPLIT="$SCRIPTS_PATH/predict_split.py"

DATA_IN_FOLDER="/large_experiments/mmt/data/monolingual/cc100-xl"

LID_MODEL="/large_experiments/nllb/mmt/lidruns/2021-09-20-23-26-goal124-filter-percentile/result/model.8.8.bin"
THRESHOLDS_FILE="/large_experiments/nllb/mmt/lidruns/2021-09-29-22-40-optim-threshold/result/thresholds_2.npy"

CC100_XL_LANGS=(kon khm)


FILTER_CHAR_HISTOGRAM="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/filtering/filter_char_histogram.py"
HISTOGRAMS_VALID_FOLDER="/large_experiments/mmt/lidruns/2021-09-20-14-14-histogram-baseline/histograms/valid"

THRESH_SCORE="0.8"

ANNOTATION_FOLDER="/large_experiments/nllb/mmt/lidruns/2021-10-01-00-00-annotation"

RESULT_FOLDER="/large_experiments/nllb/mmt/lidruns/cc100-xl-classif4/"



analyze_lang(){
    A_LANG=$1

    ISO_639_3_LANG="$A_LANG"

    MONO_CAT_FILE="$RESULT_FOLDER/$A_LANG/$ISO_639_3_LANG.accepted.txt"
    CLEANED_TRAIN_FILE="$RESULT_FOLDER/$A_LANG/$ISO_639_3_LANG.cleaned.$THRESH_SCORE.txt"
    REJECTD_TRAIN_FILE="$RESULT_FOLDER/$A_LANG/$ISO_639_3_LANG.rejectd.$THRESH_SCORE.txt"

    mkdir -p "$RESULT_FOLDER/$A_LANG/"
    touch $MONO_CAT_FILE

    xzcat "$DATA_IN_FOLDER/$A_LANG.txt.xz" \
    | $PREDICT_SPLIT --original-lang $A_LANG \
                     --original-result-folder $RESULT_FOLDER \
                     --model $LID_MODEL \
                     --thresholds $THRESHOLDS_FILE


    # tail -f $MONO_CAT_FILE | cut -f 5- | python $FILTER_CHAR_HISTOGRAM \
    #     --lang $ISO_639_3_LANG \
    #     --threshold $THRESH_SCORE \
    #     --histogram-threshold 0.98 \
    #     --histograms $HISTOGRAMS_VALID_FOLDER \
    #         2> $REJECTD_TRAIN_FILE \
    #         1> $CLEANED_TRAIN_FILE

}

clean_lang() {
    A_LANG=$1

    ISO_639_3_LANG="$A_LANG"
    MONO_CAT_FILE="$RESULT_FOLDER/$A_LANG/$ISO_639_3_LANG.accepted.txt"
    CLEANED_TRAIN_FILE="$RESULT_FOLDER/$A_LANG/$ISO_639_3_LANG.cleaned.$THRESH_SCORE.txt"
    REJECTD_TRAIN_FILE="$RESULT_FOLDER/$A_LANG/$ISO_639_3_LANG.rejectd.$THRESH_SCORE.txt"

    mkdir -p "$RESULT_FOLDER/$A_LANG/"
    touch $MONO_CAT_FILE


    cat $MONO_CAT_FILE | cut -f 5- | python $FILTER_CHAR_HISTOGRAM \
        --lang $ISO_639_3_LANG \
        --threshold $THRESH_SCORE \
        --histogram-threshold 0.98 \
        --histograms $HISTOGRAMS_VALID_FOLDER \
            2> $REJECTD_TRAIN_FILE \
            1> $CLEANED_TRAIN_FILE

    echo "Cleaned $A_LANG"
}

analyze_langs(){
    for LANG in ${CC100_XL_LANGS[@]}
    do
        echo "$LANG"
        analyze_lang $LANG && clean_lang $LANG &
    done

    wait
}

analyze_langs_all(){
    CC100_XL_LANGS=(ewo sat bod kik alt xmf nia mos tog kbp arn ssw)
    analyze_langs
    CC100_XL_LANGS=(kmb abk dyu kac nav ady udm cjk sag bxr tum nan aym)
    analyze_langs
    CC100_XL_LANGS=(fon bho lua krc min yid srd pag grn bem bis fij oss)
    analyze_langs
    CC100_XL_LANGS=(run tir kal ewe twi kin zza vec smo war tsn lim kab)
    analyze_langs
    CC100_XL_LANGS=(pcm que bak tso roh kur san gla chv mai sot sun tuk)
    analyze_langs
    CC100_XL_LANGS=(uig tah che mlt lmo lao lus gom pap hat kir tgk scn)
    analyze_langs
    CC100_XL_LANGS=(mlg mya ilo azb jav ceb bos ltz oci pus ckb uzb)
    analyze_langs
    CC100_XL_LANGS=(eus sin mon nno arz glg afr bel mkd kat kaz aze sqi)
    analyze_langs
    CC100_XL_LANGS=(lav slv hrv heb)
    analyze_langs
}





clean_langs(){
    for LANG in ${CC100_XL_LANGS[@]}
    do
        echo "$LANG"
        clean_lang $LANG &
    done

    wait
}

collect_all_annotation(){
    for LANG in ${CC100_XL_LANGS[@]}
    do
        LANG_2="$LANG"
        if [ $LANG = "nan" ]
        then
            LANG_2="nah"
        fi

        if [ $LANG = "aze" ]
        then
            LANG_2="azj"
        fi

        echo "$LANG $LANG_2"

        ORIG_FILE="$RESULT_FOLDER/$LANG/$LANG_2.cleaned.0.8.txt"
        echo $ORIG_FILE

        RESULT_FILE="$ANNOTATION_FOLDER/$LANG.txt"
        cat $ORIG_FILE | shuf | head -n 500 > $RESULT_FILE

    done
}

analyze_langs
analyze_langs_all

clean_langs
collect_all_annotation

