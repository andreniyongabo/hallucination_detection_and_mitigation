#!/bin/bash


EXPERIMENT_NAME="2021-09-27-15-30-goal124-optim-threshold"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

# temporary solution because of read-only problem
EXPERIMENT_FOLDER="/checkpoint/celebio/projects/nlp/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER



TRAINED_LID_PATH="/large_experiments/nllb/mmt/lidruns/2021-09-20-23-26-goal124-filter-percentile/"


THRESHOLD_OPTIM_PATH="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/optim/threshold_finder"

FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"
PREDICTIONS="$RESULT_FOLDER/valid.all.predictions.txt"
PREDICTIONS_THRESHOLD="$RESULT_FOLDER/valid.all.predictions.thresholded.txt"
GOLD="$RESULT_FOLDER/valid.all.gold.txt"

PREDICTIONS_FLORESFILL="$RESULT_FOLDER/floresfill.all.predictions.txt"
PREDICTIONS_THRESHOLD_FLORESFILL="$RESULT_FOLDER/floresfill.all.predictions.thresholded.txt"
PREDICTIONS_THRESHOLD_FLORESFILL_2="$RESULT_FOLDER/floresfill.all.predictions.thresholded.2.txt"
PREDICTIONS_THRESHOLD_FLORESFILL_3="$RESULT_FOLDER/floresfill.all.predictions.thresholded.3.txt"
GOLD_FLORESFILL="$RESULT_FOLDER/floresfill.all.gold.txt"


PREDICTIONS_HEAD="$RESULT_FOLDER/valid.all.predictions.head"
GOLD_HEAD="$RESULT_FOLDER/valid.all.gold.head"


prepare_predictions() {
    echo $TRAINED_LID_PATH
    VALID_FILE="$TRAINED_LID_PATH/data/valid/all.txt"
    LID_MODEL="$TRAINED_LID_PATH/result/model.5.bin"


    cat "$VALID_FILE" | cut -f 1 -d" " > $GOLD

    ls -la $VALID_FILE
    RESULT_GATHER="$FASTTEXT_BIN predict-prob $LID_MODEL $VALID_FILE -1"
    $RESULT_GATHER > $PREDICTIONS
}

prepare_predictions_floresfill() {
    echo $TRAINED_LID_PATH
    TEST_FILE="$TRAINED_LID_PATH/data/test/flores-filled.txt"
    LID_MODEL="$TRAINED_LID_PATH/result/model.5.bin"

    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD_FLORESFILL

    ls -la $TEST_FILE
    RESULT_GATHER="$FASTTEXT_BIN predict-prob $LID_MODEL $TEST_FILE -1"
    $RESULT_GATHER > $PREDICTIONS_FLORESFILL
}


collect_prediction_as_numpy_data() {
    COLLECTOR="$THRESHOLD_OPTIM_PATH/collect_pred_data.py"

    $COLLECTOR --prediction $PREDICTIONS_HEAD \
               --gold $GOLD_HEAD \
               --output "$RESULT_FOLDER/predictions_head"

}


compute_best_thresholds() {
    OPTIMIZER="$THRESHOLD_OPTIM_PATH/compute_best_thresholds.py"

    $OPTIMIZER train \
        --prediction-data "$RESULT_FOLDER/predictions_head.npy" \
        --output-threshold "$RESULT_FOLDER/thresholds_head" \
        --budget 100 \
}

compute_best_thresholds_2() {

    # loss = np.abs(PPV[PPV < 0.98] - 0.98).sum()
    # loss2 = np.abs(1.0 - PPV[labels.index('__label__eng')])
    # loss += loss2 * 10

    OPTIMIZER="$THRESHOLD_OPTIM_PATH/compute_best_thresholds.py"

    $OPTIMIZER train \
        --prediction-data "$RESULT_FOLDER/predictions_head.npy" \
        --output-threshold "$RESULT_FOLDER/thresholds_head2" \
        --budget 2000

}

compute_best_thresholds_3() {
    OPTIMIZER="$THRESHOLD_OPTIM_PATH/compute_best_thresholds.py"

    $OPTIMIZER train \
        --prediction-data "$RESULT_FOLDER/predictions_head.npy" \
        --output-threshold "$RESULT_FOLDER/thresholds_head3" \
        --budget 500

}


compute_best_thresholds_4() {
    OPTIMIZER="$THRESHOLD_OPTIM_PATH/compute_best_thresholds.py"

    $OPTIMIZER train \
        --prediction-data "$RESULT_FOLDER/predictions_head.npy" \
        --output-threshold "$RESULT_FOLDER/thresholds_head4" \
        --budget 500

}


apply_thresholds () {
    APPLIER="$THRESHOLD_OPTIM_PATH/apply_thresholds_to_predictions.py"

    cat $PREDICTIONS | $APPLIER --input-threshold "$RESULT_FOLDER/thresholds_head.npy" > $PREDICTIONS_THRESHOLD
}

apply_thresholds_floresfill () {
    APPLIER="$THRESHOLD_OPTIM_PATH/apply_thresholds_to_predictions.py"
    cat $PREDICTIONS_FLORESFILL | $APPLIER --input-threshold "$RESULT_FOLDER/thresholds_head.npy" > $PREDICTIONS_THRESHOLD_FLORESFILL
}

apply_thresholds_floresfill_2 () {
    APPLIER="$THRESHOLD_OPTIM_PATH/apply_thresholds_to_predictions.py"
    cat $PREDICTIONS_FLORESFILL | $APPLIER --input-threshold "$RESULT_FOLDER/thresholds_head2.npy" > $PREDICTIONS_THRESHOLD_FLORESFILL_2
}

apply_thresholds_floresfill_3 () {
    APPLIER="$THRESHOLD_OPTIM_PATH/apply_thresholds_to_predictions.py"
    cat $PREDICTIONS_FLORESFILL | $APPLIER --input-threshold "$RESULT_FOLDER/thresholds_head3.npy" > $PREDICTIONS_THRESHOLD_FLORESFILL_3
}

eval_floresfill(){
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-fill.threshold.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS_THRESHOLD_FLORESFILL --gold $GOLD_FLORESFILL > $RESULT_TXT
}

eval_floresfill_2(){
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-fill.threshold.2.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS_THRESHOLD_FLORESFILL_2 --gold $GOLD_FLORESFILL > $RESULT_TXT
}

eval_floresfill_3(){
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-flores-fill.threshold.3.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS_THRESHOLD_FLORESFILL_3 --gold $GOLD_FLORESFILL > $RESULT_TXT
}




prepare_predictions_8_8() {
    echo $TRAINED_LID_PATH
    VALID_FILE="$TRAINED_LID_PATH/data/valid/all.txt"
    LID_MODEL="$TRAINED_LID_PATH/result/model.8.8.bin"

    cat "$VALID_FILE" | cut -f 1 -d" " > $GOLD

    ls -la $VALID_FILE
    RESULT_GATHER="$FASTTEXT_BIN predict-prob $LID_MODEL $VALID_FILE -1"
    $RESULT_GATHER > "$PREDICTIONS.8.8.txt"
}


prepare_predictions
collect_prediction_as_numpy_data
compute_best_thresholds
compute_best_thresholds_2
compute_best_thresholds_3
prepare_predictions_floresfill
apply_thresholds
apply_thresholds_floresfill
eval_floresfill
apply_thresholds_floresfill_2
eval_floresfill
eval_floresfill_2
apply_thresholds_floresfill_3
eval_floresfill_3
prepare_predictions_8_8
compute_best_thresholds_4



