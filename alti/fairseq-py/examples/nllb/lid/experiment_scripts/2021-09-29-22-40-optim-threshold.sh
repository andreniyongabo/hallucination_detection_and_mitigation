#!/bin/bash


EXPERIMENT_NAME="2021-09-29-22-40-optim-threshold"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"
mkdir -p $EXPERIMENT_FOLDER
RESULT_FOLDER="$EXPERIMENT_FOLDER/result"
mkdir -p $RESULT_FOLDER




FASTTEXT_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/fastText/fasttext"
PREDICTIONS="$RESULT_FOLDER/valid.all.predictions.txt"
PREDICTIONS_WITHOUT_THRESHOLD="$RESULT_FOLDER/valid.all.predictions.nth.txt"
PREDICTIONS_THRESHOLD="$RESULT_FOLDER/valid.all.predictions.thresholded.txt"
GOLD="$RESULT_FOLDER/valid.all.gold.txt"
VALID_HEAD="$RESULT_FOLDER/valid.head"


PREDICTIONS_TEST="$RESULT_FOLDER/test.all.predictions.txt"
PREDICTIONS_WITHOUT_THRESHOLD_TEST="$RESULT_FOLDER/test.all.predictions.nth.txt"
PREDICTIONS_THRESHOLD_TEST="$RESULT_FOLDER/test.all.predictions.thresholded.txt"
GOLD_TEST="$RESULT_FOLDER/test.all.gold.txt"

THRESHOLD_OPTIM_PATH="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/optim/threshold_finder"


TRAINED_LID_PATH="/large_experiments/nllb/mmt/lidruns/2021-09-20-23-26-goal124-filter-percentile/"
LID_MODEL="$TRAINED_LID_PATH/result/model.8.8.bin"

VALID_FILE="$TRAINED_LID_PATH/data/valid/all.txt"
TEST_FILE="$TRAINED_LID_PATH/data/test/flores-filled.txt"

prepare_predictions_valid() {
    head -n 1000000 $VALID_FILE > $VALID_HEAD

    cat "$VALID_HEAD" | cut -f 1 -d" " > $GOLD

    RESULT_GATHER="$FASTTEXT_BIN predict-prob $LID_MODEL $VALID_HEAD -1"
    $RESULT_GATHER > $PREDICTIONS
}

prepare_predictions_test() {
    cat "$TEST_FILE" | cut -f 1 -d" " > $GOLD_TEST

    ls -la $TEST_FILE
    RESULT_GATHER="$FASTTEXT_BIN predict-prob $LID_MODEL $TEST_FILE -1"
    $RESULT_GATHER > $PREDICTIONS_TEST
}

collect_prediction_as_numpy_data() {
    COLLECTOR="$THRESHOLD_OPTIM_PATH/collect_pred_data.py"

    $COLLECTOR --prediction $PREDICTIONS \
               --gold $GOLD \
               --output "$RESULT_FOLDER/predictions_pkl"
}

compute_best_thresholds_1() {
    OPTIMIZER="$THRESHOLD_OPTIM_PATH/compute_best_thresholds.py"

    # loss = (PPV < 0.98).sum()
    $OPTIMIZER train \
        --prediction-data "$RESULT_FOLDER/predictions_pkl.npy" \
        --output-threshold "$RESULT_FOLDER/thresholds_1" \
        --budget 500
}

compute_best_thresholds_2() {
    OPTIMIZER="$THRESHOLD_OPTIM_PATH/compute_best_thresholds.py"

    # loss1 = 3.0 * np.abs(PPV[PPV < 0.98] - 0.98).sum() + \
    #         0.5 * np.abs(PPV[PPV < 0.97] - 0.97).sum() + \
    #         0.3 * np.abs(PPV[PPV < 0.95] - 0.95).sum() + \
    #         10.0 * np.abs(PPV[PPV < 0.50] - 0.50).sum()

    # loss2 = 2.0 * np.abs(TPR[TPR < 0.80] - 0.80).sum() + \
    #         0.5 * np.abs(TPR[TPR < 0.70] - 0.70).sum() + \
    #         0.2 * np.abs(TPR[TPR < 0.50] - 0.50).sum()

    # loss = loss1 + loss2

    $OPTIMIZER train \
        --prediction-data "$RESULT_FOLDER/predictions_pkl.npy" \
        --output-threshold "$RESULT_FOLDER/thresholds_2" \
        --budget 5000
}


apply_thresholds () {
    APPLIER="$THRESHOLD_OPTIM_PATH/apply_thresholds_to_predictions.py"
    cat $PREDICTIONS | $APPLIER --input-threshold "$RESULT_FOLDER/thresholds_1.npy" > "$PREDICTIONS_THRESHOLD.1"
}

apply_thresholds_2 () {
    APPLIER="$THRESHOLD_OPTIM_PATH/apply_thresholds_to_predictions.py"
    cat $PREDICTIONS | $APPLIER --input-threshold "$RESULT_FOLDER/thresholds_2.npy" > "$PREDICTIONS_THRESHOLD.2"
}

apply_test_thresholds_2 () {
    APPLIER="$THRESHOLD_OPTIM_PATH/apply_thresholds_to_predictions.py"
    cat $PREDICTIONS_TEST | $APPLIER --input-threshold "$RESULT_FOLDER/thresholds_2.npy" > "$PREDICTIONS_THRESHOLD_TEST.2"
}

eval_valid(){
    cat $PREDICTIONS | cut -f1 -d" " > $PREDICTIONS_WITHOUT_THRESHOLD
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-valid.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS_WITHOUT_THRESHOLD --gold $GOLD > $RESULT_TXT
}

eval_test(){
    cat $PREDICTIONS_TEST | cut -f1 -d" " > $PREDICTIONS_WITHOUT_THRESHOLD_TEST
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-test.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS_WITHOUT_THRESHOLD_TEST --gold $GOLD_TEST > $RESULT_TXT
}


eval_valid_threshold(){
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-valid.threshold.1.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction "$PREDICTIONS_THRESHOLD.1" --gold $GOLD > $RESULT_TXT
}

eval_valid_threshold_2(){
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-valid.threshold.2.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction "$PREDICTIONS_THRESHOLD.2" --gold $GOLD > $RESULT_TXT
}


eval_test_threshold_2(){
    RESULT_TXT="$RESULT_FOLDER/result.classifiermetrics-test.threshold2.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS_THRESHOLD_TEST.2 --gold $GOLD_TEST > $RESULT_TXT
}


# prepare_predictions_valid
# prepare_predictions_test
# collect_prediction_as_numpy_data

# compute_best_thresholds_1

#eval_valid
#apply_thresholds
# eval_valid_threshold

# compute_best_thresholds_2
# apply_thresholds_2
# eval_valid_threshold_2


apply_test_thresholds_2
eval_test
eval_test_threshold_2

