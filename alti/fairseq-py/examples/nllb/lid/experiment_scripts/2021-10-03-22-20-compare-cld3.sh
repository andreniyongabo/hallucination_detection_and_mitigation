#!/bin/bash


EXPERIMENT_NAME="2021-10-03-22-20-compare-cld3"
EXPERIMENT_FOLDER="/large_experiments/nllb/mmt/lidruns/$EXPERIMENT_NAME"

mkdir -p $EXPERIMENT_FOLDER

RESULT_FOLDER="$EXPERIMENT_FOLDER/result"

mkdir -p $RESULT_FOLDER



TRAINED_LID_PATH="/large_experiments/nllb/mmt/lidruns/2021-09-20-23-26-goal124-filter-percentile/"

VALID_FILE="$TRAINED_LID_PATH/data/valid/all.txt"
TEST_FILE_FLORES_FILLED="$TRAINED_LID_PATH/data/test/flores-filled.txt"
TEST_FILE_FLORES_DEVTEST="$TRAINED_LID_PATH/data/test/flores-devtest.txt"


PREDICTIONS_TEST_FLORES_FILLED="$RESULT_FOLDER/test.flores-filled.all.cld3.predictions.txt"
GOLD_TEST_FLORES_FILLED="$RESULT_FOLDER/test.flores-filled.all.cld3.gold.txt"


PREDICTIONS_TEST_DEVTEST="$RESULT_FOLDER/test.flores-devtest.all.cld3.predictions.txt"
GOLD_TEST_FLORES_DEVTEST="$RESULT_FOLDER/test.flores-devtest.all.cld3.gold.txt"



CLD3_BIN="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/models/run_cld3.py"

predictions_cld3_flores_filled () {
    cat "$TEST_FILE_FLORES_FILLED" | cut -f 1 -d" " > $GOLD_TEST_FLORES_FILLED
    cat "$TEST_FILE_FLORES_FILLED" | cut -f 2- -d" " | $CLD3_BIN > $PREDICTIONS_TEST_FLORES_FILLED
}



eval_cld3_test_flores_filled (){
    RESULT_TXT_FLORES_FILLED="$RESULT_FOLDER/result.classifiermetrics-test.cld3.flores-filled.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS_TEST_FLORES_FILLED --gold $GOLD_TEST_FLORES_FILLED > $RESULT_TXT_FLORES_FILLED
}



predictions_cld3_flores_devtest () {
    cat "$TEST_FILE_FLORES_DEVTEST" | cut -f 1 -d" " > $GOLD_TEST_FLORES_DEVTEST
    cat "$TEST_FILE_FLORES_DEVTEST" | cut -f 2- -d" " | $CLD3_BIN > $PREDICTIONS_TEST_DEVTEST
}

eval_cld3_test_flores_devtest () {
    RESULT_TXT_FLORES_DEVTEST="$RESULT_FOLDER/result.classifiermetrics-test.cld3.flores-devtest.txt"
    CLASSIFIER_METRICS="/private/home/celebio/nlp/nllb/fairseq-py-lid/examples/nllb/lid/eval/classifier_metrics.py"
    $CLASSIFIER_METRICS --prediction $PREDICTIONS_TEST_DEVTEST --gold $GOLD_TEST_FLORES_DEVTEST > $RESULT_TXT_FLORES_DEVTEST
}


predictions_cld3_flores_filled
eval_cld3_test_flores_filled

predictions_cld3_flores_devtest
eval_cld3_test_flores_devtest

