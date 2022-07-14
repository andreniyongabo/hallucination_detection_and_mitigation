#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

args = None

"""
./classifier_metrics.py \
    --prediction /large_experiments/mmt/lidruns/2021-06-10-11-38-flores-99-match-cld3/result/flores-dev.predictions \
    --gold /large_experiments/mmt/lidruns/2021-06-10-11-38-flores-99-match-cld3/result/flores-dev.gold \
"""


def read_file_labels(path):
    return [ln.rstrip().split()[0] for ln in open(path, "r").readlines()]


def read_and_compute_metrics():
    prediction = read_file_labels(args.prediction)
    gold = read_file_labels(args.gold)

    assert len(prediction) == len(gold), "Prediction and gold have different sizes"
    labels = sorted(list(set(prediction).union(set(gold))))
    lang_index_map = {labels[i]: i for i in range(len(labels))}

    gold_n = list(map(lang_index_map.__getitem__, gold))
    prediction_n = list(map(lang_index_map.__getitem__, prediction))

    # From:
    # https://stackoverflow.com/a/50671617/411264
    confusion = confusion_matrix(gold_n, prediction_n)

    if args.ignore_lang:
        ignore_lang_id = lang_index_map[args.ignore_lang]
        confusion = np.delete(confusion, ignore_lang_id, 0)
        confusion = np.delete(confusion, ignore_lang_id, 1)
        labels.remove(args.ignore_lang)

    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    def get_val_str(val):
        if np.isnan(val) or val is None:
            return "-" * 8
        else:
            return f"{val:.6f}"

    for i, label in enumerate(labels):
        if label == args.ignore_lang:
            continue
        f1 = get_val_str(2 * PPV[i] * TPR[i] / (PPV[i] + TPR[i]))
        precision = get_val_str(PPV[i])
        recall = get_val_str(TPR[i])
        fpr = get_val_str(FPR[i])
        print(
            f"F1-Score : {f1}  Precision : {precision}  Recall : {recall}  FPR : {fpr}   {label}"
        )

    overall_precision = TP.sum() / (TP.sum() + FP.sum())
    overall_recall = TP.sum() / (TP.sum() + FN.sum())
    overall_fpr = FP.sum() / (FP.sum() + TN.sum())

    print(f"Precision : {overall_precision:.6f}")
    print(f"Recall    : {overall_recall:.6f}")
    print(f"FPR       : {overall_fpr:.6f}")


def main():
    global args

    parser = argparse.ArgumentParser(
        description="Display classifier metrics based on prediction and gold files"
    )
    parser.add_argument("--prediction", type=str, help="prediction file")
    parser.add_argument("--gold", type=str, help="gold file")
    parser.add_argument("--ignore-lang", type=str, help="ignore lang")
    args = parser.parse_args()

    read_and_compute_metrics()


if __name__ == "__main__":
    main()
