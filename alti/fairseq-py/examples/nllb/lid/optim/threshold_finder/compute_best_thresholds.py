#!/usr/bin/env python


import argparse

import nevergrad as ng
import numpy as np
import pandas as pd

from concurrent import futures

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

args = None
X = None
gold = None
labels = []

label_unk = "__label__unk"


def add_unknown(labels):
    if label_unk not in labels:
        labels.append(label_unk)


def eval_with_thresholds(thresholds):

    if args.threshold_after_prediction:
        label_unk_id = labels.index(label_unk)

        X2 = X
        predictions = np.argmax(X2, axis=1)
        pred_probas = X2[range(len(predictions)), predictions]

        pred_thresholds = thresholds[predictions]
        predictions[pred_thresholds > pred_probas] = label_unk_id
    else:
        X2 = X.copy()
        X2[X2 < thresholds] = 0.0
        predictions = np.argmax(X2, axis=1)

    confusion = confusion_matrix(gold, predictions)

    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    np.seterr(divide="ignore", invalid="ignore")

    TPR = TP / (TP + FN)  # recall
    PPV = TP / (TP + FP)  # Precision
    FPR = FP / (FP + TN)

    loss1 = 3.0 * np.abs(PPV[PPV < 0.98] - 0.98).sum() + \
            0.5 * np.abs(PPV[PPV < 0.97] - 0.97).sum() + \
            0.3 * np.abs(PPV[PPV < 0.95] - 0.95).sum() + \
            10.0 * np.abs(PPV[PPV < 0.50] - 0.50).sum()

    loss2 = 2.0 * np.abs(TPR[TPR < 0.80] - 0.80).sum() + \
            0.5 * np.abs(TPR[TPR < 0.70] - 0.70).sum() + \
            0.2 * np.abs(TPR[TPR < 0.50] - 0.50).sum()

    loss = loss1 + loss2

    return (loss, PPV, TPR, FPR)


def eval_with_thresholds_ng(thresholds):
    loss, PPV, TPR, FPR = eval_with_thresholds(thresholds)
    return loss


def print_candidate_and_value(optimizer, candidate, value):
    print(value)



def train():
    global X
    global args

    nb_langs = X.shape[1]
    instrum = ng.p.Instrumentation(
        ng.p.Array(shape=(nb_langs,)).set_bounds(lower=0.0, upper=1.0),
    )

    num_workers = 20

    # ng.optimizers.NGOpt
    optimizer = ng.optimizers.CMA(parametrization=instrum, budget=args.budget, num_workers=num_workers)
    optimizer.register_callback("tell", print_candidate_and_value)

    if num_workers > 1:
        with futures.ThreadPoolExecutor(max_workers=optimizer.num_workers) as executor:
            recommendation = optimizer.minimize(eval_with_thresholds_ng, executor=executor, batch_mode=False)
    else:
        recommendation = optimizer.minimize(eval_with_thresholds_ng)

    print(recommendation.value)
    print(recommendation)

    threshold_data = np.array([labels, recommendation.value[0][0], args.threshold_after_prediction], dtype=object)
    np.save(args.output_threshold, threshold_data)


def eval():
    global X
    global labels
    global args

    th0 = np.zeros(X.shape[1])
    loss0, PPV0, TPR0, FPR0 = eval_with_thresholds(th0)

    threshold_after_prediction = False
    checkpoint_data = np.load(args.input_threshold, allow_pickle=True)

    if len(checkpoint_data) == 2:   # backward compatibility
        labels_th, th = checkpoint_data
    else:
        labels_th, th, threshold_after_prediction = checkpoint_data
        args.threshold_after_prediction = threshold_after_prediction

    add_unknown(labels_th)
    assert labels == labels_th, "Threshold file's labels do not correspond to the ones in prediction-data"

    loss1, PPV1, TPR1, FPR1 = eval_with_thresholds(th)

    print(f"loss without thresholds = {loss0}")
    print(f"loss with thresholds = {loss1}")

    F1_s0 = 2 * PPV0 * TPR0 / (PPV0 + TPR0)
    F1_s1 = 2 * PPV1 * TPR1 / (PPV1 + TPR1)

    fig, ax = plt.subplots()
    threshold_no_label = "Threshold=0"
    threshold_best_label = "Threshold=best"
    ax.plot(np.sort(PPV0), label=threshold_no_label)
    ax.plot(np.sort(PPV1), label=threshold_best_label)

    plt.legend()
    plt.xlabel("Ordered langs")
    plt.ylabel("Precision")

    plt.savefig("compare_threshold_ppv.png", dpi=300)

    fig, ax = plt.subplots()
    ax.plot(np.sort(TPR0), label=threshold_no_label)
    ax.plot(np.sort(TPR1), label=threshold_best_label)
    plt.legend()
    plt.xlabel("Ordered langs")
    plt.ylabel("Recall")
    plt.savefig("compare_threshold_tpr.png", dpi=300)

    fig, ax = plt.subplots()
    ax.plot(-np.sort(-FPR0), label=threshold_no_label)
    ax.plot(-np.sort(-FPR1), label=threshold_best_label)
    plt.legend()
    plt.xlabel("Ordered langs")
    plt.ylabel("FPR")
    plt.savefig("compare_threshold_fpr.png", dpi=300)

    fig, ax = plt.subplots()
    srt = np.argsort(F1_s0 - F1_s1)
    ax.plot(F1_s0[srt], label=threshold_no_label)
    ax.plot(F1_s1[srt], label=threshold_best_label)
    plt.legend()
    plt.xlabel("Ordered langs")
    plt.ylabel("F1-Score")
    plt.savefig("compare_threshold_f1.png", dpi=300)

    labels = [lbl[9:] for lbl in labels]
    d = {"labels": labels, "precision": PPV1, "recall": TPR1, "f1": F1_s1}
    df = pd.DataFrame(data=d)
    df.set_index("labels", inplace=True)

    # fmt: off
    out_order_labels = ["ell", "srp", "spa", "hrv", "eng", "ara", "ita", "nld", "por", "dan", "sot", "zho", "mar", "fij", "fra", "fin", "deu", "tah", "fas", "ady", "nor", "slv", "rus", "run", "sqi", "afr", "xho", "amh", "war", "chv", "pol", "msa", "kor", "ind", "lav", "hye", "zul", "jpn", "tat", "glg", "bis", "swe", "smo", "tpi", "cat", "sna", "swh", "slk", "ces", "tgl", "epo", "hin", "kin", "est", "pcm", "vie", "hau", "ukr", "que", "kat", "lin", "lit", "aym", "mai", "kir", "ast", "bho", "tso", "ron", "mos", "hun", "mkd", "mlt", "ilo", "tog", "kaz", "roh", "tuk", "wes", "orm", "ben", "tgk", "san", "kam", "fon", "pap", "tur", "luo", "wol", "ceb", "npi", "bul", "sin", "isl", "yor", "zza", "gla", "fao", "kur", "bxr", "tsn", "lus", "kan", "nya", "mlg", "bos", "dyu", "kbp", "xmf", "pan", "krc", "mon", "nan", "gom", "twi", "grn", "jav", "nso", "bem", "lmo", "gle", "tum", "aze", "heb", "bel", "scn", "som", "kab", "srd", "ewe", "vec", "umb", "udm", "kmb", "uzb", "oci", "abk", "snd", "ssw", "sun", "kik", "kon", "bak", "tha", "asm", "mal", "urd", "ibo", "uig", "nno", "tam", "arz", "lug", "ltz", "azb", "yid", "min", "tir", "cym", "pag", "kea", "yue", "cjk", "pus", "che", "lua", "hat", "kac", "kal", "nia", "guj", "tel", "oss", "nav", "eus", "lim", "bod", "mya", "ckb", "lao", "sag", "khm", "sat", "ory", "ewo", "arn", "alt"]
    # out_order_labels = ["hau", "orm", "som", "amh", "ful", "ibo", "wol", "yor", "lug", "swh", "sna", "xho", "zul", "luo", "lin", "nya"]
    # fmt: on

    df = df.reindex(out_order_labels)
    df.to_excel("lidresults_thresholds.xlsx", sheet_name="LID results")


def main():
    global args
    global X
    global gold
    global labels

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("action", type=str, choices=["train", "eval"], help="")
    parser.add_argument(
        "--prediction-data",
        type=str,
        help="predictions for all classes, produced by the script collect_pred_data.py",
    )
    parser.add_argument(
        "--input-threshold",
        type=str,
        default="thresholds",
        help="For `eval`: file name for thresholds to evaluate",
    )
    parser.add_argument(
        "--output-threshold",
        type=str,
        default="thresholds",
        help="For `train`: file name to save the best thresholds",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=1000,
        help="For `train`: how many trials should nevergrad do",
    )
    parser.add_argument("--threshold-after-prediction", action="store_true")

    args = parser.parse_args()

    labels, gold, X = np.load(args.prediction_data, allow_pickle=True)
    add_unknown(labels)

    if args.action == "train":
        train()
    elif args.action == "eval":
        eval()


if __name__ == "__main__":
    main()
