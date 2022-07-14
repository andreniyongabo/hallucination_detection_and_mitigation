#!/usr/bin/env python

import argparse
import sys

import numpy as np

args = None


def collect_prediction_data():
    ctr = 0
    lang_index_map = None
    ordered_langs = None
    nlines = []

    label_unknown = "__label__unk"

    for line in open(args.prediction, "r"):
        line = line.strip("\n")

        l = line.split()
        labels = l[::2]
        probas = list(map(float, l[1::2]))

        if not lang_index_map:
            if label_unknown not in labels:
                labels.append(label_unknown)
            lang_index_map = {labels[i] : i for i in range(len(labels))}
            ordered_langs = labels

            print(lang_index_map)
        else:
            assert len(labels) == len(lang_index_map) or len(labels) + 1 == len(lang_index_map) # unknown

        nline = np.zeros(len(lang_index_map))

        for label, proba in zip(labels, probas):
            nline[lang_index_map[label]] = proba

        nlines.append(nline)

        if ctr % 100000 == 0:
            print(ctr)
        ctr += 1
        # if ctr > 1:
        #     break

    X = np.stack(nlines)

    gold_classes = []
    for line in open(args.gold, "r"):
        line = line.strip("\n")
        gold_classes.append(lang_index_map.get(line, lang_index_map[label_unknown]))

    gold_classes = np.array(gold_classes)

    prediction_data = np.array([labels, gold_classes, X], dtype=object)
    np.save(args.output, prediction_data)


def main():
    global args

    parser = argparse.ArgumentParser(
        description="Read fasttext format output for prediction probabilities and construct a numpy array from it"
    )
    parser.add_argument("--prediction", type=str, help="prediction file with probabilities")
    parser.add_argument("--gold", type=str, help="gold file that contains 'real' labels")
    parser.add_argument("--output", type=str, help="output numpy array storage file for prediction")
    args = parser.parse_args()

    collect_prediction_data()


if __name__ == '__main__':
    main()
