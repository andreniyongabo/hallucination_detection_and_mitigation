#!/usr/bin/env python

import argparse
import sys

import numpy as np


label_unk = "__label__unk"


def add_unknown(labels):
    if label_unk not in labels:
        labels.append(label_unk)


def main():
    parser = argparse.ArgumentParser(description="Apply thresholds per language")
    parser.add_argument("--input-threshold")

    args = parser.parse_args()

    checkpoint_data = np.load(args.input_threshold, allow_pickle=True)

    threshold_after_prediction = False
    if len(checkpoint_data) == 2:   # backward compatibility
        labels, th = checkpoint_data
    else:
        labels, th, threshold_after_prediction = checkpoint_data

    add_unknown(labels)
    thresholds_map = dict(zip(labels, th.tolist()))

    for line in sys.stdin:
        line = line.rstrip()

        l = line.split()
        labels = l[::2]
        probas = list(map(float, l[1::2]))

        if threshold_after_prediction:
            if probas[0] < thresholds_map[labels[0]]:
                print(f"{label_unk} 0.0")
            else:
                print(f"{labels[0]} {probas[0]}")
        else:
            n_probas = []
            for label, proba in zip(labels, probas):
                n_proba = -1
                if proba >= thresholds_map[label]:
                    n_proba = proba
                n_probas.append(n_proba)
            n_probas = np.array(n_probas)

            if not (n_probas > 0.0).any():
                print(f"{label_unk} 0.0")
            else:
                bst = np.argmax(n_probas)
                print(f"{labels[bst]} {probas[bst]}")


if __name__ == "__main__":
    main()
