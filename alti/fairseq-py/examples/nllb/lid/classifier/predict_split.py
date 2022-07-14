#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import os
import sys

import fasttext
import numpy as np


label_unk = "__label__unk"


def main():
    parser = argparse.ArgumentParser(
        description="Run fasttext predict on a file that contains prod lid predictions"
    )
    parser.add_argument("--original-lang", type=str)
    parser.add_argument("--original-result-folder", default="/large_experiments/nllb/mmt/lidruns/cc100-xl-classif3/", type=str)
    parser.add_argument("--model", type=str, help="fasttext model")
    parser.add_argument("--thresholds", type=str)
    parser.add_argument("--stop-at", type=int, default=0)
    args = parser.parse_args()

    print("Loading model", file=sys.stderr)
    ft = fasttext.load_model(args.model)
    print("Model loaded", file=sys.stderr)

    original_lang = args.original_lang
    result_folder = os.path.join(args.original_result_folder, original_lang)
    os.makedirs(result_folder, exist_ok=True)

    labels, th = np.load(args.thresholds, allow_pickle=True)
    thresholds_map = dict(zip(labels, th.tolist()))

    out_files_accepted = {}
    out_files_untrusted = {}

    ctr = 0
    ignore_rest = False
    accepted_ctr = 0
    for line in sys.stdin:
        if ignore_rest:
            continue
        line = line.rstrip()
        prev_ft_score, prod_lid, line = line.split("\t")
        prev_ft_score = float(prev_ft_score)

        labels, probas = ft.predict(line, k=1, threshold=0.0)

        n_probas = []
        for label, proba in zip(labels, probas):
            n_proba = -1
            if proba >= thresholds_map[label]:
                n_proba = proba
            n_probas.append(n_proba)
        n_probas = np.array(n_probas)

        if not (n_probas > 0.0).any():
            predicted_label = label_unk
            predicted_prob = 0.000
        else:
            bst = np.argmax(n_probas)
            predicted_label = labels[bst]
            predicted_prob = probas[bst]

        out_line = f"{predicted_label}\t{predicted_prob:.5f}\t{prev_ft_score:.5f}\t{prod_lid}\t{line}"

        if ctr % 10000 == 0:
            print(f"{original_lang}\t{ctr}")

        def add_to_files(files, status_name):
            if predicted_label not in files:
                files[predicted_label] = open(os.path.join(result_folder, predicted_label[9:]+"."+status_name+".txt"), 'w')
            files[predicted_label].write(out_line+"\n")

        if predicted_prob > thresholds_map[predicted_label]:
            add_to_files(out_files_accepted, "accepted")
            if predicted_label[9:] == original_lang:
                accepted_ctr += 1
                if args.stop_at and accepted_ctr >= args.stop_at:
                    ignore_rest = True
        else:
            add_to_files(out_files_untrusted, "untrusted")

        ctr += 1

    for _, fl in out_files_accepted.items():
        fl.close()
    for _, fl in out_files_untrusted.items():
        fl.close()

    print(f"Finished: {original_lang}")

if __name__ == "__main__":
    main()
