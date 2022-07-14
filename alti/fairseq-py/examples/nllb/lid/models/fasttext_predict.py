#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import sys

import fasttext


def main():
    parser = argparse.ArgumentParser(
        description="Run fasttext predict on a file that contains prod lid predictions"
    )
    parser.add_argument("--model", type=str, help="fasttext model")
    args = parser.parse_args()

    ft = fasttext.load_model(args.model)
    for line in sys.stdin:
        line = line.rstrip()
        prod_lid, line = line.split("\t")

        labels, probs = ft.predict(line, k=1, threshold=0.0)
        print(f"{labels[0]}\t{probs[0]:.5f}\t{prod_lid}\t{line}")


if __name__ == "__main__":
    main()
