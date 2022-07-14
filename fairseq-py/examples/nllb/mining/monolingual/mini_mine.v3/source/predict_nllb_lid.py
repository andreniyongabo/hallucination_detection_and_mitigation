#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

from examples.nllb.mining.monolingual.utils.predict_lid import (
    get_lid_predictor,
    get_lid_predictor_date,
)


def main():
    parser = argparse.ArgumentParser(description="Run LID predict on a file")
    parser.add_argument("--model", type=str, default=None, help="fasttext model")
    parser.add_argument("--thresholds", type=str, default=None, help="thresholds file")
    parser.add_argument(
        "--model-date",
        type=str,
        help="specify the model and threshold by train date. Set 'last' for the latest one.",
    )
    parser.add_argument("--print-proba", action="store_true", help="show probabilities")
    parser.add_argument(
        "--filter-mode",
        action="store_true",
        help="show the original text and the probability",
    )
    parser.add_argument(
        "--meta-fields",
        type=int,
        default=0,
        help="Number of tab-separated meta fields to pass through",
    )
    args = parser.parse_args()

    assert (
        args.model_date or args.model
    ), "Select a model: for example `--model-date last`"

    if args.model_date:
        assert (
            args.model is None and args.thresholds is None
        ), "You can't specify the model or the thresholds with the `--model-date` argument"

        predict_fn = get_lid_predictor_date(args.model_date)
    else:
        predict_fn = get_lid_predictor(
            Path(args.model), Path(args.thresholds) if args.thresholds else None
        )

    n_inp = 0
    for line in sys.stdin:
        line = line.rstrip()

        if args.meta_fields > 0:
            fields = line.split("\t", args.meta_fields)
            assert (
                len(fields) > args.meta_fields
            ), f"expected {args.meta_fields} meta fields in input line {n_inp}"
            sent = fields[args.meta_fields]
        else:
            sent = line

        n_inp += 1

        (predicted_label, predicted_prob) = predict_fn(sent)

        if args.filter_mode:
            print(f"{predicted_label}\t{predicted_prob:.5f}\t{line}")
        else:
            if args.print_proba:
                print(f"{predicted_label} {predicted_prob:.5f}")
            else:
                print(predicted_label)


if __name__ == "__main__":
    main()
