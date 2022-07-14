#!/usr/bin/env python

import argparse
import sys

from examples.nllb.mining.monolingual.utils.predict_script import get_script_predictor


def main():
    parser = argparse.ArgumentParser(description="Run LID predict on a file")
    parser.add_argument(
        "--filter-mode", action="store_true", help="show the original text"
    )
    parser.add_argument(
        "--meta-fields",
        type=int,
        default=0,
        help="Number of tab-separated meta fields to pass through",
    )
    args = parser.parse_args()

    predict_fn = get_script_predictor()
    print("hist built.", file=sys.stderr)

    n_inp = 0
    for line in sys.stdin:
        line = line.rstrip()
        line_orig = line

        if args.meta_fields > 0:
            fields = line.split("\t", args.meta_fields)
            assert (
                len(fields) > args.meta_fields
            ), f"expected {args.meta_fields} meta fields in input line {n_inp}"
            sent = fields[args.meta_fields]
        else:
            sent = line

        n_inp += 1

        (predicted_script, _) = predict_fn(sent)

        if args.filter_mode:
            print(f"{predicted_script}\t{line_orig}")
        else:
            print(predicted_script)


if __name__ == "__main__":
    main()
