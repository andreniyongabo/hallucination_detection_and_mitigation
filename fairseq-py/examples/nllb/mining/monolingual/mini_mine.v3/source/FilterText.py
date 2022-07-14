#!/usr/bin/python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Helper functions for tokenization and BPE

import argparse
import sys
from string import punctuation

PUNCT = punctuation + "—|–"


def CountPunct(s):
    return len([ch for ch in s if ch in PUNCT])


NUMBER = "0123456789"


def CountNumber(s):
    return len([ch for ch in s if ch in NUMBER])


# ------------------------------------------------
def FilterGetArgs(args_list=None):
    parser = argparse.ArgumentParser(description="NLLB: filter lines")
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        default="eng",
        help="Language used for filter rules",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=1,
        help="Minimum number of characters in sentence",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=500,
        help="Maximum number of characters in sentence",
    )
    parser.add_argument(
        "--max-punct-ratio",
        type=float,
        default=0.2,
        help="Maximum ratio of punctation in sentence",
    )
    parser.add_argument(
        "--max-number-ratio",
        type=float,
        default=0.2,
        help="Maximum ratio of numbers in sentence",
    )
    parser.add_argument(
        "--meta-fields",
        type=int,
        default=0,
        help="Number of tab-separarted meta fields to pass through",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print statistics on stderr"
    )
    parser.add_argument(
        "--encoding", default="utf-8", help="character encoding for input/output"
    )

    return parser.parse_args(args_list)


# ----------------------------------------------------


def FilterText(
    min_chars=1,
    max_chars=500,
    max_punct_ratio=1.0,
    max_number_ratio=-0.2,
    meta_fields=0,
    verbose=False,
):

    n_inp = n_minchar = n_maxchar = n_punct = n_number = n_out = 0
    for line in sys.stdin:
        line = line.strip()
        if meta_fields > 0:
            fields = line.split("\t", meta_fields)
            assert (
                len(fields) > meta_fields
            ), f"expected {meta_fields} meta fields in input line {n_inp}"
            sent = fields[meta_fields]
        else:
            sent = line

        n_inp += 1
        slen = len(sent)
        if slen < min_chars:
            n_minchar += 1  # sentence too short
            continue
        if slen > max_chars:
            n_maxchar += 1  # sentence too long
            continue

        if CountPunct(sent) / slen > abs(max_punct_ratio):
            n_punct += 1  # too much punctuation
            if max_punct_ratio < 0:  # debug ouptut
                print(f"PUNCT: {sent}")
            continue

        if CountNumber(sent) / slen > abs(max_number_ratio):
            n_number += 1  # too much numbers
            if max_number_ratio < 0:  # debug ouptut
                print(f"NUMBER: {sent}")
            continue

        n_out += 1
        print(line)

    if verbose:
        print(
            f" - {n_inp} lines, {n_minchar} shorter then {min_chars}, {n_maxchar} longer then {max_chars} tokens, {n_punct} more than {abs(max_punct_ratio)}% punctuations, {n_number} more than {abs(max_number_ratio)}% numbers, {n_out} lines kept",
            file=sys.stderr,
        )


# ------------------------------
if __name__ == "__main__":
    args = FilterGetArgs()
    FilterText(
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        max_punct_ratio=args.max_punct_ratio,
        max_number_ratio=args.max_number_ratio,
        meta_fields=args.meta_fields,
        verbose=args.verbose,
    )
