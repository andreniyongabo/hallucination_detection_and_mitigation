#!/bin/python3
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
# Dependencies:
#  pip install indic-nlp-library
#  pip install pythainlp
#  pip install laonlp
#  pip install khmer-nltk
#  pip install laonlp
#  pip install python-crfsuite

import argparse
import os
import sys
import time

from examples.nllb.mining.monolingual.utils.sentence_split import (
    get_split_algo,
    map_lang,
)

BASE_DIR = os.path.realpath(__file__).replace("SentenceSplit.py", "")


def info(msg: str):
    print(msg, file=sys.stderr)


def SplitLines(args) -> None:
    lang = map_lang(args.lang, args.lang_equiv)

    splitter = get_split_algo(lang, args.split_algo)

    nlines = 0  # input lines
    nempty = 0  # empty lines
    nsplit = 0  # split sentences
    nwords = 0  # number of words
    t = int(time.time())
    for line in sys.stdin:
        line = line.strip()
        line = line.replace("\t", " ")
        if len(line) == 0:
            nlines += 1
            nempty += 1
            continue

        sents = splitter(line)
        ni = 0
        for sent in sents:
            if args.add_meta:
                print(f"{args.add_meta}\t{nlines}\t{sent}")
            else:
                print(sent)
            ni += 1
            nsplit += 1
            nw = len(line.split())  # split into words
            nwords += nw

        nlines += 1

    t = int(time.time() - t)
    info(
        f" - {nlines} lines, {nempty} empty, output {nsplit} sentences, avg length {float(nwords)/nsplit:.1f} words, time: {t // 60}m{t % 60:02d}s"
    )


###############################################################################


def SplitGetArgs(args_list=None):
    parser = argparse.ArgumentParser(
        description="NLLB: split long lines into logical sentences and deduplicate"
    )
    parser.add_argument(
        "-l",
        "--lang",
        type=str,
        default="eng",
        help="Language used for splitting rule",
    )
    parser.add_argument(
        "--split-algo",
        default="default",
        choices=["none", "default", "moses", "indic"],
        help="Specify algorithm used to split sentences",
    )
    parser.add_argument(
        "--lang-equiv",
        default=BASE_DIR + "/language_equivalences.tsv",
        help="TSV file which gives language equivalences",
    )
    parser.add_argument(
        "--add-meta",
        type=str,
        help="Add meta information to output file (specified corpus name and line number)",
    )
    parser.add_argument(
        "-d",
        "--deduplicate",
        action="store_true",
        help="Deduplicate sentences (and exclude them)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")

    return parser.parse_args(args_list)


if __name__ == "__main__":
    args = SplitGetArgs()
    SplitLines(args)
