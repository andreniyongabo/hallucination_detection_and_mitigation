#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import sys
import re


def gencsde():
    ENDQUOTE = re.compile(r'(^|[ ({\[])("|,,|\'\'|``)')
    STARTQUOTE = re.compile(r'("|\'\'|‟)($|[ ,.?!:;)}\]])')
    HYPHEN = re.compile(r" --? ")

    def cs(input):
        input = ENDQUOTE.sub(r"\1„", input)
        input = STARTQUOTE.sub(r"“\2", input)
        input = HYPHEN.sub(r" – ", input)
        return input

    return cs


def genzh():
    QUOTES = re.compile(r'"|,,|\'\'|``|‟|“|”')
    HYPHEN = re.compile(r" -\s*- ")
    STARTQUOTE = re.compile(r'(^|[ ({\[])("|,,|\'\'|``)')
    ENDQUOTE = re.compile(r'("|\'\'|‟)($|[ ,.?!:;)}\]])')
    NUMERALS = re.compile(r"([\d]+[\d\-\.%\,]*)")
    LATIN = re.compile(r"([a-zA-Z’\'@_\-]+)")
    SPACE = re.compile(r"\s+")
    PUNCT = re.compile(r"\s([\.)”。])")
    PUNCT2 = re.compile(r"([(“])\s")
    COMMA1 = re.compile(r"(\D),")
    COMMA2 = re.compile(r",(\D)")

    def zh(input):
        if len(QUOTES.findall(input)) == 2:
            s = QUOTES.split(input)
            input = "{}“{}”{}".format(s[0], s[1], s[2])
        if len(QUOTES.findall(input)) == 4:
            s = QUOTES.split(input)
            input = "{}“{}”{}“{}”{}".format(s[0], s[1], s[2], s[3], s[4])

        input = STARTQUOTE.sub(r"“\1", input)
        input = ENDQUOTE.sub(r"\2“", input)
        input = HYPHEN.sub(r"——", input)
        input = NUMERALS.sub(r" \1 ", input)
        input = LATIN.sub(r" \1 ", input)
        input = SPACE.sub(r" ", input)
        input = PUNCT.sub(r"\1", input)
        input = PUNCT2.sub(r"\1", input)
        # harmful        input = input.replace('.', '。')
        input = input.replace(":", "：")

        input = COMMA1.sub(r"\1，", input)
        input = COMMA2.sub(r"，\1", input)
        input = input.replace("?", "？")
        input = input.replace("!", "！")
        input = input.replace(";", "；")
        input = input.replace("(", "（")
        input = input.replace(")", "）")

        return input.strip()

    return zh


def genja():
    STARTQUOTE = re.compile(r'(^|[ ({\[])("|,,|\'\'|``)')
    ENDQUOTE = re.compile(r'("|\'\'|‟)($|[ ,.?!:;)}\]])')
    QUOTES = re.compile(r'"|,,|\'\'|``|‟|“|”')
    COMMA1 = re.compile(r"(\D),")
    COMMA2 = re.compile(r",(\D)")

    def ja(input):
        if len(QUOTES.findall(input)) == 2:
            s = QUOTES.split(input)
            input = "{}「{}」{}".format(s[0], s[1], s[2])
        if len(QUOTES.findall(input)) == 4:
            s = QUOTES.split(input)
            input = "{}「{}」{}「{}」{}".format(s[0], s[1], s[2], s[3], s[4])

        input = STARTQUOTE.sub(r"「\1", input)
        input = ENDQUOTE.sub(r"\2」", input)
        # harmful        input = input.replace('.', '。')

        input = COMMA1.sub(r"\1、", input)
        input = COMMA2.sub(r"、\1", input)

        input = input.replace("!", "！")
        input = input.replace("?", "？")
        input = input.replace(";", "；")
        input = input.replace(":", "：")
        input = input.replace("(", "（")
        input = input.replace(")", "）")
        return input.strip()

    return ja


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Text language")
    args = parser.parse_args()

    def lang(x):
        return x

    if args.lang == "zh":
        lang = genzh()
    elif args.lang in ["de", "cs"]:
        lang = gencsde()
    elif args.lang == "ja":
        lang = genja()
    else:
        print(sys.stderr, "language {} is unsupported".format(args.lang))

    for line in sys.stdin:
        line = line.strip()
        print(lang(line))


if __name__ == "__main__":
    main()
