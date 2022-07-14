# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import contextlib
import sys
import io

import sentencepiece as spm
from multiprocessing import Pool
from functools import partial

def initialize_sp(args):
    global sp
    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

def encode_lines(encode, valid, lines):
    def encode_line(line):
        line = line.strip()
        if len(line) > 0:
            line_toks = encode(line)
            if valid(line_toks):
                return " ".join(line_toks), False, False
            else:
                return None, True, False
        else:
            return "", False, True

    results = list(map(encode_line, lines))
    return (
        [encoded_lines for encoded_lines, _, _ in results],
        any(is_filtered for _, is_filtered, _ in results),
        any(is_empty for _, _, is_empty in results)
    )

def valid_length_filter(args, line):
    return (
        (args.min_len is None or len(line) >= args.min_len)
        and (args.max_len is None or len(line) <= args.max_len)
    )

def valid_true(args, lines):
    return True

def encode_as_pieces(l):
    return sp.EncodeAsPieces(l)

def encode_as_ids(l):
    return list(map(str, sp.EncodeAsIds(l)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="sentencepiece model to use for encoding")
    parser.add_argument("--inputs", nargs="+", default=['-'],
                        help="input files to filter/encode")
    parser.add_argument("--outputs", nargs="+", default=['-'],
                        help="path to save encoded outputs")
    parser.add_argument("--output_format", choices=["piece", "id"], default="piece")
    parser.add_argument("--min-len", type=int, metavar="N",
                        help="filter sentence pairs with fewer than N tokens")
    parser.add_argument("--max-len", type=int, metavar="N",
                        help="filter sentence pairs with more than N tokens")
    parser.add_argument("--processes", type=int, metavar="N", default=12,
                        help="number of process in parallel (the output data will keep the same order as input)")

    parser.add_argument("--keep-empty", action="store_true", default=False)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.outputs), \
            "number of input and output paths should match"

    sp = spm.SentencePieceProcessor()
    sp.Load(args.model)

    if args.output_format == "piece":
        encode = encode_as_pieces
    elif args.output_format == "id":
        encode = encode_as_ids
    else:
        raise NotImplementedError

    if args.min_len is not None or args.max_len is not None:
        valid = partial(valid_length_filter, args)
    else:
        valid = partial(valid_true, args)

    with contextlib.ExitStack() as stack:
        inputs = [
            stack.enter_context(open(input, "r", encoding="utf-8", newline="\n", errors="replace"))
                if input != "-" else io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors="replace")
            for input in args.inputs
        ]
        outputs = [
            stack.enter_context(open(output, "w", encoding="utf-8", newline="\n"))
                if output != "-" else sys.stdout
            for output in args.outputs
        ]

        stats = {
            "num_empty": 0,
            "num_filtered": 0,
        }

        with Pool(args.processes, initializer=partial(initialize_sp, args)) as p:
            iterable = p.imap(partial(encode_lines, encode, valid), list(zip(*inputs)), chunksize=500)
            for i, (enc_lines, is_filtered, is_empty) in enumerate(iterable, start=1):
                if is_filtered:
                    stats["num_filtered"] += 1
                elif is_empty and not args.keep_empty:
                    stats["num_empty"] += 1
                else:
                    for enc_line, output_h in zip(enc_lines, outputs):
                        print(enc_line, file=output_h)
                if i % 10000 == 0:
                    print("processed {} lines".format(i), file=sys.stderr)

        print("skipped {} empty lines".format(stats["num_empty"]), file=sys.stderr)
        print("filtered {} lines".format(stats["num_filtered"]), file=sys.stderr)


if __name__ == "__main__":
    main()

