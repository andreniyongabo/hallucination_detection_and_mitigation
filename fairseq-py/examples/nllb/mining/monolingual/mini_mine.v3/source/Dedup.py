#!/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# --------------------------------------------------------

import argparse
import hashlib
import lzma
import sys
import time

###############################################################################


def get_args(args_list=None):
    parser = argparse.ArgumentParser(description="Simple sentence deduplication tool")
    parser.add_argument("-v", "--verbose", action="store_true", help="Detailed output")
    parser.add_argument(
        "--meta-fields",
        type=int,
        default=0,
        help="Number of tab-separated meta fields to pass through",
    )
    parser.add_argument("--meta-out", type=str, help="Output file for meta information")
    return parser.parse_args(args_list)


###############################################################################
def Deduplicate(meta_fields=0, meta_out=None, verbose=False):

    nlines = 0  # input lines
    nempty = 0  # empty lines
    ndedup = 0  # deuplicated sentences

    if meta_out:
        metaf = lzma.open(meta_out, "wt", encoding="utf-8", errors="surrogateescape")
    else:
        metaf = None

    t = int(time.time())
    seen = set()
    for line in sys.stdin:
        line = line.strip()

        if meta_fields > 0:
            fields = line.split("\t", meta_fields)
            assert (
                len(fields) > meta_fields
            ), f"expected {meta_fields} meta fields in input line {nlines}"
            sent = fields[meta_fields]
        else:
            sent = line

        nlines += 1
        if len(sent) == 0:
            nempty += 1
            continue

        hash_val = hashlib.md5(sent.encode("utf-8")).hexdigest()
        if hash_val not in seen:
            seen.add(hash_val)
            print(sent)
            ndedup += 1
            if metaf:
                print(f"{fields[0]}\t{fields[1]}", file=metaf)

    t = int(time.time() - t)
    if verbose:
        print(
            f" - {nlines} lines, {nempty} empty, deduplicated {ndedup} sentences, time: {t // 60}m{t % 60:02d}s",
            file=sys.stderr,
        )
        if metaf:
            metaf.close()
            print(f" - wrote meta-information to {meta_out}", file=sys.stderr)


###############################################################################
if __name__ == "__main__":
    args = get_args()

    Deduplicate(
        meta_fields=args.meta_fields, meta_out=args.meta_out, verbose=args.verbose
    )
