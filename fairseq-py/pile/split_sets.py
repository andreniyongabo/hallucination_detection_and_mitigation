#!/usr/bin/python3

import argparse
import fileinput
import json
from multiprocessing import Pool
import sys
import re
import os


def parse_json(raw_line):
    x = json.loads(raw_line)
    # collapse multiple newlines in the original with a single newline
    text = x["text"].strip().replace("\n", "\u2581")
    text = re.sub("\u2581+", "\n", text)
    return (x["meta"]["pile_set_name"], text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="input files")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sets", required=True)
    args = parser.parse_args()

    with open(args.sets) as h:
        set_names = set(map(lambda s: s.strip(), h.readlines()))

    outputs = {}
    for set_name in set_names:
        out_dir = os.path.join(
            "data", set_name.replace(" (", "_").replace(" ", "_").replace(")", "")
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        outputs[set_name] = open(f"{out_dir}/{args.output}.txt", "w")

    newline_prefix = {set_name: "" for set_name in set_names}

    h = fileinput.input(args.files, openhook=fileinput.hook_compressed)
    pool = Pool(args.workers)
    results = pool.imap_unordered(parse_json, h, 1000)
    for i, line in enumerate(results):
        if line is None:
            continue
        (set_name, text) = line

        assert set_name in set_names
        assert set_name in newline_prefix
        assert set_name in outputs

        print(newline_prefix[set_name] + text, file=outputs[set_name])
        newline_prefix[set_name] = "\n"
        if i % 1000000 == 0:
            print(i, file=sys.stderr, end="", flush=True)
        elif i % 100000 == 0:
            print(".", file=sys.stderr, end="", flush=True)

    for output in outputs.values():
        print(file=output, flush=True)
        output.close()
