#!/usr/bin/python3

import argparse
import fileinput
import json
import sys
import os

from fairseq import fb_hub


class Encoder:
    def __init__(self):
        self.initializer()

    def initializer(self):
        global bpe_fn
        bpe_fn = fb_hub.load("roberta.base").bpe.encode

    def encode(self, raw_line):
        global bpe_fn
        x = json.loads(raw_line)
        return (x["meta"]["pile_set_name"], bpe_fn(x["text"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="input files")
    # parser.add_argument('--workers', type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sets", required=True)
    args = parser.parse_args()

    bpe_fn = fb_hub.load("roberta.base").bpe.encode

    with open(args.sets) as h:
        set_names = set(map(lambda s: s.strip(), h.readlines()))

    outputs = {}
    for set_name in set_names:
        out_dir = os.path.join(
            args.output_dir,
            set_name.replace(" (", "_").replace(" ", "_").replace(")", ""),
        )
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        outputs[set_name] = open(f"{out_dir}/{args.output}.txt", "w")

    h = fileinput.input(args.files, openhook=fileinput.hook_compressed)
    encoder = Encoder()
    # pool = Pool(args.workers, initializer=encoder.initializer)
    # results = pool.imap(encoder.encode, h, 1000)
    results = map(encoder.encode, h)
    for i, line in enumerate(results):
        if line is None:
            continue

        (set_name, text) = line
        assert set_name in set_names
        assert set_name in outputs
        print(text, file=outputs[set_name])

        if i % 1000000 == 0:
            print(i, file=sys.stderr, end="", flush=True)
        elif i % 100000 == 0:
            print(".", file=sys.stderr, end="", flush=True)

    for output in outputs.values():
        print(file=output, flush=True)
        output.close()
