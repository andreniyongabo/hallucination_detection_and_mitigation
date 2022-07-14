# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
import random
import time

import tqdm
from joblib import Parallel, delayed
from nltk.util import ngrams

random.seed(50)


def chunk_func(l, n):
    for i in range(0, len(l), n):
        yield i, l[i : i + n]


def extract_ngrams(args, start_idx, content):
    with open(f"{args.output_path}.partial.{start_idx}", "w") as out_f:
        for i, line in tqdm.tqdm(enumerate(content)):
            sentence = line.strip()
            tokens = sentence.split(" ")
            ngram_list = list(ngrams(tokens, args.ngram_order))
            for ngram in ngram_list:
                ngram_str = " ".join(ngram)
                print(ngram_str, file=out_f)


def extract_phrases_parallel(args):
    now = time.time()
    with open(args.input_path) as fi:
        sentences = fi.readlines()
    chunk_size = -(-len(sentences) // args.workers)  # ceil operation
    chunks = chunk_func(sentences, chunk_size)

    # Map with each worker
    Parallel(n_jobs=args.workers, verbose=100)(
        delayed(extract_ngrams)(args, i[0], i[1]) for i in chunks
    )

    print(f"Time to complete parallel map step: {round(time.time() - now, 2)} seconds.")
    partial_files = glob.glob(f"{args.output_path}.partial.*")
    with open(args.output_path, "w") as out_f:
        for partial_file in partial_files:
            for line in open(partial_file):
                print(line.strip(), file=out_f)
            os.remove(partial_file)


"""
Given sentences, extrace phrases

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-path",
        default="/large_experiments/nllb/mmt/data/monolingual/mini_mine.v1/cld3_filtered/wol.txt/0",
    )
    parser.add_argument(
        "--output-path",
        default="/large_experiments/nllb/mmt/phrase_mining/phrases/wol/0",
    )
    parser.add_argument("--ngram-order", default=4, type=int)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    extract_phrases_parallel(args)
