import argparse
import os
import random
import subprocess
import re
from nltk.util import ngrams
from collections import defaultdict
import pickle
import bz2
import time

from glob import glob

random.seed(50)
DELIM = "\t"


def load_sharded_dict(paths):
    results = {}
    sharded_paths = glob(paths)
    for sharded_path in sharded_paths:
        sharded_results = pickle.load(bz2.open(sharded_path, "rb"))
        results.update(sharded_results)
    return results


def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    src_index = load_sharded_dict(args.src_index_path)
    print(f"Loaded {args.src_index_path}, size: {len(src_index)}")
    tgt_index = load_sharded_dict(args.tgt_index_path)
    print(f"Loaded {args.tgt_index_path}, size: {len(tgt_index)}")
    shared_ngrams = list(set(src_index.keys()) & set(tgt_index.keys()))
    print(f"Matching ngrams: {len(shared_ngrams)}")

    selected_ngrams = []
    for ngram in shared_ngrams:
        if (
            len(src_index[ngram]) <= args.ngram_max_freq
            and len(tgt_index[ngram]) <= args.ngram_max_freq
        ):
            selected_ngrams.append(ngram)
    print(f"Using ngram_max_freq={args.ngram_max_freq}.")
    print(f"Filtered matching ngrams: {len(selected_ngrams)}")

    out_files = [
        bz2.open(os.path.join(args.output_path, f"shard{i}.bz2"), "wt")
        for i in range(args.num_shards)
    ]
    for ngram in selected_ngrams:
        for src_file_name, src_line_num in src_index[ngram]:
            for tgt_file_name, tgt_line_num in tgt_index[ngram]:
                tgt_file_id = int(tgt_file_name.split(".")[0])
                shard_id = tgt_file_id % args.num_shards
                print(
                    src_file_name,
                    src_line_num,
                    tgt_file_name,
                    tgt_line_num,
                    file=out_files[shard_id],
                )
    for out_file in out_files:
        out_file.close()

    print(f"Finished finding matching candidates.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-index-path", default="ngram_mining_outputs/hau_index/shard0_combined.bin"
    )
    parser.add_argument(
        "--tgt-index-path", default="ngram_mining_outputs/eng_index/shard0_combined.bin"
    )
    parser.add_argument(
        "--output-path", default="ngram_mining_outputs/mining_outputs/results.0.0"
    )
    parser.add_argument("--num-shards", type=int, default=1)

    parser.add_argument("--ngram-max-freq", default=1, type=float)

    main(parser.parse_args())
