# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import bz2
import glob
import hashlib
import os
import pickle
import random
import tempfile
import time
from collections import defaultdict

import tqdm
from joblib import Parallel, delayed
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

random.seed(50)


def get_hash(string):
    return int(hashlib.sha1(string.encode()).hexdigest(), 16)


def chunk_func(l, n):
    for i in range(0, len(l), n):
        yield i, l[i : i + n]


class Mapper:
    def __init__(self, args):
        super().__init__()
        self.args = args
        os.makedirs(args.output_path, exist_ok=True)
        self.path = args.input_path
        _, self.file_name = os.path.split(self.path)

    def _mapper(self, start_idx, content):
        ngram_to_sentence_ids = [defaultdict(list) for i in range(self.args.num_shards)]
        for i, line in tqdm.tqdm(enumerate(content)):
            sentence = line.strip()
            sentence_id = (self.file_name, i + start_idx)
            tokens = sentence.split(" ")
            ngram_list = list(ngrams(tokens, self.args.ngram_order))
            for ngram in ngram_list:
                ngram_str = " ".join(ngram)
                shard_id = get_hash(ngram_str) % self.args.num_shards
                ngram_to_sentence_ids[shard_id][ngram_str].append(sentence_id)

        # save shard wise dict for this worker
        for i in range(self.args.num_shards):
            output_file = os.path.join(
                self.args.output_path, f"shard{i}_{os.getpid()}.bin"
            )
            pickle.dump(ngram_to_sentence_ids[i], open(output_file, "wb"))

    def _reducer(self, shard_id):
        paths = glob.glob(os.path.join(self.args.output_path, f"shard{shard_id}_*.bin"))
        results = defaultdict(list)
        for path in tqdm.tqdm(paths):
            index = pickle.load(open(path, "rb"))
            for ngram_str, sentence_ids in index.items():
                results[ngram_str].extend(sentence_ids)

        output_path = os.path.join(self.args.output_path, f"shard{shard_id}.bin")
        print(f"Saving to {output_path}, size: {len(results)}")
        pickle.dump(results, open(output_path, "wb"))

        for pth in paths:
            os.remove(pth)

    def parallel_map(self):
        now = time.time()
        with open(self.path) as fi:
            sentences = fi.readlines()
        chunk_size = -(-len(sentences) // self.args.workers)  # ceil operation
        chunks = chunk_func(sentences, chunk_size)

        # Map with each worker
        Parallel(n_jobs=self.args.workers, verbose=100)(
            delayed(self._mapper)(i[0], i[1]) for i in chunks
        )

        # Reduce for each shard
        Parallel(n_jobs=self.args.workers, verbose=100)(
            delayed(self._reducer)(i) for i in range(0, self.args.num_shards)
        )

        print(
            f"Time to complete parallel map step: {round(time.time() - now, 2)} seconds."
        )

    def sequential_map(self):
        # Map from ngram to sentence hash
        ngram_to_sentence_ids = [defaultdict(list) for i in range(args.num_shards)]

        now = time.time()
        with open(self.path) as fi:
            for i, line in enumerate(fi):
                sentence = line.strip()
                sentence_id = (self.file_name, i)
                tokens = sentence.split(" ")
                ngram_list = list(ngrams(tokens, self.args.ngram_order))
                for ngram in ngram_list:
                    ngram_str = " ".join(ngram)
                    shard_id = get_hash(ngram_str) % self.args.num_shards
                    ngram_to_sentence_ids[shard_id][ngram_str].append(sentence_id)

        print(f"Done with {self.path}. Processed {i+1} lines")

        for i in range(self.args.num_shards):
            output_file = os.path.join(self.args.output_path, f"shard{i}.bin")
            print(
                f"Saving ngram_to_sentence_ids size={len(ngram_to_sentence_ids[i])} to {output_file}"
            )
            pickle.dump(ngram_to_sentence_ids[i], open(output_file, "wb"))
        print(
            f"Time to complete sequential map step: {round(time.time() - now, 2)} seconds."
        )


class Reducer:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.results = defaultdict(list)

    def _mapper(self, paths, directory):
        tmp_results = defaultdict(list)
        for path in tqdm.tqdm(paths):
            index = pickle.load(open(path, "rb"))
            for ngram_str, sentence_ids in index.items():
                tmp_results[ngram_str].extend(sentence_ids)

        # save shard wise dict for this worker
        output_file = os.path.join(directory, f"tmp_{os.getpid()}.bin")
        pickle.dump(tmp_results, open(output_file, "wb"))

    def _reducer(self, path):
        index = pickle.load(open(path, "rb"))
        for ngram_str, sentence_ids in tqdm.tqdm(index.items()):
            self.results[ngram_str].extend(sentence_ids)

    def parallel_reduce(self):
        now = time.time()
        paths = glob.glob(os.path.join(self.args.input_path))
        chunk_size = -(-len(paths) // self.args.workers)  # ceil operation
        chunks = chunk_func(paths, chunk_size)

        with tempfile.TemporaryDirectory() as tmpdirname:
            # First reduce over each chunk_size of files
            Parallel(n_jobs=self.args.workers, verbose=100)(
                delayed(self._mapper)(i[1], tmpdirname) for i in chunks
            )

            # Full reduce over all chunks, multithreaded op
            tmp_files = glob.glob(os.path.join(tmpdirname, "tmp_*.bin"))
            Parallel(n_jobs=self.args.workers, verbose=100, require="sharedmem")(
                delayed(self._reducer)(i) for i in tmp_files
            )

            print(f"Saving to {self.args.output_path}, size: {len(self.results)}")
            with bz2.BZ2File(self.args.output_path, "wb") as f:
                pickle.dump(self.results, f)

        print(
            f"Time to complete parallel reduce step: {round(time.time() - now, 2)} seconds."
        )

    def sequential_reduce(self):
        now = time.time()
        paths = glob.glob(os.path.join(self.args.input_path))
        results = defaultdict(list)
        for path in paths:
            index = pickle.load(open(path, "rb"))
            for ngram_str, sentence_ids in index.items():
                results[ngram_str].extend(sentence_ids)
            print(f"Processed {path}, size: {len(results)}")

        print(f"Saving to {self.args.output_path}, size: {len(results)}")
        pickle.dump(results, bz2.BZ2File(self.args.output_path, "wb"))
        print(
            f"Time to complete sequential reduce step: {round(time.time() - now, 2)} seconds."
        )


"""
Given a list of tokenized text, build inverted indexes from ngrams to sentence-id

"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", default="ngram_mining_outputs/eng_txt/0.tok")
    parser.add_argument("--step")
    parser.add_argument("--output-path", default="ngram_mining_outputs/eng_index/0")
    parser.add_argument("--ngram-order", default=4, type=int)
    parser.add_argument("--num-shards", default=1024, type=int)
    parser.add_argument("--mode", default="parallel", type=str)
    parser.add_argument("--workers", default=8, type=int)
    args = parser.parse_args()
    if args.step == "map":
        mapper = Mapper(args)
        if args.mode == "parallel":
            mapper.parallel_map()
        else:
            mapper.sequential_map()
    elif args.step == "reduce":
        reducer = Reducer(args)
        if args.mode == "parallel":
            reducer.parallel_reduce()
        else:
            reducer.sequential_reduce()
    else:
        print(f"Unknown step {args.step}")
