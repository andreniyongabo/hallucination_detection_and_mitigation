#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Compute language data statistics of the CC100 corpora.

import json
import os
import sys

from fairseq.data.data_utils import load_indexed_dataset


data_dirs = {
    "cc100": "/datasets01/cc100-bin/072820/250/",
    "cc100_xl": "/large_experiments/flores/namangoyal/cc100_combined/final-bin/",
}
ckpt_dir = "/checkpoint/victorialin/multilingual_moe_lm/lang_stats/"


def comp_cc100_data_stats():
    dataset = sys.argv[1]
    data_dir = data_dirs[dataset]
    out_dir = os.path.join(ckpt_dir, dataset)
    num_shards = 40

    for shard_id in range(num_shards):
        shard_dir = os.path.join(data_dir, "shard{}".format(shard_id))
        num_tokens_in_shard = dict()
        for lang_name in os.listdir(shard_dir):
            lang_dir = os.path.join(shard_dir, lang_name)
            if os.path.isdir(lang_dir):
                fs_data = load_indexed_dataset(os.path.join(lang_dir, "train"))
                shard_size = sum([len(x) for x in fs_data])
                num_tokens_in_shard[lang_name] = shard_size
                print("{}: {:,} tokens".format(lang_dir, shard_size))

        out_json = os.path.join(out_dir, "shard{}_stats.json".format(shard_id))
        with open(out_json, "w") as o_f:
            json.dump(num_tokens_in_shard, o_f)
            print("shard stats saved to {}".format(out_json))
        print()


if __name__ == "__main__":
    comp_cc100_data_stats()
