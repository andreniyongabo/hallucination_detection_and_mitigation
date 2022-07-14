#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Compute language vector representation of the CC100 corpora.

import numpy as np
import os
import sys
import tqdm

import torch
from fairseq.data.data_utils import load_indexed_dataset


data_dirs = {
    "cc100": "/datasets01/cc100-bin/072820/250/",
    "cc100_xl": "/large_experiments/flores/namangoyal/cc100_combined/final-bin/",
    "cc100_combined_roberta": "/large_experiments/moe/cc100_xl_roberta/final_bin",
}
ckpt_dir = "/checkpoint/victorialin/multilingual_moe_lm/lang_stats/"


def comp_cc100_data_stats():
    dataset = sys.argv[1]
    data_dir = data_dirs[dataset]
    out_dir = os.path.join(ckpt_dir, dataset)
    shard_id = 0
    shard_dir = os.path.join(data_dir, "shard{}".format(shard_id))

    for lang in os.listdir(shard_dir):
        lang_dir = os.path.join(shard_dir, lang)
        if not os.path.isdir(lang_dir):
            continue
        with open(os.path.join(lang_dir, "dict.txt")) as f:
            vocab_size = len(f.readlines())
        word_count = torch.zeros(vocab_size)
        one_block = torch.ones(vocab_size)

        if os.path.isdir(lang_dir):
            for data_tag in ["", "1"]:
                if not os.path.exists(
                    os.path.join(lang_dir, "train{}.bin".format(data_tag))
                ):
                    continue
                fs_data = load_indexed_dataset(
                    os.path.join(lang_dir, "train{}".format(data_tag))
                )
                for x in tqdm.tqdm(fs_data):
                    try:
                        word_count.scatter_add_(dim=0, index=(x - 1), src=one_block)
                    except RuntimeError as e:
                        print(e)
                out_dat = os.path.join(out_dir, "{}{}.bow.dat".format(lang, data_tag))
                np.save(out_dat, word_count.numpy())
                print("{} BOW representation saved to {}".format(lang, out_dat))


if __name__ == "__main__":
    comp_cc100_data_stats()
