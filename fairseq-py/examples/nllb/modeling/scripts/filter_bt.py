#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fileinput
import gzip
import hashlib
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from glob import glob
from typing import List, Optional

import fasttext
import sentencepiece as spm
import yaml
from submitit import AutoExecutor, helpers


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    empty_filtered: int = 0
    min_len_filtered: int = 0
    max_len_filtered: int = 0
    len_ratio_filtered: int = 0
    unique_ratio_filtered: int = 0
    lid_filtered: int = 0
    duplicate_bt_filtered: int = 0


def try_decode_pair(bt_toks, orig_toks, lid, spm_model, bt_lang, filter_stats, args):
    bt_toks = bt_toks.split(" ")
    orig_toks = orig_toks.split(" ")

    # Filter out if either side is empty
    if not bt_toks or not orig_toks:
        filter_stats.empty_filtered += 1
        return None

    # Filter out if lengths are not in the acceptable range or their ratio is extreme
    bt_len = len(bt_toks)
    orig_len = len(orig_toks)
    min_len = min(bt_len, orig_len)
    max_len = max(bt_len, orig_len)
    if args.min_len > 0 and min_len < args.min_len:
        filter_stats.min_len_filtered += 1
        return None
    if args.max_len > 0 and max_len > args.max_len:
        filter_stats.max_len_filtered += 1
        return None
    if args.max_len_ratio is not None and max_len / min_len > args.max_len_ratio:
        filter_stats.len_ratio_filtered += 1
        return None

    # Filter out if the backtranslated side has too many repeated tokens
    if (
        args.min_unique_ratio is not None
        and len(set(bt_toks)) / bt_len < args.min_unique_ratio
    ):
        filter_stats.unique_ratio_filtered += 1
        return None

    # Strip special tokens
    if args.strip_bt_tag:
        bt_toks = bt_toks[1:]
    if args.strip_orig_tag:
        orig_toks = orig_toks[1:]
    while orig_toks[0] in args.strip_special_tags:
        orig_toks = orig_toks[1:]
    while bt_toks[0] in args.strip_special_tags:
        bt_toks = bt_toks[1:]

    # Decode
    if spm_model is not None:
        bt = spm_model.decode(bt_toks)
        orig = spm_model.decode(orig_toks)
    else:
        bt = " ".join(bt_toks)
        orig = " ".join(orig_toks)

    # Filter based on LID score
    if args.lid_threshold > 0 and len(bt) > args.lid_min_len:
        lid_l, lid_p = lid.predict(bt, k=-1, threshold=0.0)
        lid_probs = {label[9:]: prob for label, prob in zip(lid_l, lid_p)}
        if lid_probs.get(bt_lang, 0) < args.lid_threshold:
            filter_stats.lid_filtered += 1
            return None

    return bt, orig


def safe_index(toks, index, default):
    try:
        return toks[index]
    except IndexError:
        return default


def process_lang(bt_lang, orig_lang, args):
    print(args)

    ft_lid = None
    if args.lid_threshold > 0:
        assert args.lid_model is not None
        ft_lid = fasttext.load_model(args.lid_model)

    # Get the list of shard outputs, ensuring that we only use one output per shard
    # in case there are multiple – prioritising the file that was modified last.
    shard_list = defaultdict(lambda: None)
    pattern = os.path.join(
        args.base_folder, "bt_out", f"{orig_lang}-{bt_lang}", "*.out"
    )
    for path in glob(pattern):
        shard = path[path.rindex("_") + 1 : -4]
        mtime = os.path.getmtime(path)
        _, old_mtime = shard_list.get(shard, (None, 0))
        if mtime > old_mtime:
            shard_list[shard] = (path, mtime)
    file_list = [x[0] for x in shard_list.values()]

    if args.spm_decode is not None:
        spm_model = spm.SentencePieceProcessor()
        spm_model.Load(args.spm_decode)
    else:
        spm_model = None

    bt_outpath = os.path.join(
        args.base_folder, "bt_filtered", f"{bt_lang}-{orig_lang}", f"bt.{bt_lang}.gz"
    )
    orig_outpath = os.path.join(
        args.base_folder, "bt_filtered", f"{bt_lang}-{orig_lang}", f"bt.{orig_lang}.gz",
    )
    direction_folder = os.path.dirname(bt_outpath)
    os.makedirs(direction_folder, exist_ok=True)
    bt_counts = Counter()
    filter_stats = FilterStats()
    with gzip.open(bt_outpath, "wt") as fout_bt, gzip.open(
        orig_outpath, "wt"
    ) as fout_orig:
        for line in fileinput.input(file_list):
            if line.startswith("S-"):
                orig = safe_index(line.rstrip().split("\t"), 1, "")
            elif line.startswith("H-"):
                if orig is not None:
                    bt = safe_index(line.rstrip().split("\t"), 2, "")
                    filter_stats.total += 1
                    decoded_pair = try_decode_pair(
                        bt, orig, ft_lid, spm_model, bt_lang, filter_stats, args
                    )
                    if decoded_pair is not None:
                        bt, orig = decoded_pair
                        md5 = hashlib.md5(bt.encode())
                        bt_counts[md5] += 1
                        if args.max_repeat < 0 or bt_counts[md5] <= args.max_repeat:
                            print(bt, file=fout_bt)
                            print(orig, file=fout_orig)
                            filter_stats.kept += 1
                        else:
                            filter_stats.duplicate_bt_filtered += 1
                    orig = None

    print(filter_stats)
    return f"{bt_lang}-{orig_lang}", filter_stats


def main(args):
    executor = AutoExecutor(
        os.path.join(args.base_folder, "executor_logs"),
        cluster="slurm" if not args.local_run else "local",
    )
    executor.update_parameters(
        slurm_partition=args.slurm_partition, timeout_min=args.slurm_timeout, nodes=1,
    )

    jobs = []
    for direction in args.directions:
        bt_lang, orig_lang = direction.split("-")
        jobs.append(executor.submit(process_lang, bt_lang, orig_lang, args))

    results = [job.result() for job in jobs]
    for direction, filter_stats in results:
        kept_pct = {"kept_pct": round(100 * filter_stats.kept / filter_stats.total, 2)}
        print(yaml.dump({direction: dict(filter_stats.__dict__, **kept_pct)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Extract backtranslations from the outputs of fairseq-generate. "
            "If there are multiple hypotheses for a source, we only keep the first one. "
            "If there are multiple outputs for a shard, we only keep the most recent. "
        )
    )
    parser.add_argument(
        "--directions",
        required=True,
        nargs="*",
        help="Space-separated list of directions in the format $bt_lang-$original_lang"
        " ($original_lang corresponds to S-* lines, $bt_lang to H-* lines).",
    )
    parser.add_argument(
        "--min-len", type=Optional[int], default=2, help="Min length filter (tokens)."
    )
    parser.add_argument(
        "--max-len", type=Optional[int], default=150, help="Max length filter (tokens)."
    )
    parser.add_argument(
        "--max-len-ratio",
        type=float,
        default=2.0,
        help="Maximum ratio of token length between bt and original:"
        " max(bt_len, orig_len) / min(bt_len, orig_len).",
    )
    parser.add_argument(
        "--min-unique-ratio",
        type=float,
        default=0.3,
        help="Minimum fraction of unique tokens: unique_tokens / total_tokens",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=20,
        help="Maximum number of times the same backtranslation (H-* line) can appear "
        "in a corpus – ignore after that.",
    )
    parser.add_argument(
        "--spm-decode", type=str, help="Path to the SPM model to be used for decoding.",
    )
    parser.add_argument(
        "--strip-bt-tag",
        action="store_true",
        help="Strip the first token in the backtranslated sentence (used for language tokens).",
    )
    parser.add_argument(
        "--strip-orig-tag",
        action="store_true",
        help="Strip the first token in the original sentence (used for language tokens).",
    )
    parser.add_argument(
        "--strip-special-tags",
        type=List[str],
        default=[
            "▁SecondaryData",
            "▁<nmt_bt_data>",
            "▁<binmt_bt_data>",
            "▁<smt_bt_data>",
            "▁<mined_data>",
        ],
        help="List of tags to strip if they appear at the start of a sentence.",
    )
    parser.add_argument(
        "--local-run", action="store_true", help="Run locally instead of on SLURM.",
    )
    parser.add_argument(
        "--slurm-partition", type=str, default="devaccel",
    )
    parser.add_argument(
        "--slurm-timeout", type=int, default=1440,
    )
    parser.add_argument(
        "--lid-min-len",
        type=int,
        default=15,
        help="Only perform LID filtering on sentences of at least this length",
    )
    parser.add_argument(
        "--lid-threshold",
        type=float,
        default=-1.0,
        help="Filter sentences where LID probabilities are below this threshold.",
    )
    parser.add_argument(
        "--lid-model",
        type=Optional[str],
        default=None,
        help="Location of the FT LID *.bin model.",
    )
    parser.add_argument(
        "base_folder",
        type=str,
        help="Base folder with binarised data and BT output. Fairseq outputs from "
        "which to extract BT are expected in bt_out/$direction/*_{0..999}.out. "
        "The filtered outputs will be placed under bt_filtered/$direction/bt.$lang.gz.",
    )

    args = parser.parse_args()
    main(args)
