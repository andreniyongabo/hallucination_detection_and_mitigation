import argparse
import bz2
import linecache
import os
import random
import subprocess
import sacrebleu
import pyter
import re
from nltk.util import ngrams
from collections import defaultdict
import pickle
import csv
import time
from joblib import Parallel, delayed
import tqdm
from typing import Dict, Tuple


from glob import glob

random.seed(50)


def get_score(scorer, src_translated_sent, tgt_sent):
    if scorer == "bleu":
        return sacrebleu.sentence_bleu(src_translated_sent, [tgt_sent]).score
    elif scorer == "ter":
        return pyter.ter(src_translated_sent.split(), tgt_sent.split())


def compare_score(scorer, score1, score2):
    """Returns True if score1 is 'better' than score2."""
    if scorer == "bleu":
        return score1 > score2
    elif scorer == "ter":
        return score1 < score2


def score(args):
    candidate_files = glob(args.candidates_paths)
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"scores.{args.src}-{args.tgt}",
    )
    print(f"Going through {len(candidate_files)} files")
    candidate_set = set()
    for result_file in candidate_files:
        with bz2.open(result_file, "rt") as result_f:
            for line in result_f:
                src_file_name, src_line_num, tgt_file_name, tgt_line_num = line.split()
                candidate_set.add(
                    (src_file_name, int(src_line_num), tgt_file_name, int(tgt_line_num))
                )
    print(f"{len(candidate_set)} unique candidate pairs")
    with bz2.open(output_file, "wt") as out_f:
        for src_file_name, src_line_num, tgt_file_name, tgt_line_num in tqdm.tqdm(
            candidate_set
        ):
            src_translated_path = os.path.join(
                args.src_txt_dir, src_file_name.replace(args.tok_suffix, "raw")
            )
            src_sent_path = os.path.join(
                args.src_txt_dir, src_file_name.replace(args.tok_suffix, "src")
            )
            tgt_sent_path = os.path.join(
                args.tgt_txt_dir, tgt_file_name.replace(args.tok_suffix, "raw")
            )
            src_sent = linecache.getline(src_sent_path, src_line_num + 1).strip()
            src_translated_sent = linecache.getline(
                src_translated_path, src_line_num + 1
            ).strip()
            tgt_sent = linecache.getline(tgt_sent_path, tgt_line_num + 1).strip()
            score = get_score(args.scorer, src_translated_sent, tgt_sent)
            if compare_score(args.scorer, score, args.threshold):
                print(
                    f"{score}\t{src_sent}\t{src_translated_sent}\t{tgt_sent}",
                    file=out_f,
                )
    print(f"Done with threshold {args.threshold}")


def merge_filter(args):
    score_files = glob(args.candidates_paths)
    mapping: Dict[str, Tuple[str, float]] = {}
    original_count = 0
    for score_file in score_files:
        with bz2.open(score_file, "rt") as score_f:
            for line in score_f:
                original_count += 1
                score, src_sent, _, tgt_sent = line.split("\t")
                score = float(score)
                src_sent = src_sent.strip()
                tgt_sent= tgt_sent.strip()
                if compare_score(args.scorer, score, args.threshold):
                    if src_sent not in mapping or compare_score(args.scorer, score, mapping[src_sent][1]):
                        mapping[src_sent] = (tgt_sent, score)

    print(f"Filtered {original_count} pairs to {len(mapping)} pairs")
    output_prefix = os.path.join(
        args.output_dir, f"mined.{args.scorer}.{args.threshold}.{args.src}-{args.tgt}"
    )
    with open(f"{output_prefix}.{args.src}", "w") as src_out, open(
        f"{output_prefix}.{args.tgt}", "w"
    ) as tgt_out, open(f"{output_prefix}.score", "w") as score_out:
        for src_sent in mapping:
            tgt_sent, score = mapping[src_sent]
            print(src_sent, file=src_out)
            print(tgt_sent, file=tgt_out)
            print(score, file=score_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step")
    parser.add_argument(
        "--candidates-paths", default="ngram_mining_outputs/mining_outputs/results.*"
    )
    parser.add_argument(
        "--output-dir", default="ngram_mining_outputs/filtered_mining_outputs"
    )
    parser.add_argument("--src-txt-dir", default="ngram_mining_outputs/hau_txt")
    parser.add_argument("--tgt-txt-dir", default="ngram_mining_outputs/eng_txt")
    parser.add_argument("--src", default="hau")
    parser.add_argument("--tgt", default="eng")
    parser.add_argument("--scorer", default="bleu")
    parser.add_argument("--threshold", default=30, type=float)
    parser.add_argument("--tok-suffix", default="tok")
    parser.add_argument("--workers", default=32, type=int)
    args = parser.parse_args()
    if args.step == "score":
        score(args)
    elif args.step == "merge_filter":
        merge_filter(args)
