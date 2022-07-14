#!/usr/bin/env python3

import errno
import glob
import re
from os import path, remove, symlink
from typing import List, Optional

CHECKPOINTS = "/checkpoint/jeanm/nllb/bt_data_req/checkpoints"
LANGS = ["asm", "zul", "amh", "tel", "urd", "vie", "ita", "fra", "deu"]
SIZES = ["1k", "5k", "10k", "25k", "50k"]
MONO_SIZES = ["150k", "300k"]


def get_bleu(f):
    for line in f:
        pass
    match = re.search(r"BLEU4 = ([0-9\.]+),", line)
    if match is None:
        return None
    return float(match.group(1))


def removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix) :]
    return s


directions_valid = {}
directions_test = {}
best_valid_checkpoints = {}
for lang in LANGS:
    for mono_size in MONO_SIZES:
        bleus_valid: List[Optional[float]] = [None] * len(SIZES)
        bleus_test: List[Optional[float]] = [None] * len(SIZES)
        best_checkpoints: List[Optional[float]] = [None] * len(SIZES)
        for i, size in enumerate(SIZES):
            for checkpoint in glob.glob(
                f"{CHECKPOINTS}/eng-{lang}.{size}+{mono_size}*"
            ):
                p_test = path.join(checkpoint, "out_test")
                p_valid = path.join(checkpoint, "out_valid")
                try:
                    with open(p_test, "rt") as ftest, open(p_valid, "rt") as fvalid:
                        bleu_valid = get_bleu(fvalid)
                        bleu_test = get_bleu(ftest)
                        if bleu_valid is None or bleu_test is None:
                            continue
                        checkpoint_name = removeprefix(checkpoint, f"{CHECKPOINTS}/")
                        print(f"{checkpoint_name}: {bleu_valid}")
                        if bleus_valid[i] is None or bleu_valid > bleus_valid[i]:
                            bleus_valid[i] = bleu_valid
                            bleus_test[i] = bleu_test
                            best_checkpoints[i] = checkpoint_name
                except FileNotFoundError:
                    pass
        direction = f"eng-{lang}+{mono_size}"
        directions_valid[direction] = bleus_valid
        directions_test[direction] = bleus_test
        best_valid_checkpoints[direction] = best_checkpoints

print("Test:")
print(directions_test)
print("Valid:")
print(directions_valid)
print("\nBest valid checkpoints:")
print(best_valid_checkpoints)
