#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional, Set, Tuple

import xxhash
from sentencepiece import SentencePieceProcessor

from examples.nllb.mining.monolingual.utils.text_normalizer import (
    normalize_for_dedup,
    replace_unicode_punct,
)
from examples.nllb.modeling.filtering.dataset import DatasetLine
from examples.nllb.modeling.filtering.filters.base import Filter, FilteringCounts


class BasicFilter(Filter):
    def __init__(
        self,
        normalize_punctuation: bool,
        spm: Optional[str],
        min_tok: Optional[int],
        max_tok: Optional[int],
        max_tok_len_ratio: Optional[float],
        tgt_min_unique_tok_ratio: Optional[float],
        dedup_pairs: bool,
    ):
        self.normalize_punctuation = normalize_punctuation
        self.spm: Optional[SentencePieceProcessor] = None
        self.min_tok = min_tok
        self.max_tok = max_tok
        self.max_tok_len_ratio = max_tok_len_ratio
        self.tgt_min_unique_tok_ratio = tgt_min_unique_tok_ratio
        self.dedup_pairs = dedup_pairs

        if spm is not None:
            self.spm = SentencePieceProcessor(model_file=spm)
        self.seen_hashes: Set[int] = set()

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        # filter empty
        line.src = line.src.strip()
        if self.normalize_punctuation:
            line.src = replace_unicode_punct(line.src)
        if not line.src or (line.tgt is not None and not line.tgt):
            counts.empty += 1
            return None

        if line.tgt is not None:
            line.tgt = line.tgt.strip()
            if self.normalize_punctuation:
                line.tgt = replace_unicode_punct(line.tgt)
            if not line.tgt:
                counts.empty += 1
                return None

        # normalize + dedup
        if self.dedup_pairs:
            normalized = normalize_for_dedup(line.src)
            if line.tgt is not None:
                normalized += "\t" + normalize_for_dedup(line.tgt)
            line_hash = xxhash.xxh3_64_intdigest(normalized)
            if line_hash in self.seen_hashes:
                counts.dedup += 1
                return None
            else:
                self.seen_hashes.add(line_hash)

        if (
            self.min_tok
            or self.max_tok
            or self.max_tok_len_ratio
            or self.tgt_min_unique_tok_ratio
        ):
            # min len, max len
            assert self.spm is not None
            src_toks = self.spm.encode(line.src)
            src_len = len(src_toks)
            if self.min_tok is not None and src_len < self.min_tok:
                counts.min_tok += 1
                return None
            if self.max_tok is not None and src_len > self.max_tok:
                counts.max_tok += 1
                return None
            # same as above, but for tgt if set
            if line.tgt is not None:
                tgt_toks = self.spm.encode(line.tgt)
                tgt_len = len(tgt_toks)
                if self.min_tok is not None and tgt_len < self.min_tok:
                    counts.min_tok += 1
                    return None
                if self.max_tok is not None and tgt_len > self.max_tok:
                    counts.max_tok += 1
                    return None
                # len ratio
                if self.max_tok_len_ratio is not None:
                    ratio = (
                        src_len / tgt_len if src_len > tgt_len else tgt_len / src_len
                    )
                    if ratio > self.max_tok_len_ratio:
                        counts.max_tok_len_ratio += 1
                        return None

            # target minimum unique tokens
            if line.tgt is not None and self.tgt_min_unique_tok_ratio is not None:
                if len(set(tgt_toks)) / tgt_len > self.tgt_min_unique_tok_ratio:
                    counts.tgt_min_unique_tok_ratio += 1
                    return None

        return line
