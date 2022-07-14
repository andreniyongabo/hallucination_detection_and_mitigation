#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Optional, Tuple

from examples.nllb.modeling.filtering.dataset import DatasetLine


# because of some hydra weirdness (can't pass dataclasses to hydra.utils.instantiate)
# this cannot be a dataclass
class FilteringCounts:
    def __init__(
        self,
        total_before: int = 0,  # total examples before filtering
        total_after: int = 0,  # total examples after filtering
        empty: int = 0,  # num of examples filtered due to being empty
        dedup: int = 0,
        min_tok: int = 0,
        max_tok: int = 0,
        max_tok_len_ratio: int = 0,
        tgt_min_unique_tok_ratio: int = 0,
        max_toxicity: int = 0,
        max_toxicity_difference: int = 0,
        lid_threshold: int = 0,
        laser_threshold: int = 0,  # num of examples filtered due to LASER score
        max_target_dups: int = 0,
    ):
        self.total_before = total_before
        self.total_after = total_after

        # BasicFilter
        self.empty = empty
        self.dedup = dedup
        self.min_tok = min_tok
        self.max_tok = max_tok
        self.max_tok_len_ratio = max_tok_len_ratio
        self.tgt_min_unique_tok_ratio = tgt_min_unique_tok_ratio

        # ToxicityFilter
        self.max_toxicity = max_toxicity
        self.max_toxicity_difference = max_toxicity_difference

        # LidFilter
        self.lid_threshold = lid_threshold

        # LaserFilter
        self.laser_threshold = laser_threshold
        self.max_target_dups = max_target_dups

    def __add__(self, other):
        return FilteringCounts(
            total_before=self.total_before + other.total_before,
            total_after=self.total_after + other.total_after,
            empty=self.empty + other.empty,
            dedup=self.dedup + other.dedup,
            min_tok=self.min_tok + other.min_tok,
            max_tok=self.max_tok + other.max_tok,
            max_tok_len_ratio=self.max_tok_len_ratio + other.max_tok_len_ratio,
            tgt_min_unique_tok_ratio=self.tgt_min_unique_tok_ratio
            + other.tgt_min_unique_tok_ratio,
            max_toxicity=self.max_toxicity + other.max_toxicity,
            max_toxicity_difference=self.max_toxicity_difference
            + other.max_toxicity_difference,
            lid_threshold=self.lid_threshold + other.lid_threshold,
            laser_threshold=self.laser_threshold + other.laser_threshold,
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return other.__add__(self)


class Filter:
    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        raise NotImplementedError
