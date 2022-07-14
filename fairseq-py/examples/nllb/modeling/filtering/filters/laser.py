#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import xxhash

from examples.nllb.mining.monolingual.utils.text_normalizer import normalize_for_dedup
from examples.nllb.modeling.filtering.dataset import DatasetLine
from examples.nllb.modeling.filtering.filters.base import Filter, FilteringCounts


class LaserFilter(Filter):
    def __init__(self, threshold: float, max_target_dups: Optional[int]):
        self.threshold = threshold

        # target-side deduplication
        self.max_target_dups = max_target_dups
        self.dup_counts: Dict[int, int] = Counter()

    def filter_line(
        self, line: DatasetLine, counts: FilteringCounts
    ) -> Optional[DatasetLine]:
        if not line.score:
            return line
        if line.score < self.threshold:
            counts.laser_threshold += 1
            return None
        if self.max_target_dups:
            line_hash = xxhash.xxh3_64_intdigest(normalize_for_dedup(line.tgt))
            if self.dup_counts[line_hash] >= self.max_target_dups:
                counts.max_target_dups += 1
                return None
            self.dup_counts[line_hash] += 1
        return line
