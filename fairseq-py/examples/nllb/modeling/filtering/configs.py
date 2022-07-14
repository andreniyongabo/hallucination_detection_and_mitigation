#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from examples.nllb.modeling.prepare_data.data_types import ExecutorConfig


@dataclass
class LaserFilterConfig:
    _target_: str = "examples.nllb.modeling.filtering.filters.LaserFilter"
    threshold: float = 1.06
    max_target_dups: Optional[int] = None


@dataclass
class BasicFilterConfig:
    _target_: str = "examples.nllb.modeling.filtering.filters.BasicFilter"
    normalize_punctuation: bool = True
    spm: Optional[str] = None
    min_tok: Optional[int] = 1
    max_tok: Optional[int] = 250
    max_tok_len_ratio: Optional[float] = None
    tgt_min_unique_tok_ratio: Optional[float] = None
    dedup_pairs: bool = False


@dataclass
class LidFilterConfig:
    _target_: str = "examples.nllb.modeling.filtering.filters.LidFilter"
    model_path: str = "/large_experiments/nllb/mmt/lidruns/lid_models/2022-02-18_ft_model.bin"
    excluded_corpora: Optional[List[str]] = None
    excluded_languages: Optional[List[str]] = None
    default_threshold: float = 0.0
    thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class ToxicityFilterConfig:
    _target_: str = "examples.nllb.modeling.filtering.filters.ToxicityFilter"
    twl_path_template: str = "/large_experiments/nllb/mmt/data/toxicity/{lang}_twl.txt"
    eng_porn_twl_path: Optional[
        str
    ] = "/large_experiments/nllb/mmt/data/toxicity/eng_twl_short_porn.txt"
    max_toxicity: Optional[int] = None
    max_toxicity_difference: Optional[int] = 2


@dataclass
class GroupFilterConfig:
    # one (and only one) of these should be set, the other should be None
    included_corpora: Optional[List[str]] = None
    excluded_corpora: Optional[List[str]] = None

    laser_filter: Optional[LaserFilterConfig] = None
    basic_filter: BasicFilterConfig = BasicFilterConfig()
    lid_filter: Optional[LidFilterConfig] = None
    toxicity_filter: Optional[ToxicityFilterConfig] = None


@dataclass
class FilterConfig:
    data_conf_dir: str
    output_dir: str
    executor: ExecutorConfig
    directions: List[str]
    train_primary: Optional[GroupFilterConfig]
    train_mined: Optional[GroupFilterConfig]
    # train_nmt_bt: Optional[GroupFilterConfig]
    # train_smt_bt: Optional[GroupFilterConfig]


def register_configs():
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=FilterConfig)

    cs.store(name="default", group="train_primary/laser_filter", node=LaserFilterConfig)
    cs.store(name="default", group="train_primary/basic_filter", node=BasicFilterConfig)
    cs.store(
        name="default", group="train_primary/toxicity_filter", node=ToxicityFilterConfig
    )
    cs.store(name="default", group="train_primary/lid_filter", node=LidFilterConfig)

    cs.store(name="default", group="train_mined/laser_filter", node=LaserFilterConfig)
    cs.store(name="default", group="train_mined/basic_filter", node=BasicFilterConfig)
    cs.store(
        name="default", group="train_mined/toxicity_filter", node=ToxicityFilterConfig
    )
    cs.store(name="default", group="train_mined/lid_filter", node=LidFilterConfig)
