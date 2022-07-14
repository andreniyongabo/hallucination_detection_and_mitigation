# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class DataConfig:
    data_version: str = MISSING
    iteration: int = MISSING
    data_shard_dir: str = MISSING
    shard_type: str = MISSING
    bname: str = MISSING
