# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import MISSING

from examples.nllb.mining.global_mining.mining_utils import extract_shard_id
from examples.nllb.mining.global_mining.modules.mlenc_port.mine_bitext_indexes_utils import (  # noqa
    DISTANCES_FILE_SUFFIX,
    INDICES_FILE_SUFFIX,
    mine,
)
from examples.nllb.mining.nllb_lib.nllb_module import (
    DistributedRequirements,
    NLLBModule,
)
from examples.nllb.mining.nllb_lib.utils import ensure_dir

logger = logging.getLogger("mine_bitext_indexes")


@dataclass
class MineBitextConfig:
    src_lang: str = MISSING
    tgt_lang: str = MISSING
    index_type: str = MISSING
    dists_idxs_basename: str = MISSING  # the precomputed distances
    data_version: str = "V32m"
    output_dir: str = "mine.${data_version}"
    knn_dist: int = 16
    src_k: int = 16
    tgt_k: int = 16
    k_extract: int = 1
    margin: str = "ratio"
    margin_norm: str = "mean"
    num_probe: int = 128
    gpu_type: str = "fp16-shard"
    mine_threshold: float = 1.06


class MineBitextIndexesModule(NLLBModule):
    def __init__(self, config):
        super().__init__(config)
        ensure_dir(self.config.output_dir)

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=40,
            timeout_min=600,
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        # TODO: use Path to build path names
        out_base_name = os.path.abspath(
            os.path.join(
                self.config.output_dir,
                f"{self.config.src_lang}-{self.config.tgt_lang}"
                f".{self.config.index_type}.k{self.config.src_k}-{self.config.tgt_k}"
                f".{self.config.margin_norm}.np{self.config.num_probe}"
                f".{self.config.gpu_type}",
            )
        )
        # TODO: this should be removed and use filenames passed from the previous step
        # of the mining pipeline instead. It can be done once the ports have been merged
        # loading in all the precalculated distance, indices files
        wkdir = Path(self.config.dists_idxs_basename).parent
        dists_file_template = (
            f"{self.config.src_lang}-{self.config.tgt_lang}"
            f".{{direction}}.*{DISTANCES_FILE_SUFFIX}.npy"
        )
        dists_x2y_files = sorted(
            [str(f) for f in wkdir.glob(dists_file_template.format(direction="x2y"))],
            key=extract_shard_id,
        )
        dists_y2x_files = sorted(
            [str(f) for f in wkdir.glob(dists_file_template.format(direction="y2x"))],
            key=extract_shard_id,
        )
        indices_file_template = (
            f"{self.config.src_lang}-{self.config.tgt_lang}"
            f".{{direction}}.*{INDICES_FILE_SUFFIX}.npy"
        )
        indices_x2y_files = sorted(
            [str(f) for f in wkdir.glob(indices_file_template.format(direction="x2y"))],
            key=extract_shard_id,
        )
        indices_y2x_files = sorted(
            [str(f) for f in wkdir.glob(indices_file_template.format(direction="y2x"))],
            key=extract_shard_id,
        )

        # mining, extracting sentence indices
        scores, src_idx, trg_idx = mine(
            dists_x2y_files,
            dists_y2x_files,
            indices_x2y_files,
            indices_y2x_files,
            self.config.src_k,
            self.config.tgt_k,
            self.config.k_extract,
            self.config.mine_threshold,
            self.config.margin_norm == "last",
            logger,
        )

        # persisting results to disk
        meta = f"{out_base_name}.align"
        np.savez(meta, scores=scores, src_idx=src_idx, trg_idx=trg_idx)

        return meta

    def version(self):
        return "0.1"

    def name(self):
        return (
            f"mineD.{self.config.src_lang}-{self.config.tgt_lang}"
            f".{self.config.margin_norm}.meta"
        )
