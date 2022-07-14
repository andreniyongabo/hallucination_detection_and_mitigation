# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import os
import typing as tp
from dataclasses import dataclass
from enum import Enum

import numpy as np
from omegaconf.omegaconf import MISSING

from examples.nllb.mining.global_mining.modules.mlenc_port.calculate_distances_utils import (  # noqa
    compute_distances,
    load_index,
    save_to_disk,
)
from examples.nllb.mining.nllb_lib.nllb_module import (
    DistributedRequirements,
    NLLBModule,
)
from examples.nllb.mining.nllb_lib.utils import ensure_dir

logger = logging.getLogger("calculate_distances")


class DistanceType(Enum):
    src2tgt = "x2y"
    tgt2src = "y2x"


@dataclass
class CalculateDistancesConfig:
    lang: str = MISSING
    other_lang: str = MISSING  # mostly for logging
    lang_embeddings: tp.List[str] = MISSING  # list of embedding files
    distance_type: DistanceType = MISSING  # mostly for logging
    index_other_lang: str = MISSING  # "path/to/index"

    output_dir: tp.Optional[
        str
    ] = None  # If None, will be set to dist.src-tgt.knn.numprobe.gpu_type

    num_probe: int = 128
    knn: int = 16
    gpu_type: str = "fp16-shard"
    save_dists_as_fp16: bool = True
    embedding_dimensions: int = 1024
    normalize_query_embeddings: bool = True


class CalculateDistancesModule(NLLBModule):
    def __init__(
        self,
        config: CalculateDistancesConfig = CalculateDistancesConfig(),
    ):
        super().__init__(config)
        self.bi = "-".join(sorted([self.config.lang, self.config.other_lang]))
        self.output_dir = os.path.abspath(
            f"dist-{self.bi}.k{self.config.knn}.np{self.config.num_probe}"
            f".{self.config.gpu_type}"
            if self.config.output_dir is None
            else self.config.output_dir
        )
        ensure_dir(self.output_dir)

        fp16 = getattr(self.config, "fp16_embeddings", False)
        self.embedding_dtype = np.float16 if fp16 else np.float32

        index_size = os.path.getsize(self.config.index_other_lang) >> 30  # size in GB
        # TODO: add gpu_memory_gb to either a preset or this module's config
        num_gpu = math.ceil(index_size / self.config.gpu_memory_gb)
        # max to 8, min to 1
        self.num_gpu = 8 if num_gpu > 8 else 1 if num_gpu < 1 else num_gpu

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            mem_gb=self.num_gpu * 50,
            tasks_per_node=1,
            gpus_per_node=self.num_gpu,
            cpus_per_task=10,
            timeout_min=500,
            constraint="volta32gb",
        )

    def array(self):
        return self.config.lang_embeddings

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,  # the embedding shard
        iteration_index: int = 0,
    ):
        # loading the index in memory
        current_index = load_index(
            self.config.index_other_lang,
            self.config.num_probe,
            self.config.gpu_type,
        )

        # computing distances
        distances, indices = compute_distances(
            iteration_value,
            current_index,
            self.config.embedding_dimensions,
            self.embedding_dtype,
            self.config.knn,
            self.config.normalize_query_embeddings,
        )

        # persisting to disk
        distance_file = os.path.join(
            self.output_dir,
            f"{self.bi}.{self.config.distance_type.value}.{iteration_index:03d}",
        )
        save_to_disk(
            distances,
            indices,
            distance_file,
            self.config.save_dists_as_fp16,
        )

        return distance_file

    def name(self):
        return (
            f"knn.{self.bi}.{self.config.distance_type.value}"
            f".x{len(self.config.lang_embeddings)}.k{self.config.knn}"
            f".np{self.config.num_probe}"
        )

    def comment(self):
        return "Calculating distances between embeddings and FAISS index"

    def version(self):
        return "0.3"
