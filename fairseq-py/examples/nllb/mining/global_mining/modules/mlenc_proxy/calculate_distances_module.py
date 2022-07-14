# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import math
import os
import subprocess
import typing as tp
from dataclasses import dataclass
from enum import Enum

from omegaconf.omegaconf import MISSING

from examples.nllb.mining.global_mining.mining_utils import FakeEmbedName
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
    # TODO this should be a python import instead
    laser_dir: str = "/private/home/schwenk/projects/mlenc"

    num_probe: int = 128
    knn: int = 16
    gpu_type: str = "fp16-shard"
    dist_opt: str = "--dist-fp16"


class CalculateDistancesModule(NLLBModule):
    def __init__(
        self,
        config: CalculateDistancesConfig = CalculateDistancesConfig(),
    ):
        super().__init__(config)
        self.bi = "-".join(sorted([self.config.lang, self.config.other_lang]))
        self.output_dir = os.path.abspath(
            f"dist-{self.bi}.k{self.config.knn}.np{self.config.num_probe}.{self.config.gpu_type}"
            if self.config.output_dir is None
            else self.config.output_dir
        )

        ensure_dir(self.output_dir)
        index_size = os.path.getsize(self.config.index_other_lang) >> 30  # size in GB
        num_gpu = math.ceil(index_size / self.config.gpu_memory_gb)
        # max to 8, min to 1
        self.num_gpu = 8 if num_gpu > 8 else 1 if num_gpu < 1 else num_gpu
        self.index_opt = " --index-k 4" if self.num_gpu >= 7 else " --index-k 8"

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            # mem_gb=self.num_gpu * 50,
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
        distance_file = os.path.join(
            self.output_dir,
            f"{self.bi}.{self.config.distance_type.value}.{iteration_index:03d}",
        )

        log_file = f"{distance_file}.log"
        with FakeEmbedName(
            true_filename=iteration_value,
            file_index=iteration_index,
            lang=self.config.lang,
        ) as (embedding_basename, _embedding_file):
            # TODO just call the python directly
            try:
                subprocess.run(
                    f"python3 -u {self.config.laser_dir}/source/graph_create.py "
                    f"--embed {embedding_basename} "
                    f"--src-lang {self.config.lang} "
                    f"--index {self.config.index_other_lang} "
                    f"--k {self.config.knn} "
                    f"--nprobe {self.config.num_probe} "
                    f"--gpu {self.config.gpu_type} "
                    f"{self.config.dist_opt} "
                    f"{self.index_opt} "
                    f"--graph {distance_file} "
                    f"--verbose "
                    f"--embed-fp16 "
                    f">> {log_file} 2>&1",
                    shell=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"ERROR during distance calculation using embeddings: {iteration_value}, and index: {self.config.index_other_lang}, see {log_file}.",
                    exc_info=e,
                )
                raise e

        return distance_file

    def name(self):
        return f"knn.{self.bi}.{self.config.distance_type.value}.x{len(self.config.lang_embeddings)}.k{self.config.knn}.np{self.config.num_probe}"

    def comment(self):
        return "Calculating distances between embeddings and FAISS index"

    def version(self):
        return "0.3"
