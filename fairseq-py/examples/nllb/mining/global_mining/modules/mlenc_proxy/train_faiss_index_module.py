# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import subprocess
import typing as tp
from dataclasses import dataclass

from omegaconf.omegaconf import MISSING

from examples.nllb.mining.global_mining.mining_utils import (
    get_cached_line_count,
    get_faiss_index_type,
)

from examples.nllb.mining.global_mining.data_utils import DataConfig
from examples.nllb.mining.nllb_lib.nllb_module import NLLBModule, DistributedRequirements
from examples.nllb.mining.nllb_lib.utils import ensure_dir

logger = logging.getLogger("train_faiss_index")


@dataclass
class TrainFAISSIndexConfig:
    lang: str = MISSING
    embedding_file: str = MISSING
    data: DataConfig = MISSING
    output_dir: str = "ts.index.iteration_${iteration}"
    # TODO this should be a python import instead
    laser_dir: str = "/private/home/schwenk/projects/mlenc"
    sample_size: int = 40000000
    num_cpu: int = 40
    embedding_dimensions: int = 1024


class TrainFAISSIndexModule(NLLBModule):
    def __init__(self, config: TrainFAISSIndexConfig = TrainFAISSIndexConfig()):
        super().__init__(config)
        self.lang_output_dir = os.path.join(self.config.output_dir, self.config.lang)
        ensure_dir(self.lang_output_dir)

        nb_sent = get_cached_line_count(
            data_cfg=self.config.data,
            lang=self.config.lang,
        )
        self.index_type = get_faiss_index_type(
            data_cfg=self.config.data,
            lang=self.config.lang,
        )
        self.sample_ratio = self.config.sample_size / nb_sent
        logger.info(
            f"lang={self.config.lang}, "
            f"sents={nb_sent}, "
            f"required={self.config.sample_size}, "
            f"index type={self.index_type}"
        )

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            mem_gb=700,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=self.config.num_cpu,
            timeout_min=1000,
            constraint="ib2",
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        log_file = os.path.abspath(
            os.path.join(
                self.lang_output_dir,
                f"train.{self.index_type}.{self.config.lang}.log",
            )
        )
        index_file = os.path.abspath(
            os.path.join(
                self.lang_output_dir,
                f"{self.config.data.bname}.{self.index_type}.{self.config.lang}",
            )
        )

        # TODO just call the python directly
        subprocess.run(
            f"OMP_NUM_THREADS={self.config.num_cpu} "
            f"python3 -u {self.config.laser_dir}/source/index.py "
            f"--dim {self.config.embedding_dimensions} "
            f"--lang {self.config.lang} "
            f"--index {index_file} "
            f"--index-type {self.index_type} "
            f"--train {self.config.embedding_file} "
            f"--sample {self.sample_ratio:.6f} "
            f"--gpu "
            f" >> {log_file} 2>&1",
            check=True,
            shell=True,
        )
        return index_file + ".train.idx"

    def name(self):
        return f"index-train.{self.config.lang}.iteration_{self.config.data.iteration}"

    def comment(self):
        return (
            f"Creating FAISS index using student encoder v{self.config.data.iteration}"
        )
