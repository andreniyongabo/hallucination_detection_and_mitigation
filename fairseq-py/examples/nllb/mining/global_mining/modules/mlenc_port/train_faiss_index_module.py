# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import typing as tp
from dataclasses import dataclass

import faiss
import numpy as np
from omegaconf.omegaconf import MISSING

from examples.nllb.mining.global_mining.data_utils import DataConfig
from examples.nllb.mining.global_mining.mining_utils import (
    get_cached_line_count,
    get_faiss_index_type,
)
from examples.nllb.mining.global_mining.modules.indexing.train_index import train_index
from examples.nllb.mining.nllb_lib.nllb_module import (
    DistributedRequirements,
    NLLBModule,
)
from examples.nllb.mining.nllb_lib.utils import ensure_dir

logger = logging.getLogger("train_faiss_index_port")


@dataclass
class TrainFAISSIndexConfig:
    lang: str = MISSING
    embedding_file: str = MISSING
    data: DataConfig = MISSING
    output_dir: str = "ts.index.iteration_${iteration}"
    num_cpu: int = 40
    embedding_dimensions: int = 1024
    use_gpu: bool = True
    fp16_storage: bool = True


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

        logger.info(
            f"lang={self.config.lang}, "
            f"sents={nb_sent}, "
            f"index type={self.index_type}"
        )

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            # mem_gb=700,
            tasks_per_node=1,
            gpus_per_node=1 if self.config.use_gpu else 0,
            cpus_per_task=self.config.num_cpu,
            timeout_min=1000,
            constraint="ib2",
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        index_output_file = os.path.abspath(
            os.path.join(
                self.lang_output_dir,
                f"{self.config.data.bname}.{self.index_type}.{self.config.lang}.train.idx",
            )
        )

        returned_index = train_index(
            self.config.embedding_file,
            self.index_type,
            self.config.embedding_dimensions,
            self.config.use_gpu,
            np.float16 if self.config.fp16_storage else np.float32,
        )
        if self.config.use_gpu:
            returned_index = faiss.index_gpu_to_cpu(returned_index)

        faiss.write_index(returned_index, str(index_output_file))

        index_output_file_path = str(index_output_file)
        logger.info(
            f"Trained index of type: {self.index_type} and lang: {self.config.lang}, can be found in output file: {index_output_file_path}"
        )

        return index_output_file

    def name(self):
        return f"index-train-port.{self.config.lang}.iteration_{self.config.data.iteration}"

    def comment(self):
        return f"Creating FAISS index (ported) using student encoder v{self.config.data.iteration}"

    def version(cls):
        return "0.1"
