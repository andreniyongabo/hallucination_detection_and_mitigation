# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import shutil
import subprocess
import typing as tp
from dataclasses import dataclass

import faiss
from examples.nllb.mining.global_mining.embedding_utils import Embedding
from examples.nllb.mining.global_mining.mining_utils import FakeEmbedName
from examples.nllb.mining.nllb_lib.nllb_module import NLLBModule, DistributedRequirements
from examples.nllb.mining.nllb_lib.utils import ensure_dir
from omegaconf.omegaconf import MISSING

logger = logging.getLogger("populate_faiss_index")


@dataclass
class PopulateFAISSIndexConfig:
    lang: str = MISSING
    output_dir: str = MISSING
    index: str = MISSING
    index_type: str = MISSING
    embedding_files: tp.List[str] = MISSING

    # TODO this should be a python import instead
    laser_dir: str = "/private/home/schwenk/projects/mlenc"

    sample_size: int = 40000000
    num_cpu: int = 40
    embedding_dimensions: int = 1024



class PopulateFAISSIndexModule(NLLBModule):
    def __init__(
        self,
        config: PopulateFAISSIndexConfig = PopulateFAISSIndexConfig(),
    ):
        super().__init__(config)
        self.lang_output_dir = os.path.abspath(
            os.path.join(self.config.output_dir, self.config.lang)
        )
        ensure_dir(self.lang_output_dir)

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            mem_gb=500,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=self.config.num_cpu,
            timeout_min=1000,
            constraint="ib2",
        )

    def array(self):
        return self.config.embedding_files

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        log_file = os.path.join(
            self.lang_output_dir,
            f"populate_index.{self.config.index_type}.{self.config.lang}.{iteration_index:03d}.log",
        )
        populated_index = os.path.join(
            self.lang_output_dir,
            f"populate_index.{self.config.index_type}.{self.config.lang}.{iteration_index:03d}",
        )
        if os.path.getsize(iteration_value) == 0:
            return None
        # copy the trained index to a separate file to populate it with a single shard
        shutil.copyfile(self.config.index, populated_index + ".train.idx")
        with FakeEmbedName(
            true_filename=iteration_value,
            file_index=iteration_index,
            lang=self.config.lang,
        ) as (embedding_basename, _embedding_file):
            # TODO just call the python directly
            try:
                subprocess.run(
                    f"python3 -u {self.config.laser_dir}/source/index.py "
                    f"--embed {embedding_basename} "
                    f"--dim {self.config.embedding_dimensions} "
                    f"--lang {self.config.lang} "
                    f"--index {populated_index} "
                    f"--index-type {self.config.index_type} "
                    f"--train UNUSED "
                    f"--sample 0 "
                    f">> {log_file} 2>&1",
                    shell=True,
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"ERROR during index population using embeddings: {iteration_value}, and index: {self.config.index}, see {log_file}.",
                    exc_info=e,
                )
                raise e
        # the train.idx is empty now, just remove it
        os.remove(populated_index + ".train.idx")
        return os.path.abspath(populated_index + ".data.idx")

    def name(self):
        return f"populate_index.{self.config.index_type}.{self.config.lang}"

    def comment(self):
        return f"Populating FAISS index {self.config.index} for {self.config.lang}"

    def version(self):
        return "0.14"

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        if os.path.getsize(iteration_value) == 0:
            assert output is None, "embedding is empty, shouldn't populate anything"
            return True
        assert os.path.exists(output), f"index file {output} is missing"
        idx = faiss.read_index(output)
        nbex = len(Embedding(iteration_value, self.config.embedding_dimensions))
        assert (
            idx.ntotal == nbex
        ), f"expected {nbex} sentences in index, only found {idx.ntotal}."
        return True
