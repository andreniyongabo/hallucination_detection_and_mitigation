# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import subprocess
import tempfile
import typing as tp
import glob
import shutil
from dataclasses import dataclass

from examples.nllb.mining.nllb_lib.nllb_module import NLLBModule, DistributedRequirements
from examples.nllb.mining.nllb_lib.utils import ensure_dir
from omegaconf import MISSING

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
    margin: str = "ratio"
    margin_norm: str = "mean"
    L2: int = 0
    num_probe: int = 128
    gpu_type: str = "fp16-shard"
    retrieval: str = "fastmax"
    mine_threshold: float = 1.06
    # TODO this should be a python import instead
    laser_dir: str = "/private/home/schwenk/projects/mlenc"


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
        out_base_name = os.path.abspath(
            os.path.join(
                self.config.output_dir,
                f"{self.config.src_lang}-{self.config.tgt_lang}.{self.config.index_type}.k{self.config.src_k}-{self.config.tgt_k}.{self.config.margin_norm}.np{self.config.num_probe}.{self.config.retrieval}.{self.config.gpu_type}",
            )
        )
        log_file = f"{out_base_name}.log"
        meta = f"{out_base_name}.align"

        with tempfile.TemporaryDirectory() as tmp_dir:
            for index_file in glob.glob(f"{self.config.dists_idxs_basename}*npy"):
                shutil.copy(index_file, tmp_dir)
            tmpdir_dists_idxs_basename = os.path.join(tmp_dir, os.path.basename(self.config.dists_idxs_basename))
            try:
                subprocess.run(
                    f"python3 -u {self.config.laser_dir}/source/mine_bitexts.py "
                    f"--src-lang {self.config.src_lang} "
                    f"--trg-lang {self.config.tgt_lang} "
                    f"--dists-idxs {tmpdir_dists_idxs_basename} "
                    f"--mode mine "
                    f"--retrieval {self.config.retrieval} "
                    f"--margin {self.config.margin} "
                    f"--margin-norm {self.config.margin_norm} "
                    f"--src-k {self.config.src_k} "
                    f"--trg-k {self.config.tgt_k}  "
                    f"--L2 {self.config.L2} "
                    f"--threshold {self.config.mine_threshold} "
                    f"--nprobe {self.config.num_probe} "
                    f"--meta {meta} "
                    f"--verbose "
                    f"--gpu {self.config.gpu_type} "
                    f">> {log_file} 2>&1",
                    check=True,
                    shell=True,
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"ERROR during mining of bitext indexes; see {log_file}.",
                    exc_info=e,
                )
                raise e

        return meta

    def version(self):
        return "0.1"

    def name(self):
        return f"mineD.{self.config.src_lang}-{self.config.tgt_lang}.{self.config.margin_norm}.{self.config.retrieval}.meta"
