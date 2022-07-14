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

from examples.nllb.mining.global_mining.data_utils import DataConfig
from examples.nllb.mining.nllb_lib.nllb_module import NLLBModule, DistributedRequirements
from examples.nllb.mining.nllb_lib.utils import ensure_dir
from omegaconf import MISSING

logger = logging.getLogger("mine_bitext_sentences")


@dataclass
class MineBitextSentencesConfig:
    src_lang: str = MISSING
    tgt_lang: str = MISSING
    meta_file: str = MISSING  # the mined indexes, without npz extension
    data: DataConfig = MISSING
    output_dir: str = "mine.${data.data_version}"
    margin_norm: str = "mean"
    mine_threshold: float = 1.04
    score_max: float = 1.25
    # TODO this should be a python import instead
    laser_dir: str = "/private/home/schwenk/projects/mlenc"
    # find text boundaries
    targets_first: int = 0
    targets_last: int = 999
    source_last: int = 999


class MineBitextSentencesModule(NLLBModule):
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
                f"{self.config.src_lang}-{self.config.tgt_lang}.TH-{self.config.mine_threshold}",
            )
        )
        log_file = f"{out_base_name}.log"
        bitext = f"{out_base_name}.bitext.tsv"
        bimeta = f"{out_base_name}.bimeta.tsv"

        try:
            subprocess.run(
                f"python3 -u {self.config.laser_dir}/source/npz2text+meta.py "
                "--verbose "
                f"--align {self.config.meta_file}.npz "
                f"--bitext-tsv {bitext}.gz "
                f"--bimeta-tsv {bimeta}.gz "
                f"--unify-bitext "
                f"--meta {self.config.data.data_shard_dir}/{self.config.data.bname}.meta "
                f"--texts {self.config.data.data_shard_dir}/{self.config.data.bname}.{self.config.data.shard_type} "
                f"--lang-src {self.config.src_lang} "
                f"--lang-trg {self.config.tgt_lang}  "
                f"--score-min {self.config.mine_threshold} "
                f"--score-max {self.config.score_max} "
                f"--source-last {self.config.source_last} "
                f"--targets-first {self.config.targets_first} "
                f"--targets-last {self.config.targets_last} "
                f"> {log_file} 2>&1",
                check=True,
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(
                f"ERROR during mining of bitext sentences; see {log_file}.",
                exc_info=e,
            )
            raise e
        return bitext + ".gz"

    def version(self):
        return "0.3"

    def name(self):
        return f"mineD.{self.config.src_lang}-{self.config.tgt_lang}.TH-{self.config.mine_threshold}.sents"
