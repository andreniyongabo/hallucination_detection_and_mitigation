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
from examples.nllb.mining.global_mining.embedding_utils import Embedding
from examples.nllb.mining.global_mining.mining_utils import (
    extract_shard_id,
    get_cached_line_count,
    tokenization_type,
    iso3_to_iso2,
)
from examples.nllb.mining.nllb_lib.nllb_module import NLLBModule, DistributedRequirements
from examples.nllb.mining.nllb_lib.utils import ensure_dir
from omegaconf import MISSING

logger = logging.getLogger("embed_text")


@dataclass
class EmbedTextConfig:
    # core conf
    lang: str = MISSING
    shards: tp.List[str] = MISSING
    # useful confs
    data: DataConfig = MISSING
    output_dir: str = "ts.embed.iteration_${data.iteration}"
    encodings_name: str = "encodings"
    # TODO this should be a python import instead
    laser_dir: str = "/private/home/schwenk/projects/mlenc"
    model_dir: str = "${laser_dir}/models"
    encoder_model: str = "/checkpoint/kevinheffernan/data/laser2/spm.cc100xx50/african15_added_mini_mine_v1.opus2en.2000000.th.1.07.v1.a0.4.lr0005.upf1.t10000.n16/checkpoint20.enc.pt"
    embedding_dimensions: int = 1024
    # TODO these should come from a previous step
    spm_model: str = "${model_dir}/spm100v1.xx.50k.model"
    token_lang_file: str = (
        "/private/home/kevinheffernan/mining/scripts/map_token_lang.V32m.sed"
    )


class EmbedTextModule(NLLBModule):
    def __init__(self, config: EmbedTextConfig = EmbedTextConfig()):
        super().__init__(config)

        N = len(self.config.shards)
        assert (
            N > 0
        ), f"no {config.lang} texts for embedding in {config.data.data_shard_dir}"
        logger.info(f"Number of shards: {N}")
        lang_iso2 = iso3_to_iso2(self.config.lang)
        self.tokenizer = tokenization_type(lang_iso2, self.config.token_lang_file)
        logger.info(f"Embed {self.config.lang} ({self.tokenizer}), {N} files")

        lang_output = os.path.abspath(
            os.path.join(
                config.output_dir,
                config.lang,
            )
        )
        ensure_dir(lang_output)
        self.output_prefix = os.path.join(
            lang_output,
            f"{config.data.bname}.{self.config.encodings_name}",
        )

    def array(self):
        return self.config.shards

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=1,
            cpus_per_task=4,
            timeout_min=120,
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        input_file = iteration_value
        shard_idx = extract_shard_id(input_file)
        output_file = f"{self.output_prefix}.{shard_idx:03d}.{self.config.lang}"

        # TODO move this to pure python + hydra config
        log_file = f"{output_file}.log"
        try:
            subprocess.run(
                f"/bin/bash -o pipefail -c '"  # we need this to be sure to catch zcat failures
                f"zcat {input_file} "
                f"| python3 -u {self.config.laser_dir}/source/embed.py "
                f"--encoder {self.config.encoder_model} "
                f"--spm-model {self.config.spm_model} "
                f"--token-lang {self.tokenizer} "
                f"--output {output_file} "
                f"--verbose "
                f"' >> {log_file} 2>&1",
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            if os.path.isfile(output_file):
                os.remove(output_file)
            logger.error(
                f"ERROR during encoding of {input_file}, see {log_file}; removed encoding.",
                exc_info=e,
            )
            raise e

        return output_file

    def name(self):
        return f"encode.{self.config.lang}.{len(self.config.shards)}"

    def comment(self):
        return "Embedding texts using LASER v2"

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        assert os.path.exists(output), f"embedding file {output} is missing"
        nbex = len(Embedding(output, self.config.embedding_dimensions))
        shard_idx = extract_shard_id(iteration_value)
        expected_line_count = get_cached_line_count(
            self.config.lang,
            self.config.data,
            shard=shard_idx,
        )
        assert expected_line_count == nbex, (
            f"expected {expected_line_count} sentences from {iteration_value},"
            f" only found {nbex} in embedding {output}."
        )
        return True
