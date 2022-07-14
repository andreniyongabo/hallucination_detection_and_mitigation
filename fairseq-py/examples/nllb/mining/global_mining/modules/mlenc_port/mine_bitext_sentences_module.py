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

from omegaconf import MISSING

from examples.nllb.mining.global_mining.data_utils import DataConfig
from examples.nllb.mining.global_mining.mining_utils import extract_shard_id
from examples.nllb.mining.global_mining.modules.mlenc_port.mine_bitext_sentences_utils import (  # noqa
    Alignments,
)
from examples.nllb.mining.nllb_lib.nllb_module import (
    DistributedRequirements,
    NLLBModule,
)
from examples.nllb.mining.nllb_lib.utils import ensure_dir

logger = logging.getLogger("mine_bitext_sentences")


@dataclass
class MineBitextSentencesConfig:
    src_lang: str = MISSING
    tgt_lang: str = MISSING
    alignment_file: str = MISSING  # the mined indexes & distances without npz extension
    data: DataConfig = MISSING
    output_dir: str = "mine.${data.data_version}"
    mine_threshold: float = 1.06
    score_max: float = 1.25
    dedup_bitexts: bool = True
    compress_output: bool = True


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
                f"{self.config.src_lang}-{self.config.tgt_lang}"
                f".TH-{self.config.mine_threshold}",
            )
        )

        # TODO: ideally the .npz should not be hardcoded and we should be using
        # the filename given by the previous module directly

        # loading alignments from the previous step (mine_indexes)
        # also applies some filtering
        alignments = Alignments.from_npz(
            f"{self.config.alignment_file}.npz",
            self.config.mine_threshold,
            self.config.score_max,
        )

        # TODO: use the files returned by the previous step in the pipeline instead
        # loading text and metadata files (if there are any)
        glo = (
            f"{self.config.data.bname}.{self.config.data.shard_type}"
            "{meta}.{lang}.[0-9]*.gz"
        )
        pth = Path(self.config.data.data_shard_dir)

        src_text_files = sorted(
            [str(f) for f in pth.glob(glo.format(meta="", lang=self.config.src_lang))],
            key=extract_shard_id,
        )
        tgt_text_files = sorted(
            [str(f) for f in pth.glob(glo.format(meta="", lang=self.config.tgt_lang))],
            key=extract_shard_id,
        )
        src_meta_files = sorted(
            [
                str(f)
                for f in pth.glob(glo.format(meta=".meta", lang=self.config.src_lang))
            ],
            key=extract_shard_id,
        )
        tgt_meta_files = sorted(
            [
                str(f)
                for f in pth.glob(glo.format(meta=".meta", lang=self.config.tgt_lang))
            ],
            key=extract_shard_id,
        )

        # persisting the mined sentences & corresponding metadata to disk
        bitexts_tsv = f"{out_base_name}.bitext.tsv"
        bimeta_tsv = f"{out_base_name}.bimeta.tsv"
        if self.config.compress_output:
            bitexts_tsv += ".gz"
            bimeta_tsv += ".gz"
        alignments.save_texts(
            src_text_files,
            tgt_text_files,
            src_meta_files,
            tgt_meta_files,
            bitexts_tsv,
            bimeta_tsv,
            self.config.dedup_bitexts,
            logger,
        )

        return bitexts_tsv, bimeta_tsv

    def version(self):
        return "0.3"

    def name(self):
        return (
            f"mineD.{self.config.src_lang}-{self.config.tgt_lang}"
            f".TH-{self.config.mine_threshold}.sents"
        )
