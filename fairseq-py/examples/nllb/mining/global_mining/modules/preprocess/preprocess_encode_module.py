# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import typing as tp
from pathlib import Path

import hydra
import logging

from examples.nllb.mining.nllb_lib.nllb_module import LocalOnlyRequirements, NLLBModule

logger = logging.getLogger("preprocess_encode")

class PreprocessEncodeModule(NLLBModule):
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.launcher = hydra.utils.instantiate(config.launcher)
        self.shards = self.config.shards
        if isinstance(self.shards, str):
            # it's a glob instead of a list of files
            self.shards = list(glob.glob(self.shards))

        # Create output dir so that child module can use it
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

    def requirements(self):
        #  this just stiches other modules together
        #  so we really just want it to run inline with the local
        #  coordinator script
        return LocalOnlyRequirements()

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        preprocess_module = NLLBModule.build(
            self.config.preprocess,
            lang=self.config.lang,
            shards=self.shards,
        )
        preprocessed_files = await self.launcher.schedule(preprocess_module)

        embed_module = NLLBModule.build(
            self.config.embed_text,
            outfile_prefix=f"{self.config.embed_text.config.outfile_prefix}",
            shards=[str(f) for f in preprocessed_files],
            # maybe we have lang overrides for encoder/spm
            encoder_model=getattr(
                self.config.embed_text.config.line_processor,
                'encoder_model',
                None,
            ),
            spm_model=getattr(
                self.config.embed_text.config.line_processor,
                'spm_model',
                None,
            ),
            spm_vocab=getattr(
                self.config.embed_text.config.line_processor,
                'spm_vocab',
                None,
            ),
        )
        return await self.launcher.schedule(embed_module)

    def version(cls):
        return "0.2"

    def name(self):
        return f"moses_and_encode.{self.config.lang}.{len(self.shards)}"
