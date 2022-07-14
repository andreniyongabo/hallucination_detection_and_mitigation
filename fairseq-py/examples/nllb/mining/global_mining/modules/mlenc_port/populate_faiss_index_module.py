# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import shutil
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import submitit
from omegaconf.omegaconf import MISSING

from examples.nllb.mining.global_mining.embedding_utils import Embedding
from examples.nllb.mining.global_mining.modules.mlenc_port.populate_faiss_index_utils import (
    CheckpointSummary,
    add_embedding_to_index,
)
from examples.nllb.mining.nllb_lib.nllb_module import (
    DistributedRequirements,
    NLLBModule,
)
from examples.nllb.mining.nllb_lib.utils import ensure_dir

logger = logging.getLogger("populate_faiss_index_port")


@dataclass
class PopulateFAISSIndexConfig:
    lang: str = MISSING
    output_dir: str = MISSING
    index: str = MISSING
    index_type: str = MISSING
    embedding_files: tp.List[str] = MISSING

    num_cpu: int = 40
    embedding_dimensions: int = 1024


class PopulateFAISSIndexModule(NLLBModule):
    def __init__(
        self,
        config: PopulateFAISSIndexConfig = PopulateFAISSIndexConfig(),
        checkpoint_summary: tp.Optional[CheckpointSummary] = None,
    ):
        super().__init__(config)
        self.lang_output_dir = (
            Path(self.config.output_dir) / self.config.lang
        ).resolve()

        fp16 = getattr(self.config, "fp16_embeddings", False)
        self.dtype = np.float16 if fp16 else np.float32
        logger.info(f"embedding dtype: {self.dtype}")
        ensure_dir(self.lang_output_dir)
        self.checkpoint_summary = (
            CheckpointSummary(
                partial_idx=None,
                partial_idx_file=None,
                idx_size_before_populating_embedding=None,
                is_partial_file_valid=False,
                is_partial_idx_valid=False,
            )
            if checkpoint_summary is None
            else checkpoint_summary
        )

        # Calculate original size of index (i.e. size of index before we start populating the embedding onto it)
        self.checkpoint_summary.idx_size_before_populating_embedding = (
            self.checkpoint_summary.idx_size_before_populating_embedding
            or faiss.read_index(str(self.config.index)).ntotal
        )

    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            # mem_gb=500,
            tasks_per_node=1,
            gpus_per_node=1 if self.config.use_gpu else 0,
            cpus_per_task=self.config.num_cpu,
            timeout_min=48 * 60,
        )

    def array(self):
        return self.config.embedding_files

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ):
        lang_output_dir = Path(self.lang_output_dir)
        file_name = f"populate_index.{self.config.index_type}.{self.config.lang}.{iteration_index:03d}.data.idx"
        populated_index = lang_output_dir / file_name

        if Path(iteration_value).stat().st_size == 0:
            logger.warning(
                f"Within populate_faiss_index (port) module run: Embedding shard is empty, so None is returned. Embedding Shard path is: {iteration_value}"
            )
            return None

        # If both the partial file and index are None OR
        #    both the partial file and index are INVALID
        if (
            self.checkpoint_summary.partial_idx_file is None
            and self.checkpoint_summary.partial_idx is None
        ) or (
            (not self.checkpoint_summary.is_partial_file_valid)
            and (not self.checkpoint_summary.is_partial_idx_valid)
        ):
            # this means we're entering the job for the very first time (haven't even started the first checkpoint)
            self.checkpoint_summary.partial_idx_file = populated_index.with_suffix(
                ".checkpoint"
            )
            # since we're going to start the 1st checkpoint, copy the trained index to the partial_idx_file. This will be populated with a single shard
            shutil.copyfile(self.config.index, self.checkpoint_summary.partial_idx_file)
            self.checkpoint_summary.is_partial_file_valid = True

        # By now, at least one of is_partial_idx_valid or is_partial_file_valid must be true at any one time
        try:
            # If the partial_file is valid, the partial_idx should have the same contents as the file
            if self.checkpoint_summary.is_partial_file_valid:
                self.checkpoint_summary.partial_idx = faiss.read_index(
                    str(self.checkpoint_summary.partial_idx_file)
                )
                # Note that partial_idx is overwritten regardless of whether it is valid from before already or not, because, there's an edge case where:
                # both partial_idx and partial_idx_file are valid but partial_idx is one checkpoint ahead of partial_idx_file. So we calibrate both to be on the same checkpoint.
                # This occurs when the job is interrupted right after adding the chunk to partial index but before writing it to the file.
                self.checkpoint_summary.is_partial_idx_valid = True
            else:  # If the partial_idx is valid and the partial_idx_file isn't we write the partial_idx to the file.
                faiss.write_index(
                    self.checkpoint_summary.partial_idx,
                    str(self.checkpoint_summary.partial_idx_file),
                )
                self.checkpoint_summary.is_partial_file_valid = True

            # Since now both the partial idx and file are valid, we can populate the embedding onto the index:
            add_embedding_to_index(
                self.checkpoint_summary,
                iteration_value,
                self.config.embedding_dimensions,
                dtype=self.dtype,
                gpu=self.config.use_gpu,
            )

        except Exception as e:
            logger.exception(
                f"Error in index population with embeddings: {iteration_value}, & index: {self.config.index}",
            )
            raise e

        # The process is complete: copying completed (checkpointed) index onto return value file
        shutil.copyfile(
            str(self.checkpoint_summary.partial_idx_file),
            str(populated_index),
        )
        self.checkpoint_summary.is_partial_file_valid = False
        self.checkpoint_summary.partial_idx_file.unlink()
        logger.info(f"Populated index, can be found in output file: {populated_index}")
        return populated_index

    def name(self):
        return f"populate_index.{self.config.index_type}.{self.config.lang}"

    def comment(self):
        return f"Populating FAISS index {self.config.index} for {self.config.lang}"

    def checkpoint(
        self, *args: tp.Any, **kwargs: tp.Any
    ) -> submitit.helpers.DelayedSubmission:
        return submitit.helpers.DelayedSubmission(
            PopulateFAISSIndexModule(
                config=self.config,
                checkpoint_summary=self.checkpoint_summary,
            ),
            *args,
            **kwargs,
        )

    def validate(
        self,
        output: tp.Any,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> bool:
        if Path(iteration_value).stat().st_size == 0:
            logger.info(
                "Within populate_faiss_index (port) module validate: Embedding shard is empty"
            )
            assert output is None, "embedding is empty, shouldn't populate anything"
            return True
        assert output.exists(), f"index file {output} is missing"
        idx = faiss.read_index(str(output))
        nbex = len(
            Embedding(
                iteration_value, self.config.embedding_dimensions, dtype=self.dtype
            )
        )
        assert (
            idx.ntotal == nbex
        ), f"expected {nbex} sentences, only found {idx.ntotal} in index {output} populated from {iteration_value}."

        return True
