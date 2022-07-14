# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import logging
import os
from collections import defaultdict
from typing import Dict, Tuple

from submitit import AutoExecutor

import examples.nllb.modeling.prepare_data.data_types as data_types
from examples.nllb.modeling.prepare_data.cache import cache_step
from examples.nllb.modeling.prepare_data.utils import count_lines_async

logger = logging.getLogger("prepare_data_validation")


async def validate_parallel_path(
    direction: str,
    corpus_name: str,
    parallel_dataset: data_types.ParallelDataset,
    executor: AutoExecutor,
):
    src_path = parallel_dataset.source
    tgt_path = parallel_dataset.target
    is_gzip = parallel_dataset.is_gzip
    assert os.path.exists(src_path), f"nonexistent source path: {src_path}"
    assert os.path.exists(tgt_path), f"nonexistent target path: {tgt_path}"
    num_src_lines, num_tgt_lines = await asyncio.gather(
        count_lines_async(src_path, is_gzip, executor),
        count_lines_async(tgt_path, is_gzip, executor),
    )
    if num_src_lines == num_tgt_lines:
        return direction, corpus_name, num_src_lines
    return direction, corpus_name, None


async def _validate_corpora_items(corpora, executor):
    return await asyncio.gather(
        *[
            validate_parallel_path(
                direction,
                corpus_name,
                parallel_paths["local_paths"],
                executor,
            )
            for direction, corpora_confs in corpora.items()
            for corpus_name, parallel_paths in corpora_confs["values"].items()
        ]
    )


@cache_step("validate_data_config")
async def validate_data_config(
    data_config: data_types.DataConfig,
    output_dir: str,
) -> Tuple[data_types.DataConfig, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """
    Validate the following aspects of data_config:
        correct schema
        same list of directions in train_corpora and valid_corpora
        existence of training files
        same number of lines of source and target file
    Returns validated data_config or raises error for invalid data_config
    """
    logger.info("Validating values in input data_config\n")
    executor = AutoExecutor(
        folder=os.path.join(output_dir, data_config.executor_config.log_folder),
        cluster=data_config.executor_config.cluster,
    )
    executor.update_parameters(
        slurm_partition=data_config.executor_config.slurm_partition,
        timeout_min=1440,  # TODO: we need to tune this
        nodes=1,  # we only need one node for this
        cpus_per_task=8,
    )

    train_directions = set(data_config.train_corpora.keys())
    if data_config.get("valid_corpora"):
        valid_directions = set(data_config.valid_corpora.keys())
        assert (
            train_directions == valid_directions
        ), "inconsistent direction lists between train_corpora and valid_corpora"

    train_src_counts_map = defaultdict(int)
    train_tgt_counts_map = defaultdict(int)
    train_counts_map = defaultdict(int)
    logger.info("Checking line counts in training data")
    train_counts_list = await _validate_corpora_items(
        data_config.train_corpora, executor
    )
    for direction, corpus, num_lines in train_counts_list:
        if num_lines is None:
            assert (
                False
            ), f"{direction}.{corpus} has inconsistent number of lines between source and target"
        source, target = direction.split("-")
        train_src_counts_map[source] += num_lines
        train_tgt_counts_map[target] += num_lines
        train_counts_map[direction] += num_lines

    secondary_train_counts_map = defaultdict(int)
    if data_config.secondary_train_corpora is not None:
        logger.info("Checking line counts in secondary train data")
        secondary_train_counts_list = await _validate_corpora_items(
            data_config.secondary_train_corpora, executor
        )
        for direction, corpus, num_lines in secondary_train_counts_list:
            if num_lines is None:
                assert (
                    False
                ), f"{direction}.{corpus} has inconsistent number of lines between source and target"
            source, target = direction.split("-")
            train_src_counts_map[source] += num_lines
            train_tgt_counts_map[target] += num_lines
            secondary_train_counts_map[direction] += num_lines

    if data_config.valid_corpora is not None:
        logger.info("Checking line counts in validation data")
        await _validate_corpora_items(data_config.valid_corpora, executor)
    if data_config.test_corpora is not None:
        logger.info("Checking line counts in test data")
        await _validate_corpora_items(data_config.test_corpora, executor)

    return (
        train_src_counts_map,
        train_tgt_counts_map,
        dict(train_counts_map),
        dict(secondary_train_counts_map),
        executor,
    )
