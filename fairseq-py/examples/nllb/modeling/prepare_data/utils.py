# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import hashlib
import logging
import os
from functools import lru_cache
from typing import List, Set, Tuple

import numpy as np
from hydra import compose
from omegaconf import OmegaConf
from submitit import AutoExecutor

import examples.nllb.modeling.prepare_data.data_types as data_types
from examples.nllb.modeling.prepare_data.cache import cache_step_sync
from examples.nllb.modeling.utils import execute_in_shell

logger = logging.getLogger("prepare_data_utils")


@lru_cache
def split_direction(direction: str) -> Tuple[str, str]:
    source, target = direction.split("-", maxsplit=1)
    return (source, target)


def setup_config(config_name: str, overrides: List[str]) -> data_types.DataConfig:
    data_config = compose(config_name=config_name, overrides=overrides)
    schema = OmegaConf.structured(data_types.DataConfig)
    return OmegaConf.merge(schema, data_config)


def count_lines(fi: str, is_gzip: bool) -> int:
    open_func = gzip.open if is_gzip else open
    with open_func(fi, "rb") as f:
        count = sum(1 for line in f)
    return count


async def count_lines_async(
    file_name: str, is_gzip: bool, executor: AutoExecutor
) -> int:
    return count_lines(file_name, is_gzip)


def hash_sentence_pair(source_line: str, target_line: str) -> str:
    concated = f"{source_line} {target_line}"
    hashed = hashlib.md5(concated.encode()).hexdigest()
    return hashed


def hash_parallel_data(parallel_data: data_types.CorporaMap) -> Set[str]:
    """
    Get the set of hashed lines in valid/test data
    """
    seen = set()
    with open(parallel_data.source) as s_f, open(parallel_data.target) as t_f:
        for source_line, target_line in zip(s_f, t_f):
            hashed = hash_sentence_pair(source_line, target_line)
            seen.add(hashed)
    return seen


@cache_step_sync("dedup_sharding")
def dedup_sharding(
    direction: str,
    train_parallel: data_types.ParallelDataset,
    seen: Set[str],
    num_shards: int,
    binarization_config: data_types.BinarizationConfig,
    sharding_output_dir: str,
    output_dir: str,
    custom_step_name: str,
) -> List[data_types.ParallelDataset]:
    """
    Deduplicate training data appeared in valid & test data;
    execute random sharding to training data for one direction
    Returns list of sharded files prefixes
    """
    logger.info("dedup and sharding")
    logger.info(f"train_parallel: {train_parallel}\n")
    if train_parallel is None:
        logger.info(f"No data provided {custom_step_name}")
        return []

    if not os.path.exists(sharding_output_dir):
        os.makedirs(sharding_output_dir, exist_ok=True)
    src, tgt = direction.split("-")

    # sharding prepare
    source_shards = []
    target_shards = []
    for i in range(num_shards):
        execute_in_shell(f"mkdir -p {sharding_output_dir}/shard{i:03d}")
        source_shards.append(
            open(
                f"{sharding_output_dir}/shard{i:03d}/train.shard{i:03d}.{src}-{tgt}.{src}",
                "wb",
            )
        )
        target_shards.append(
            open(
                f"{sharding_output_dir}/shard{i:03d}/train.shard{i:03d}.{src}-{tgt}.{tgt}",
                "wb",
            )
        )
    np.random.seed(binarization_config.random_seed)
    num_lines = count_lines(
        train_parallel.source, train_parallel.is_gzip
    )  # using non-deduped data for random sharding
    idx_to_shards = np.random.randint(0, num_shards, num_lines)

    idx = 0
    with open(train_parallel.source, "rb") as src_f, open(
        train_parallel.target, "rb"
    ) as tgt_f:
        for source_line, target_line in zip(src_f, tgt_f):
            # dedup
            hashed = hash_sentence_pair(source_line, target_line)
            if (
                hashed in seen
                or len(source_line.strip()) < 1
                or len(target_line.strip()) < 1
            ):
                pass
            else:
                seen.add(hashed)
                # sharding
                shard_id = idx_to_shards[idx]
                source_shards[shard_id].write(source_line)
                target_shards[shard_id].write(target_line)
            idx += 1

    for fi in source_shards + target_shards:
        fi.flush()
        fi.close()

    return [
        data_types.ParallelDataset(
            source=f"{sharding_output_dir}/shard{i:03d}/train.shard{i:03d}.{src}-{tgt}.{src}",
            target=f"{sharding_output_dir}/shard{i:03d}/train.shard{i:03d}.{src}-{tgt}.{tgt}",
        )
        for i in range(num_shards)
    ]


async def async_noop():
    """
    useful to return none in awaitable ternaries
    """
    return None
