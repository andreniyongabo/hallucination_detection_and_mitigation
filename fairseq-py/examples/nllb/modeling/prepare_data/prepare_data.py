# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import asyncio
import logging
import os
import shutil
import time
from typing import Dict, Set

from hydra import initialize
from joblib import Parallel, delayed
from omegaconf import OmegaConf

import examples.nllb.modeling.prepare_data.data_types as data_types
from examples.nllb.modeling.prepare_data.cache import cache_step, cache_step_sync
from examples.nllb.modeling.prepare_data.encode_and_binarize import encode_and_binarize
from examples.nllb.modeling.prepare_data.prepare_vocab import get_vocab
from examples.nllb.modeling.prepare_data.retrieve_data import retrieve_data
from examples.nllb.modeling.prepare_data.sharding import (
    get_all_num_shards,
    write_to_all_shards,
)
from examples.nllb.modeling.prepare_data.utils import (
    async_noop,
    dedup_sharding,
    hash_parallel_data,
    setup_config,
)
from examples.nllb.modeling.prepare_data.validation import validate_data_config
from examples.nllb.modeling.utils import awaitable_job, execute_in_shell

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("prepare_data")


@cache_step_sync("prepare_valid_test_direction")
def prepare_valid_test_direction(
    direction: str,
    valid_data: Dict[str, data_types.CorporaMap],
    test_data: Dict[str, data_types.CorporaMap],
    sampled_train: Dict[str, data_types.CorporaMap],
    src_vocab: data_types.BuiltVocab,
    tgt_vocab: data_types.BuiltVocab,
    all_num_shards: Dict[str, int],
    data_config: data_types.DataConfig,
    temp_dir: str,
    output_dir: str,
    custom_step_name: str,
) -> None:

    parent_dir = os.path.join(temp_dir, "temp_binarized")
    valid_test_dir = os.path.join(temp_dir, "encoded_valid_test")

    if valid_data is not None and direction in valid_data:
        binarized_valid = encode_and_binarize(
            direction=direction,
            parallel_data=valid_data[direction],
            tag="valid",
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            binarize_workers=data_config.binarization_config.binarize_workers,
            output_dir=output_dir,
            encoded_outdir=valid_test_dir,
            binarized_outdir=parent_dir,
            shard_id=0,
            custom_step_name=f"encode_and_binarize.valid.{direction}",
        )
    else:
        binarized_valid = None

    if sampled_train is not None and direction in sampled_train:
        binarized_sampled_train = encode_and_binarize(
            direction=direction,
            parallel_data=sampled_train[direction],
            tag="sampled_train",
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            binarize_workers=data_config.binarization_config.binarize_workers,
            output_dir=output_dir,
            encoded_outdir=valid_test_dir,
            binarized_outdir=f"{temp_dir}/temp_binarized/sampled_train",
            shard_id=0,
            custom_step_name=f"encode_and_binarize.sampled_train.{direction}",
        )
    else:
        binarized_sampled_train = None

    if test_data is not None and direction in test_data:
        binarized_test = encode_and_binarize(
            direction=direction,
            parallel_data=test_data[direction],
            tag="test",
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            binarize_workers=data_config.binarization_config.binarize_workers,
            output_dir=output_dir,
            encoded_outdir=valid_test_dir,
            binarized_outdir=parent_dir,
            shard_id=0,
            custom_step_name=f"encode_and_binarize.test.{direction}",
        )
    else:
        binarized_test = None

    source, target = direction.split("-")
    if binarized_sampled_train:
        # Move secondary data into the main folder
        secondary_dir = os.path.join(parent_dir, "sampled_train")
        for ext in ["bin", "idx"]:
            for lang in [source, target]:
                execute_in_shell(
                    f"mv {secondary_dir}/valid.{source}-{target}.{lang}.{ext} {secondary_dir}/sampled_train.{source}-{target}.{lang}.{ext}"
                )

    if binarized_valid is not None:
        write_to_all_shards(
            binarized_valid,
            all_num_shards[direction],
            f"{output_dir}/data_bin",
        )
    if binarized_sampled_train is not None:
        write_to_all_shards(
            binarized_sampled_train,
            all_num_shards[direction],
            f"{output_dir}/data_bin",
        )

    if binarized_test is not None:
        write_to_all_shards(
            binarized_test,
            all_num_shards[direction],
            f"{output_dir}/data_bin",
        )


@cache_step_sync("prepare_train_directory")
def prepare_train_direction(
    direction: str,
    train_data: Dict[str, data_types.CorporaMap],
    secondary_train_data: Dict[str, data_types.CorporaMap],
    src_vocab: data_types.BuiltVocab,
    tgt_vocab: data_types.BuiltVocab,
    all_num_shards: Dict[str, int],
    all_num_secondary_shards: Dict[str, int],
    data_config: data_types.DataConfig,
    temp_dir: str,
    output_dir: str,
    seen: Set[str],
    custom_step_name: str,
) -> None:
    if train_data and direction in train_data:
        train_shards = dedup_sharding(
            output_dir=output_dir,
            custom_step_name=f"dedup_sharding.{direction}",
            direction=direction,
            train_parallel=train_data[direction],
            seen=seen,
            num_shards=all_num_shards[direction],
            binarization_config=data_config.binarization_config,
            sharding_output_dir=f"{temp_dir}/deduped_train",
        )
    else:
        train_shards = []
    if secondary_train_data and direction in secondary_train_data:
        secondary_train_shards = dedup_sharding(
            output_dir=output_dir,
            custom_step_name=f"dedup_sharding_secondary.{direction}",
            direction=direction,
            train_parallel=secondary_train_data.get(direction, None),
            seen=seen,
            num_shards=all_num_secondary_shards[direction],
            binarization_config=data_config.binarization_config,
            sharding_output_dir=f"{temp_dir}/deduped_secondary_train",
        )
    else:
        secondary_train_shards = []

    Parallel(n_jobs=8, verbose=100)(
        delayed(encode_and_binarize)(
            direction=direction,
            parallel_data=shard,
            tag="train",
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            binarize_workers=data_config.binarization_config.binarize_workers,
            output_dir=output_dir,
            encoded_outdir=f"{temp_dir}/encoded_train/shard{i:03d}",
            binarized_outdir=f"{output_dir}/data_bin/shard{i:03d}",
            shard_id=i,
            custom_step_name=f"encode_and_binarize.train.{direction}.{i}",
            encoded_filtered_outdir=f"{temp_dir}/encoded_filtered_train/shard{i:03d}",
        )
        for i, shard in enumerate(train_shards)
    )
    if len(secondary_train_shards) > 0:
        Parallel(n_jobs=8, verbose=100)(
            delayed(encode_and_binarize)(
                direction=direction,
                parallel_data=shard,
                tag="train_secondary",
                src_vocab=src_vocab,
                tgt_vocab=tgt_vocab,
                binarize_workers=data_config.binarization_config.binarize_workers,
                output_dir=output_dir,
                encoded_outdir=f"{temp_dir}/encoded_train/shard{i:03d}/secondary",
                binarized_outdir=f"{output_dir}/data_bin/shard{i:03d}/secondary",
                shard_id=i,
                custom_step_name=f"encode_and_binarize.train_secondary.{direction}.{i}",
                encoded_filtered_outdir=f"{temp_dir}/encoded_filtered_train/shard{i:03d}/secondary",
            )
            for i, shard in enumerate(secondary_train_shards)
        )

    source, target = direction.split("-")
    if secondary_train_shards:
        # Move secondary data into the main folder
        for i in range(len(secondary_train_shards)):
            parent_dir = f"{output_dir}/data_bin/shard{i:03d}"
            secondary_dir = f"{parent_dir}/secondary"
            for ext in ["bin", "idx"]:
                for lang in [source, target]:
                    execute_in_shell(
                        f"mv {secondary_dir}/train.{source}-{target}.{lang}.{ext} {parent_dir}/train1.{source}-{target}.{lang}.{ext}"
                    )

        num_main_shards = len(train_shards)
        num_secondary_shards = len(secondary_train_shards)
        if num_main_shards > 0 and num_main_shards < num_secondary_shards:
            for i in range(num_main_shards, num_secondary_shards):
                for ext in ["bin", "idx"]:
                    for lang in [source, target]:
                        source_shard = i % num_main_shards
                        execute_in_shell(
                            f"cp {output_dir}/data_bin/shard{source_shard:03d}/train.{source}-{target}.{lang}.{ext} {output_dir}/data_bin/shard{i:03d}/train.{source}-{target}.{lang}.{ext}"
                        )


def prepare_data_direction(
    direction: str,
    train_data: Dict[str, data_types.CorporaMap],
    secondary_train_data: Dict[str, data_types.CorporaMap],
    valid_data: Dict[str, data_types.CorporaMap],
    sampled_train: Dict[str, data_types.CorporaMap],
    test_data: Dict[str, data_types.CorporaMap],
    src_vocab: data_types.BuiltVocab,
    tgt_vocab: data_types.BuiltVocab,
    all_num_shards: Dict[str, int],
    all_num_secondary_shards: Dict[str, int],
    data_config: data_types.DataConfig,
    temp_dir: str,
    output_dir: str,
) -> None:
    logger.info(f"Preparing data for {direction}")
    seen_valid = set()
    seen_test = set()
    reverse_direction = "-".join(direction.split("-")[::-1])
    if valid_data and test_data:
        if direction in valid_data and direction in test_data:
            seen_valid = hash_parallel_data(valid_data[direction])
            seen_test = hash_parallel_data(test_data[direction])
        elif reverse_direction in valid_data and reverse_direction in test_data:
            seen_valid = hash_parallel_data(valid_data[reverse_direction])
            seen_test = hash_parallel_data(test_data[reverse_direction])
    seen = seen_valid.union(seen_test)
    max_num_shards = {
        direction: max(all_num_shards[direction], all_num_secondary_shards[direction])
        for direction in all_num_shards
    }
    prepare_valid_test_direction(
        direction,
        valid_data,
        test_data,
        sampled_train,
        src_vocab,
        tgt_vocab,
        max_num_shards,
        data_config,
        temp_dir,
        output_dir=output_dir,
        custom_step_name=f"prepare_valid_test_direction.{direction}",
    )
    prepare_train_direction(
        direction,
        train_data,
        secondary_train_data,
        src_vocab,
        tgt_vocab,
        all_num_shards,
        all_num_secondary_shards,
        data_config,
        temp_dir,
        output_dir=output_dir,
        seen=seen,
        custom_step_name=f"prepare_train_directory.{direction}",
    )

    return "Done prepare data"


async def main(data_config: data_types.DataConfig, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "progress"), exist_ok=True)
    temp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(temp_dir, exist_ok=True)

    (
        train_src_counts_map,
        train_tgt_counts_map,
        train_counts_map,
        secondary_train_counts_map,
        executor,
    ) = await validate_data_config(data_config, output_dir=output_dir)

    logger.info(f"Running prepare data with:\n{OmegaConf.to_yaml(data_config)}")
    print(
        OmegaConf.to_yaml(data_config),
        file=open(os.path.join(output_dir, "config.yaml"), "w"),
    )

    # wait for all retrieval
    retrieve_outdir = f"{output_dir}/retrieved_data"
    if not os.path.exists(retrieve_outdir):
        os.makedirs(retrieve_outdir, exist_ok=True)

    @cache_step("retrieve_data")
    async def retrieve_data_step(data_config: data_types.DataConfig, output_dir: str):
        return await asyncio.gather(
            *[
                retrieve_data(
                    all_corpora_map=data_config.train_corpora,
                    output_prefix=os.path.join(retrieve_outdir, "train"),
                    preprocess_config=data_config.preprocessing_config,
                    tag="train",
                    output_dir=output_dir,
                    executor=executor,
                ),
                retrieve_data(
                    all_corpora_map=data_config.secondary_train_corpora,
                    output_prefix=os.path.join(retrieve_outdir, "secondary_train"),
                    preprocess_config=data_config.preprocessing_config,
                    tag="secondary_train",
                    output_dir=output_dir,
                    executor=executor,
                )
                if data_config.secondary_train_corpora
                else async_noop(),
                retrieve_data(
                    all_corpora_map=data_config.valid_corpora,
                    output_prefix=os.path.join(retrieve_outdir, "valid"),
                    preprocess_config=data_config.preprocessing_config,
                    tag="valid",
                    output_dir=output_dir,
                    executor=None,
                )
                if data_config.valid_corpora
                else async_noop(),
                retrieve_data(
                    all_corpora_map=data_config.test_corpora,
                    output_prefix=os.path.join(retrieve_outdir, "test"),
                    preprocess_config=data_config.preprocessing_config,
                    tag="test",
                    output_dir=output_dir,
                    executor=None,
                )
                if data_config.test_corpora
                else async_noop(),
            ]
        )

    (
        full_train_data,
        secondary_train_data,
        valid_data,
        test_data,
    ) = await retrieve_data_step(
        data_config=data_config,
        output_dir=output_dir,
    )

    train_data = full_train_data[0]
    sampled_train = full_train_data[1]
    if secondary_train_data is not None:
        secondary_train_data = secondary_train_data[0]
    valid_data = valid_data[0]
    test_data = test_data[0]

    src_vocab, tgt_vocab = await get_vocab(
        data_config=data_config,
        train_corpora=train_data,
        secondary_train_corpora=secondary_train_data,
        src_counts_map=train_src_counts_map,
        tgt_counts_map=train_tgt_counts_map,
        output_dir=output_dir,
    )

    all_num_shards, all_num_secondary_shards = get_all_num_shards(
        train_counts_map, secondary_train_counts_map, data_config
    )
    directions = list(
        set(list(train_counts_map.keys()) + list(secondary_train_counts_map.keys()))
    )

    jobs = []
    for direction in directions:
        jobs.append(
            executor.submit(
                prepare_data_direction,
                direction,
                train_data,
                secondary_train_data,
                valid_data,
                sampled_train,
                test_data,
                src_vocab,
                tgt_vocab,
                all_num_shards,
                all_num_secondary_shards,
                data_config,
                temp_dir,
                output_dir,
            )
        )

    await asyncio.gather(*[awaitable_job(j) for j in jobs])

    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-config",
        default="baselines_conf/data_configs/demo/flores2_x_en.32k.yaml",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--log-file", default="prepare_data.log")
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config_path, config_name = os.path.split(args.data_config)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else f"{config_name}_{int(time.time())}"
    )
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    fh = logging.FileHandler(filename=os.path.join(output_dir, args.log_file))
    logger.addHandler(fh)

    with initialize(config_path=config_path):
        data_config = setup_config(config_name, args.rest)
        asyncio.run(main(data_config, output_dir))
