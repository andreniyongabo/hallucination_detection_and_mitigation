#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import itertools
import logging
import os
from contextlib import ExitStack
from pathlib import Path
from typing import Dict, Optional

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf
from submitit import AutoExecutor

from examples.nllb.mining.monolingual.utils.predict_script import find_lang_script
from examples.nllb.modeling.filtering.configs import (
    FilterConfig,
    GroupFilterConfig,
    register_configs,
)
from examples.nllb.modeling.filtering.dataset import Dataset, DatasetLine, DatasetReader
from examples.nllb.modeling.filtering.filters import FilteringCounts
from examples.nllb.modeling.filtering.utils import smart_open
from examples.nllb.modeling.prepare_data.cache import cache_step_sync

logger = logging.getLogger(__name__)


register_configs()


@cache_step_sync("filter")
def filter_datasets(
    group_name: str,
    src_lang: str,
    tgt_lang: Optional[str],  # if None, treat as monolingual datasets
    datasets: Dict[str, Dataset],
    config: GroupFilterConfig,
    dataset_output_dir: Path,
    custom_step_name: str,
    output_dir: Path,
):
    direction = f"{src_lang}-{tgt_lang}" if tgt_lang is not None else src_lang
    print(f"Filtering {group_name}.{direction}")

    # build the list of filters to be applied to this group
    filters = [
        hydra.utils.instantiate(config.laser_filter),
        hydra.utils.instantiate(config.basic_filter),
        hydra.utils.instantiate(
            config.lid_filter, src_lang=src_lang, tgt_lang=tgt_lang
        ),
        hydra.utils.instantiate(
            config.toxicity_filter, src_lang=src_lang, tgt_lang=tgt_lang
        ),
    ]

    # filter datasets sequentially
    counts: Dict[str, FilteringCounts] = {}
    for corpus_name, dataset in datasets.items():
        these_counts = FilteringCounts()  # filtering counts for the current dataset

        path_out_src = dataset_output_dir / f"{corpus_name}.{src_lang}.gz"
        path_out_tgt = dataset_output_dir / f"{corpus_name}.{tgt_lang}.gz"

        print(f"Processing {corpus_name}")
        with ExitStack() as outputs, DatasetReader(dataset, corpus_name) as inputs:
            fout_src = outputs.enter_context(gzip.open(path_out_src, "wt"))
            fout_tgt = None

            # if tgt_lang is not provided, we have a monolingual dataset;
            # otherwise, the parallel file needs to be opened
            if tgt_lang is not None:
                fout_tgt = outputs.enter_context(gzip.open(path_out_tgt, "wt"))

            for line in inputs:
                these_counts.total_before += 1
                # apply filters sequentially
                for fltr in filters:
                    if fltr is None:
                        continue
                    line = fltr.filter_line(line, these_counts)
                    # no need to keep filtering if the line was already discarded
                    if line is None:
                        break
                if line is None:
                    continue

                fout_src.write(line.src + "\n")
                if fout_tgt is not None:
                    fout_tgt.write(line.tgt + "\n")
                these_counts.total_after += 1

            counts[corpus_name] = these_counts
    if counts:
        print(f"Total counts: {sum(counts.values()).__dict__}")
    return direction, counts


def filter_group(group_name: str, config: DictConfig):
    assert group_name in config, f"unknown data group {group_name}"
    executor = AutoExecutor(
        folder=Path(config.output_dir) / config.executor.log_folder / group_name,
        cluster=config.executor.cluster,
    )
    executor.update_parameters(
        slurm_partition=config.executor.slurm_partition,
        timeout_min=2880,
        nodes=1,
        cpus_per_task=4,
        name=f"filter.{group_name}",
    )
    logger.info(f"Filtering {group_name}")

    group_config = config.get(group_name)
    assert (
        group_config.included_corpora is None or group_config.excluded_corpora is None
    )
    datasets = OmegaConf.load(Path(config.data_conf_dir) / f"{group_name}.yaml")
    # submit all directions as part of the same array
    jobs = []
    with executor.batch():
        for direction, corpora in datasets.items():
            if direction not in config.directions:
                continue

            try:
                src, tgt = direction.split("-")
            except ValueError:  # this is monolingual data
                src = direction
                tgt = None

            assert group_config.included_corpora or group_config.excluded_corpora
            # select the datasets we want to include
            if group_config.included_corpora is not None:
                group_datasets = {
                    corpus_name: dataset
                    for corpus_name, dataset in corpora.items()
                    if corpus_name in group_config.included_corpora
                }
            else:
                group_datasets = {
                    corpus_name: dataset
                    for corpus_name, dataset in corpora.items()
                    if corpus_name not in group_config.excluded_corpora
                }
            if not group_datasets:
                logger.warning(f"Skipping empty {group_name}.{direction}")
                continue

            dataset_output_dir = Path(config.output_dir) / group_name / direction
            os.makedirs(dataset_output_dir, exist_ok=True)
            logger.info(f"Preparing {group_name}.{direction} job")
            job = executor.submit(
                filter_datasets,
                group_name=group_name,
                src_lang=src,
                tgt_lang=tgt,
                datasets=group_datasets,
                config=config.get(group_name),
                dataset_output_dir=dataset_output_dir,
                custom_step_name=f"{group_name}.{direction}",
                output_dir=Path(config.output_dir),
            )
            jobs.append(job)
    logger.info(f"All jobs for {group_name} have been scheduled")
    results = [job.result() for job in jobs]
    logger.info(f"All jobs for {group_name} are done")
    # direction -> counts
    per_direction_counts = {
        direction: sum(direction_counts.values()).__dict__
        for direction, direction_counts in results
        if direction_counts
    }
    # direction -> corpus_name -> counts
    per_corpus_counts = {
        direction: {
            corpus_name: corpus_counts.__dict__
            for corpus_name, corpus_counts in direction_counts.items()
            if isinstance(corpus_counts, FilteringCounts)
        }
        for direction, direction_counts in results
    }
    return per_direction_counts, per_corpus_counts


@hydra.main(config_path="conf", config_name="example")
def main(config: DictConfig) -> None:
    os.makedirs(config.output_dir, exist_ok=True)
    logger.info(f"Running with config:\n{OmegaConf.to_yaml(config)}")
    with open(Path(config.output_dir) / "config.yaml", "wt") as fout:
        fout.write(OmegaConf.to_yaml(config, sort_keys=True))

    total_counts = {}
    corpus_counts = {}
    for group_name in ("train_primary", "train_mined"):
        if config.get(group_name, None):
            total_counts[group_name], corpus_counts[group_name] = filter_group(
                group_name=group_name, config=config
            )

    total_counts_path = Path(config.output_dir) / "total_counts.yaml"
    corpus_counts_path = Path(config.output_dir) / "corpus_counts.yaml"
    with open(total_counts_path, "wt") as fout:
        yaml.safe_dump(total_counts, fout)
    with open(corpus_counts_path, "wt") as fout:
        yaml.safe_dump(corpus_counts, fout)
    logger.info(f"All jobs done – counts written to {total_counts_path}")


if __name__ == "__main__":
    main()
