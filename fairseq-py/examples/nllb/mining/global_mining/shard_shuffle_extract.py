# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import glob
import logging
import random
import shutil
import typing as tp
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import hydra
from omegaconf import MISSING, DictConfig, OmegaConf

from examples.nllb.mining.global_mining.modules.preprocess.multiproc_line_processor import (  # noqa
    MultiprocLineProcessorCallback,
    MultiprocLineProcessorConfig,
    MultiprocLineProcessorModule,
)
from examples.nllb.mining.nllb_lib import nllb_module, utils

logger = logging.getLogger("shard_shuffle_extract")


@dataclass
class ShardShuffleExtractConfig:
    launcher: DictConfig
    nb_shards: int
    output_dir: str = MISSING
    input_files_glob: str = MISSING
    outfile_prefix: str = "shard"
    text_starts_at_col: int = 6
    log_every: int = 1_000_000


class Shard:
    """
    Class handling most of the file IO for shard chunks when processing file chunks
    and final shards when merging chunks
    """

    def __init__(
        self,
        output_file: Path,
        shard_id: int,
        mode: str = "wt",
        encoding: str = "utf-8",
    ):
        # creating filepaths for meta, text and nl files
        self.meta_path = Path(f"{output_file}.{shard_id:03d}.meta")
        self.text_path = Path(f"{output_file}.{shard_id:03d}.text")
        self.nl_path = Path(f"{output_file}.{shard_id:03d}.nl")
        self.nl_cnt = 0
        logger.info(
            f"Initialized shard {shard_id} at {self.meta_path}, {self.text_path},"
            f" {self.nl_path}"
        )
        self.mode = mode
        self.encoding = encoding

    def __enter__(self):
        self.meta_fd = utils.open(
            self.meta_path, mode=self.mode, encoding=self.encoding
        )
        self.text_fd = utils.open(
            self.text_path, mode=self.mode, encoding=self.encoding
        )
        return self

    def __exit__(self, *exc):
        # cleaning up
        self.meta_fd.close()
        self.text_fd.close()
        self.meta_fd, self.text_fd = None, None
        # saving the number of line for that shard
        with utils.open(self.nl_path, mode="wt") as nl_fd:
            print(str(self.nl_cnt), file=nl_fd)

    # TODO: ensure write and append are atomic operations
    def write(self, meta: tp.List[str], text: tp.List[str]):
        # persisting metadata content
        print(*meta, file=self.meta_fd, sep="\t")
        # persisting text content
        print(*text, file=self.text_fd, sep="\t")
        # keeping track of the number of lines within the shard
        self.nl_cnt += 1

    def append(self, chunk: "Shard"):
        # appending metadata from the chunk
        with utils.open(chunk.meta_path, mode="rb") as meta_shard_fd:
            shutil.copyfileobj(meta_shard_fd, self.meta_fd)
        # appending text from the chunk
        with utils.open(chunk.text_path, mode="rb") as text_shard_fd:
            shutil.copyfileobj(text_shard_fd, self.text_fd)
        # keeping track of the number of lines within the shard
        # the total count will be persisted after stitching all chunks
        self.nl_cnt += chunk.nl_cnt


class ShardShuffleExtractMC(MultiprocLineProcessorCallback):
    """
    splits a single input file into multiple shards,
    shuffling and extracting metadata on the fly
    """

    def __init__(
        self,
        # set by LineProcessorModule
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        offset_start: tp.Optional[int],
        offset_end: tp.Optional[int],
        merging: bool,
        # our params
        # controls how many shards the input file will be split into
        nb_shards: int,
        # column to split on, 0-indexed
        text_starts_at_col: int,
        log_every: int = 10_000,
    ):
        super().__init__(
            outfile_prefix=outfile_prefix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            offset_start=offset_start,
            offset_end=offset_end,
            merging=merging,
        )
        utils.ensure_dir(self.output_dir)
        # TODO: harmonize file naming. For now, files might need to be
        # renamed before starting mining.
        self.output_file = (
            Path(output_dir)
            / f"{Path(self.input_file).stem}.{self.outfile_prefix}.{offset_start}_{offset_end}"
        )
        self.shards = []
        self.nb_shards = nb_shards
        self.text_starts_at_col = text_starts_at_col
        self.log_every = log_every

    def __enter__(self):
        self.stack = ExitStack()
        for shard_id in range(self.nb_shards):
            self.shards.append(
                self.stack.enter_context(Shard(self.output_file, shard_id))
            )
        return self

    def __exit__(self, *exc):
        self.stack.close()

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        """
        processes a chunk of lines, separates metadata and text, picks a random shard
        and persists everything to the corresponding chunk temp files
        """
        random.seed(42)

        for idx, line in lines_with_number:
            fields = line.strip().split("\t", maxsplit=self.text_starts_at_col)
            # TODO: keep track of how many lines get processed vs how many get thrown
            # out and log that in final_result
            if len(fields) < 2:
                logger.info(f"Found meaningless text on line: {idx} with text: {line}")
                continue

            # selecting one shard at random, putting contents into it
            r = random.randint(0, self.nb_shards - 1)
            self.shards[r].write(
                meta=fields[: self.text_starts_at_col],
                text=fields[self.text_starts_at_col :],
            )

            if idx % self.log_every == 0:
                logger.info(f"{self.output_file} : processed {idx} lines")

    def final_result(self) -> tp.List[Shard]:
        """
        passes references to shard chunks, used by merge_results
        """
        return self.shards

    def merge_results(self, splits: tp.List[tp.List[Shard]]) -> tp.List[Shard]:
        """
        stitches shard chunk temp files together
        """

        merge = (
            Path(self.output_dir)
            / f"{Path(self.input_file).stem}.{self.outfile_prefix}"
        )

        # initializing the actual output shards
        final_shards = []
        with ExitStack() as stack:
            for shard_id in range(self.nb_shards):
                final_shards.append(
                    stack.enter_context(
                        Shard(merge, shard_id, mode="wb", encoding=None)
                    )
                )

            # splits contains shards for each chunk of the original data
            for shards in splits:
                for idx, s in enumerate(shards):
                    f_shard = final_shards[idx]
                    f_shard.append(chunk=s)

        # saving the number of lines of each shard
        nl_path = f"{merge}.xxx.nl"
        with utils.open(nl_path, mode="wt") as nl_fd:
            for idx, f_shard in enumerate(final_shards):
                print(str(f_shard.nl_cnt), file=nl_fd)

        return final_shards


async def shard_shuffle_extract(config: ShardShuffleExtractConfig):
    launcher = hydra.utils.instantiate(config.launcher)

    OmegaConf.save(
        config=config,
        f=str(Path(launcher.config_dump_dir) / "shard_shuffle_extract.yaml"),
    )
    files_to_process = glob.glob(config.input_files_glob)
    file_processor = MultiprocLineProcessorModule(
        config=MultiprocLineProcessorConfig(
            line_processor=DictConfig(
                {
                    "_target_": "examples.nllb.mining.global_mining.shard_shuffle_extract.ShardShuffleExtractMC",
                    "nb_shards": config.nb_shards,
                    "text_starts_at_col": config.text_starts_at_col,
                    "log_every": config.log_every,
                }
            ),
            custom_name="shard_shuffle_extract",
            output_dir=str(config.output_dir),
            outfile_prefix=config.outfile_prefix,
            shards=files_to_process,
            requirements=nllb_module.DistributedRequirements(
                nodes=1,
                mem_gb=getattr(config, "mem_gb", 1),
                tasks_per_node=1,
                cpus_per_task=getattr(config, "num_cpu", 40),
                gpus_per_node=0,
                timeout_min=getattr(config, "timeout_min", 14400),
            ),
            tmp_dir=str(config.tmp_dir),
        )
    )

    logger.info(f"Sharding, shuffling and extracting metadata from {files_to_process}")
    all_results = await launcher.schedule(file_processor)
    logger.info("Done sharding, shuffling and extracting metadata into shards")

    for f_idx, shards in enumerate(all_results):
        logger.info(f"File processed: {files_to_process[f_idx]}")
        for s_idx, s in enumerate(shards):
            logger.info(
                f"Shard nb {s_idx} - files: {str(s.meta_path)}, {str(s.text_path)},"
                f"nb elements: {s.nl_cnt}"
            )


@hydra.main(config_path="conf", config_name="shard_shuffle_extract")
def main(config: ShardShuffleExtractConfig) -> None:
    asyncio.run(shard_shuffle_extract(config))


if __name__ == "__main__":
    main()
