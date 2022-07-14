#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.file_io import recursively_cast_dictconfigs

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.reshard")


def main(cfg, **unused_kwargs):
    if isinstance(cfg, Namespace):
        cfg = convert_namespace_to_omegaconf(cfg)
    utils.import_user_module(cfg.common)  # allow --user-dir
    assert (
        cfg.distributed_training.ddp_backend == "fully_sharded"
    ), "pass --ddp-backend fully_sharded"

    # These should be clargs
    target_size = cfg.reshard.target_world_size
    src_path = cfg.common_eval.path
    save_dir = cfg.reshard.save_dir
    prefix = cfg.reshard.save_prefix
    do_pad = cfg.reshard.do_pad
    # Note: See top of eval_lm to resurrect model_overrides
    load_sharded = cfg.distributed_training.use_sharded_state
    rank = distributed_utils.get_data_parallel_rank()
    suffix = f"-shard{rank}" if load_sharded else ""
    task = tasks.setup_task(cfg.task)  # Need this to build the dict to build the model

    def build_with_fsdp(train_cfg, *ignored):
        """train_cfg is read from disk, but cfg.distributed_training is passed through command line"""
        with fsdp_enable_wrap(
            cfg.distributed_training, use_sharded_state=load_sharded, is_moe=False
        ):
            model = fsdp_wrap(task.build_model(train_cfg.model))
        return model

    # TODO: HANDLE SUFFIX IF NOT CONSOLIDATED
    original_state = checkpoint_utils.load_checkpoint_to_cpu(src_path)
    models, _, __ = checkpoint_utils.load_model_ensemble_and_task(
        [src_path],
        suffix=suffix,
        state=original_state,
        task=task,
        build_model_hook=build_with_fsdp,
        strict=False,  # hack around Missing key(s) in state_dict: "_fpw_module.decoder.output_projection.weight"
    )
    fsdp_lm = models[0]
    original_state.pop("model")

    # Don't save raw omegaconf objects (mimicing behavior of Trainer)
    original_state["cfg"] = recursively_cast_dictconfigs(original_state["cfg"])
    local_model_state = fsdp_lm.local_state_dict()
    del models, fsdp_lm  # Maybe save CPU Ram
    assert "decoder.embed_tokens.weight" not in original_state
    assert "decoder.embed_tokens.weight" not in local_model_state

    shards: List[dict] = [{} for _ in range(target_size)]
    for k, v in local_model_state.items():
        v = v.half()
        if "flat_param" in k:
            if v.numel() % target_size != 0 and not do_pad:
                raise AssertionError(
                    f"Cannot split {k}, sized {v.numel()} into {target_size} even chunks. Try passing a smaller --target-world-size."
                )
            sharded_param = list(torch.flatten(v).chunk(target_size))
            # Same logic as https://tinyurl.com/fairscale
            # Why not always pad? https://github.com/fairinternal/fairseq-py/issues/2894
            for rank, param in enumerate(sharded_param):
                # This clone is essential. Not sure why.
                shards[rank][k] = param.clone()

            num_to_pad = shards[0][k].numel() - shards[-1][k].numel()
            if num_to_pad > 0 and do_pad:
                shards[-1][k] = F.pad(shards[-1][k], [0, num_to_pad])
                logger.info(f"Padding {k} with {num_to_pad} zeros")
            elif num_to_pad > 0:
                logger.info(f"Not padding {k} even though it wants {num_to_pad} zeros.")

        else:
            logger.info(f"Skipping non-flat parameter {k}")
    del local_model_state

    Path(save_dir).mkdir(exist_ok=True)
    # TODO: swap to PathManager
    for i, shard_state in enumerate(shards):
        save_path = f"{save_dir}/{prefix}-shard{i}.pt"
        full_state = {"model": shard_state}
        full_state.update(original_state)
        torch.save(full_state, save_path)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    task_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    task_parser.add_argument("--task", default="streaming_language_modeling", type=str)
    task_args, _ = task_parser.parse_known_args()
    parser = options.get_reshard_parser(task=task_args.task)
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    main(cfg)
