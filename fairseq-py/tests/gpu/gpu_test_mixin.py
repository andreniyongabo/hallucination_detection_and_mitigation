# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import logging
import os
import tempfile
import unittest
from io import StringIO

import torch

from fairseq.distributed.stitch_fsdp_ckpt import consolidate_fsdp_shards
from fairseq.file_io import load_json
from tests.utils import (
    create_dummy_data,
    eval_lm_main,
    preprocess_lm_data,
    train_language_model,
)

DEVICE_COUNT = torch.cuda.device_count()

try:
    import bitsandbytes as bnb  # noqa

    HAS_BNB = True
except ImportError:
    HAS_BNB = False


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class LMTestMixin(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @property
    def eval_clargs(self):
        return [
            "--fp16",
            "--max-valid-steps",
            "5",
            "--pad-to-fixed-length",
            "--model-overrides",
            "{'track_expert_stats': True}",
            "--batch-size",
            "1",
            "--tokens-per-sample",
            "20",
        ]

    def _train_and_eval_lm(
        self, extra_train_flags, extra_eval_flags, arch="transformer_lm_gpt2_tiny"
    ):
        mu = 5  # use a small max_update count
        eval_flags = self.eval_clargs + extra_eval_flags
        train_flags = self.train_clargs(mu) + extra_train_flags
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fp16") as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(
                    data_dir, num_examples=int(mu * 20 * self.world_size * 1.5)
                )  # make sure enough data for max updates
                preprocess_lm_data(data_dir)
                train_language_model(
                    data_dir,
                    arch,
                    train_flags + ["--log-file", log],
                    world_size=self.world_size,
                )
                stats_path = os.path.join(data_dir, "eval_stats")
                eval_lm_main(
                    data_dir,
                    eval_flags + ["--sp", stats_path],
                    world_size=self.world_size,
                )
                assert os.path.exists(stats_path), os.listdir(data_dir)
                stats = load_json(stats_path)
        print(stats)

    @staticmethod
    def parse_logs(logfile):
        logs = []
        for ln in open(logfile, "r").readlines():
            try:
                logs.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return logs

    @property
    def world_size(self):
        return DEVICE_COUNT

    @property
    def moe_clargs(self):
        return [
            "--moe-freq",
            "2",
            "--decoder-layers",
            "2",
            "--criterion",
            "moe_cross_entropy",
            "--moe-gate-loss-wt",
            ".01",
            "--moe-gate-loss-combine-method",
            "sum",
            "--moe-second-expert-policy",
            "all",
            "--moe-gating-use-fp32",
            "--record-a2a-perf-stats",
        ]

    @property
    def moe_clargs_1_expert_per_gpu_clargs(self):
        return self.moe_clargs + ["--moe-expert-count", str(self.world_size)]

    def train_clargs(self, mu):
        return [
            "--memory-efficient-fp16",
            "--update-freq",
            "1",
            "--seed",
            "1",
            "--log-format",
            "json",
            "--max-update",
            str(mu),
            "--tokens-per-sample",
            "20",
            "--batch-size",
            "2",
            "--share-decoder-input-output-embed",
            "--optimizer",
            "adam",
            "--max-valid-steps",
            "1",
            "--pad-to-fixed-length",
            "--sample-break-mode",
            "none",
        ]

    def _test_resume_training(
        self,
        extra_clargs,
        arch="transformer_lm_gpt2_tiny",
        consolidate_and_eval=False,
        eval_sharded=False,
        second_world_size=None,
        assert_losses_match=True,
        save_interval=5,
        mu=10,
    ):
        train_clargs = (
            self.train_clargs(mu)
            + [
                "--save-interval-updates",
                str(save_interval),
                "--log-interval",
                "1",
                "--init-model-on-gpu",
            ]
            + extra_clargs
        )
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fp16") as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(
                    data_dir, num_examples=int(mu * 20 * self.world_size * 1.5)
                )  # make sure enough data for 10 updates
                preprocess_lm_data(data_dir)
                train_language_model(
                    data_dir,
                    arch,
                    train_clargs + ["--log-file", log],
                    world_size=self.world_size,
                )
                ckpt_prefix = f"checkpoint_1_{save_interval}"
                for file in os.listdir(data_dir):
                    if file.startswith(ckpt_prefix):
                        ckpt_last_file = os.path.join(
                            data_dir, file.replace(ckpt_prefix, "checkpoint_last")
                        )
                        assert os.path.exists(
                            ckpt_last_file
                        ), f"missing {ckpt_last_file}"
                log2 = os.path.join(data_dir, "resume.log")
                ckpt_name = f"{ckpt_prefix}.pt"
                restore_file = os.path.join(data_dir, ckpt_name)
                if second_world_size is None:
                    second_world_size = self.world_size
                else:
                    train_clargs.extend(
                        ["--update-freq", str(self.world_size // second_world_size)]
                    )

                train_language_model(
                    data_dir,
                    arch,
                    train_clargs
                    + ["--log-file", log2, "--restore-file", restore_file, "--no-save"],
                    world_size=second_world_size,
                )

                if assert_losses_match:
                    self.assert_resumed_loss_equals_original_loss(
                        ckpt_name, data_dir, log, log2, mu, save_interval
                    )
                if consolidate_and_eval:
                    consolidated_path = consolidate_fsdp_shards(
                        f"{data_dir}/{ckpt_name}"
                    )
                    eval_lm_main(
                        data_dir,
                        path=consolidated_path,
                        extra_flags=[
                            "--sp",
                            f"{data_dir}/stats",
                            "--tokens-per-sample",
                            "20",
                        ],
                    )

                if eval_sharded:
                    eval_lm_main(
                        data_dir,
                        path=f"{data_dir}/{ckpt_name}",
                        extra_flags=[
                            "--sp",
                            f"{data_dir}/sharded_stats",
                            "--tokens-per-sample",
                            "20",
                            "--ddp-backend",
                            "fully_sharded",
                            "--use-sharded-state",
                            "--fp16",
                            "--memory-efficient-fp16",
                        ],
                        world_size=self.world_size,
                    )

    def assert_resumed_loss_equals_original_loss(
        self, ckpt_name, data_dir, log, log2, mu, save_interval
    ):
        l1 = self.parse_logs(log)
        assert (
            int(l1[-1]["train_num_updates"]) == mu
        ), f"The first run did not complete {mu} updates. Add more data"
        l2 = self.parse_logs(log2)
        if not l2:
            raise ValueError(f"No second train.log at {log2}")
        if int(l2[0]["num_updates"]) != save_interval + 1:
            all_ckpt_files = [x for x in os.listdir(data_dir) if x.endswith(".pt")]
            import shutil

            shutil.move(data_dir, "last_failed_resume")
            raise AssertionError(
                f"Likely failed to load {ckpt_name}. {all_ckpt_files} \n LOGS: {l1} \n\n {l2}. "
            )
        for k in [
            "train_loss",
            "train_num_updates",
            # "train_ppl",  TODO: fails for unknown reasons
            "train_gnorm",
        ]:
            from_scratch, resumed = float(l1[-1][k]), float(l2[-1][k])
            # This fails without rounding!
            assert (
                from_scratch == resumed
            ), f"difference at {k} {from_scratch} != {resumed}"
