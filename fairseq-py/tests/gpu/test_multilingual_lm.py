import contextlib
import os
import tempfile
import unittest
from io import StringIO

import torch

from fairseq.file_io import load_json
from tests.gpu.gpu_test_mixin import LMTestMixin
from tests.utils import (
    create_dummy_data,
    eval_lm_main,
    preprocess_lm_data,
    train_language_model,
)


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestMultilingualLM(LMTestMixin):
    def _train_and_eval_multilingual_lm(
        self, extra_train_clargs, extra_eval_clargs, arch="transformer_lm_gpt2_tiny"
    ):
        languages = ["en_XX", "fr_XX", "zh_CN"]
        mu = 5  # use a small max_update count
        eval_clargs = self.eval_clargs + extra_eval_clargs
        train_clargs = self.train_clargs(mu) + extra_train_clargs
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fp16") as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(
                    data_dir,
                    num_examples=int(
                        mu * 20 * self.world_size * 1.5
                    ),  # make sure enough data for max updates
                    languages=languages,
                )
                preprocess_lm_data(data_dir, languages)
                train_language_model(
                    data_dir,
                    arch,
                    extra_flags=train_clargs + ["--log-file", log],
                    task="multilingual_language_modeling",
                    world_size=self.world_size,
                )
                stats_path = os.path.join(data_dir, "eval_stats")
                eval_lm_main(
                    data_dir,
                    eval_clargs + ["--langs", ",".join(languages), "--sp", stats_path],
                    task="multilingual_language_modeling",
                    world_size=self.world_size,
                )
                assert os.path.exists(stats_path), os.listdir(data_dir)
                stats = load_json(stats_path)
        print(stats)

    @unittest.skip("Currently broken")
    def test_eval_multilingual_lm(self):
        self._train_and_eval_multilingual_lm([], [])

    @unittest.skip("Currently broken")
    def test_eval_mutlilingual_lm_moe(self):
        self._train_and_eval_multilingual_lm(
            self.moe_clargs_1_expert_per_gpu_clargs, ["--is-moe"]
        )

    def test_resume_training_multilingual_moe_noc10d(self):
        self._test_resume_multilingual_training(
            self.moe_clargs_1_expert_per_gpu_clargs + ["--fp16-no-flatten-grads"]
        )

    def test_resume_training_multilingual_moe_fsdp(self):
        self._test_resume_multilingual_training(
            self.moe_clargs_1_expert_per_gpu_clargs + ["--ddp-backend", "fully_sharded"]
        )

    def _test_resume_multilingual_training(
        self, extra_clargs, arch="transformer_lm_gpt2_tiny"
    ):
        languages = ["en_XX", "fr_XX", "zh_CN"]
        save_interval = 5
        mu = 10
        train_clargs = (
            self.train_clargs(mu)
            + ["--save-interval-updates", str(save_interval), "--log-interval", "1"]
            + extra_clargs
        )
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fp16") as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(
                    data_dir,
                    num_examples=int(
                        mu * 20 * self.world_size * 1.5
                    ),  # make sure enough data for max updates
                    languages=languages,
                )
                preprocess_lm_data(data_dir, languages)
                train_language_model(
                    data_dir,
                    arch,
                    train_clargs + ["--log-file", log],
                    task="multilingual_language_modeling",
                    world_size=self.world_size,
                )
                log2 = os.path.join(data_dir, "resume.log")
                ckpt_name = f"checkpoint_1_{save_interval}.pt"
                restore_file = os.path.join(data_dir, ckpt_name)
                train_language_model(
                    data_dir,
                    arch,
                    train_clargs
                    + ["--log-file", log2, "--restore-file", restore_file, "--no-save"],
                    task="multilingual_language_modeling",
                    world_size=self.world_size,
                )

                self.assert_resumed_loss_equals_original_loss(
                    ckpt_name, data_dir, log, log2, mu, save_interval
                )
