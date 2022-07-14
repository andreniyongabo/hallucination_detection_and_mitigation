# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import json
import logging
import os
import random
import tempfile
import unittest
from io import StringIO
from typing import Dict, List

import torch

from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as distributed_utils
from fairseq_cli import eval_lm, train
from tests.utils import (
    create_dummy_data,
    generate_main,
    preprocess_lm_data,
    preprocess_translation_data,
    train_language_model,
    train_translation_model,
)

try:
    import transformers  # noqa

    has_hf_transformers = True
except ImportError:
    has_hf_transformers = False


class TestPromptTuning(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def _train(self, model_name, downstream_task, extra_flags=None, world_size=1):
        train_parser = options.get_training_parser()
        train_args = options.parse_args_and_arch(
            train_parser,
            [
                "--user-dir",
                "examples/few_shot/finetune",  # not used
                "--task",
                "prompt_tuning",
                "--criterion",
                "prompt_tuning",
                "--report-accuracy",
                "--model-name",
                model_name,
                "--downstream-task",
                downstream_task,
                "--sample-break-mode",
                "eos",
                "--arch",
                "transformer_lm",  # not used
                "--optimizer",
                "adam",
                "--lr",
                "0.0001",
                "--batch-size",
                "2",
                "--no-save",
                "--max-epoch",
                "1",
                "--no-progress-bar",
                "--distributed-world-size",
                str(world_size),
                "--ddp-backend",
                "no_c10d",
                "--num-workers",
                "0",
            ]
            + (extra_flags or []),
        )
        cfg = convert_namespace_to_omegaconf(train_args)
        distributed_utils.call_main(cfg, train.main)

    @unittest.skip("Disabled as currently broken")
    def test_prefix_tokens(self):
        with contextlib.redirect_stdout(StringIO()):
            self._train(
                model_name="125M_gpt3_setting",
                downstream_task="cb",
                extra_flags=["--num-prefix-tokens", "5"],
            )

    @unittest.skip("Disabled as currently broken")
    def test_prefix_tokens_with_finetuning(self):
        with contextlib.redirect_stdout(StringIO()):
            self._train(
                model_name="125M_gpt3_setting",
                downstream_task="cb",
                extra_flags=["--num-prefix-tokens", "5", "--finetune-model-weights"],
            )


class TestStories(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_fconv_self_att_wp(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fconv_self_att_wp") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                config = [
                    "--encoder-layers",
                    "[(128, 3)] * 2",
                    "--decoder-layers",
                    "[(128, 3)] * 2",
                    "--decoder-attention",
                    "True",
                    "--encoder-attention",
                    "False",
                    "--gated-attention",
                    "True",
                    "--self-attention",
                    "True",
                    "--project-input",
                    "True",
                    "--encoder-embed-dim",
                    "8",
                    "--decoder-embed-dim",
                    "8",
                    "--decoder-out-embed-dim",
                    "8",
                    "--multihead-self-attention-nheads",
                    "2",
                ]
                train_translation_model(data_dir, "fconv_self_att_wp", config)
                generate_main(data_dir)

                # fusion model
                os.rename(
                    os.path.join(data_dir, "checkpoint_last.pt"),
                    os.path.join(data_dir, "pretrained.pt"),
                )
                config.extend(
                    [
                        "--pretrained",
                        "True",
                        "--pretrained-checkpoint",
                        os.path.join(data_dir, "pretrained.pt"),
                        "--save-dir",
                        os.path.join(data_dir, "fusion_model"),
                    ]
                )
                train_translation_model(data_dir, "fconv_self_att_wp", config)


class TestLanguageModeling(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @unittest.skip("Disabled as currently broken")
    def test_transformer_lm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer_lm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_language_model(
                    data_dir,
                    "transformer_lm",
                    ["--add-bos-token", "--nval", "1"],
                    run_validation=True,
                )
                eval_lm_main(data_dir)
                eval_lm_main(data_dir, extra_flags=["--context-window", "25"])
                generate_main(
                    data_dir,
                    [
                        "--task",
                        "language_modeling",
                        "--sample-break-mode",
                        "eos",
                        "--tokens-per-sample",
                        "500",
                    ],
                )

    @unittest.skip("Disabled as currently broken")
    def test_transformer_lm_with_adaptive_softmax(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_transformer_lm_with_adaptive_softmax"
            ) as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_language_model(
                    data_dir,
                    "transformer_lm",
                    [
                        "--add-bos-token",
                        "--criterion",
                        "adaptive_loss",
                        "--adaptive-softmax-cutoff",
                        "5,10,15",
                    ],
                    run_validation=True,
                )
                eval_lm_main(data_dir)
                generate_main(
                    data_dir,
                    [
                        "--task",
                        "language_modeling",
                        "--sample-break-mode",
                        "eos",
                        "--tokens-per-sample",
                        "500",
                    ],
                )

    @unittest.skipIf(not has_hf_transformers, "skip test if transformers is missing")
    @unittest.skip("Disabled as currently broken")
    def test_transformer_xl_bptt_lm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer_xl_bptt_lm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                task_flags = [
                    "--user-dir",
                    "examples/truncated_bptt",
                    "--task",
                    "truncated_bptt_lm",
                    "--batch-size",
                    "2",
                    "--tokens-per-sample",
                    "50",
                ]
                train_language_model(
                    data_dir=data_dir,
                    arch="transformer_xl",
                    extra_flags=task_flags
                    + [
                        "--n-layer",
                        "2",
                    ],
                    task="truncated_bptt_lm",
                    run_validation=True,
                    extra_valid_flags=task_flags,
                )
                eval_lm_main(data_dir, extra_flags=task_flags)
                # Train with activation offloading
                train_language_model(
                    data_dir=data_dir,
                    arch="transformer_xl",
                    extra_flags=task_flags
                    + [
                        "--n-layer",
                        "2",
                        "--offload-activations",
                    ],
                    task="truncated_bptt_lm",
                    run_validation=True,
                    extra_valid_flags=task_flags,
                )


class TestMaskedLanguageModel(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_legacy_masked_lm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_legacy_mlm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_legacy_masked_language_model(data_dir, "masked_lm")

    @unittest.skip("Disabled as currently broken")
    def test_roberta_masked_lm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_roberta_mlm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_masked_lm(
                    data_dir, "roberta_base", extra_flags=["--encoder-layers", "2"]
                )

    @unittest.skip("Disabled as currently broken")
    def test_roberta_sentence_prediction(self):
        num_classes = 3
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_roberta_head") as data_dir:
                create_dummy_roberta_head_data(data_dir, num_classes=num_classes)
                preprocess_lm_data(os.path.join(data_dir, "input0"))
                preprocess_lm_data(os.path.join(data_dir, "label"))
                train_roberta_head(data_dir, "roberta_base", num_classes=num_classes)

    @unittest.skip("Disabled as currently broken")
    def test_roberta_regression_single(self):
        num_classes = 1
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_roberta_regression_single"
            ) as data_dir:
                create_dummy_roberta_head_data(
                    data_dir, num_classes=num_classes, regression=True
                )
                preprocess_lm_data(os.path.join(data_dir, "input0"))
                train_roberta_head(
                    data_dir,
                    "roberta_base",
                    num_classes=num_classes,
                    extra_flags=["--regression-target"],
                )

    @unittest.skip("Disabled as currently broken")
    def test_roberta_regression_multiple(self):
        num_classes = 3
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_roberta_regression_multiple"
            ) as data_dir:
                create_dummy_roberta_head_data(
                    data_dir, num_classes=num_classes, regression=True
                )
                preprocess_lm_data(os.path.join(data_dir, "input0"))
                train_roberta_head(
                    data_dir,
                    "roberta_base",
                    num_classes=num_classes,
                    extra_flags=["--regression-target"],
                )

    @unittest.skip("Disabled as currently broken")
    def test_linformer_roberta_masked_lm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_linformer_roberta_mlm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_masked_lm(
                    data_dir,
                    "linformer_roberta_base",
                    extra_flags=[
                        "--user-dir",
                        "examples/linformer/linformer_src",
                        "--encoder-layers",
                        "2",
                    ],
                )

    @unittest.skip("Disabled as currently broken")
    def test_linformer_roberta_sentence_prediction(self):
        num_classes = 3
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_linformer_roberta_head") as data_dir:
                create_dummy_roberta_head_data(data_dir, num_classes=num_classes)
                preprocess_lm_data(os.path.join(data_dir, "input0"))
                preprocess_lm_data(os.path.join(data_dir, "label"))
                train_roberta_head(
                    data_dir,
                    "linformer_roberta_base",
                    num_classes=num_classes,
                    extra_flags=["--user-dir", "examples/linformer/linformer_src"],
                )

    @unittest.skip("Disabled as currently broken")
    def test_linformer_roberta_regression_single(self):
        num_classes = 1
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_linformer_roberta_regression_single"
            ) as data_dir:
                create_dummy_roberta_head_data(
                    data_dir, num_classes=num_classes, regression=True
                )
                preprocess_lm_data(os.path.join(data_dir, "input0"))
                train_roberta_head(
                    data_dir,
                    "linformer_roberta_base",
                    num_classes=num_classes,
                    extra_flags=[
                        "--regression-target",
                        "--user-dir",
                        "examples/linformer/linformer_src",
                    ],
                )

    @unittest.skip("Disabled as currently broken")
    def test_linformer_roberta_regression_multiple(self):
        num_classes = 3
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_linformer_roberta_regression_multiple"
            ) as data_dir:
                create_dummy_roberta_head_data(
                    data_dir, num_classes=num_classes, regression=True
                )
                preprocess_lm_data(os.path.join(data_dir, "input0"))
                train_roberta_head(
                    data_dir,
                    "linformer_roberta_base",
                    num_classes=num_classes,
                    extra_flags=[
                        "--regression-target",
                        "--user-dir",
                        "examples/linformer/linformer_src",
                    ],
                )

    def _test_pretrained_masked_lm_for_translation(self, learned_pos_emb, encoder_only):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_mlm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                train_legacy_masked_language_model(
                    data_dir,
                    arch="masked_lm",
                    extra_args=("--encoder-learned-pos",) if learned_pos_emb else (),
                )
                with tempfile.TemporaryDirectory(
                    "test_mlm_translation"
                ) as translation_dir:
                    create_dummy_data(translation_dir)
                    preprocess_translation_data(
                        translation_dir, extra_flags=["--joined-dictionary"]
                    )
                    # Train transformer with data_dir/checkpoint_last.pt
                    train_translation_model(
                        translation_dir,
                        arch="transformer_from_pretrained_xlm",
                        extra_flags=[
                            "--decoder-layers",
                            "1",
                            "--decoder-embed-dim",
                            "32",
                            "--decoder-attention-heads",
                            "1",
                            "--decoder-ffn-embed-dim",
                            "32",
                            "--encoder-layers",
                            "1",
                            "--encoder-embed-dim",
                            "32",
                            "--encoder-attention-heads",
                            "1",
                            "--encoder-ffn-embed-dim",
                            "32",
                            "--pretrained-xlm-checkpoint",
                            "{}/checkpoint_last.pt".format(data_dir),
                            "--activation-fn",
                            "gelu",
                            "--max-source-positions",
                            "500",
                            "--max-target-positions",
                            "500",
                        ]
                        + (
                            ["--encoder-learned-pos", "--decoder-learned-pos"]
                            if learned_pos_emb
                            else []
                        )
                        + (["--init-encoder-only"] if encoder_only else []),
                        task="translation_from_pretrained_xlm",
                    )

    def test_pretrained_masked_lm_for_translation_learned_pos_emb(self):
        self._test_pretrained_masked_lm_for_translation(True, False)

    @unittest.skip("Disabled as currently broken")
    def test_pretrained_masked_lm_for_translation_sinusoidal_pos_emb(self):
        self._test_pretrained_masked_lm_for_translation(False, False)

    @unittest.skip("Disabled as currently broken")
    def test_pretrained_masked_lm_for_translation_encoder_only(self):
        self._test_pretrained_masked_lm_for_translation(True, True)

    @unittest.skip("Disabled as currently broken")
    def test_r4f_roberta(self):
        num_classes = 3
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_r4f_roberta_head") as data_dir:
                create_dummy_roberta_head_data(data_dir, num_classes=num_classes)
                preprocess_lm_data(os.path.join(data_dir, "input0"))
                preprocess_lm_data(os.path.join(data_dir, "label"))
                train_roberta_head(
                    data_dir,
                    "roberta_base",
                    num_classes=num_classes,
                    extra_flags=[
                        "--user-dir",
                        "examples/rxf/rxf_src",
                        "--criterion",
                        "sentence_prediction_r3f",
                        "--spectral-norm-classification-head",
                    ],
                )


def train_legacy_masked_language_model(data_dir, arch, extra_args=()):
    train_parser = options.get_training_parser()
    # TODO: langs should be in and out right?
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            "cross_lingual_lm",
            data_dir,
            "--arch",
            arch,
            # Optimizer args
            "--optimizer",
            "adam",
            "--lr-scheduler",
            "reduce_lr_on_plateau",
            "--lr-shrink",
            "0.5",
            "--lr",
            "0.0001",
            "--stop-min-lr",
            "1e-09",
            # dropout, attention args
            "--dropout",
            "0.1",
            "--attention-dropout",
            "0.1",
            # MLM args
            "--criterion",
            "legacy_masked_lm_loss",
            "--masked-lm-only",
            "--monolingual-langs",
            "in,out",
            "--num-segment",
            "5",
            # Transformer args: use a small transformer model for fast training
            "--encoder-layers",
            "1",
            "--encoder-embed-dim",
            "32",
            "--encoder-attention-heads",
            "1",
            "--encoder-ffn-embed-dim",
            "32",
            # Other training args
            "--max-tokens",
            "500",
            "--tokens-per-sample",
            "500",
            "--save-dir",
            data_dir,
            "--max-epoch",
            "1",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--dataset-impl",
            "raw",
            "--num-workers",
            "0",
        ]
        + list(extra_args),
    )
    train.main(train_args)


class TestOptimizers(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_optimizers(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_optimizers") as data_dir:
                # Use just a bit of data and tiny model to keep this test runtime reasonable
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                optimizers = ["adafactor", "adam", "nag", "adagrad", "sgd", "adadelta"]
                last_checkpoint = os.path.join(data_dir, "checkpoint_last.pt")
                for optimizer in optimizers:
                    if os.path.exists(last_checkpoint):
                        os.remove(last_checkpoint)
                    train_translation_model(
                        data_dir,
                        "lstm",
                        [
                            "--required-batch-size-multiple",
                            "1",
                            "--encoder-layers",
                            "1",
                            "--encoder-hidden-size",
                            "32",
                            "--decoder-layers",
                            "1",
                            "--optimizer",
                            optimizer,
                        ],
                    )
                    generate_main(data_dir)


def read_last_log_entry(
    logs: List[logging.LogRecord], logger_name: str
) -> Dict[str, float]:
    for x in reversed(logs):
        if x.name == logger_name:
            return json.loads(x.message)
    raise ValueError(f"No entries from {logger_name} found in captured logs")


class TestActivationCheckpointing(unittest.TestCase):
    base_flags = [
        "--encoder-layers",
        "2",
        "--decoder-layers",
        "2",
        "--encoder-embed-dim",
        "8",
        "--decoder-embed-dim",
        "8",
        "--restore-file",
        "x.pt",
        "--log-format",
        "json",
        "--log-interval",
        "1",
        "--max-update",
        "2",
    ]

    def _train(self, data_dir, extra_flags):
        with self.assertLogs() as logs:
            train_translation_model(
                data_dir,
                "transformer_iwslt_de_en",
                self.base_flags + extra_flags,
                run_validation=True,
                extra_valid_flags=["--log-format", "json"],
            )
        return logs.records

    def test_activation_offloading_does_not_change_metrics(self):
        """Neither ----checkpoint-activations nor --offload-activations should change loss"""
        with tempfile.TemporaryDirectory("test_transformer_with_act_cpt") as data_dir:

            with self.assertLogs():
                create_dummy_data(data_dir, num_examples=20)
                preprocess_translation_data(data_dir)
            offload_logs = self._train(data_dir, ["--offload-activations"])
            baseline_logs = self._train(data_dir, [])

            assert len(baseline_logs) == len(offload_logs)

            baseline_valid_stats = read_last_log_entry(baseline_logs, "valid")
            offload_valid_stats = read_last_log_entry(offload_logs, "valid")
            baseline_train_stats = read_last_log_entry(baseline_logs, "train")
            offload_train_stats = read_last_log_entry(offload_logs, "train")

            assert (
                baseline_train_stats["train_loss"] == offload_train_stats["train_loss"]
            )
            assert (
                baseline_valid_stats["valid_loss"] == offload_valid_stats["valid_loss"]
            )

    def test_activation_checkpointing_does_not_change_metrics(self):
        """--checkpoint-activations should not change loss"""

        with tempfile.TemporaryDirectory("test_transformer_with_act_cpt") as data_dir:
            with self.assertLogs():
                create_dummy_data(data_dir, num_examples=20)
                preprocess_translation_data(data_dir)
            ckpt_logs = self._train(data_dir, ["--checkpoint-activations"])
            baseline_logs = self._train(data_dir, [])
            assert len(baseline_logs) == len(ckpt_logs)

            baseline_train_stats = read_last_log_entry(baseline_logs, "train")
            ckpt_train_stats = read_last_log_entry(ckpt_logs, "train")
            assert baseline_train_stats["train_loss"] == ckpt_train_stats["train_loss"]

            baseline_valid_stats = read_last_log_entry(baseline_logs, "valid")
            ckpt_valid_stats = read_last_log_entry(ckpt_logs, "valid")
            assert baseline_valid_stats["valid_loss"] == ckpt_valid_stats["valid_loss"]


def create_dummy_roberta_head_data(
    data_dir, num_examples=100, maxlen=10, num_classes=2, regression=False
):
    input_dir = "input0"

    def _create_dummy_data(filename):
        random_data = torch.rand(num_examples * maxlen)
        input_data = 97 + torch.floor(26 * random_data).int()
        if regression:
            output_data = torch.rand((num_examples, num_classes))
        else:
            output_data = 1 + torch.floor(num_classes * torch.rand(num_examples)).int()
        with open(os.path.join(data_dir, input_dir, filename + ".out"), "w") as f_in:
            label_filename = filename + ".label" if regression else filename + ".out"
            with open(os.path.join(data_dir, "label", label_filename), "w") as f_out:
                offset = 0
                for i in range(num_examples):
                    # write example input
                    ex_len = random.randint(1, maxlen)
                    ex_str = " ".join(map(chr, input_data[offset : offset + ex_len]))
                    print(ex_str, file=f_in)
                    # write example label
                    if regression:
                        class_str = " ".join(map(str, output_data[i].numpy()))
                        print(class_str, file=f_out)
                    else:
                        class_str = "class{}".format(output_data[i])
                        print(class_str, file=f_out)
                    offset += ex_len

    os.mkdir(os.path.join(data_dir, input_dir))
    os.mkdir(os.path.join(data_dir, "label"))
    _create_dummy_data("train")
    _create_dummy_data("valid")
    _create_dummy_data("test")


def train_masked_lm(data_dir, arch, extra_flags=None):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            "masked_lm",
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--criterion",
            "masked_lm",
            "--batch-size",
            "500",
            "--save-dir",
            data_dir,
            "--max-epoch",
            "1",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
        ]
        + (extra_flags or []),
    )
    train.main(train_args)


def train_roberta_head(data_dir, arch, num_classes=2, extra_flags=None):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            "sentence_prediction",
            data_dir,
            "--arch",
            arch,
            "--encoder-layers",
            "2",
            "--num-classes",
            str(num_classes),
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--criterion",
            "sentence_prediction",
            "--max-tokens",
            "500",
            "--max-positions",
            "500",
            "--batch-size",
            "500",
            "--save-dir",
            data_dir,
            "--max-epoch",
            "1",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
        ]
        + (extra_flags or []),
    )
    train.main(train_args)


def eval_lm_main(data_dir, extra_flags=None):
    eval_lm_parser = options.get_eval_lm_parser()
    eval_lm_args = options.parse_args_and_arch(
        eval_lm_parser,
        [
            data_dir,
            "--path",
            os.path.join(data_dir, "checkpoint_last.pt"),
            "--no-progress-bar",
            "--num-workers",
            "0",
        ]
        + (extra_flags or []),
    )
    eval_lm.main(eval_lm_args)


if __name__ == "__main__":
    unittest.main()
