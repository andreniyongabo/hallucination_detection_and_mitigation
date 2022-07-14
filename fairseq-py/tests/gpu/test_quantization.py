import contextlib
import logging
import os
import tempfile
import unittest
from io import StringIO

import torch

from fairseq import options
from fairseq_cli import train
from tests.utils import create_dummy_data, preprocess_lm_data


@unittest.skip("Disabled as currently broken")
@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestQuantization(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_quantization(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_quantization") as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                # tests both scalar and iterative PQ quantization
                _quantize_language_model(data_dir, "transformer_lm")


def _quantize_language_model(data_dir, arch, extra_flags=None):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            "--task",
            "language_modeling",
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--criterion",
            "adaptive_loss",
            "--adaptive-softmax-cutoff",
            "5,10,15",
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
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
        ]
        + (extra_flags or []),
    )
    train.main(train_args)

    # try scalar quantization
    scalar_quant_train_parser = options.get_training_parser()
    scalar_quant_train_args = options.parse_args_and_arch(
        scalar_quant_train_parser,
        [
            "--task",
            "language_modeling",
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--criterion",
            "adaptive_loss",
            "--adaptive-softmax-cutoff",
            "5,10,15",
            "--max-tokens",
            "500",
            "--tokens-per-sample",
            "500",
            "--save-dir",
            data_dir,
            "--max-update",
            "3",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
            "--quant-noise-scalar",
            "0.5",
        ]
        + (extra_flags or []),
    )
    train.main(scalar_quant_train_args)

    # try iterative PQ quantization
    quantize_parser = options.get_training_parser()
    quantize_args = options.parse_args_and_arch(
        quantize_parser,
        [
            "--task",
            "language_modeling",
            data_dir,
            "--arch",
            arch,
            "--optimizer",
            "adam",
            "--lr",
            "0.0001",
            "--criterion",
            "adaptive_loss",
            "--adaptive-softmax-cutoff",
            "5,10,15",
            "--max-tokens",
            "50",
            "--tokens-per-sample",
            "50",
            "--max-update",
            "6",
            "--no-progress-bar",
            "--distributed-world-size",
            "1",
            "--ddp-backend",
            "no_c10d",
            "--num-workers",
            "0",
            "--restore-file",
            os.path.join(data_dir, "checkpoint_last.pt"),
            "--reset-optimizer",
            "--quantization-config-path",
            os.path.join(
                os.path.dirname(__file__), "transformer_quantization_config.yaml"
            ),
        ]
        + (extra_flags or []),
    )
    train.main(quantize_args)
