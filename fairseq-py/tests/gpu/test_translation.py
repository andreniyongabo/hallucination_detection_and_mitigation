import contextlib
import os
import tempfile
import unittest
from io import StringIO

import torch

from tests.utils import (
    create_dummy_data,
    generate_main,
    preprocess_translation_data,
    train_translation_model,
)


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestTranslation(unittest.TestCase):
    def test_fp16_multigpu_dense_translation(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fp16") as data_dir:
                log = os.path.join(data_dir, "train.log")
                create_dummy_data(data_dir, num_examples=100)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "fconv_iwslt_de_en",
                    ["--fp16", "--log-file", log],
                    world_size=min(torch.cuda.device_count(), 2),
                )
                generate_main(data_dir)
                assert os.path.exists(log)

    def test_memory_efficient_fp16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_memory_efficient_fp16") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir, "fconv_iwslt_de_en", ["--memory-efficient-fp16"]
                )
                generate_main(data_dir)

    def test_transformer_fp16(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "transformer_iwslt_de_en",
                    [
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "64",
                        "--decoder-embed-dim",
                        "64",
                        "--fp16",
                    ],
                    run_validation=True,
                )
                generate_main(data_dir)
