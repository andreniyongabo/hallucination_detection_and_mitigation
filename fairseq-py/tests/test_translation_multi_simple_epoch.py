import contextlib
import logging
import sys
import tempfile
import unittest
from io import StringIO

from fairseq import options
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as distributed_utils
from fairseq_cli import eval_lm, train
from tests.utils import (
    create_dummy_data,
    create_laser_data_and_config_json,
    generate_main,
    preprocess_summarization_data,
    preprocess_translation_data,
    train_translation_model,
)


class TestTranslation(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_translation_multi_simple_epoch(self):
        # test with all combinations of encoder/decoder lang tokens
        encoder_langtok_flags = [
            [],
            ["--encoder-langtok", "src"],
            ["--encoder-langtok", "tgt"],
        ]
        decoder_langtok_flags = [[], ["--decoder-langtok"]]
        with contextlib.redirect_stdout(StringIO()):
            for i in range(len(encoder_langtok_flags)):
                for j in range(len(decoder_langtok_flags)):
                    enc_ltok_flag = encoder_langtok_flags[i]
                    dec_ltok_flag = decoder_langtok_flags[j]
                    with tempfile.TemporaryDirectory(
                        f"test_translation_multi_simple_epoch_{i}_{j}"
                    ) as data_dir:
                        create_dummy_data(data_dir)
                        preprocess_translation_data(
                            data_dir, extra_flags=["--joined-dictionary"]
                        )
                        train_translation_model(
                            data_dir,
                            arch="transformer",
                            task="translation_multi_simple_epoch",
                            extra_flags=[
                                "--encoder-layers",
                                "2",
                                "--decoder-layers",
                                "2",
                                "--encoder-embed-dim",
                                "8",
                                "--decoder-embed-dim",
                                "8",
                                "--sampling-method",
                                "temperature",
                                "--sampling-temperature",
                                "1.5",
                                "--virtual-epoch-size",
                                "1000",
                            ]
                            + enc_ltok_flag
                            + dec_ltok_flag,
                            lang_flags=["--lang-pairs", "in-out,out-in"],
                            run_validation=True,
                            extra_valid_flags=enc_ltok_flag + dec_ltok_flag,
                        )
                        generate_main(
                            data_dir,
                            extra_flags=[
                                "--task",
                                "translation_multi_simple_epoch",
                                "--lang-pairs",
                                "in-out,out-in",
                                "--source-lang",
                                "in",
                                "--target-lang",
                                "out",
                            ]
                            + enc_ltok_flag
                            + dec_ltok_flag,
                        )

    def test_translation_multi_simple_epoch_no_vepoch(self):
        # test with all combinations of encoder/decoder lang tokens
        with contextlib.redirect_stdout(StringIO()):
            enc_ltok_flag = ["--encoder-langtok", "src"]
            dec_ltok_flag = ["--decoder-langtok"]
            with tempfile.TemporaryDirectory(
                "test_translation_multi_simple_epoch_dict"
            ) as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, extra_flags=[])
                train_translation_model(
                    data_dir,
                    arch="transformer",
                    task="translation_multi_simple_epoch",
                    extra_flags=[
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--sampling-method",
                        "temperature",
                        "--sampling-temperature",
                        "1.5",
                    ]
                    + enc_ltok_flag
                    + dec_ltok_flag,
                    lang_flags=["--lang-pairs", "in-out"],
                    run_validation=True,
                    extra_valid_flags=enc_ltok_flag + dec_ltok_flag,
                )
                generate_main(
                    data_dir,
                    extra_flags=[
                        "--task",
                        "translation_multi_simple_epoch",
                        "--lang-pairs",
                        "in-out",
                        "--source-lang",
                        "in",
                        "--target-lang",
                        "out",
                    ]
                    + enc_ltok_flag
                    + dec_ltok_flag,
                )

    def test_translation_multi_simple_epoch_dicts(self):
        # test with all combinations of encoder/decoder lang tokens
        with contextlib.redirect_stdout(StringIO()):
            enc_ltok_flag = ["--encoder-langtok", "src"]
            dec_ltok_flag = ["--decoder-langtok"]
            with tempfile.TemporaryDirectory(
                "test_translation_multi_simple_epoch_dict"
            ) as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, extra_flags=[])
                train_translation_model(
                    data_dir,
                    arch="transformer",
                    task="translation_multi_simple_epoch",
                    extra_flags=[
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--sampling-method",
                        "temperature",
                        "--sampling-temperature",
                        "1.5",
                        "--virtual-epoch-size",
                        "1000",
                    ]
                    + enc_ltok_flag
                    + dec_ltok_flag,
                    lang_flags=["--lang-pairs", "in-out"],
                    run_validation=True,
                    extra_valid_flags=enc_ltok_flag + dec_ltok_flag,
                )
                generate_main(
                    data_dir,
                    extra_flags=[
                        "--task",
                        "translation_multi_simple_epoch",
                        "--lang-pairs",
                        "in-out",
                        "--source-lang",
                        "in",
                        "--target-lang",
                        "out",
                    ]
                    + enc_ltok_flag
                    + dec_ltok_flag,
                )

    def test_translation_multi_simple_epoch_src_tgt_dict_spec(self):
        # test the specification of explicit --src-dict and --tgt-dict
        with contextlib.redirect_stdout(StringIO()):
            enc_ltok_flag = ["--encoder-langtok", "src"]
            dec_ltok_flag = ["--decoder-langtok"]
            with tempfile.TemporaryDirectory(
                "test_translation_multi_simple_epoch_dict"
            ) as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, extra_flags=[])
                train_translation_model(
                    data_dir,
                    arch="transformer",
                    task="translation_multi_simple_epoch",
                    extra_flags=[
                        "--source-dict",
                        f"{data_dir}/dict.in.txt",
                        "--target-dict",
                        f"{data_dir}/dict.out.txt",
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--sampling-method",
                        "temperature",
                        "--sampling-temperature",
                        "1.5",
                        "--virtual-epoch-size",
                        "1000",
                    ]
                    + enc_ltok_flag
                    + dec_ltok_flag,
                    lang_flags=["--lang-pairs", "in-out"],
                    run_validation=True,
                    extra_valid_flags=enc_ltok_flag + dec_ltok_flag,
                )
                generate_main(
                    data_dir,
                    extra_flags=[
                        "--task",
                        "translation_multi_simple_epoch",
                        "--lang-pairs",
                        "in-out",
                        "--source-lang",
                        "in",
                        "--target-lang",
                        "out",
                    ]
                    + enc_ltok_flag
                    + dec_ltok_flag,
                )
