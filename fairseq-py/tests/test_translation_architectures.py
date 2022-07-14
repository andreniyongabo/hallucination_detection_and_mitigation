import contextlib
import logging
import tempfile
import unittest
from io import StringIO

from tests.utils import (
    create_dummy_data,
    create_laser_data_and_config_json,
    generate_main,
    preprocess_translation_data,
    train_translation_model,
)


class TestTranslationArchitectures(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_fconv(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fconv") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, "fconv_iwslt_de_en")
                generate_main(data_dir)

    def test_lstm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_lstm") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "lstm_wiseman_iwslt_de_en",
                    [
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--decoder-out-embed-dim",
                        "8",
                    ],
                )
                generate_main(data_dir)

    def test_lstm_bidirectional(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_lstm_bidirectional") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "lstm",
                    [
                        "--encoder-layers",
                        "2",
                        "--encoder-bidirectional",
                        "--encoder-hidden-size",
                        "16",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--decoder-out-embed-dim",
                        "8",
                        "--decoder-layers",
                        "2",
                    ],
                )
                generate_main(data_dir)

    def test_lightconv(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_lightconv") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "lightconv_iwslt_de_en",
                    [
                        "--encoder-conv-type",
                        "lightweight",
                        "--decoder-conv-type",
                        "lightweight",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                    ],
                )
                generate_main(data_dir)

    def test_dynamicconv(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_dynamicconv") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "lightconv_iwslt_de_en",
                    [
                        "--encoder-conv-type",
                        "dynamic",
                        "--decoder-conv-type",
                        "dynamic",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                    ],
                )
                generate_main(data_dir)

    def test_cmlm_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_cmlm_transformer") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, ["--joined-dictionary"])
                train_translation_model(
                    data_dir,
                    "cmlm_transformer",
                    [
                        "--apply-bert-init",
                        "--criterion",
                        "nat_loss",
                        "--noise",
                        "full_mask",
                        "--pred-length-offset",
                        "--length-loss-factor",
                        "0.1",
                    ],
                    task="translation_lev",
                )
                generate_main(
                    data_dir,
                    [
                        "--task",
                        "translation_lev",
                        "--iter-decode-max-iter",
                        "9",
                        "--iter-decode-eos-penalty",
                        "0",
                        "--print-step",
                    ],
                )

    def test_nonautoregressive_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_nonautoregressive_transformer"
            ) as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, ["--joined-dictionary"])
                train_translation_model(
                    data_dir,
                    "nonautoregressive_transformer",
                    [
                        "--apply-bert-init",
                        "--src-embedding-copy",
                        "--criterion",
                        "nat_loss",
                        "--noise",
                        "full_mask",
                        "--pred-length-offset",
                        "--length-loss-factor",
                        "0.1",
                    ],
                    task="translation_lev",
                )
                generate_main(
                    data_dir,
                    [
                        "--task",
                        "translation_lev",
                        "--iter-decode-max-iter",
                        "0",
                        "--iter-decode-eos-penalty",
                        "0",
                        "--print-step",
                    ],
                )

    def test_iterative_nonautoregressive_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_iterative_nonautoregressive_transformer"
            ) as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, ["--joined-dictionary"])
                train_translation_model(
                    data_dir,
                    "iterative_nonautoregressive_transformer",
                    [
                        "--apply-bert-init",
                        "--src-embedding-copy",
                        "--criterion",
                        "nat_loss",
                        "--noise",
                        "full_mask",
                        "--stochastic-approx",
                        "--dae-ratio",
                        "0.5",
                        "--train-step",
                        "3",
                    ],
                    task="translation_lev",
                )
                generate_main(
                    data_dir,
                    [
                        "--task",
                        "translation_lev",
                        "--iter-decode-max-iter",
                        "9",
                        "--iter-decode-eos-penalty",
                        "0",
                        "--print-step",
                    ],
                )

    @unittest.skip("Disabled as currently broken")
    def test_insertion_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_insertion_transformer") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, ["--joined-dictionary"])
                train_translation_model(
                    data_dir,
                    "insertion_transformer",
                    [
                        "--apply-bert-init",
                        "--criterion",
                        "nat_loss",
                        "--noise",
                        "random_mask",
                    ],
                    task="translation_lev",
                )
                generate_main(
                    data_dir,
                    [
                        "--task",
                        "translation_lev",
                        "--iter-decode-max-iter",
                        "9",
                        "--iter-decode-eos-penalty",
                        "0",
                        "--print-step",
                    ],
                )

    def test_mixture_of_experts(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_moe") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "transformer_iwslt_de_en",
                    [
                        "--task",
                        "translation_moe",
                        "--user-dir",
                        "examples/translation_moe/translation_moe_src",
                        "--method",
                        "hMoElp",
                        "--mean-pool-gating-network",
                        "--num-experts",
                        "3",
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                    ],
                )
                generate_main(
                    data_dir,
                    [
                        "--task",
                        "translation_moe",
                        "--user-dir",
                        "examples/translation_moe/translation_moe_src",
                        "--method",
                        "hMoElp",
                        "--mean-pool-gating-network",
                        "--num-experts",
                        "3",
                        "--gen-expert",
                        "0",
                    ],
                )

    def test_alignment(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_alignment") as data_dir:
                create_dummy_data(data_dir, alignment=True)
                preprocess_translation_data(data_dir, ["--align-suffix", "align"])
                train_translation_model(
                    data_dir,
                    "transformer_align",
                    [
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--load-alignments",
                        "--alignment-layer",
                        "1",
                        "--criterion",
                        "label_smoothed_cross_entropy_with_alignment",
                    ],
                    run_validation=True,
                )
                generate_main(data_dir)

    def test_laser_lstm(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_laser_lstm") as data_dir:
                laser_config_file = create_laser_data_and_config_json(data_dir)
                train_translation_model(
                    laser_config_file.name,
                    "laser_lstm",
                    [
                        "--user-dir",
                        "examples/laser/laser_src",
                        "--weighting-alpha",
                        "0.3",
                        "--encoder-bidirectional",
                        "--encoder-hidden-size",
                        "512",
                        "--encoder-layers",
                        "5",
                        "--decoder-layers",
                        "1",
                        "--encoder-embed-dim",
                        "320",
                        "--decoder-embed-dim",
                        "320",
                        "--decoder-lang-embed-dim",
                        "32",
                        "--save-dir",
                        data_dir,
                        "--disable-validation",
                    ],
                    task="laser",
                    lang_flags=[],
                )

    def test_laser_transformer(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_laser_transformer") as data_dir:
                laser_config_file = create_laser_data_and_config_json(data_dir)
                train_translation_model(
                    laser_config_file.name,
                    "laser_transformer",
                    [
                        "--user-dir",
                        "examples/laser/laser_src",
                        "--sentemb-criterion",
                        "maxpool",
                        "--weighting-alpha",
                        "0.3",
                        "--encoder-embed-dim",
                        "320",
                        "--decoder-embed-dim",
                        "320",
                        "--decoder-lang-embed-dim",
                        "32",
                        "--save-dir",
                        data_dir,
                        "--batch-size",
                        "8",
                        "--disable-validation",
                    ],
                    task="laser",
                    lang_flags=[],
                )

    def test_translation_distillation(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_translation_distillation"
            ) as data_dir:
                config_file = create_laser_data_and_config_json(data_dir)
                train_translation_model(
                    config_file.name,
                    "laser_lstm",
                    [
                        "--user-dir",
                        "examples/laser/laser_src",
                        "--weighting-alpha",
                        "0.3",
                        "--encoder-bidirectional",
                        "--encoder-hidden-size",
                        "512",
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "1",
                        "--encoder-embed-dim",
                        "320",
                        "--decoder-embed-dim",
                        "320",
                        "--decoder-lang-embed-dim",
                        "32",
                        "--save-dir",
                        f"{data_dir}/teacher",
                        "--disable-validation",
                    ],
                    task="laser",
                    lang_flags=[],
                )
                train_translation_model(
                    config_file.name,
                    "laser_transformer",
                    [
                        "--user-dir",
                        "examples/translation_distillation/translation_distillation_src",
                        "--criterion",
                        "encoder_similarity",
                        "--sentemb-criterion",
                        "maxpool",
                        "--weighting-alpha",
                        "0.3",
                        "--encoder-embed-dim",
                        "1024",
                        "--encoder-attention-heads",
                        "4",
                        "--decoder-embed-dim",
                        "1",
                        "--decoder-layers",
                        "1",
                        "--decoder-attention-heads",
                        "1",
                        "--save-dir",
                        data_dir,
                        "--batch-size",
                        "8",
                        "--student-teacher-config",
                        "self:fr-en,distil:zh-en",
                        "--teacher-checkpoint-path",
                        f"{data_dir}/teacher/checkpoint1.pt",
                        "--lambda-self",
                        "1",
                        "--lambda-mask",
                        "1",
                        "--lambda-distil",
                        "1",
                        "--disable-validation",
                    ],
                    task="translation_distillation",
                    lang_flags=[],
                )

    def test_alignment_full_context(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_alignment") as data_dir:
                create_dummy_data(data_dir, alignment=True)
                preprocess_translation_data(data_dir, ["--align-suffix", "align"])
                train_translation_model(
                    data_dir,
                    "transformer_align",
                    [
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--load-alignments",
                        "--alignment-layer",
                        "1",
                        "--criterion",
                        "label_smoothed_cross_entropy_with_alignment",
                        "--full-context-alignment",
                    ],
                    run_validation=True,
                )
                generate_main(data_dir)
