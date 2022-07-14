import contextlib
import logging
import sys
import tempfile
import unittest
from io import StringIO

from tests.utils import (
    create_dummy_data,
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

    def test_raw(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_fconv_raw") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir, ["--dataset-impl", "raw"])
                train_translation_model(
                    data_dir, "fconv_iwslt_de_en", ["--dataset-impl", "raw"]
                )
                generate_main(data_dir, ["--dataset-impl", "raw"])

    @unittest.skip("Disabled as currently flaky")
    def test_max_positions(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_max_positions") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                with self.assertRaises(Exception) as context:
                    train_translation_model(
                        data_dir,
                        "fconv_iwslt_de_en",
                        ["--max-target-positions", "5"],
                    )
                self.assertTrue(
                    "skip this example with --skip-invalid-size-inputs-valid-test"
                    in str(context.exception)
                )
                train_translation_model(
                    data_dir,
                    "fconv_iwslt_de_en",
                    [
                        "--max-target-positions",
                        "5",
                        "--skip-invalid-size-inputs-valid-test",
                    ],
                )
                with self.assertRaises(Exception) as context:
                    generate_main(data_dir)
                generate_main(data_dir, ["--skip-invalid-size-inputs-valid-test"])

    def test_generation(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_sampling") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(data_dir, "fconv_iwslt_de_en")
                generate_main(
                    data_dir,
                    [
                        "--sampling",
                        "--temperature",
                        "2",
                        "--beam",
                        "2",
                        "--nbest",
                        "2",
                    ],
                )
                generate_main(
                    data_dir,
                    [
                        "--sampling",
                        "--sampling-topk",
                        "3",
                        "--beam",
                        "2",
                        "--nbest",
                        "2",
                    ],
                )
                generate_main(
                    data_dir,
                    [
                        "--sampling",
                        "--sampling-topp",
                        "0.2",
                        "--beam",
                        "2",
                        "--nbest",
                        "2",
                    ],
                )
                generate_main(
                    data_dir,
                    [
                        "--diversity-rate",
                        "0.5",
                        "--beam",
                        "6",
                    ],
                )
                with self.assertRaises(ValueError):
                    generate_main(
                        data_dir,
                        [
                            "--diverse-beam-groups",
                            "4",
                            "--match-source-len",
                        ],
                    )
                generate_main(data_dir, ["--prefix-size", "2"])
                generate_main(data_dir, ["--retain-dropout"])

    def test_eval_bleu(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_eval_bleu") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "fconv_iwslt_de_en",
                    [
                        "--eval-bleu",
                        "--eval-bleu-print-samples",
                        "--eval-bleu-remove-bpe",
                        "--eval-bleu-detok",
                        "space",
                        "--eval-bleu-args",
                        '{"beam": 4, "min_len": 10}',
                    ],
                )

    def test_transformer(self):
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
                        "8",
                        "--decoder-embed-dim",
                        "8",
                    ],
                    run_validation=True,
                )
                generate_main(data_dir)

    def test_multilingual_transformer(self):
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
                        f"test_multilingual_transformer_{i}_{j}"
                    ) as data_dir:
                        create_dummy_data(data_dir)
                        preprocess_translation_data(data_dir)
                        train_translation_model(
                            data_dir,
                            arch="multilingual_transformer",
                            task="multilingual_translation",
                            extra_flags=[
                                "--encoder-layers",
                                "2",
                                "--decoder-layers",
                                "2",
                                "--encoder-embed-dim",
                                "8",
                                "--decoder-embed-dim",
                                "8",
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
                                "multilingual_translation",
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

    @unittest.skip("Disabled as currently broken")
    @unittest.skipIf(
        sys.platform.lower() == "darwin", "skip latent depth test on MacOS"
    )
    def test_multilingual_translation_latent_depth(self):
        # test with latent depth in encoder, decoder, or both
        encoder_latent_layer = [[], ["--encoder-latent-layer"]]
        decoder_latent_layer = [[], ["--decoder-latent-layer"]]
        with contextlib.redirect_stdout(StringIO()):
            for i in range(len(encoder_latent_layer)):
                for j in range(len(decoder_latent_layer)):
                    if i == 0 and j == 0:
                        continue
                    enc_ll_flag = encoder_latent_layer[i]
                    dec_ll_flag = decoder_latent_layer[j]
                    with tempfile.TemporaryDirectory(
                        f"test_multilingual_translation_latent_depth_{i}_{j}"
                    ) as data_dir:
                        create_dummy_data(data_dir)
                        preprocess_translation_data(
                            data_dir, extra_flags=["--joined-dictionary"]
                        )
                        train_translation_model(
                            data_dir,
                            arch="latent_multilingual_transformer",
                            task="multilingual_translation_latent_depth",
                            extra_flags=[
                                "--user-dir",
                                "examples/latent_depth/latent_depth_src",
                                "--encoder-layers",
                                "2",
                                "--decoder-layers",
                                "2",
                                "--encoder-embed-dim",
                                "8",
                                "--decoder-embed-dim",
                                "8",
                                "--share-encoders",
                                "--share-decoders",
                                "--sparsity-weight",
                                "0.1",
                            ]
                            + enc_ll_flag
                            + dec_ll_flag,
                            lang_flags=["--lang-pairs", "in-out,out-in"],
                            run_validation=True,
                            extra_valid_flags=[
                                "--user-dir",
                                "examples/latent_depth/latent_depth_src",
                            ]
                            + enc_ll_flag
                            + dec_ll_flag,
                        )
                        generate_main(
                            data_dir,
                            extra_flags=[
                                "--user-dir",
                                "examples/latent_depth/latent_depth_src",
                                "--task",
                                "multilingual_translation_latent_depth",
                                "--lang-pairs",
                                "in-out,out-in",
                                "--source-lang",
                                "in",
                                "--target-lang",
                                "out",
                            ]
                            + enc_ll_flag
                            + dec_ll_flag,
                        )

    @unittest.skip("Disabled as currently broken")
    def test_transformer_cross_self_attention(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_transformer_cross_self_attention"
            ) as data_dir:
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
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--no-cross-attention",
                        "--cross-self-attention",
                    ],
                    run_validation=True,
                )
                generate_main(data_dir, extra_flags=[])

    def test_transformer_pointer_generator(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory(
                "test_transformer_pointer_generator"
            ) as data_dir:
                create_dummy_data(data_dir)
                preprocess_summarization_data(data_dir)
                train_translation_model(
                    data_dir,
                    "transformer_pointer_generator",
                    extra_flags=[
                        "--user-dir",
                        "examples/pointer_generator/pointer_generator_src",
                        "--encoder-layers",
                        "2",
                        "--decoder-layers",
                        "2",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--alignment-layer",
                        "-1",
                        "--alignment-heads",
                        "1",
                        "--source-position-markers",
                        "0",
                    ],
                    run_validation=True,
                    extra_valid_flags=[
                        "--user-dir",
                        "examples/pointer_generator/pointer_generator_src",
                    ],
                )
                generate_main(
                    data_dir,
                    extra_flags=[
                        "--user-dir",
                        "examples/pointer_generator/pointer_generator_src",
                    ],
                )

    def test_transformer_layerdrop(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory("test_transformer_layerdrop") as data_dir:
                create_dummy_data(data_dir)
                preprocess_translation_data(data_dir)
                train_translation_model(
                    data_dir,
                    "transformer_iwslt_de_en",
                    [
                        "--encoder-layers",
                        "3",
                        "--decoder-layers",
                        "3",
                        "--encoder-embed-dim",
                        "8",
                        "--decoder-embed-dim",
                        "8",
                        "--encoder-layerdrop",
                        "0.01",
                        "--decoder-layerdrop",
                        "0.01",
                    ],
                )
                generate_main(data_dir)
                generate_main(
                    data_dir,
                    [
                        "--model-overrides",
                        "{'encoder_layers_to_keep':'0,2','decoder_layers_to_keep':'1'}",
                    ],
                )
