# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import copy
import glob
import logging
import os
import tempfile
import unittest

import numpy as np
import sentencepiece as spm

try:
    import examples.nllb.modeling.prepare_data.data_types as data_types
    from examples.nllb.modeling.prepare_data.encode_and_binarize import (
        encode_and_binarize,
        encode_spm,
    )
    from examples.nllb.modeling.prepare_data.prepare_vocab import get_vocab
    from examples.nllb.modeling.prepare_data.retrieve_data import retrieve_data
    from examples.nllb.modeling.prepare_data.sharding import (
        get_all_num_shards,
        get_num_shards,
    )
    from examples.nllb.modeling.prepare_data.utils import (
        count_lines,
        dedup_sharding,
        hash_parallel_data,
        setup_config,
    )
    from examples.nllb.modeling.prepare_data.validation import validate_data_config
    from examples.nllb.modeling.utils import execute_in_shell
except ImportError:
    raise unittest.SkipTest("Requires Hydra >= 1.1")


import hydra

from fairseq.data import Dictionary, data_utils

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_nllb")


class TestPrepareData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("common setup\n")
        hydra.initialize(config_path="nllb/baselines_conf/data_configs")
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.regular_outputdir = cls.tempdir.name
        cls.data_config = setup_config(config_name="regular.yaml", overrides=[])

        cls.loop = asyncio.get_event_loop()

        (
            cls.train_src_counts_map,
            cls.train_tgt_counts_map,
            cls.train_counts_map,
            cls.secondary_train_counts_map,
            cls.executor,
        ) = asyncio.run(
            validate_data_config(
                data_config=cls.data_config,
                output_dir=cls.regular_outputdir,
            )
        )

        # retrieve data
        retrieve_outdir = f"{cls.regular_outputdir}/retrieved_data"
        if not os.path.exists(retrieve_outdir):
            os.makedirs(retrieve_outdir, exist_ok=True)

        preprocess_config = copy.deepcopy(cls.data_config.preprocessing_config)

        preprocess_config.moses_config.normalize_punctuation = True
        (cls.train_data, _) = asyncio.run(
            retrieve_data(
                all_corpora_map=cls.data_config.train_corpora,
                output_prefix=os.path.join(retrieve_outdir, "train"),
                preprocess_config=preprocess_config,
                tag="train",
                output_dir=retrieve_outdir,
                executor=cls.executor,
            )
        )

        preprocess_config.moses_config.normalize_punctuation = False
        (cls.valid_data,) = asyncio.run(
            retrieve_data(
                all_corpora_map=cls.data_config.valid_corpora,
                output_prefix=os.path.join(retrieve_outdir, "valid"),
                preprocess_config=preprocess_config,
                tag="valid",
                output_dir=retrieve_outdir,
                executor=cls.executor,
            )
        )

        cls.test_corpora = cls.data_config.get("test_corpora", None)
        cls.test_data = (
            asyncio.run(
                retrieve_data(
                    all_corpora_map=cls.data_config.test_corpora,
                    output_prefix=os.path.join(retrieve_outdir, "test"),
                    preprocess_config=preprocess_config,
                    tag="test",
                    output_dir=retrieve_outdir,
                    executor=cls.executor,
                )
            )[0]
            if cls.test_corpora is not None
            else None
        )

        # get_vocab
        cls.src_vocab, cls.tgt_vocab = asyncio.run(
            get_vocab(
                data_config=cls.data_config,
                train_corpora=cls.train_data,
                secondary_train_corpora={},
                src_counts_map=cls.train_src_counts_map,
                tgt_counts_map=cls.train_tgt_counts_map,
                output_dir=f"{cls.regular_outputdir}/vocab_bin",
            )
        )

        cls.binarization_config = cls.data_config.binarization_config

    @classmethod
    def tearDownClass(cls):
        cls.tempdir.cleanup()
        cls.loop.close()

    def test_retrieve_data(self):
        """
        Test return values of function retrieve_data
        Returns None or raises error if fails testing
        """
        logger.info("test_retrieve_data")
        print(TestPrepareData.regular_outputdir)

        self.assertListEqual(
            list(TestPrepareData.train_data.keys()), ["en-de", "en-cs"]
        )
        self.assertEqual(
            TestPrepareData.train_data["en-de"].source,
            f"{TestPrepareData.regular_outputdir}/retrieved_data/train.en-de.en",
        )
        self.assertEqual(
            TestPrepareData.train_data["en-de"].target,
            f"{TestPrepareData.regular_outputdir}/retrieved_data/train.en-de.de",
        )

        self.assertListEqual(
            list(TestPrepareData.valid_data.keys()), ["en-de", "en-cs"]
        )
        self.assertEqual(
            TestPrepareData.valid_data["en-de"].source,
            f"{TestPrepareData.regular_outputdir}/retrieved_data/valid.en-de.en",
        )
        self.assertEqual(
            TestPrepareData.valid_data["en-de"].target,
            f"{TestPrepareData.regular_outputdir}/retrieved_data/valid.en-de.de",
        )

    def test_validate_data_config(self):
        """
        Test validate_data_config function (data_config validator)
            inconsistent direction
            nonexistent file
            unequal num of lines between source and target corpora
        Returns None or raises error if fails testing
        """
        logger.info("test_validate_data_config")

        inconsis_config = setup_config(config_name="inconsis.yaml", overrides=[])
        with self.assertRaises(AssertionError):
            asyncio.run(
                validate_data_config(
                    data_config=inconsis_config,
                    output_dir=TestPrepareData.regular_outputdir + "/inconsis",
                )
            )

        nonexistent_config = setup_config(config_name="nonexistent.yaml", overrides=[])
        with self.assertRaises(AssertionError):
            asyncio.run(
                validate_data_config(
                    data_config=nonexistent_config,
                    output_dir=TestPrepareData.regular_outputdir + "/nonexistent",
                )
            )

        difflines_config = setup_config(
            config_name="difflines.yaml",
            overrides=[],
        )
        with self.assertRaises(AssertionError):
            asyncio.run(
                validate_data_config(
                    data_config=difflines_config,
                    output_dir=TestPrepareData.regular_outputdir + "/difflines",
                )
            )

    def test_get_vocab(self):
        """
        Test get_vocab function
            pre-trained vocab exist
            train new source & target vocab
        Returns None or raises error if fails testing
        """
        logger.info("test_get_vocab\n")

        # test for pretrained vocab existed
        self.assertTrue(os.path.exists(TestPrepareData.src_vocab.model_file))
        self.assertTrue(os.path.exists(TestPrepareData.src_vocab.vocab_file))
        self.assertEqual(
            count_lines(TestPrepareData.src_vocab.vocab_file, is_gzip=False),
            TestPrepareData.data_config.source_vocab_config.vocab_build_params.vocab_size,
        )
        self.assertTrue(os.path.exists(TestPrepareData.tgt_vocab.model_file))
        self.assertTrue(os.path.exists(TestPrepareData.tgt_vocab.vocab_file))
        self.assertEqual(
            count_lines(TestPrepareData.tgt_vocab.vocab_file, is_gzip=False),
            TestPrepareData.data_config.target_vocab_config.vocab_build_params.vocab_size,
        )

        # check for training new source & target vocab
        with tempfile.TemporaryDirectory() as output_dir:
            get_vocab_config = setup_config(config_name="get_vocab.yaml", overrides=[])
            (
                train_src_counts_map,
                train_tgt_counts_map,
                train_counts_map,
                secondary_train_counts_map,
                executor,
            ) = asyncio.run(
                validate_data_config(
                    data_config=get_vocab_config,
                    output_dir=TestPrepareData.regular_outputdir,
                )
            )
            outdir = os.path.join(output_dir, "train")
            (train_data, _) = asyncio.run(
                retrieve_data(
                    all_corpora_map=get_vocab_config.train_corpora,
                    output_prefix=outdir,
                    preprocess_config=TestPrepareData.data_config.preprocessing_config,
                    tag="train",
                    output_dir=outdir,
                    executor=executor,
                )
            )
            src_vocab, tgt_vocab = asyncio.run(
                get_vocab(
                    get_vocab_config,
                    train_data,
                    {},  # secondary_train_corpora
                    train_src_counts_map,
                    train_tgt_counts_map,
                    output_dir,
                )
            )
            self.assertTrue(os.path.exists(src_vocab.model_file))
            self.assertTrue(os.path.exists(src_vocab.vocab_file))
            self.assertEqual(
                count_lines(src_vocab.vocab_file, is_gzip=False),
                get_vocab_config.source_vocab_config.vocab_build_params.vocab_size,
            )
            self.assertTrue(os.path.exists(tgt_vocab.model_file))
            self.assertTrue(os.path.exists(tgt_vocab.vocab_file))
            self.assertEqual(
                count_lines(tgt_vocab.vocab_file, is_gzip=False),
                get_vocab_config.target_vocab_config.vocab_build_params.vocab_size,
            )

    def test_get_num_shards(self):
        """
        Test get_num_shards function
            check the calculated num of shards
        Returns None or raises error if fails testing
        """
        logger.info("test_get_num_shards\n")
        smallest_shard = 16
        max_num_shards = 64
        lines_list = [4, 16, 20, 32, 63, 100000]
        correct_num_shards = [1, 1, 2, 2, 4, 64]

        for idx, num_lines in enumerate(lines_list):
            self.assertEqual(
                correct_num_shards[idx],
                get_num_shards(num_lines, smallest_shard, max_num_shards),
            )

    def test_get_all_num_shards(self):
        train_counts_map = {
            "en-cs": 100,
            "en-de": 1000,
        }
        secondary_train_counts_map = {
            "en-cs": 1000,
        }
        all_num_shards, all_num_secondary_shards = get_all_num_shards(
            train_counts_map, secondary_train_counts_map, TestPrepareData.data_config
        )
        # max_examples_per_shard = 600, smallest_shard=600
        self.assertEqual({"en-cs": 1, "en-de": 2}, all_num_shards)
        self.assertEqual({"en-cs": 2, "en-de": 0}, all_num_secondary_shards)

        train_counts_map = {
            "en-cs": 100,
            "en-de": 1000,
        }
        secondary_train_counts_map = {}
        all_num_shards, all_num_secondary_shards = get_all_num_shards(
            train_counts_map, secondary_train_counts_map, TestPrepareData.data_config
        )
        self.assertEqual({"en-cs": 1, "en-de": 2}, all_num_shards)
        self.assertEqual({"en-cs": 0, "en-de": 0}, all_num_secondary_shards)

        train_counts_map = {
            "en-cs": 100,
            "en-de": 1000,
        }
        secondary_train_counts_map = {
            "en-cs": 10000,
            "en-de": 20000,
        }
        all_num_shards, all_num_secondary_shards = get_all_num_shards(
            train_counts_map, secondary_train_counts_map, TestPrepareData.data_config
        )
        self.assertEqual({"en-cs": 1, "en-de": 2}, all_num_shards)
        self.assertEqual({"en-cs": 32, "en-de": 64}, all_num_secondary_shards)

    def test_dedup(self):
        """
        Test dedup_sharding function
            able to eliminate duplicated data appeared in valid or test dataset
        Returns None or raises error if fails testing
        """
        logger.info("test_dedup\n")

        with tempfile.TemporaryDirectory() as output_dir:

            for direction in TestPrepareData.train_data.keys():
                source, target = direction.split("-")
                train_parallel = data_types.ParallelDataset(
                    source=f"{output_dir}/test_dedup.{direction}.{source}",
                    target=f"{output_dir}/test_dedup.{direction}.{target}",
                )

                # make duplicaate lines
                execute_in_shell(
                    f"head -n 10 {TestPrepareData.train_data[direction].source} >> {train_parallel.source}"
                )
                execute_in_shell(
                    f"head -n 10 {TestPrepareData.train_data[direction].target} >> {train_parallel.target}"
                )
                distinct_lines = count_lines(
                    train_parallel.source, train_parallel.is_gzip
                )

                execute_in_shell(
                    f"head -n 1 {TestPrepareData.valid_data[direction].source} >> {train_parallel.source}"
                )
                execute_in_shell(
                    f"head -n 1 {TestPrepareData.valid_data[direction].target} >> {train_parallel.target}"
                )

                # dedup
                seen_valid = hash_parallel_data(TestPrepareData.valid_data[direction])
                seen_test = (
                    hash_parallel_data(TestPrepareData.test_data[direction])
                    if TestPrepareData.test_data
                    else set()
                )
                seen = seen_valid.union(seen_test)

                deduped_train = dedup_sharding(
                    direction=direction,
                    train_parallel=train_parallel,
                    seen=seen,
                    num_shards=1,
                    binarization_config=TestPrepareData.binarization_config,
                    sharding_output_dir=f"{output_dir}/sharded_bin",
                    output_dir=output_dir,
                    custom_step_name=f"dedup_sharding.{direction}",
                )

                deduped_src_lines = count_lines(
                    deduped_train[0].source, deduped_train[0].is_gzip
                )
                deduped_tgt_lines = count_lines(
                    deduped_train[0].target, deduped_train[0].is_gzip
                )
                self.assertEqual(deduped_src_lines, deduped_tgt_lines)
                self.assertEqual(distinct_lines, deduped_src_lines)

    def test_sharding(self):
        """
        Test dedup_sharding function
            able to correctly split data into shards without changing content
        Returns None or raises error if fails testing
        """
        logger.info("test_sharding\n")

        with tempfile.TemporaryDirectory() as output_dir:
            num_lines = 10
            num_shards = 2
            for idx, direction in enumerate(TestPrepareData.train_data.keys()):
                src, tgt = direction.split("-")

                train_parallel = data_types.ParallelDataset(
                    source=f"{output_dir}/sharded_bin.{direction}.{src}",
                    target=f"{output_dir}/sharded_bin.{direction}.{tgt}",
                )
                execute_in_shell(
                    f"head -n {num_lines} {TestPrepareData.train_data[direction].source} >> {train_parallel.source}"
                )
                execute_in_shell(
                    f"head -n {num_lines} {TestPrepareData.train_data[direction].target} >> {train_parallel.target}"
                )

                deduped_train = dedup_sharding(
                    direction=direction,
                    train_parallel=train_parallel,
                    seen=set(),
                    num_shards=num_shards,
                    binarization_config=TestPrepareData.binarization_config,
                    sharding_output_dir=f"{output_dir}/sharded_bin",
                    output_dir=output_dir,
                    custom_step_name=f"test_sharding.{direction}",
                )

                # check # of files
                src_shard_files = glob.glob(
                    f"{output_dir}/sharded_bin/shard*/train.shard*.{direction}.{src}"
                )
                tgt_shard_files = glob.glob(
                    f"{output_dir}/sharded_bin/shard*/train.shard*.{direction}.{tgt}"
                )
                self.assertEqual(len(src_shard_files), len(tgt_shard_files))
                self.assertEqual(len(src_shard_files), num_shards)

                # check total # of lines
                sharded_lines = sum(
                    [count_lines(fi, is_gzip=False) for fi in src_shard_files]
                )
                self.assertEqual(sharded_lines, num_lines)

            # check # of shard dirs
            num_shard_dirs = len(next(os.walk(f"{output_dir}/sharded_bin"))[1])
            self.assertEqual(num_shard_dirs, num_shards)

    def test_encode_spm(self):
        """
        Test encode_spm function
            successful encoding
            keep content unchanged after decoding encoded files
        Returns None or raises error if fails testing
        """

        logger.info("test_encode_spm\n")

        with tempfile.TemporaryDirectory() as output_dir:

            data_list = list(TestPrepareData.train_data.values())
            random_data_list = np.random.choice(data_list, 2, replace=False)

            src_sp = spm.SentencePieceProcessor()
            src_sp.Load(TestPrepareData.src_vocab.model_file)
            tgt_sp = spm.SentencePieceProcessor()
            tgt_sp.Load(TestPrepareData.tgt_vocab.model_file)

            for idx, parallel_data in enumerate(random_data_list):
                test_parallel = data_types.ParallelDataset(
                    source=f"{output_dir}/test_encode_spm_{idx}.source",
                    target=f"{output_dir}/test_encode_spm_{idx}.target",
                )

                execute_in_shell(
                    f"head -n 2 {parallel_data.source} > {test_parallel.source}"
                )
                execute_in_shell(
                    f"head -n 2 {parallel_data.target} > {test_parallel.target}"
                )
                # encode
                src_encoded = encode_spm(
                    test_parallel.source,
                    TestPrepareData.src_vocab,
                    output_dir,
                )
                tgt_encoded = encode_spm(
                    test_parallel.target,
                    TestPrepareData.tgt_vocab,
                    output_dir,
                )

                # decode
                with open(test_parallel.source) as raw_src_f, open(
                    test_parallel.target
                ) as raw_tgt_f, open(src_encoded) as encoded_src_f, open(
                    tgt_encoded
                ) as encoded_tgt_f:
                    for (
                        raw_src_line,
                        raw_tgt_line,
                        encoded_src_line,
                        encoded_tgt_line,
                    ) in zip(raw_src_f, raw_tgt_f, encoded_src_f, encoded_tgt_f):
                        decode_src_line = f"{src_sp.DecodePieces(encoded_src_line.strip().split(' '))}"
                        self.assertTrue(decode_src_line == raw_src_line.strip())

                        decode_tgt_line = f"{src_sp.DecodePieces(encoded_tgt_line.strip().split(' '))}"
                        self.assertTrue(decode_tgt_line == raw_tgt_line.strip())

    def test_binarize(self):
        """
        Test binarize function
            successfully binarize data
            keep content unchanged after load binarized files
        """

        logger.info("test_encode_spm\n")

        with tempfile.TemporaryDirectory() as output_dir:
            for direction in TestPrepareData.train_data.keys():
                src, tgt = direction.split("-")

                train_parallel = data_types.ParallelDataset(
                    source=f"{output_dir}/test_binarize.{direction}.{src}",
                    target=f"{output_dir}/test_binarize.{direction}.{tgt}",
                )
                execute_in_shell(
                    f"head -n 2 {TestPrepareData.train_data[direction].source} > {train_parallel.source}"
                )
                execute_in_shell(
                    f"head -n 2 {TestPrepareData.train_data[direction].target} > {train_parallel.target}"
                )
                encode_and_binarize(
                    direction=direction,
                    parallel_data=train_parallel,
                    tag="train",
                    src_vocab=TestPrepareData.src_vocab,
                    tgt_vocab=TestPrepareData.tgt_vocab,
                    binarize_workers=TestPrepareData.binarization_config.binarize_workers,
                    output_dir=output_dir,
                    encoded_outdir=f"{output_dir}/encoded_bin",
                    binarized_outdir=f"{output_dir}/binarized_bin",
                    shard_id=0,
                    custom_step_name="test_binarize0",
                )

                encoded_data = data_types.ParallelDataset(
                    source=f"{output_dir}/encoded_bin/spm.test_binarize.{direction}.{src}",
                    target=f"{output_dir}/encoded_bin/spm.test_binarize.{direction}.{tgt}",
                )

                src_dic = Dictionary.load(f"{output_dir}/binarized_bin/dict.{src}.txt")
                tgt_dic = Dictionary.load(f"{output_dir}/binarized_bin/dict.{tgt}.txt")

                binarized_data = data_types.ParallelDataset(
                    source=f"{output_dir}/binarized_bin/train.{direction}.{src}",
                    target=f"{output_dir}/binarized_bin/train.{direction}.{tgt}",
                )

                dataset = data_utils.load_indexed_dataset(
                    binarized_data.source, src_dic, dataset_impl=None
                )
                with open(train_parallel.source) as origin_f, open(
                    encoded_data.source
                ) as encoded_f:
                    encoded_lines = encoded_f.readlines()
                    origin_lines = origin_f.readlines()
                    for idx, tensor_line in enumerate(dataset):
                        loaded_line = src_dic.string(tensor_line)
                        encoded_line = encoded_lines[idx].strip()
                        self.assertEqual(encoded_line, loaded_line)

                        origin_line = origin_lines[idx].strip()
                        post_line = data_utils.post_process(
                            loaded_line, "sentencepiece"
                        )
                        self.assertEqual(post_line, origin_line)

                dataset = data_utils.load_indexed_dataset(
                    binarized_data.target, tgt_dic, dataset_impl=None
                )
                with open(train_parallel.target) as origin_f, open(
                    encoded_data.target
                ) as encoded_f:
                    encoded_lines = encoded_f.readlines()
                    origin_lines = origin_f.readlines()
                    for idx, tensor_line in enumerate(dataset):
                        loaded_line = tgt_dic.string(tensor_line)
                        encoded_line = encoded_lines[idx].strip()
                        self.assertEqual(encoded_line, loaded_line)

                        origin_line = origin_lines[idx].strip()
                        post_line = data_utils.post_process(
                            loaded_line, "sentencepiece"
                        )
                        self.assertEqual(post_line, origin_line.strip())


if __name__ == "__main__":
    unittest.main()
