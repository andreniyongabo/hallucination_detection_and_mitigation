# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import tempfile
import unittest
from pathlib import Path

import faiss

from examples.nllb.mining.global_mining.modules.indexing.train_index import train_index
from tests.nllb.test_modules_utils import (
    generate_embedding,
    test_dim,
    test_dtype,
    test_idx_type,
    test_lang,
)


def generate_train_index(
    dir_name: str,
    use_gpu: bool,
    embedding_outfile: Path,
):
    index_out_file = os.path.abspath(
        os.path.join(
            dir_name,
            f"{test_idx_type}.{test_lang}.train.idx",
        )
    )

    returned_index = train_index(
        embedding_outfile, test_idx_type, test_dim, use_gpu, test_dtype
    )

    if use_gpu:
        returned_index = faiss.index_gpu_to_cpu(returned_index)
    faiss.write_index(returned_index, str(index_out_file))

    return index_out_file


class TestTrainIndexPortModule(unittest.TestCase):
    def test_generate_index(self):
        with tempfile.TemporaryDirectory() as dir_name:
            embedding_outfile = os.path.join(dir_name, "embedding.bin")
            generate_embedding(file=embedding_outfile)

            with self.subTest("Test Case: used_gpu = True"):
                index_out_file = generate_train_index(dir_name, True, embedding_outfile)
                self.assertTrue(
                    os.path.exists(index_out_file),
                    f"index file {index_out_file} missing",
                )

            with self.subTest("Test Case: used_gpu = False"):
                index_out_file = generate_train_index(
                    dir_name, False, embedding_outfile
                )
                self.assertTrue(
                    os.path.exists(index_out_file),
                    f"index file {index_out_file} is missing",
                )
