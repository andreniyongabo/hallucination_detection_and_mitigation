# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import numpy as np

from examples.nllb.mining.global_mining.embedding_utils import Embedding
from tests.nllb.test_modules_utils import generate_embedding, test_dim, test_dtype


class TestEmbedding(unittest.TestCase):
    def test_len(self):
        with tempfile.TemporaryDirectory() as dir_name:
            outfile = os.path.join(dir_name, "embedding.bin")
            test_data = generate_embedding(file=outfile)
            emb = Embedding(
                outfile,
                test_dim,
                dtype=test_dtype,
            )
            self.assertEqual(len(emb), test_data.shape[0])

    def test_read_mmap(self):
        with tempfile.TemporaryDirectory() as dir_name:
            outfile = os.path.join(dir_name, "embedding.bin")
            test_data = generate_embedding(file=outfile)
            emb = Embedding(
                outfile,
                test_dim,
                dtype=test_dtype,
            )
            with emb.open_for_read() as npy_array:
                self.assertTrue(np.array_equal(npy_array, test_data))

    def test_read_memory(self):
        with tempfile.TemporaryDirectory() as dir_name:
            outfile = os.path.join(dir_name, "embedding.bin")
            test_data = generate_embedding(file=outfile)
            emb = Embedding(
                outfile,
                test_dim,
                dtype=test_dtype,
            )
            with emb.open_for_read(mode="memory") as npy_array:
                self.assertTrue(np.array_equal(npy_array, test_data))
