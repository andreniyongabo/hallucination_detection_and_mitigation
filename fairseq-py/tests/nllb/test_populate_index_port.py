# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import shutil
import tempfile
import unittest
from pathlib import Path

import faiss

from examples.nllb.mining.global_mining.embedding_utils import Embedding
from examples.nllb.mining.global_mining.modules.mlenc_port.populate_faiss_index_utils import (
    add_emb_to_index,
)
from tests.nllb.test_modules_utils import (
    generate_embedding,
    test_dim,
    test_dtype,
    test_idx_type,
    test_lang,
)
from tests.nllb.test_train_index_port import generate_train_index

test_iteration_index = 0


def generate_populated_index(
    embedding_outfile: Path, trained_index_out_file: Path, dir_name: str
) -> Path:
    populated_index = os.path.join(
        dir_name,
        f"populate_index.{test_idx_type}.{test_lang}.{test_iteration_index}.data.idx",
    )
    shutil.copyfile(trained_index_out_file, populated_index)

    index_loaded = faiss.read_index(trained_index_out_file)
    returned_index = add_emb_to_index(
        index_loaded, embedding_outfile, test_dim, test_dtype
    )

    faiss.write_index(returned_index, str(populated_index))

    return populated_index


class TestPopulateIndexPortModule(unittest.TestCase):
    def test_generate_populated_index(self):
        use_gpu = True
        with tempfile.TemporaryDirectory() as dir_name:
            embedding_outfile = os.path.join(dir_name, "embedding.bin")
            generate_embedding(file=embedding_outfile)

            trained_index_out_file = generate_train_index(
                dir_name, use_gpu, embedding_outfile
            )
            self.assertTrue(
                os.path.exists(trained_index_out_file),
                f"index file {trained_index_out_file} missing",
            )

            populated_index = generate_populated_index(
                embedding_outfile, trained_index_out_file, dir_name
            )
            self.assertTrue(
                os.path.exists(populated_index),
                f"populated index file {populated_index} missing",
            )

            read_populated_index = faiss.read_index(populated_index)
            nbex = len(Embedding(embedding_outfile, test_dim))

            self.assertEqual(
                read_populated_index.ntotal,
                nbex,
                f"expected {nbex} sentences in index, only found {read_populated_index.ntotal}.",
            )
