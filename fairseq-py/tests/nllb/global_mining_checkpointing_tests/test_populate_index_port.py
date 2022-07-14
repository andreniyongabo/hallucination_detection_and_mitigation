# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import filecmp
import tempfile
import time
import unittest
from pathlib import Path

import omegaconf
import submitit

from examples.nllb.mining.global_mining.modules.mlenc_port.populate_faiss_index_module import (
    PopulateFAISSIndexConfig,
    PopulateFAISSIndexModule,
)
from examples.nllb.mining.global_mining.modules.nmt_bitext_eval_utils.preproc_binarized_mined_utils import (
    clone_config,
)
from examples.nllb.mining.nllb_lib.utils import ensure_dir
from tests.nllb.test_modules_utils import generate_embedding
from tests.nllb.test_train_index_port import generate_train_index

test_slurm_max_num_timeout = 3
test_cluster = "local"
test_timeout_min = 60
test_chkpt_embedding_length = 1310720
# Embedding is divided into chunks, each of size close to 2^14.
# So emb length of 1310720 means approx 80 checkpoints (since 1310720/(2^14) = 80).
# This gives enough time to interrupt in the midst of the job
test_sleep_time_before_interrupt = 100

populate_test_config = omegaconf.OmegaConf.create(
    {
        "_target_": "examples.nllb.mining.global_mining.modules.mlenc_port.populate_faiss_index_module.PopulateFAISSIndexModule",
        "config": {
            "lang": "bn",
            "index": "???",  # will be generated
            "index_type": "OPQ64,IVF65536,PQ64",
            "embedding_files": [],  # will be generated
            "output_dir": "???",  # will be tmp dir
            "num_cpu": 40,
            "embedding_dimensions": 1024,
        },
    }
)


class TestCheckpointingPopulateIndexPortModule(unittest.TestCase):
    def _run_populate_job(
        self,
        populate_test_config: PopulateFAISSIndexConfig,
        interrupt: bool,
        sleep_time_before_interrupt: int = test_sleep_time_before_interrupt,
    ):
        """
        This function runs a populate index module job and waits for result.
        If interrupt parameter is True, it will interrupt the job to trigger the checkpoint method
        """
        output_dir = populate_test_config.config.output_dir
        ensure_dir(output_dir)
        path_for_embedding = populate_test_config.config.embedding_files[0]

        callable_job = PopulateFAISSIndexModule(populate_test_config.config, None)
        executor = submitit.AutoExecutor(
            folder=f"{output_dir}",
            slurm_max_num_timeout=test_slurm_max_num_timeout,
            cluster=test_cluster,
        )
        executor.update_parameters(timeout_min=test_timeout_min)
        job = executor.submit(callable_job, path_for_embedding)

        if interrupt:
            # We let the unittest sleep a little to ensure the job is in the midst of running
            time.sleep(sleep_time_before_interrupt)
            self.assertEqual(job.state, "RUNNING")
            # Since now we know the job is running, we can interrupt
            job._interrupt()

        outputted_populated_idx_file = job.result()
        return outputted_populated_idx_file

    def test_generate_populated_index(self):
        """
        This test ensures that the outputs of running the job twice,
        (with and without checkpoints) is identical
        """
        with tempfile.TemporaryDirectory() as dir_name:
            # first generate an embedding and index
            embedding_outfile = Path(dir_name) / "embedding.bin"
            generate_embedding(file=embedding_outfile)
            index_out_file = generate_train_index(dir_name, True, embedding_outfile)
            populate_test_config.config.index = index_out_file

            # generate new embedding to populate onto the index
            generated_embedding_file = (
                Path(dir_name) / "generated_embedding_to_populate.bin"
            )
            generate_embedding(
                file=generated_embedding_file, emb_length=test_chkpt_embedding_length
            )
            populate_test_config.config.embedding_files.append(
                str(generated_embedding_file)
            )

            # Now run two jobs: one with and one without interruption
            # Job with an interruption
            with clone_config(populate_test_config) as config_interrupt:
                config_interrupt.config.output_dir = str(
                    Path(dir_name) / "job_one_interruption"
                )
            populated_idx_file_with_interrupt = self._run_populate_job(
                config_interrupt, True
            )

            # Job wihout interruption
            with clone_config(populate_test_config) as config_no_interrupt:
                config_no_interrupt.config.output_dir = str(
                    Path(dir_name) / "job_without_interruption"
                )
            no_interrupt_populated_idx_file = self._run_populate_job(
                config_no_interrupt, False
            )

            # If both files are identical, this indicates that the checkpointing worked
            are_files_identical = filecmp.cmp(
                populated_idx_file_with_interrupt,
                no_interrupt_populated_idx_file,
                shallow=False,
            )

            self.assertTrue(
                are_files_identical,
                f"Test failed: the populated index files produced by each job differ",
            )
