import unittest

import torch

from tests.gpu.gpu_test_mixin import DEVICE_COUNT, LMTestMixin


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestMoeLM(LMTestMixin):
    def test_resume_training_moe_noc10d(self):
        self._test_resume_training(
            self.moe_clargs_1_expert_per_gpu_clargs + ["--fp16-no-flatten-grads"]
        )

    def test_resume_training_moe_fsdp_normal(self):
        self._test_resume_training(
            self.moe_clargs_1_expert_per_gpu_clargs
            + [
                "--ddp-backend",
                "fully_sharded",
                "--scale-heads",
                "--scale-attn",
                "--scale-fc",
            ]
        )

    def test_resume_training_moe_fsdp_sharded(self):
        self._test_resume_training(
            self.moe_clargs_1_expert_per_gpu_clargs
            + ["--ddp-backend", "fully_sharded", "--use-sharded-state"]
        )

    # Replicated Experts
    def test_resume_training_moe_noc10d_replication_raises(self):
        # Feel free to delete this if you fix the bug (loss should be the same as training with FSDP).
        with self.assertRaises(
            torch.multiprocessing.ProcessRaisedException
        ):  # Swallows AssertionError
            self._test_resume_training(
                self.moe_clargs
                + ["--ddp-backend", "no_c10d", "--moe-expert-count", "1"]
            )

    def test_resume_training_moe_replication_one_expert(self):
        self._test_resume_training(
            self.moe_clargs
            + ["--ddp-backend", "fully_sharded", "--moe-expert-count", "1"]
        )

    @unittest.skip("Disabled as currently broken")
    @unittest.skipIf(DEVICE_COUNT <= 2, "cannot replicate experts")
    def test_resume_training_moe_replication(self):
        self._test_resume_training(
            self.moe_clargs
            + [
                "--ddp-backend",
                "fully_sharded",
                "--moe-expert-count",
                str(int(self.world_size / 2)),
            ]
        )

    @unittest.skipIf(DEVICE_COUNT < 2, "cannot replicate experts")
    def test_resume_training_moe_fsdp_replication_sharded_state(self):
        self._test_resume_training(
            self.moe_clargs
            + [
                "--ddp-backend",
                "fully_sharded",
                "--use-sharded-state",
                "--moe-expert-count",
                str(int(self.world_size / 2)),
            ]
        )

    def test_resume_training_base_moe(self):
        self._test_resume_training(
            ["--ddp-backend", "no_c10d", "--base-layers", "1", "--base-sublayers", "2"]
        )

    @unittest.skipIf(DEVICE_COUNT < 2, "require at least two gpus")
    def test_resume_training_moe_top1gate(self):
        self._test_resume_training(
            self.moe_clargs
            + [
                "--ddp-backend",
                "fully_sharded",
                "--moe-expert-count",
                str(self.world_size),
                "--moe-top1-expert",
            ]
        )
