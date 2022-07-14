import unittest

import torch

from tests.gpu.gpu_test_mixin import DEVICE_COUNT, HAS_BNB, LMTestMixin


@unittest.skipIf(not torch.cuda.is_available(), "test requires a GPU")
class TestDenseLM(LMTestMixin):
    @unittest.skip("Disabled as currently broken")
    def test_resume_training_dense_fsdp_sharded_alibi(self):
        self._test_resume_training(
            [
                "--ddp-backend",
                "fully_sharded",
                "--use-sharded-state",
                "--alibi",
                "--decoder-attention-heads",
                "4",
                "--decoder-embed-dim",
                "128",
                # fused softmax asserts that its input are bigger than this
            ],
            consolidate_and_eval=True,
        )

    @unittest.skipUnless(
        HAS_BNB and DEVICE_COUNT > 1, "adam8bit requires bits and bytes"
    )
    def test_resume_training_dense_fsdp_sharded_adam8bit_smaller_world_size(self):
        self._test_resume_training(
            [
                "--ddp-backend",
                "fully_sharded",
                "--use-sharded-state",
                "--optimizer",
                "adam8bit",
                "--block-wise",
                "--stable-emb",
                "--no-scale-embedding",
                "--memory-efficient-fp16",
                "--decoder-attention-heads",
                "1",
                "--decoder-embed-dim",
                "32",
            ],
            second_world_size=self.world_size // 2,
            eval_sharded=True,
            assert_losses_match=False,
        )

    @unittest.skipUnless(HAS_BNB, "adam8bit requires bits and bytes")
    def test_resume_training_dense_fsdp_sharded_adam8bit(self):
        self._test_resume_training(
            [
                "--ddp-backend",
                "fully_sharded",
                "--use-sharded-state",
                "--optimizer",
                "adam8bit",
                "--block-wise",
                "--stable-emb",
                "--no-scale-embedding",
                "--memory-efficient-fp16",
                "--decoder-attention-heads",
                "1",
                "--decoder-embed-dim",
                "32",
            ],
            eval_sharded=True,
        )

    def test_resume_training_dense_fsdp_sharded_adam32bit(self):
        self._test_resume_training(
            [
                "--ddp-backend",
                "fully_sharded",
                "--use-sharded-state",
            ],
            second_world_size=self.world_size // 2,
            assert_losses_match=False,  # TODO: they match in bash, why not here?
        )

    def test_resume_training_dense_fsdp(self):
        self._test_resume_training(["--ddp-backend", "fully_sharded"])

    def test_resume_training_dense_noc10d(self):
        self._test_resume_training(["--ddp-backend", "no_c10d"])

    def test_fp16_adafactor_noc10d(self):
        self._test_resume_training(
            [
                "--ddp-backend",
                "no_c10d",
                "--optimizer",
                "adafactor",
                "--first-moment-fp16",
                "--beta1",
                "0.1",
            ]
        )
