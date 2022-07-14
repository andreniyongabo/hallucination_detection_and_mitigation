# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import uuid

from examples.few_shot import utils, models
from fairseq import metrics


class TestModels(unittest.TestCase):
    @property
    def test_config(self):
        return {
            "dummy_key": {
                "model_path": "/dev/null",
                "dict_path": utils.PATH_TO_ROBERTA_DICT,
                "model_overrides": {"bpe": "gpt2"},
                "extra_args": [
                    "--distributed-world-size",
                    "8",
                ],
            }
        }

    def test_get_lm_config(self):
        fairseq_cfg, model_config = models.get_lm_config(
            "dummy_key",
            fsdp=True,
            model_configs=self.test_config,
            distributed_port=12,
        )

        self.assertEqual(
            set(model_config["extra_args"]),
            {
                "--distributed-world-size",
                "8",
                # fp16 gets set by default
                "--fp16",
                # The following arguments get added since fsdp is True
                "--ddp-backend",
                "fully_sharded",
                "--distributed-port",
                "12",
            },
        )

        fairseq_cfg, model_config = models.get_lm_config(
            "dummy_key",
            fsdp=False,
            model_configs=self.test_config,
            distributed_port=12,
        )
        self.assertEqual(
            set(model_config["extra_args"]),
            {"--distributed-world-size", "8", "--fp16", "--distributed-port", "12"},
        )


if __name__ == "__main__":
    unittest.main()
