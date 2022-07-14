# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import unittest
from packaging import version

import hydra

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_nllb")


@unittest.skipIf(
    version.parse(hydra.__version__) < version.parse("1.1"), "Requires Hydra > 1.1"
)
class TestTrainPipeline(unittest.TestCase):
    def test_traverse_values_from_cfg(self):
        """
        Test traverse_values_from_cfg function
            recursively traverse all config values
            connect the hierarchy with dots, in "xxx.xxx.xxx" format
        """
        from examples.nllb.modeling.train import traverse_values_from_cfg

        # traverse all types of single-valued parameters
        all_types_cfg = {
            "batch_size": 2,
            "optimizer": "adam",
            "langs": "'cs,de,en'",
            "lr": [0.01],
        }
        all_types_params = traverse_values_from_cfg("", all_types_cfg)
        self.assertListEqual(
            all_types_params,
            ["batch_size=2", "optimizer=adam", "langs='cs,de,en'", "lr=[0.01]"],
        )

        # traverse all types of multi-valued parameters (for sweeping)
        all_sweep_cfg = {
            "batch_size": "2,3",
            "optimizer": "adam,cpu_adam",
            "lr": "[0.01],[0.02]",
        }
        all_sweep_params = traverse_values_from_cfg("", all_sweep_cfg)
        self.assertListEqual(
            all_sweep_params,
            ["batch_size=2,3", "optimizer=adam,cpu_adam", "lr=[0.01],[0.02]"],
        )

        # special transformation of key "_name"
        name_cfg = {"lr_scheduler": {"_name": "cosine"}}
        name_param = traverse_values_from_cfg("", name_cfg)
        self.assertListEqual(name_param, ["lr_scheduler=cosine"])

        # hierarchical representation
        hierachical_cfg = {
            "level1": {
                "_name": "level1",
                "level2": {"_name": "level2", "level3": "level3"},
            }
        }
        hierachical_params = traverse_values_from_cfg("", hierachical_cfg)
        self.assertListEqual(
            hierachical_params,
            ["level1=level1", "level1.level2=level2", "level1.level2.level3=level3"],
        )


if __name__ == "__main__":
    unittest.main()
