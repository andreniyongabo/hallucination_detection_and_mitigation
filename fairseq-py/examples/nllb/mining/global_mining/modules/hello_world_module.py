# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import typing as tp

from omegaconf.omegaconf import OmegaConf

from examples.nllb.mining.nllb_lib.launcher import Launcher
from examples.nllb.mining.nllb_lib.nllb_module import (
    DistributedRequirements,
    NLLBModule,
)


class HelloWorldModule(NLLBModule):
    def requirements(self):
        return DistributedRequirements(
            nodes=1,
            mem_gb=10,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=1,
            timeout_min=60,
        )

    async def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
        launcher: tp.Optional[Launcher] = None,
    ):
        print(OmegaConf.to_yaml(self.config))
        # Let the job sleep for a bit to simulate the module doing work
        time.sleep(60)
        return iteration_value
