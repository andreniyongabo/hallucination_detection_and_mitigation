# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import asyncio

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from examples.nllb.mining.nllb_lib.nllb_module import NLLBModule


@hydra.main(config_path="conf", config_name="launch_conf")
def main(config: DictConfig) -> None:
    config_keys = [k for k in config.keys() if k != 'launcher']
    assert len(config_keys) == 1, "should only specify one module config"
    launcher = hydra.utils.instantiate(config.launcher)
    module_conf = config[config_keys[0]]
    module = NLLBModule.build(module_conf)
    asyncio.run(launcher.schedule(module))


if __name__ == "__main__":
    main()
