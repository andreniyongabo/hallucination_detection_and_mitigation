# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import TASK_REGISTRY

if "laser" not in TASK_REGISTRY:
    from .laser_lstm import *  # noqa
    from .laser_task import *  # noqa
    from .laser_transformer import *  # noqa
