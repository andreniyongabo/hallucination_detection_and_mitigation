# Copyright (c) Facebook Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

from .transformer_config import (
    DEFAULT_MAX_SOURCE_POSITIONS,
    DEFAULT_MAX_TARGET_POSITIONS,
    DEFAULT_MIN_PARAMS_TO_WRAP,
)
from .transformer_decoder import TransformerDecoder, Linear
from .transformer_encoder import TransformerEncoder
from .transformer_base import (
    Embedding,
    TransformerModel,
    base_architecture,
    tiny_architecture,
    transformer_iwslt_de_en,
    transformer_wmt_en_de,
    transformer_vaswani_wmt_en_de_big,
    transformer_vaswani_wmt_en_fr_big,
    transformer_wmt_en_de_big,
    transformer_wmt_en_de_big_t2t,
)


__all__ = [
    "base_architecture",
    "DEFAULT_MAX_SOURCE_POSITIONS",
    "DEFAULT_MAX_TARGET_POSITIONS",
    "DEFAULT_MIN_PARAMS_TO_WRAP",
    "Embedding",
    "fsdpwrap_expert",
    "Linear",
    "tiny_architecture",
    "transformer_iwslt_de_en",
    "transformer_vaswani_wmt_en_de_big",
    "transformer_vaswani_wmt_en_fr_big",
    "transformer_wmt_en_de_big_t2t",
    "transformer_wmt_en_de_big",
    "transformer_wmt_en_de",
    "TransformerConfig",
    "TransformerDecoder",
    "TransformerEncoder",
    "TransformerModel",
]
