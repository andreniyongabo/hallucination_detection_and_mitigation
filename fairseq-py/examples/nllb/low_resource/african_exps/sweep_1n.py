#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam

PREDIFINED_GRID_FUNCTION = {}


def register_grid(name):
    def register_grid_func(fn):
        if name not in PREDIFINED_GRID_FUNCTION:
            PREDIFINED_GRID_FUNCTION[name] = fn
        return fn

    return register_grid_func


def get_predefined_grid(name):
    if name not in PREDIFINED_GRID_FUNCTION:
        return []
    else:
        return PREDIFINED_GRID_FUNCTION[name]()


def add_extra_options_func(parser):
    parser.add_argument("--max-update", help="max update", default=None)
    parser.add_argument("--max-epoch", help="max epoch", default=None)
    parser.add_argument("--warmup-updates", help="warmup", default=4000)
    parser.add_argument(
        "--finetune-from-model",
        help="finetune from a pretrained model",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--lang-dict",
        help="a file containing a list of languages to support",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max-tokens", help="max tokens per batch", type=int, default=None
    )
    parser.add_argument(
        "--batch-size", help="max sentences per batch", type=int, default=None
    )
    parser.add_argument("--arch", default="transformer")
    parser.add_argument("--task", default="translation_multi_simple_epoch")
    parser.add_argument(
        "--lang-pairs", help="lang pairs for multilingual training", type=str
    )
    parser.add_argument(
        "--sampling-method", help="sampling method", default="temperature"
    )
    parser.add_argument(
        "--sampling-temperature", help="sampling temperature", default=5
    )
    parser.add_argument(
        "--encoder-langtok", help="add src language token to encoder", default="src"
    )
    parser.add_argument("--decoder-langtok", default=True, action="store_true")
    parser.add_argument("--virtual-epoch-size", default=None)
    parser.add_argument("--virtual-data-size", default=None)
    # use double the default learning rate, since we're using --update-freq=16
    # per token learning should be approximately constant;
    # ideally momentent and 2nd momentent of adam should be adjusted accordingly but less important
    parser.add_argument("--lr", default=10e-4)
    parser.add_argument("--weight-decay", default=0)
    parser.add_argument("--upsample-primary", default=1)
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--dropout", default=0.1)
    parser.add_argument("--label-smoothing", default=0.1)
    parser.add_argument(
        "--ddp-backend", default=None,
    )
    parser.add_argument(
        "--moe", action="store_true", default=False,
    )
    parser.add_argument(
        "--enable-reservsed-directions-shared-datasets",
        default=False,
        action="store_true",
    )
    parser.add_argument("--save-interval-updates", default=None)
    parser.add_argument("--save-interval", default=1)
    parser.add_argument("--keep-last-epochs", default=None)


@register_grid("transformer_24_24")
def get_transformer_24_24_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 24, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 24, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
        hyperparam("--memory-efficient-fp16", True, binary_flag=True),
    ]


@register_grid("transformer_24_24_wide")
def get_transformer_24_24_wide_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 24, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 24, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            16 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            16 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 2048, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 2048),
        hyperparam("--encoder-attention-heads", 32, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 32),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
        hyperparam("--memory-efficient-fp16", True, binary_flag=True),
    ]


@register_grid("transformer_48_48")
def get_transformer_48_48_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 48, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 48, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
        hyperparam("--memory-efficient-fp16", True, binary_flag=True),
    ]


@register_grid("transformer_48_48_wide")
def get_transformer_48_48_wide_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 48, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 48, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            16 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            16 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 32, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 32),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
        hyperparam("--memory-efficient-fp16", True, binary_flag=True),
    ]


@register_grid("transformer_96_96_8k_16")
def get_transformer_96_96_8k_16_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 96, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 96, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
        hyperparam("--memory-efficient-fp16", True, binary_flag=True),
    ]


@register_grid("transformer_96_96_16k_16")
def get_transformer_96_96_16k_16_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 96, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 96, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            16 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            16 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
        hyperparam("--memory-efficient-fp16", True, binary_flag=True),
    ]


@register_grid("transformer_96_96_16k_32")
def get_transformer_96_96_16k_32_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 96, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 96, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            16 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            16 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 2048, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 2048),
        hyperparam("--encoder-attention-heads", 32, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 32),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
        hyperparam("--memory-efficient-fp16", True, binary_flag=True),
    ]


@register_grid("transformer_16_16")
def get_transformer_16_16_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 16, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 16, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("mbart_large")
def get_transformer_mbart_large_grid():
    return [
        hyperparam("--arch", "mbart_large", save_dir_key=lambda val: val),
        hyperparam("--lang-tok-style", "mbart"),
        hyperparam(
            "--layernorm-embedding", binary_flag=True, save_dir_key=lambda val: "lnemb"
        ),
        hyperparam("--encoder-learned-pos"),
        hyperparam("--decoder-learned-pos"),
        hyperparam("--encoder-normalize-before"),
        hyperparam("--decoder-normalize-before"),
        hyperparam("--share-all-embeddings"),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
        hyperparam("--warmup-updates", 2000, save_dir_key=lambda val: f"warmup{val}"),
    ]


@register_grid("transformer_12_12_no_share")
def get_transformer_12_12_no_share_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_12_12")
def get_transformer_12_12_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_12_12_8k_no_share")
def get_transformer_12_12_8k_no_share_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_12_12_8k")
def get_transformer_12_12_8k_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_12_12_8k_02")
def get_transformer_12_12_8k_02_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.2, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.2, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_12_12_8k_03")
def get_transformer_12_12_8k_03_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            8 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.3, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.3, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_12_12_03")
def get_transformer_12_12_03_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.3, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.3, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_12_12_02")
def get_transformer_12_12_02_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 12, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.2, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.2, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_6_6")
def get_transformer_6_6_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 6, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 6, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            2 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            2 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 512, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 512),
        hyperparam("--encoder-attention-heads", 8, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_6_6_wide")
def get_transformer_6_6_wide_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 6, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 6, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.0, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_6_6_wide_03")
def get_transformer_6_6_wide_03_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 6, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 6, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.3, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.3, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_6_6_wide_02")
def get_transformer_6_6_wide_02_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 6, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 6, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            4 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 1024, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 1024),
        hyperparam("--encoder-attention-heads", 16, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 16),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.2, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.2, save_dir_key=lambda val: f"rdrp{val}"),
    ]


@register_grid("transformer_4_4")
def get_transformer_4_4_grid():
    return [
        hyperparam("--arch", "transformer", save_dir_key=lambda val: val),
        hyperparam(
            "--share-all-embeddings",
            True,
            binary_flag=True,
            save_dir_key=lambda val: "shem",
        ),
        hyperparam("--encoder-layers", 4, save_dir_key=lambda val: f"els{val}"),
        hyperparam("--decoder-layers", 4, save_dir_key=lambda val: f"dls{val}"),
        # this is a multiplier of embed dim
        hyperparam(
            "--encoder-ffn-embed-dim",
            2 * 1024,
            save_dir_key=lambda val: f"encffnx{val}",
        ),
        hyperparam(
            "--decoder-ffn-embed-dim",
            2 * 1024,
            save_dir_key=lambda val: f"decffnx{val}",
        ),
        hyperparam("--encoder-embed-dim", 512, save_dir_key=lambda val: f"E{val}"),
        hyperparam("--decoder-embed-dim", 512),
        hyperparam("--encoder-attention-heads", 8, save_dir_key=lambda val: f"H{val}"),
        hyperparam("--decoder-attention-heads", 8),
        hyperparam(
            "--encoder-normalize-before",
            True,
            binary_flag=True,
            save_dir_key=lambda _: "NBF",
        ),
        hyperparam("--decoder-normalize-before", True, binary_flag=True),
        hyperparam("--attention-dropout", 0.3, save_dir_key=lambda val: f"adrp{val}"),
        hyperparam("--relu-dropout", 0.3, save_dir_key=lambda val: f"rdrp{val}"),
    ]


def get_grid(args):
    task = args.task
    sampling_method = args.sampling_method
    sampling_temperature = args.sampling_temperature
    encoder_langtok = args.encoder_langtok

    grids = [
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "mfp16"),
        hyperparam("--fp16", save_dir_key=lambda val: "fp16"),
        hyperparam("--checkpoint-activations"),
        hyperparam("--fp16-no-flatten-grads"),
        hyperparam("--fp16-adam-stats"),
        hyperparam("--max-source-positions", 1024),
        hyperparam("--max-target-positions", 1024),
        hyperparam("--save-interval", args.save_interval),
        hyperparam("--keep-last-epochs", args.keep_last_epochs),
        hyperparam(
            "--update-freq", args.update_freq, save_dir_key=lambda val: f"uf{val}"
        ),
        hyperparam("--task", task),
        hyperparam("--lang-pairs", args.lang_pairs),
        hyperparam(
            "--encoder-langtok", encoder_langtok, save_dir_key=lambda val: f"ent{val}"
        ),
        hyperparam(
            "--sampling-method", sampling_method, save_dir_key=lambda val: f"SPL_{val}"
        ),
        hyperparam(
            "--sampling-temperature",
            sampling_temperature,
            save_dir_key=lambda val: f"tmp{val}",
        ),
        hyperparam("--optimizer", "adam"),
        hyperparam("--adam-eps", 1e-06),
        hyperparam("--adam-betas", "(0.9, 0.98)"),
        hyperparam("--lr-scheduler", "inverse_sqrt"),
        hyperparam("--warmup-init-lr", 1e-7, save_dir_key=lambda val: f"ilr{val}"),
        hyperparam(
            "--warmup-updates", args.warmup_updates, save_dir_key=lambda val: f"wu{val}"
        ),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--upsample-primary", args.upsample_primary),
        hyperparam("--wandb-project", args.wandb_project),
        hyperparam("--stop-min-lr", 1e-9),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--dropout", args.dropout, save_dir_key=lambda val: f"drp{val}"),
        hyperparam(
            "--weight-decay", args.weight_decay, save_dir_key=lambda val: f"wd{val}"
        ),
        hyperparam("--criterion", "label_smoothed_cross_entropy"),
        hyperparam(
            "--label-smoothing",
            args.label_smoothing,
            save_dir_key=lambda val: f"ls{val}",
        ),
        hyperparam("--seed", args.seed, save_dir_key=lambda val: f"seed{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 100 if not args.local else 10),
        hyperparam("--keep-best-checkpoints", 5 if not args.local else 10),
    ]

    if args.max_update:
        grids.append(hyperparam("--max-update", args.max_update),)
    if args.max_epoch:
        grids.append(hyperparam("--max-epoch", args.max_epoch),)
    if args.ddp_backend:
        grids.append(
            hyperparam(
                "--ddp-backend", args.ddp_backend, save_dir_key=lambda val: f"{val}"
            )
        )

    if args.decoder_langtok:
        grids.append(
            hyperparam(
                "--decoder-langtok",
                [True],
                binary_flag=True,
                save_dir_key=lambda val: "det",
            )
        )
    if args.virtual_data_size:
        grids.append(hyperparam("--virtual-data-size", args.virtual_data_size))
    if args.virtual_epoch_size:
        grids.append(hyperparam("--virtual-epoch-size", args.virtual_epoch_size))
    if args.lang_dict:
        grids.append(hyperparam("--lang-dict", args.lang_dict))
    if args.langs:
        grids.append(hyperparam("--langs", args.langs))
    if args.max_tokens:
        grids.append(
            hyperparam(
                "--max-tokens", args.max_tokens, save_dir_key=lambda val: f"mt{val}"
            )
        )
    if args.batch_size:
        grids.append(
            hyperparam(
                "--batch-size", args.batch_size, save_dir_key=lambda val: f"bsz{val}"
            )
        )
    if args.finetune_from_model:
        grids.append(hyperparam("--finetune-from-model", args.finetune_from_model))
    if args.enable_reservsed_directions_shared_datasets:
        grids.append(
            hyperparam(
                "--enable-reservsed-directions-shared-datasets",
                [True],
                binary_flag=True,
            )
        )
    if args.save_interval_updates:
        grids.append(hyperparam("--save-interval-updates", args.save_interval_updates),)
    arch_grid = get_predefined_grid(args.arch)
    arch_grid = (
        arch_grid
        if arch_grid
        else [hyperparam("--arch", args.arch, save_dir_key=lambda val: val),]
    )

    if args.moe:
        grids.append(hyperparam("--moe-expert-count", args.num_nodes * args.num_gpus))
        grids.append(hyperparam("--criterion", "moe_label_smoothed_cross_entropy"))
        grids.append(
            hyperparam(
                "--moe-gate-loss-wt", [0.01], save_dir_key=lambda val: f"moe_w{val}"
            )
        )
        grids.append(hyperparam("--moe-gate-loss-combine-method", "sum"))
        grids.append(
            hyperparam(
                "--moe-second-expert-policy", ["all"], save_dir_key=lambda val: val
            )
        )
        grids.append(
            hyperparam(
                "--moe-normalize-gate-prob-before-dropping",
                [False],
                binary_flag=True,
                save_dir_key=lambda val: "norm_b",
            )
        )
        grids.append(hyperparam("--moe-gating-use-fp32"))
        grids.append(hyperparam("--moe-freq", 4))

    grids += arch_grid

    return grids


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    NAME_VARS = [
        "--encoder-layers",
        "--decoder-layers",
        "--dropout",
        "--attention-dropout",
        "--relu-dropout",
        "--lr",
        "--update-freq",
        "--label-smoothing",
        "--seed",
    ]
    for k in config:
        if k not in NAME_VARS:
            config[k].save_dir_key = None


if __name__ == "__main__":
    sweep.main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
