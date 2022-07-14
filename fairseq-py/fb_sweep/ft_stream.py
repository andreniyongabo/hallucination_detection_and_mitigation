#!/usr/bin/env python

import os
from dataclasses import dataclass

import sweep
from sweep import hyperparam


@dataclass
class Size:
    # copied from gcmf
    n_layers: int
    emb_size: int
    n_heads: int

    @property
    def ffn_size(self):
        return 4 * self.emb_size


M = 1024 * 1024  # 1 million
# see table 2.1 in https://arxiv.org/pdf/2005.14165.pdf
MODEL_SIZES = {
    "125m": Size(12, 768, 12),
    "350m": Size(24, 1024, 16),
    "760m": Size(24, 1536, 16),
    "1.3b": Size(24, 2048, 32),
    "2.7b": Size(32, 2560, 32),
    "6.7b": Size(32, 4096, 32),
    "13b": Size(40, 5120, 40),
    "175b": Size(96, 12288, 96),
}
DATA_LOCATIONS = {
    # for aws: see https://fb.workplace.com/groups/aws.fair.discuss/posts/921655628752789/?comment_id=921794508738901
    "aws": "/datasets01/gptz",
    "rsc": "/checkpoint/xlmg/data/gptz",
    "azure": "/data/xlmg/gptz",
}


def _streaming_task_config(args):

    if os.path.exists("/data/xlmg/gptz/tokenizers"):
        args.data = getattr(args, "data", "/data/xlmg/flan_minus_qa_streaming")
        tok_dir = "/data/xlmg/gptz/tokenizers"
    elif os.path.exists("/datasets01/gptz/tokenizers/"):
        args.data = getattr(args, "data", "/fsx/sshleifer/flan_streaming")
        tok_dir = "/datasets01/gptz/tokenizers/"
    else:
        args.data = getattr(
            args, "data", "/checkpoint/sviyer/data/flan_minus_qa_streaming"
        )
        tok_dir = "/large_experiments/xlmg/data/gptz/tokenizers"  # on FAIR cluster

    return [
        hyperparam("--task", "streaming_language_modeling"),
        # hyperparam("--valid-subset", valid_subset_str),
        hyperparam(
            "--vocab-filename",
            f"{tok_dir}/gpt2-vocab.json",
            save_dir_key=lambda _: "gpt2",
        ),
        hyperparam("--merges-filename", f"{tok_dir}/gpt2-merges.txt"),
    ]


def _lm_task_config(args):
    # downstream_task, valid_subset_str = _flan_task_mappings(args)
    if os.path.exists("/data/xlmg/flan_txt"):
        args.data = "/data/xlmg/data-bin/flan"
    elif os.path.exists("/fsx/sshleifer"):  # AWS
        args.data = "/fsx/sshleifer/flan-bin"
    else:
        args.data = "/private/home/sshleifer/fairseq-py/flan-bin"
    assert os.path.exists(args.data), args.data
    return [
        hyperparam("--task", "language_modeling"),
        hyperparam("--shorten-method", "truncate", save_dir_key=lambda val: "truncate"),
    ]


def get_grid(args):
    grid = []

    def H(*args, **kwargs):
        grid.append(hyperparam(*args, **kwargs))

    grid += [
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--save-interval", -1),
        hyperparam("--save-interval-updates", args.interval),
        hyperparam("--validate-interval-updates", args.interval),
        hyperparam("--no-save-optimizer-state"),
        # hyperparam("--ignore-unused-valid-subsets"),
    ]
    if args.no_save:
        H("--no-save")
    else:
        H("--best-checkpoint-metric", "loss")

    if args.streaming:
        grid.extend(_streaming_task_config(args))
    else:
        grid.extend(_lm_task_config(args))

    if not args.no_combine_valid_subsets:
        valid_subset_str = "valid"
        H("--combine-valid-subsets")
    else:
        raise NotImplementedError(
            "uncombined valid-subsets not yet supported for this dataset out of laziness"
        )
    H("--sample-break-mode", args.sbm, save_dir_key=lambda val: f"sbm_{val}")
    # model settings
    grid += [
        # arch not used, we will load `--model-name` directly
        hyperparam("--arch", "transformer_lm_gpt"),
        # hyperparam("--user-dir", "examples/few_shot/finetune"),
        # hyperparam("--task", "prompt_tuning"),
        hyperparam("--criterion", "cross_entropy"),
        # ,
        hyperparam(
            "--tokens-per-sample", args.tps, save_dir_key=lambda val: f"tps_{val}"
        ),
        hyperparam("--init-model-on-gpu"),
    ]
    H("--ddp-backend", "fully_sharded")
    H("--save-async")
    from examples.few_shot.model_configs import PATH_TO_ROBERTA_DICT
    from examples.few_shot.models import get_lm_config

    for k, v in MODEL_SIZES.items():
        if k in args.model_name.lower():
            size = v
            break
    else:
        raise ValueError(f"no match for {args.model_name}")
    if "no_pretrain" not in args.model_name:
        _, config = get_lm_config(args.model_name)
        restore_file = config["model_path"]
        if (config.get("dict_path", "") == PATH_TO_ROBERTA_DICT) and args.streaming:
            H("--final-vocab-size", 51200)
        if os.path.exists(restore_file):
            H("--ignore-suffix")  # All workers load restore_file
        H("--restore-file", restore_file, save_dir_key=lambda _: args.model_name)

    grid.extend(
        [
            hyperparam("--decoder-layers", size.n_layers),
            hyperparam("--decoder-embed-dim", size.emb_size),
            hyperparam("--decoder-ffn-embed-dim", size.ffn_size),
            hyperparam("--decoder-attention-heads", size.n_heads),
            hyperparam("--arch", "transformer_lm_gpt", save_dir_key=lambda val: val),
            hyperparam("--share-decoder-input-output-embed"),
        ]
    )

    if "175B" in args.model_name or "gptz" in args.model_name:
        # GCMF Models
        H("--checkpoint-activations", binary_flag=True, save_dir_key=lambda _: "ckpt")
        # this model requires checkpoint activations to load
        H("--use-sharded-state", save_dir_key=lambda val: f"uf{val}")
        H("--activation-fn", "relu")
        H("--decoder-learned-pos")
        H("--no-scale-embedding")

    else:
        H("--activation-fn", "gelu")
    num_examples = 30000 * 4  # FIXME
    num_tokens_per_epoch = 65536 * 527  #

    # Note(srini): steps (from FLAN paper) * examples per step (approximated from 8192 tokens per batch)
    max_update = args.max_update
    tot_gpu = args.num_nodes * args.num_gpus
    if max_update is None:
        max_update = num_examples // (args.bs * args.uf * tot_gpu)
    warmup_update = int(args.wu * max_update)
    grid += [
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam("--warmup-updates", warmup_update),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--batch-size", args.bs, save_dir_key=lambda val: f"bsz{val}"),
        hyperparam("--update-freq", args.uf, save_dir_key=lambda val: f"uf{val}"),
    ]

    # regularization
    dropout = args.dropout
    grid += [
        hyperparam("--dropout", dropout, save_dir_key=lambda val: f"dr{val}"),
        # --attention-dropout will be set to mirror --dropout in postprocess_args
        hyperparam(
            "--attention-dropout", dropout, save_dir_key=lambda val: f"atdr{val}"
        ),
        hyperparam("--activation-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        # hyperparam("--weight-decay", [0.0, 0.01], save_dir_key=lambda val: f"wd{val}"),
    ]

    adam_betas = "(0.9, 0.98)" if args.model_name != "175B" else "(0.9, 0.95)"
    H("--adam-betas", adam_betas)
    H("--adam-eps", 1e-8 if args.model_name == "175B" else 1e-6)
    H("--clip-norm", args.clip_norm)
    if not args.no_fp16_adam:
        H("--fp16-adam-stats")
        H("--optimizer", "adam", save_dir_key=lambda val: "fp16adam")
    else:
        H("--optimizer", "adam", save_dir_key=lambda val: "fp32adam")

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
    ]
    H("--memory-efficient-fp16")
    H("--reset-meters")
    H("--reset-dataloader")
    H("--reset-optimizer")

    # data loading settings
    nw = 0
    grid += [hyperparam("--num-workers", nw), hyperparam("--num-workers-valid", nw)]

    # logging settings
    grid += [
        # hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
    ]
    if not args.zero3:
        H("--no-reshard-after-forward")

    if args.ckpt:
        H("--checkpoint-activations", binary_flag=True, save_dir_key=lambda _: "ckpt")
    H("--patience", args.patience)
    return grid


def postprocess_hyperparams(args, config):
    pass


def add_args(parser):
    parser.add_argument("--model-name", "--m", default="355M", type=str)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--cheat", action="store_true")
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--no-fp16-adam", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--no-combine-valid-subsets", "--vss", action="store_true")
    parser.add_argument(
        "--max-update", "--mu", type=int, default=None
    )  # FIXME: ignoresd
    parser.add_argument("--tps", "--seq-len", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ckpt", "--checkpoint-activations", action="store_true")
    parser.add_argument("--uf", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--wu", type=float, default=0.06)
    parser.add_argument("--interval", type=int, default=500)
    parser.add_argument("--zero3", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--sbm", type=str, default="none")
    parser.add_argument("--benchmark", action="store_true", default=False)


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams, add_extra_options_func=add_args)
