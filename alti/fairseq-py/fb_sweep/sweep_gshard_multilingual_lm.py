#!/usr/bin/env python

import sweep
from sweep import hyperparam
import os


def get_grid(args):
    distributed_world_size = args.num_nodes * args.num_gpus
    max_update = os.getenv("MU", 400000)
    save_updates = 2000
    args.snapshot_code = False if args.local else True

    # sanity check input lang sequence (for duplicated language specifications)
    assert len(set(args.langs.split(","))) == len(args.langs.split(","))

    experts_per_gpu = args.local_experts
    num_experts = experts_per_gpu * args.num_nodes * args.num_gpus

    # wpb = 2 * 1024 * 1024
    batch_size = 2
    tokens_per_sample = 1024

    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--fp16-no-flatten-grads'),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--num-workers", 3),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", save_updates),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-best-checkpoints"),
        hyperparam("--no-save-optimizer-state-on-training-finished"),
        hyperparam("--keep-interval-updates", 1),
        # TODO: change if needed: arguments for per-language batching
        # hyperparam("--one-language-per-gpu", save_dir_key=lambda val: "per_lbatch" if val else ""),
        # hyperparam('--sampling-func', ["alpha_sampler"], save_dir_key=lambda val: f"sf_{val}"),
        hyperparam("--task", "multilingual_language_modeling"),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"),
        hyperparam(
            "--tokens-per-sample",
            tokens_per_sample,
            save_dir_key=lambda val: f"tps{val}",
        ),
        hyperparam(
            "--multilang-sampling-alpha",
            0.7,
            save_dir_key=lambda val: f"samplealpha{val}",
        ),
        # TODO: change if needed --add-bos-token
        # hyperparam("--add-bos-token", save_dir_key=lambda val: "with_bos"),
        hyperparam(
            "--langs",
            args.langs,
            save_dir_key=lambda val: f"nlangs_{len(val.split(','))}",
        ),
        hyperparam("--arch", "transformer_lm_gpt2_small", save_dir_key=lambda val: val),
        # hyperparam('--decoder-layers', 12, save_dir_key=lambda val: f'dl{val}'),
        hyperparam("--criterion", "moe_cross_entropy"),
        hyperparam(
            "--moe-gate-loss-wt", [0.01], save_dir_key=lambda val: f"moe_w{val}"
        ),
        hyperparam("--moe-gate-loss-combine-method", "sum"),
        hyperparam("--moe-second-expert-policy", ["all"], save_dir_key=lambda val: val),
        hyperparam(
            "--moe-normalize-gate-prob-before-dropping",
            [False],
            binary_flag=True,
            save_dir_key=lambda val: "norm_b",
        ),
        hyperparam("--moe-gating-use-fp32"),
        hyperparam("--moe-freq", 2),
        hyperparam("--moe-expert-count", num_experts),
        hyperparam(
            "--share-decoder-input-output-embed", save_dir_key=lambda val: "share"
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        # hyperparam('--optimizer', 'adafactor', save_dir_key=lambda val: val),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        # TODO: change if needed
        hyperparam("--lr", [1e-3], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam("--warmup-updates", 2000, save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--dropout", 0.0, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.0, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--max-sentences", batch_size, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--max-sentences-valid", 2),
        hyperparam("--pad-to-fixed-length"),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam(
            "--update-freq", [args.update_freq], save_dir_key=lambda val: f"uf{val}"
        ),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 100),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
