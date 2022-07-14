#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam, SHARDED_ENGLISH_DATA


def get_grid(args):
    max_update = 28000
    save_updates = 2000
    args.data = SHARDED_ENGLISH_DATA
    args.snapshot_code = False if args.local else True
    experts_per_gpu = 1
    uf = 8
    return [
        hyperparam("--train-subset", "train" if not args.local else "train13"),
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--fp16-no-flatten-grads'),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--num-workers", 10),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", save_updates),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--keep-interval-updates", 1),
        hyperparam("--task", "language_modeling"),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"),
        hyperparam("--tokens-per-sample", 1024, save_dir_key=lambda val: f"tps{val}"),
        hyperparam(
            "--arch", "transformer_lm_gpt2_big_wide", save_dir_key=lambda val: val
        ),
        hyperparam("--decoder-layers", 12, save_dir_key=lambda val: f"dl{val}"),
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
        hyperparam(
            "--moe-expert-count", experts_per_gpu * args.num_nodes * args.num_gpus
        ),
        hyperparam(
            "--share-decoder-input-output-embed", save_dir_key=lambda val: "share"
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-6, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"cl{val}"),
        # hyperparam('--optimizer', 'adafactor', save_dir_key=lambda val: val),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        # hyperparam('--lr', [1e-4, 5e-4, 1e-3, 2e-3], save_dir_key=lambda val: f'lr{val}'),
        # TODO: change if needed
        hyperparam("--lr", 0.002, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam("--warmup-updates", 2000, save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--dropout", 0.0, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.0, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--max-sentences", 2, save_dir_key=lambda val: f"ms{val}"),
        hyperparam("--max-sentences-valid", 2),
        hyperparam("--pad-to-fixed-length"),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--update-freq", uf, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 50),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
