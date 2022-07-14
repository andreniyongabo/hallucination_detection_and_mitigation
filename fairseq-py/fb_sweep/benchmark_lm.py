#!/usr/bin/env python

from fb_sweep import sweep
from fb_sweep.sweep import SHARDED_ENGLISH_DATA, hyperparam


def get_grid(args):
    max_update = args.mu
    experts_per_gpu = args.epg
    embed_dim = args.embed_dim
    dl = args.dl
    save_updates = args.save_updates
    args.data = SHARDED_ENGLISH_DATA
    args.snapshot_code = False if args.local else True

    total_bsz = args.ebs
    num_gpus = args.num_gpus * args.num_nodes
    assert total_bsz % num_gpus == 0
    bsz_per_gpu = min(args.bs, total_bsz // num_gpus)
    update_freq = total_bsz // num_gpus // bsz_per_gpu
    assert bsz_per_gpu * update_freq * num_gpus == total_bsz

    moe_args = [
        hyperparam("--moe-gate-loss-wt", 0.01, save_dir_key=lambda val: f"moe_w{val}"),
        hyperparam("--moe-gate-loss-combine-method", "sum"),
        hyperparam("--moe-second-expert-policy", ["all"], save_dir_key=lambda val: val),
        hyperparam(
            "--moe-normalize-gate-prob-before-dropping",
            False,
            binary_flag=True,
            save_dir_key=lambda val: "norm_b",
        ),
        hyperparam("--moe-gating-use-fp32"),
        hyperparam("--moe-freq", min(2, dl)),
        hyperparam(
            "--moe-expert-count",
            int(experts_per_gpu * args.num_nodes * args.num_gpus),
            save_dir_key=lambda val: f"{val}e",
        ),
        # Use same capacity during validation as of training.
        # hyperparam('--moe-eval-capacity-token-fraction', args.cap),  # previous was ??
        hyperparam(
            "--moe-eval-capacity-token-fraction",
            args.cap,
            save_dir_key=lambda v: f"cap{v}",
        ),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--clip-norm", 0.1, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--criterion", "moe_cross_entropy"),
        hyperparam("--pad-to-fixed-length"),
        # hyperparam('--record-a2a-perf-stats', binary_flag=True),
    ]
    dense_args = [
        hyperparam("--clip-norm", 1.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--criterion", "cross_entropy"),
    ]
    grid = moe_args if experts_per_gpu > 0 else dense_args

    def H(*args, **kwargs):
        grid.append(hyperparam(*args, **kwargs))

    original_opt = args.opt
    if args.opt == "adam16":
        args.fp16_adam = True
        args.opt = "adam"
    elif args.opt == "adam8bit":
        grid.append(hyperparam("--no-scale-embedding"))
        grid.append(
            hyperparam("--use-stable-embedding", save_dir_key=lambda x: "stable")
        )
        grid.append(hyperparam("--block-wise"))

        if args.ddp == "fully_sharded":
            args.use_sharded_state = True
    grid.append(
        hyperparam("--optimizer", args.opt, save_dir_key=lambda val: original_opt),
    )
    grid.append(hyperparam("--ignore-unused-valid-subsets"))
    # grid.append(hyperparam('--use-sharded-state'))
    if "adam" in args.opt:
        grid.extend(
            [
                hyperparam(
                    "--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"
                ),
                hyperparam("--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"),
            ]
        )
    elif args.opt == "adafactor" and args.adafactor_use_momentum:
        grid.extend(
            [
                hyperparam("--beta1", [0.1], save_dir_key=lambda b: f"beta1_{b}"),
                hyperparam(
                    "--first-moment-fp16",
                    [True],
                    binary_flag=True,
                    save_dir_key=lambda b: "mofp16" if b else "mofp32",
                ),
                hyperparam(
                    "--no-relative-lr",
                    [True],
                    binary_flag=True,
                    save_dir_key=lambda b: "share-lr" if b else "ada-lr",
                ),
            ]
        )

    grid.append(
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16")
    )

    if args.fp16_adam:
        grid.append(hyperparam("--fp16-adam-stats", binary_flag=True))

    grid += [
        hyperparam("--save-async"),
        hyperparam(
            "--activation-fn",
            args.activation,
            save_dir_key=lambda x: x if x != "gelu" else "",
        ),
        hyperparam("--tensorboard-logdir", ""),
        hyperparam("--num-workers", args.nw),
        hyperparam("--ddp-backend", args.ddp, save_dir_key=lambda b: b[:3]),
        hyperparam("--validate-interval-updates", 2500),
        hyperparam("--save-interval-updates", save_updates),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-best-checkpoints"),
        hyperparam("--no-save-optimizer-state-on-training-finished"),
        hyperparam("--keep-interval-updates", 10),
        hyperparam("--task", "language_modeling"),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"),
        hyperparam(
            "--tokens-per-sample", args.tps, save_dir_key=lambda val: f"tps{val}"
        ),
    ]
    if args.use_sharded_state:
        grid.append(
            hyperparam("--use-sharded-state", save_dir_key=lambda v: "ss" if v else "")
        )

    grid.append(
        hyperparam(
            "--moe-batch-prioritized-routing",
            args.bpr,
            binary_flag=True,
            save_dir_key=lambda x: "bpr" if x else "",
        )
    )
    grid += [
        hyperparam(
            "--arch", "transformer_lm_gpt2_bigger", save_dir_key=lambda val: val
        ),
        # hyperparam('--decoder-layers', [400, 432, 450], save_dir_key=lambda val: f'dl{val}'),
        hyperparam("--decoder-layers", dl, save_dir_key=lambda val: f"dl{val}"),
        hyperparam(
            "--decoder-embed-dim", embed_dim, save_dir_key=lambda val: f"d{val}"
        ),
        hyperparam("--decoder-ffn-embed-dim", embed_dim * 4),
        hyperparam(
            "--share-decoder-input-output-embed", save_dir_key=lambda val: "share"
        ),
    ]
    if args.nh:
        H("--decoder-attention-heads", args.nh, save_dir_key=lambda k: f"nh_{k}")
    grid += [
        hyperparam("--lr-scheduler", args.sched),
        hyperparam(
            "--warmup-updates", args.warmup_updates, save_dir_key=lambda val: f"wu{val}"
        ),
        hyperparam("--dropout", args.dropout, save_dir_key=lambda val: f"dr{val}"),
        hyperparam(
            "--attention-dropout", args.dropout, save_dir_key=lambda val: f"atdr{val}"
        ),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),
        hyperparam("--max-sentences", bsz_per_gpu, save_dir_key=lambda val: f"bs{val}"),
        hyperparam("--max-sentences-valid", bsz_per_gpu),
        hyperparam("--pad-to-fixed-length"),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", args.log_interval),
        # hyperparam('--disable-validation', binary_flag=True),
    ]
    if args.restore_file is not None:
        from pathlib import Path

        H(
            "--restore-file",
            args.restore_file,
            save_dir_key=lambda f: f"wt_{Path(f).name[:-3]}" if f != "x.pt" else "",
        )
    if args.no_save:
        grid.append(hyperparam("--no-save"))

    if args.checkpoint_activations:
        H("--checkpoint-activations")
    if args.stable:
        H(
            "--use-stable-embedding",
            True,
            binary_flag=True,
            save_dir_key=lambda x: "stable",
        )
        H("--no-scale-embedding")
    if args.reset:
        H("--reset-dataloader")
        H("--reset-optimizer")
        H("--reset-meters")
        H("--reset-lr-scheduler")
    elif args.reset_lr:
        H("--reset-lr-scheduler")

    H(
        "--scale-attn",
        args.scale_attn,
        binary_flag=True,
        save_dir_key=lambda x: f"lat_{x}",
    )
    H("--scale-fc", args.scale_fc, binary_flag=True, save_dir_key=lambda x: f"lfc_{x}")
    H(
        "--scale-heads",
        args.scale_heads,
        binary_flag=True,
        save_dir_key=lambda x: f"lhead_{x}",
    )
    H("--fp16-init-scale", 4)

    if args.sched != "inverse_sqrt":
        H("--total-num-update", max(600000, args.mu))

    if args.restore_file is not None:
        H(
            "--restore-file",
            args.restore_file,
            save_dir_key=lambda f: f"wt_{Path(f).name[:-3]}" if f != "x.pt" else "",
        )

    grid.append(
        hyperparam(
            "--alibi",
            args.alibi,
            binary_flag=True,
            save_dir_key=lambda x: "alibi" if x else None,
        )
    )
    return grid


# RF=/checkpoint/sshleifer/2021-04-09/agbm.dl24.d1024.ngpu8/checkpoint_1_5.pt
NAME_VARS = [
    "--decoder-embed-dim",
    "--decoder-layers",
    "--optimizer",
    "--moe-expert-count",
    "--max-sentences",
    "--lr",
    "--scale-attn",
    "--scale-fc",
    "--scale-heads",
    "--warmup-updates",
    "--use-stable-embedding",
    "--decoder-attention-heads",
    "--dropout",
    "--tokens-per-sample",
]


def destroy_most_save_dir_keys(config):
    for k in config:
        if k not in NAME_VARS:
            config[k].save_dir_key = None


def add_args(parser):
    parser.add_argument("--dl", default=12, type=int)
    parser.add_argument("--activation", "--act", default="gelu", type=str)
    parser.add_argument("--embed-dim", default=768, type=int)
    parser.add_argument("--mu", default=100, type=int)
    parser.add_argument("--epg", default=0, type=float)
    parser.add_argument("--restore-file", "--rf", default=None, type=str)
    parser.add_argument(
        "--checkpoint-activations", "--ckpt", action="store_true", default=False
    )
    parser.add_argument("--ebs", default=512, type=int, help="total batch size")
    parser.add_argument("--bs", default=2, type=int, help="local batch size")
    parser.add_argument("--log-interval", "--li", default=1, type=int)
    parser.add_argument("--cap", default=-1.0, type=float)
    parser.add_argument("--ddp", default="fully_sharded", type=str)
    parser.add_argument("--nw", default=1, type=int)
    parser.add_argument("--opt", default="adam", type=str)
    parser.add_argument("--adafactor-use-momentum", action="store_true", default=False)
    parser.add_argument("--no-save", action="store_true", default=False)
    parser.add_argument("--fp16-adam", action="store_true", default=False)
    parser.add_argument(
        "--use-sharded-state", "--ss", action="store_true", default=False
    )
    parser.add_argument("--save-updates", type=int, default=5000)
    parser.add_argument("--stable", action="store_true", default=False)
    parser.add_argument("--reset", action="store_true", default=False)
    parser.add_argument("--reset-lr", action="store_true")
    parser.add_argument("--bpr", "--batch-prioritized-routing", action="store_true")
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--tps", "--tokens-per-sample", default=1024, type=int)
    parser.add_argument("--warmup-updates", "--wu", type=int, default=500)
    parser.add_argument("--alibi", action="store_true")
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--scale-attn", action="store_true", default=False)
    parser.add_argument("--scale-fc", action="store_true", default=False)
    parser.add_argument("--scale-heads", "--sh", action="store_true")
    parser.add_argument("--nh", default=None, type=int)
    parser.add_argument("--sched", type=str, default="polynomial_decay")


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    destroy_most_save_dir_keys(config)


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams, add_extra_options_func=add_args)
