#!/usr/bin/env python

import os
from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


def get_grid(args):
    target_tokens = 300e9
    num_gpus = args.num_gpus * args.num_nodes

    # some defaults
    # TODO
    tokens_per_sample = 2048

    # 2.7B equiv on 256 GPUs
    # assert num_gpus == 256
    # max_sentences = 2
    # decoder_layers = 32
    # embed_dim = 2560
    # experts_per_gpu = 2

    # 1.3B equiv
    max_sentences = 32 // num_gpus
    decoder_layers = 24
    embed_dim = 2048
    experts_per_gpu = 16 // num_gpus
    # assert max_sentences * num_gpus == 512
    assert experts_per_gpu * num_gpus == 16

    max_update = int(target_tokens // num_gpus // max_sentences // tokens_per_sample)

    train_subset = "train13"
    if train_subset != "train":
        assert args.prefix.startswith(
            "test_"
        ), 'please set train_subset="train" for non-test runs'

    save_updates = 15 if args.local else 2500
    uf = 1
    if os.path.exists("/nfs2/"):  # Azure
        args.data = "/mnt/data/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/"
        args.snapshot_code = False  # containers don't support snapshot_code
    elif os.path.exists("/fsx"):  # AWS
        args.data = (
            "/fsx/myleott/data/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin"
        )
    else:
        args.data = "/private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin"
        args.snapshot_code = False if args.local else True

    assert args.model_type is not None, "please provide model type"
    if args.model_type == "1.2T":
        embed_dim = 3072
        # assert num_gpus == 512
        experts_per_gpu = 2  # 1024 experts
        upload_path = "https://checkpoint4.blob.core.windows.net/checkpoints1?sp=racwdl&st=2021-04-23T02:47:26Z&se=2021-05-14T10:47:26Z&spr=https&sv=2020-02-10&sr=c&sig=JX58QO9xdpEupQDhFXpFPSjT8iuGy3h4MiS93kcDWQE%3D"
    elif args.model_type == "1.7T":
        embed_dim = 3584
    else:
        assert args.model_type == "2.2T"
        experts_per_gpu = 1
        max_sentences = 2
        embed_dim = 4096
        upload_path = "https://xlmgwestus2.blob.core.windows.net/checkpoints?sp=racwdl&st=2021-05-11T06:08:51Z&se=2021-06-01T14:08:51Z&spr=https&sv=2020-02-10&sr=c&sig=mPpO3Tk3JJ7ibW4mU6Kvi0H7rQRYUkjaFWXO6T36JX8%3D"

    grid = [
        hyperparam("--save-async"),
        # hyperparam('--s3-upload-path', upload_path),
        hyperparam("--tensorboard-logdir", ""),
        # hyperparam('--restore-file', '/mnt/checkpoints/xlmg.2.2T/checkpoint_2_37000.pt'),
        # hyperparam('--use-fused-scale-mask-softmax'),
        # hyperparam('--use-modularized-layers'),
        # hyperparam('--use-apex-fast-layer-norm'),
        hyperparam("--train-subset", train_subset),
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--fp16-no-flatten-grads'),
        hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "adam_fp16"),
        hyperparam("--fp16-init-scale", 1),
        hyperparam("--threshold-loss-scale", 0.25),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--num-workers", 2),
        hyperparam("--ddp-backend", "fully_sharded"),
        hyperparam("--checkpoint-activations"),
        # hyperparam('--cpu-offload'),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", save_updates),
        # hyperparam('--no-save'),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-best-checkpoints"),
        hyperparam("--no-save-optimizer-state-on-training-finished"),
        hyperparam("--keep-interval-updates", 1),
        hyperparam("--task", "language_modeling"),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"),
        hyperparam(
            "--tokens-per-sample",
            tokens_per_sample,
            save_dir_key=lambda val: f"tps{val}",
        ),
    ]

    if args.local:
        grid += [
            hyperparam(
                "--arch", "transformer_lm_gpt2_bigger", save_dir_key=lambda val: val
            ),
            hyperparam("--decoder-layers", 24, save_dir_key=lambda val: f"dl{val}"),
            hyperparam(
                "--decoder-embed-dim", 1024, save_dir_key=lambda val: f"demb{val}"
            ),
            hyperparam(
                "--decoder-ffn-embed-dim",
                1024 * 4,
                save_dir_key=lambda val: f"dffn{val}",
            ),
        ]
    else:
        grid += [
            hyperparam(
                "--arch", "transformer_lm_gpt2_bigger", save_dir_key=lambda val: val
            ),
            hyperparam(
                "--decoder-layers", decoder_layers, save_dir_key=lambda val: f"dl{val}"
            ),
            hyperparam(
                "--decoder-embed-dim", embed_dim, save_dir_key=lambda val: f"demb{val}"
            ),
            hyperparam(
                "--decoder-ffn-embed-dim",
                embed_dim * 4,
                save_dir_key=lambda val: f"dffn{val}",
            ),
        ]

    grid += [
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
        # Use same capacity during validation as of training.
        hyperparam("--moe-eval-capacity-token-fraction", -1.0),
        hyperparam(
            "--share-decoder-input-output-embed", save_dir_key=lambda val: "share"
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        # TODO: change if needed
        hyperparam("--lr", 0.00016, save_dir_key=lambda val: f"lr{val}"),
        hyperparam(
            "--moe-normalize-expert-grad",
            "sqrt_world_size",
            save_dir_key=lambda val: val,
        ),
        hyperparam("--total-num-update", max_update),
        hyperparam("--warmup-updates", 2000, save_dir_key=lambda val: f"wu{val}"),
        hyperparam("--dropout", 0.1, save_dir_key=lambda val: f"dr{val}"),
        hyperparam("--attention-dropout", 0.1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--weight-decay", 0.0, save_dir_key=lambda val: f"wd{val}"),
        hyperparam(
            "--max-sentences", max_sentences, save_dir_key=lambda val: f"ms{val}"
        ),
        hyperparam("--max-sentences-valid", 1),
        hyperparam("--pad-to-fixed-length"),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--update-freq", uf, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 3, save_dir_key=lambda val: f"s1"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 5),
    ]

    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


def add_extra_options_func(parser):
    parser.add_argument("--model-type", choices=["1.2T", "1.7T", "2.2T"])


if __name__ == "__main__":
    sweep.main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
