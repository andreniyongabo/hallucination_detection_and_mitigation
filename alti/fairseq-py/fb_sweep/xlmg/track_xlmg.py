#!/usr/bin/env python
"""
Example usage:

    PYTHONPATH=. ./fb_sweep/xlmg/sweep_xlmg_en_lm.py \
            --num-trials 1 --num-gpus 8 --num-nodes 1 \
            --model-size 125M_xlmg_h2_2021 \
            --prefix xlmg.125m \
            --partition learnaccel

This sweep script takes some additional optional arguments. See add_extra_options_func
for more details.
"""

import os
from fb_sweep import sweep
from fb_sweep.sweep import hyperparam, SHARDED_ENGLISH_DATA


def add_extra_options_func(parser):
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="use synthetic data and only train for 50 steps (for benchmarking)",
    )
    parser.add_argument(
        "--model-size", help="model configuration, see get_grid for available options"
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="tokens_per_sample")
    parser.add_argument(
        "--restore-file", help="load an existing checkpoint for continuing training"
    )
    parser.add_argument(
        "--debug-train-on-small-subset",
        action="store_true",
        help="only load a single shard of data from one datasource (OpenWebText), "
        "which reduces startup time and is useful for debugging",
    )
    parser.add_argument("--lat", action="store_true", default=False)
    parser.add_argument("--lfc", action="store_true", default=False)
    parser.add_argument("--scale-heads", "--sh", action="store_true")
    parser.add_argument("--opt", default="adam", type=str)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--no-fp16-adam", action="store_true", default=False)
    parser.add_argument("--bs", default=None, type=int)
    parser.add_argument("--no-ckpt", default=False, action="store_true")


def infer_data_path_(args):
    if os.path.exists("/nfs2/"):  # Azure (H1 2021)
        args.data = "/mnt/data/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/"
        args.snapshot_code = False  # containers don"t support snapshot_code
    elif os.path.exists("/data/xlmg"):  # Azure (H2 2021)
        args.data = "/data/xlmg/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin"
        args.snapshot_code = False  # containers don"t support snapshot_code
    else:
        args.data = SHARDED_ENGLISH_DATA


def get_base_model_config(layers, model_dim, heads):
    return [
        hyperparam("--arch", "transformer_lm_gpt", save_dir_key=lambda val: val),
        hyperparam("--activation-fn", "gelu"),
        hyperparam("--share-decoder-input-output-embed"),
        hyperparam("--decoder-layers", layers, save_dir_key=lambda val: f"nlay{val}"),
        hyperparam(
            "--decoder-embed-dim", model_dim, save_dir_key=lambda val: f"emb{val}"
        ),
        hyperparam("--decoder-ffn-embed-dim", 4 * model_dim),
        hyperparam("--decoder-attention-heads", heads),
    ]


def add_moe_config_(args, model_config, expert_count):
    model_config.extend(
        [
            # general config
            hyperparam(
                "--max-sentences-valid", 1
            ),  # not strictly necessary, but safer to avoid OOM
            hyperparam("--num-workers-valid", 0),  # this can avoid hangs in some cases
            # options exposed in model
            hyperparam(
                "--moe-expert-count",
                expert_count,
                save_dir_key=lambda val: f"nexprt{val}",
            ),
            hyperparam("--moe-freq", 2),  # MOE on every other layer
            hyperparam("--moe-gating-use-fp32"),
            hyperparam("--moe-second-expert-policy", "all"),
            # hyperparam("--moe-normalize-gate-prob-before-dropping", save_dir_key=lambda val: "norm_b"),
            hyperparam(
                "--moe-eval-capacity-token-fraction", -1.0
            ),  # use same capacity during valid and train
            # options exposed in criterion
            hyperparam("--criterion", "moe_cross_entropy"),
            hyperparam(
                "--moe-gate-loss-wt", [0.01], save_dir_key=lambda val: f"moe_w{val}"
            ),
            hyperparam("--moe-gate-loss-combine-method", "sum"),
            hyperparam(
                "--moe-normalize-expert-grad",
                "sqrt_world_size",
                save_dir_key=lambda val: val,
            ),
        ]
    )
    if not args.benchmark:
        model_config.extend(
            [
                hyperparam("--pad-to-fixed-length"),
            ]
        )


def add_adam8bit_config_(optimizer_config):
    optimizer_config.extend(
        [
            hyperparam("--use-sharded-state"),
            hyperparam("--stable-emb"),
            hyperparam("--no-scale-embedding"),
            hyperparam("--block-wise"),
        ]
    )


def add_adafactor_config(optimizer_config):
    optimizer_config.extend(
        [
            hyperparam("--optimizer", "adafactor"),
            hyperparam("--decay-rate", 0.98),
            hyperparam("--beta1", 0.1, save_dir_key=lambda b: f"beta1_{b}"),
            hyperparam(
                "--no-relative-lr",
                [True],
                binary_flag=True,
                save_dir_key=lambda b: "share-lr" if b else "ada-lr",
            ),
        ]
    )


def add_cpu_adam_config_(optimizer_config):
    optimizer_config.extend(
        [
            hyperparam("--optimizer", "cpu_adam"),
            hyperparam("--cpu-offload", save_dir_key=lambda _: "cpuoff"),
            hyperparam("--offload-activations", save_dir_key=lambda _: "offloadact"),
        ]
    )


def get_grid(args):
    num_gpus = args.num_gpus * args.num_nodes
    training_tokens = int(300e9)  # matches GPT-3

    # Set this to 0 on AWS to avoid segfaults
    num_dataloading_workers = 2 if not os.path.exists("/fsx") else 0
    if os.path.exists("/data/xlmg"):
        train_subset = "train13"

    # TODO the original dense training runs in H1 2021 used a single validation
    # set coming from CC-News. If you need to have comparable valid_ppl to those
    # runs, then set this to False. Otherwise True is preferred, since it will
    # aggregate all of the valid sets for CC-News, Books, Wikipedia, etc.
    combine_valid_sets = True

    # Infer data path if not given
    args.snapshot_code = not args.local
    if args.data is None:
        infer_data_path_(args)

    # assert os.path.exists(args.data), f"Could not find data path: {args.data}"

    # Model configuration based on size
    M = 1024 * 1024
    if args.model_size == "125M_xlmg_h2_2021":
        assert num_gpus >= 8
        model_config = get_base_model_config(layers=12, model_dim=768, heads=12)
        batch_size_tokens = int(0.5 * M)
        max_batch_size_per_gpu = 16 if args.opt != "adafactor" else 4
        learning_rate = 6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "15B_moe_xlmg_h2_2021":
        assert num_gpus >= 32
        model_config = get_base_model_config(layers=12, model_dim=768, heads=12)
        add_moe_config_(args, model_config, expert_count=512)
        batch_size_tokens = int(0.5 * M)
        max_batch_size_per_gpu = 8 if args.opt != "adafactor" else 2
        learning_rate = 6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "355M_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/dense/355M/few_shot.roberta+cc100.os.bm_none.tps2048.transformer_lm_gpt2_small.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu715.dr0.0.atdr0.0.wd0.01.ms1.uf4.mu572204.s1.ngpu64
        assert num_gpus >= 8
        model_config = get_base_model_config(layers=24, model_dim=1024, heads=16)
        batch_size_tokens = int(0.5 * M)
        max_batch_size_per_gpu = 16 if args.opt != "adafactor" else 2
        learning_rate = 3e-4
        warmup_tokens = int(375 * M)
        dropout = 0.0
        weight_decay = 0.01
    elif args.model_size == "1.3B_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/dense/1.3B/few_shot.roberta+cc100.cpt.os.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0002.wu357.dr0.1.atdr0.1.wd0.01.ms2.uf1.mu286102.s1.ngpu256
        assert num_gpus >= 32
        model_config = get_base_model_config(layers=24, model_dim=2048, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 8
        learning_rate = 2e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "207B_moe_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/moe/207B/xlmg.adam_fp16.me_fp16.bm_none.tps2048.transformer_lm_gpt2_bigger.dl24.demb2048.dffn8192.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.sqrt_world_size.wu2000.dr0.1.atdr0.1.wd0.0.ms2.uf1.mu286102.s1.ngpu256
        assert num_gpus >= 256
        model_config = get_base_model_config(layers=24, model_dim=2048, heads=32)
        add_moe_config_(args, model_config, expert_count=512)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 2
        learning_rate = 1.6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.0
    elif args.model_size == "2.7B_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/dense/2.7B/gpt3_2.7B.layers32.emb2560.head32.cpt.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu357.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu286102.s1.ngpu128
        # assert num_gpus >= 128
        model_config = get_base_model_config(layers=32, model_dim=2560, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 4
        learning_rate = 1.6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.0
        weight_decay = 0.0
    elif args.model_size == "6.7B_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/dense/6.7B/gpt3_6_7B.fp16.adam_fp16.me_fp16.bm_none.tps1024.transformer_lm_gpt2_bigger.dl32.demb4096.dffn16384.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.wu2000.dr0.0.atdr0.0.wd0.0.ms12.uf1.mu47683.s1.ngpu512
        assert num_gpus >= 128
        assert args.seq_len == 1024
        model_config = get_base_model_config(layers=32, model_dim=4096, heads=32)
        batch_size_tokens = int(6.0 * M)
        max_batch_size_per_gpu = 12
        learning_rate = 3e-4
        warmup_tokens = int(12000 * M)
        dropout = 0.0
        weight_decay = 0.0
    elif args.model_size == "6.7B_xlmg_h1_2021_resume_27k":
        # for resuming training of 6.7B_xlmg_h1_2021 from the 27k checkpoint
        assert num_gpus >= 128
        assert args.seq_len in {1024, 2048}
        model_config = get_base_model_config(layers=32, model_dim=4096, heads=32)
        batch_size_tokens = int(6.0 * M)
        max_batch_size_per_gpu = 12
        learning_rate = 5.43e-5
        warmup_tokens = int(0)
        dropout = 0.0
        weight_decay = 0.0
        training_tokens = int(300e9) - 27000 * batch_size_tokens
        combine_valid_sets = False
    elif args.model_size == "6.7B_xlmg_h2_2021":
        assert num_gpus >= 64
        model_config = get_base_model_config(layers=32, model_dim=4096, heads=32)
        batch_size_tokens = int(2.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.2e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "1.1T_moe_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/moe/1.1T/xlmg.1.1T.adam_fp16.me_fp16.bm_none.tps1024.transformer_lm_gpt2_bigger.dl32.demb4096.dffn16384.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0003.sqrt_world_size.wu2000.dr0.0.atdr0.0.wd0.0.ms12.uf1.mu47683.s1.ngpu512
        assert num_gpus >= 512
        assert args.seq_len == 1024
        model_config = get_base_model_config(layers=32, model_dim=4096, heads=32)
        add_moe_config_(args, model_config, expert_count=512)
        batch_size_tokens = int(6.0 * M)
        max_batch_size_per_gpu = 12
        learning_rate = 3e-4
        warmup_tokens = int(12000 * M)
        dropout = 0.0
        weight_decay = 0.0
    elif args.model_size == "1.1T_moe_xlmg_h2_2021":
        assert num_gpus >= 256
        model_config = get_base_model_config(layers=32, model_dim=4096, heads=32)
        add_moe_config_(args, model_config, expert_count=512)
        batch_size_tokens = int(2.0 * M)
        max_batch_size_per_gpu = 12
        learning_rate = 1.2e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "13B_xlmg_h2_2021":
        assert num_gpus >= 64
        model_config = get_base_model_config(layers=40, model_dim=5120, heads=40)
        batch_size_tokens = int(2.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.0e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "175B_xlmg_h2_2021":
        assert num_gpus >= 512
        model_config = get_base_model_config(layers=96, model_dim=12288, heads=96)
        # model_config += [
        #    hyperparam("--offload-activations", save_dir_key=lambda _: "offloadact"),
        # ]
        batch_size_tokens = int(4.0 * M)
        max_batch_size_per_gpu = 4
        # optimizer = "cpu_adam"
        learning_rate = 0.6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    # elif args.model_size == "529B_xlmg_h2_2021":
    #     # TODO currently OOMs on 640 GPUs
    #     assert num_gpus >= 640
    #     model_config = get_base_model_config(layers=105, model_dim=20480, heads=128)
    #     #model_config += [
    #     #    hyperparam("--offload-activations", save_dir_key=lambda _: "offloadact"),
    #     #]
    #     batch_size_tokens = int(5.0 * M)
    #     max_batch_size_per_gpu = 4
    #     #optimizer = "cpu_adam"
    #     learning_rate = 0.6e-4
    #     warmup_tokens = int(375 * M)
    #     dropout = 0.1
    #     weight_decay = 0.01
    else:
        raise ValueError(f"Unknown --model-size argument: {args.model_size}")

    # Batch size logic
    batch_size_seqs = batch_size_tokens // args.seq_len
    if args.bs is not None:
        max_batch_size_per_gpu = args.bs
    batch_size_per_gpu = min(max_batch_size_per_gpu, batch_size_seqs // num_gpus)
    update_freq = 1  # batch_size_seqs // (batch_size_per_gpu * num_gpus)
    # assert batch_size_tokens == update_freq * batch_size_per_gpu * num_gpus * args.seq_len

    max_update = training_tokens // batch_size_tokens
    warmup_updates = 100  # warmup_tokens // batch_size_tokens

    log_interval = 10 if not args.local else 1

    if args.benchmark:
        # Overrides for speed benchmarking
        args.data = None
        task_config = [
            hyperparam("--task", "dummy_lm", save_dir_key=lambda val: val),
            hyperparam(
                "--tokens-per-sample",
                args.seq_len,
                save_dir_key=lambda val: f"tps{val}",
            ),
            hyperparam("--dict-size", 51200 - 4),
            hyperparam("--disable-validation"),
        ]
        max_update = 10
        warmup_updates = 0

    else:
        task_config = [
            hyperparam("--task", "language_modeling"),
            hyperparam(
                "--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"
            ),
            hyperparam(
                "--tokens-per-sample",
                args.seq_len,
                save_dir_key=lambda val: f"tps{val}",
            ),
        ]
    grid = []

    def H(*args, **kwargs):
        grid.append(hyperparam(*args, **kwargs))

    # Optimizer config
    optimizer = args.opt
    optimizer_config = [
        hyperparam("--optimizer", optimizer, save_dir_key=lambda val: val)
    ]
    if optimizer == "adam":
        if not args.no_fp16_adam:
            optimizer_config.extend(
                [
                    hyperparam(
                        "--fp16-adam-stats", save_dir_key=lambda val: "fp16adam"
                    ),
                ]
            )
        H(
            "--adam-betas",
            "(0.9, 0.98)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1]),
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        H(
            "--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"
        ),  # GPT-3 used --clip-norm=1.0
    elif optimizer == "adam8bit":
        add_adam8bit_config_(model_config)
        H(
            "--adam-betas",
            "(0.9, 0.98)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1]),
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        H(
            "--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"
        ),  # GPT-3 used --clip-norm=1.0
    elif optimizer == "adafactor":
        add_adafactor_config(optimizer_config)
    elif optimizer == "cpu_adam":
        optimizer_config.extend(
            [
                hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "fp16adam"),
            ]
        )
        add_cpu_adam_config_(model_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    if args.restore_file:
        grid += [
            hyperparam("--restore-file", args.restore_file),
        ]
    if combine_valid_sets:
        grid += [hyperparam("--combine-val")]
    else:
        grid += [hyperparam("--ignore-unused-valid-subsets")]
    grid += [
        hyperparam("--num-workers", num_dataloading_workers),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", 1000),
        hyperparam(
            "--no-epoch-checkpoints"
        ),  # only save checkpoints based on num steps
        hyperparam("--no-best-checkpoints"),  # don't save checkpoint_best.pt
        # hyperparam("--keep-interval-updates", 1),  # only keep the most recent checkpoint
        # hyperparam("--no-save-optimizer-state-on-training-finished"),
        # hyperparam("--save-async"),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--memory-efficient-fp16"),
        hyperparam("--fp16-init-scale", 4),
    ]
    if os.path.exists("/data/xlmg"):
        H("--train-subset", "train13")
    if not args.no_ckpt:
        H("--checkpoint-activations")

    grid += model_config
    grid += task_config
    grid += optimizer_config

    lr_to_use = learning_rate if args.lr is None else args.lr
    grid += [
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", lr_to_use, save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", max_update),
        hyperparam(
            "--warmup-updates", warmup_updates, save_dir_key=lambda val: f"wu{val}"
        ),
        hyperparam("--dropout", dropout, save_dir_key=lambda val: f"dr{val}"),
        hyperparam(
            "--attention-dropout", dropout, save_dir_key=lambda val: f"atdr{val}"
        ),
        hyperparam("--weight-decay", weight_decay, save_dir_key=lambda val: f"wd{val}"),
        hyperparam(
            "--batch-size", batch_size_per_gpu, save_dir_key=lambda val: f"ms{val}"
        ),
        hyperparam("--required-batch-size-multiple", 1),
        hyperparam("--update-freq", update_freq, save_dir_key=lambda val: f"uf{val}"),
        hyperparam("--max-update", max_update, save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--seed", 1, save_dir_key=lambda val: f"s{val}"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", log_interval),
    ]

    H("--optimizer", args.opt, save_dir_key=lambda val: args.opt)
    H("--scale-attn", args.lat, binary_flag=True, save_dir_key=lambda x: f"lat_{x}")
    H("--scale-fc", args.lfc, binary_flag=True, save_dir_key=lambda x: f"lfc_{x}")
    H(
        "--scale-heads",
        args.scale_heads,
        binary_flag=True,
        save_dir_key=lambda x: f"lhead_{x}",
    )
    H("--track-norms", save_dir_key=lambda x: f"lhead_{x}")

    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
