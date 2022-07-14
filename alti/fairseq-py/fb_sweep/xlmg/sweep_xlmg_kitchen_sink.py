#!/usr/bin/env python
"""
Example usage:

    PYTHONPATH=. ./fb_sweep/xlmg/sweep_xlmg_kitchen_sink.py \
            --num-trials 1 --num-gpus 8 --num-nodes 32 \
            --model-size experiment2 \
            --prefix xlmg.kitchen_sink_exp2 \

This sweep script takes some additional optional arguments. See add_extra_options_func
for more details.

Command for launching exp23, to provide some of those extra args we are using:
    PYTHONPATH=. python fb_sweep/xlmg/sweep_xlmg_kitchen_sink.py --fp32-adam \
            --zero2 -t 1 --partition 175b -g 8 -n 32 -p gptz.exp23 \
            --model-size experiment23
"""

import os
from fb_sweep import sweep
from fb_sweep.sweep import hyperparam, UNSHARDED_EN_DATA


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
    parser.add_argument(
        "--optimizer",
        "--opt",
        default="adam",
        choices=["adam", "adam8bit", "cpu_adam"],
        help="which optimizer to use",
    )
    parser.add_argument("--scale-attn", action="store_true", default=False)
    parser.add_argument("--scale-fc", action="store_true", default=False)
    parser.add_argument("--scale-heads", "--sh", action="store_true")
    parser.add_argument("--lr", default=None, type=float, help="overrides default lr")
    parser.add_argument("--fp32-adam", action="store_true", default=False)
    parser.add_argument(
        "--bs", default=None, type=int, help="overrides default local batch size"
    )
    parser.add_argument(
        "--no-ckpt",
        default=False,
        action="store_true",
        help="dont checkpoint activations",
    )
    parser.add_argument(
        "--stable", default=False, action="store_true", help="use StableEmbeddingLayer"
    )
    parser.add_argument("--alibi", default=False, action="store_true")
    parser.add_argument("--use-fused-softmax", default=False, action="store_true")
    parser.add_argument("--scale-resids", default=False, action="store_true")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--save-interval", default=1000, type=int)
    parser.add_argument("--dropout", default=None)
    parser.add_argument(
        "--zero2",
        action="store_true",
        help="use ZeRO-2 instead of ZeRO-3, which speeds up training by ~5% at the "
        "cost of more memory usage; ideal for dense equiv. models <10B params",
    )


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
    data = "roberta+cc100_en"

    # Set this to 0 on AWS to avoid segfaults
    num_dataloading_workers = 2 if not os.path.exists("/fsx") else 0

    if args.debug_train_on_small_subset:
        train_subset = "train13"
        assert args.prefix.startswith(
            "test"
        ), "please ensure that --prefix starts with 'test' when using --debug-train-on-small-subset"
    else:
        train_subset = "train"

    # TODO the original dense training runs in H1 2021 used a single validation
    # set coming from CC-News. If you need to have comparable valid_ppl to those
    # runs, then set this to False. Otherwise True is preferred, since it will
    # aggregate all of the valid sets for CC-News, Books, Wikipedia, etc.
    combine_valid_sets = True

    if args.data is None:
        args.data = UNSHARDED_EN_DATA
    assert os.path.exists(args.data), f"Could not find data path: {args.data}"

    if os.path.exists("/nfs2/") or os.path.exists("/data/xlmg") or args.local:
        args.snapshot_code = False  # containers don't support snapshot_code
    else:
        args.snapshot_code = True

    task = "language_modeling"

    valid_subsets = []

    # Model configuration based on size
    M = 1024 * 1024
    if args.model_size == "2.7B_xlmg_h1_2021":
        # matches /large_experiments/xlmg/models/dense/2.7B/gpt3_2.7B.layers32.emb2560.head32.cpt.bm_none.tps2048.transformer_lm_gpt.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.00016.wu357.dr0.1.atdr0.1.wd0.01.ms4.uf1.mu286102.s1.ngpu128
        assert num_gpus >= 128
        model_config = get_base_model_config(layers=32, model_dim=2560, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 4
        learning_rate = 1.6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01
    elif args.model_size == "experiment2":
        # assert num_gpus >= 64
        assert args.fp32_adam
        assert args.seq_len == 2048
        # 2.7B config
        model_config = get_base_model_config(layers=32, model_dim=2560, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.1
        model_config.extend(
            [
                hyperparam(
                    "--activation-fn", "relu_squared", save_dir_key=lambda val: val
                ),
            ]
        )
    elif args.model_size == "experiment10":
        # assert num_gpus >= 64
        assert args.fp32_adam
        assert args.seq_len == 2048
        task = "streaming_language_modeling"
        # 2.7B config
        model_config = get_base_model_config(layers=32, model_dim=2560, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 1.6e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.1
        model_config.extend(
            [
                hyperparam(
                    "--activation-fn", "relu_squared", save_dir_key=lambda val: val
                ),
            ]
        )
    elif args.model_size in {
        "experiment14",
        "experiment15",
        "experiment16",
        "experiment17",
        "experiment18",
        "experiment19",
        "experiment20",  # old data (gpt-2 bpe)
        "experiment21",  # new data with encoding fix (gpt-2 bpe)
        "experiment22",  # old data + reddit, compare with exp 20 to see effect of reddit
        "experiment23",  # subset of new data closest to old data
        "experiment24",  # data ablations
        "experiment25",  # data ablations
        "experiment26",  # data ablations
        "experiment27",  # data ablations
    }:
        # experiment 14 config is the "baseline"
        assert args.fp32_adam
        assert args.seq_len == 2048
        assert args.zero2
        task = "streaming_language_modeling"
        data = "gptz_dedup_10_10_1_0.05"
        bpe = "vanilla"
        # 1.3B config
        model_config = get_base_model_config(layers=24, model_dim=2048, heads=32)
        batch_size_tokens = int(1.0 * M)
        max_batch_size_per_gpu = 16
        learning_rate = 2e-4
        warmup_tokens = int(375 * M)
        dropout = 0.1
        weight_decay = 0.01

        # other experiments tweak specific things
        if args.model_size == "experiment15":
            # old data, new BPE + dataloader
            data = "roberta+cc100_en"
        elif args.model_size == "experiment16":
            # learned positional embeddings
            model_config.extend(
                [
                    hyperparam("--decoder-learned-pos", save_dir_key=lambda val: "lpe"),
                ]
            )
        elif args.model_size == "experiment17":
            # learned positional embeddings + NormFormer
            args.scale_attn = True
            args.scale_heads = True
            args.scale_fc = True
            model_config.extend(
                [
                    hyperparam("--decoder-learned-pos", save_dir_key=lambda val: "lpe"),
                ]
            )
        elif args.model_size == "experiment18":
            # relu instead of gelu
            model_config.extend(
                [
                    hyperparam("--activation-fn", "relu", save_dir_key=lambda val: val),
                ]
            )
        elif args.model_size == "experiment19":
            # learned positional embeddings + no embedding dropout
            model_config.extend(
                [
                    hyperparam("--decoder-learned-pos", save_dir_key=lambda val: "lpe"),
                    hyperparam(
                        "--no-emb-dropout", save_dir_key=lambda val: "noembdrop"
                    ),
                ]
            )
        elif args.model_size == "experiment20":
            # old data, old BPE, new dataloader
            data = "roberta+cc100_en"
            bpe = "gpt2"
        elif args.model_size == "experiment21":
            # new data, old BPE, new dataloader
            data = "gptz_dedup_10_10_1_0.05_encoding_fix_v2"
            bpe = "gpt2"
        elif args.model_size == "experiment22":
            # old data + reddit, old BPE, new dataloader
            data = "roberta+cc100_en+reddit"
            bpe = "gpt2"
        elif args.model_size == "experiment23":
            # subset of data most like old data within new data, old BPE, new dataloader
            data = "gptz_dedup_10_10_1_0.05_encoding_fix_v2_old_subset"
            bpe = "gpt2"
        elif args.model_size == "experiment24":
            # see https://fb.workplace.com/groups/gogogptz/posts/381916033646932/
            valid_subsets = [
                "everything",
                "CommonCrawl.jsonl",
                "OpenWebText2.jsonl",
                "Wikipedia_en.jsonl",
                "ccnewsv2.jsonl",
                "redditflattened.jsonl",
                "stories.jsonl",
            ]
            data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_exp24"
            bpe = "gpt2"
        elif args.model_size == "experiment25":
            # see https://fb.workplace.com/groups/gogogptz/posts/381916033646932/
            valid_subsets = [
                "everything",
                "CommonCrawl.jsonl",
                "HackerNews.jsonl",
                "OpenWebText2.jsonl",
                "Wikipedia_en.jsonl",
                "stories.jsonl",
                "Gutenberg_PG-19.jsonl",
                "OpenSubtitles.jsonl",
                "USPTO.jsonl",
                "ccnewsv2.jsonl",
                "redditflattened.jsonl",
            ]
            data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_exp25"
            bpe = "gpt2"
        elif args.model_size == "experiment26":
            # see https://fb.workplace.com/groups/gogogptz/posts/381916033646932/
            valid_subsets = [
                "everything",
                "BookCorpusFair.jsonl",
                "CommonCrawl.jsonl",
                "OpenWebText2.jsonl",
                "Wikipedia_en.jsonl",
                "ccnewsv2.jsonl",
                "redditflattened.jsonl",
                "stories.jsonl",
            ]
            data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_exp26"
            bpe = "gpt2"
        elif args.model_size == "experiment27":
            # see https://fb.workplace.com/groups/gogogptz/posts/381916033646932/
            valid_subsets = [
                "everything",
                "BookCorpusFair.jsonl",
                "CommonCrawl.jsonl",
                "HackerNews.jsonl",
                "OpenWebText2.jsonl",
                "Wikipedia_en.jsonl",
                "stories.jsonl",
                "Gutenberg_PG-19.jsonl",
                "OpenSubtitles.jsonl",
                "USPTO.jsonl",
                "ccnewsv2.jsonl",
                "redditflattened.jsonl",
            ]
            data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_exp27"
            bpe = "gpt2"

        else:
            raise Exception
    else:
        raise ValueError(f"Unknown --model-size argument: {args.model_size}")

    if args.dropout is not None:
        dropout = dropout

    # Batch size logic
    batch_size_seqs = batch_size_tokens // args.seq_len
    if args.bs is not None:
        max_batch_size_per_gpu = args.bs
    batch_size_per_gpu = min(max_batch_size_per_gpu, batch_size_seqs // num_gpus)
    update_freq = batch_size_seqs // (batch_size_per_gpu * num_gpus)
    assert (
        batch_size_tokens == update_freq * batch_size_per_gpu * num_gpus * args.seq_len
    )

    max_update = training_tokens // batch_size_tokens
    warmup_updates = warmup_tokens // batch_size_tokens

    # obnoxiously verbose output in case we want to find the bad batch
    log_interval = 1

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
        max_update = 50
        warmup_updates = 50
        log_interval = 5
    elif task == "language_modeling":
        assert data == "roberta+cc100_en"
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
    elif (
        task == "streaming_language_modeling"
    ):  # TODO: did we resolve perf issues for streaming data loader?
        train_subset = "train"
        if args.debug_train_on_small_subset:
            args.data = "/data/xlmg/gptz/small_corpus"
        elif data == "roberta+cc100_en":  # old data
            args.data = "/data/xlmg/cc100_roberta_en_jsonl"
        elif data == "gptz_dedup_10_10_1_0.05":  # new data before encoding fix
            raise ValueError("this data doesn't have the encoding fix, don't use")
            args.data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05"
        elif (
            data == "gptz_dedup_10_10_1_0.05_encoding_fix_v2"
        ):  # new data with encoding fix
            args.data = "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_encoding_fix_v2"
        elif data == "roberta+cc100_en+reddit":  # old data + reddit
            args.data = "/data/xlmg/cc100_roberta_en_reddit_jsonl"
        elif (
            data == "gptz_dedup_10_10_1_0.05_encoding_fix_v2_old_subset"
        ):  # subset of new data that is most like old data
            args.data = (
                "/data/xlmg/gptz/corpus_dedup_10_10_1_0.05_encoding_fix_v2_old_subset"
            )
        elif data.startswith("/"):
            args.data = data
        else:
            raise Exception

        num_dataloading_workers = 8
        task_config = [
            hyperparam("--task", "streaming_language_modeling"),
            hyperparam(
                "--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"
            ),
            hyperparam(
                "--tokens-per-sample",
                args.seq_len,
                save_dir_key=lambda val: f"tps{val}",
            ),
        ]
        if bpe == "vanilla":
            task_config.extend(
                [
                    hyperparam(
                        "--vocab-filename",
                        "/data/xlmg/gptz/tokenizers/vanilla50k-vocab.json",
                        save_dir_key=lambda _: "vanilla",
                    ),
                    hyperparam(
                        "--merges-filename",
                        "/data/xlmg/gptz/tokenizers/vanilla50k-merges.txt",
                    ),
                ]
            )
        elif bpe == "punctsplit":
            task_config.extend(
                [
                    hyperparam(
                        "--vocab-filename",
                        "/data/xlmg/gptz/tokenizers/punctsplit50k-vocab.json",
                        save_dir_key=lambda _: "punctsplit",
                    ),
                    hyperparam(
                        "--merges-filename",
                        "/data/xlmg/gptz/tokenizers/punctsplit50k-merges.txt",
                    ),
                ]
            )
        elif bpe == "gpt2":
            task_config.extend(
                [
                    hyperparam(
                        "--vocab-filename",
                        "/data/xlmg/gptz/tokenizers/gpt2-vocab.json",
                        save_dir_key=lambda _: "gpt2",
                    ),
                    hyperparam(
                        "--merges-filename",
                        "/data/xlmg/gptz/tokenizers/gpt2-merges.txt",
                    ),
                ]
            )
        else:
            raise Exception

    # Optimizer config
    optimizer = args.optimizer
    optimizer_config = [
        hyperparam("--optimizer", optimizer, save_dir_key=lambda val: val)
    ]
    if not args.fp32_adam and optimizer != "adam8bit":
        optimizer_config.append(
            hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "fp16adam")
        )
    if optimizer == "adam":
        pass  # defaults set elsewhere
    elif optimizer == "adam8bit":
        add_adam8bit_config_(model_config)
    elif optimizer == "cpu_adam":
        optimizer_config.extend(
            [
                hyperparam("--fp16-adam-stats", save_dir_key=lambda val: "fp16adam"),
            ]
        )
        add_cpu_adam_config_(model_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    grid = []

    def H(*args, **kwargs):
        """Add a hyperparameter"""
        grid.append(hyperparam(*args, **kwargs))

    if args.stable:
        H("--stable-emb", True, binary_flag=True, save_dir_key=lambda x: "stable_emb")
        H("--no-scale-embedding")

    if args.restore_file:
        grid += [
            hyperparam("--restore-file", args.restore_file),
        ]
    if valid_subsets:
        grid += [
            hyperparam(
                "--valid-subset", ",".join(f"valid/{ss}" for ss in valid_subsets)
            )
        ]
    elif combine_valid_sets:
        grid += [hyperparam("--combine-val")]
    else:
        grid += [hyperparam("--ignore-unused-valid-subsets")]
    grid += [
        hyperparam("--train-subset", train_subset),
        hyperparam("--num-workers", num_dataloading_workers),
        hyperparam("--num-workers-valid", num_dataloading_workers),
        hyperparam("--validate-interval-updates", 1000),
        hyperparam("--save-interval-updates", args.save_interval),
        hyperparam(
            "--no-epoch-checkpoints"
        ),  # only save checkpoints based on num steps
        hyperparam("--no-best-checkpoints"),  # don't save checkpoint_best.pt
        # hyperparam("--keep-interval-updates", 1),  # only keep the most recent checkpoint
        # hyperparam("--no-save-optimizer-state-on-training-finished"),
        # hyperparam("--save-async"),
        hyperparam("--ddp-backend", "fully_sharded", save_dir_key=lambda val: "fsdp"),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--fp16-init-scale", 4),
    ]

    if not args.no_ckpt:
        H("--checkpoint-activations")

    if args.zero2:
        grid += [
            hyperparam("--no-reshard-after-forward", save_dir_key=lambda val: "zero2")
        ]
    grid += model_config
    grid += task_config
    grid += optimizer_config

    lr_to_use = learning_rate if args.lr is None else args.lr
    grid += [
        # GPT-3 uses "(0.9, 0.95)"
        hyperparam(
            "--adam-betas",
            "(0.9, 0.98)",
            save_dir_key=lambda val: "b2_{}".format(eval(val)[1]),
        ),
        # Sometimes lowering --adam-eps to 1e-6 can stabilize training
        hyperparam(
            "--adam-eps", args.eps, save_dir_key=lambda val: f"eps{val}"
        ),  # GPT-3 used --clip-norm=1.0
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

    H(
        "--scale-attn",
        args.scale_attn,
        binary_flag=True,
        save_dir_key=lambda x: "ln_attn" if x else "",
    )
    H(
        "--scale-fc",
        args.scale_fc,
        binary_flag=True,
        save_dir_key=lambda x: "ln_fc" if x else "",
    )
    H(
        "--scale-heads",
        args.scale_heads,
        binary_flag=True,
        save_dir_key=lambda x: "scale_heads" if x else "",
    )
    H(
        "--use-fused-softmax",
        args.use_fused_softmax,
        binary_flag=True,
        save_dir_key=lambda x: "fused" if x else "",
    )
    H(
        "--scale-resids",
        args.scale_resids,
        binary_flag=True,
        save_dir_key=lambda x: "scale_resids" if x else "",
    )
    H(
        "--alibi",
        args.alibi,
        binary_flag=True,
        save_dir_key=lambda x: "alibi" if x else "",
    )

    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(
        get_grid, postprocess_hyperparams, add_extra_options_func=add_extra_options_func
    )
