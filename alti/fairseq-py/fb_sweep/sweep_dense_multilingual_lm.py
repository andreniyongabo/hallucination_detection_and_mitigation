#!/usr/bin/env python

import os

from fb_sweep import sweep
from fb_sweep.sweep import hyperparam


def get_grid(args):
    max_update = os.getenv("MU", 100000)
    save_updates = 2000

    assert args.data in (
        "cc100",
        "cc100_combined",
        "cc100_xl_bpe",
        "cc100_xl_unilm",
        "cc100_xl_unilm_supershards",
    )
    lang_to_shard_ratio_file = None
    if args.data == "cc100":
        args.data = ":".join(
            [
                "/checkpoint/namangoyal/storage/cc100-bin/250/shard{}".format(i)
                for i in range(40)
            ]
        )
    elif args.data == "cc100_xl_bpe":
        args.data = ":".join(
            [
                "/large_experiments/xlmg/data/cc100_xl_comb_bpe/bin/shard{}".format(i)
                for i in range(64)
            ]
        )
    elif args.data == "cc100_xl_unilm":
        args.data = ":".join(
            [
                "/large_experiments/xlmg/data/cc100_xl_unigram/bin/shard{}".format(i)
                for i in range(64)
            ]
        )
    elif args.data == "cc100_xl_unilm_supershards":
        args.data = ":".join(
            [
                "/large_experiments/xlmg/data/cc100_xl_unigram/supershard_with_merge-bin/supershard{}".format(
                    i
                )
                for i in range(100)
            ]
        )
        lang_to_shard_ratio_file = "/large_experiments/xlmg/data/cc100_xl_unigram/supershard_with_merge-bin/lang_to_offline_shard_ratio.tsv"
    else:
        args.data = ":".join(
            [
                "/large_experiments/flores/namangoyal/cc100_combined/final-bin/shard{}".format(
                    i
                )
                for i in range(64)
            ]
        )

    args.snapshot_code = False if args.local else True

    # sanity check input lang sequence (for duplicated language specifications)
    assert len(set(args.langs.split(","))) == len(args.langs.split(","))

    batch_size = 2
    tokens_per_sample = 1024

    grid = [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--fp16-no-flatten-grads'),
        hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16"),
        hyperparam("--num-workers", 1),
        hyperparam("--ddp-backend", "no_c10d"),
        hyperparam("--validate-interval-updates", 2000),
        hyperparam("--save-interval-updates", save_updates),
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--no-best-checkpoints"),
        hyperparam("--no-save-optimizer-state-on-training-finished"),
        hyperparam("--keep-interval-updates", 1),
        hyperparam("--task", "multilingual_language_modeling"),
        hyperparam("--sample-break-mode", "none", save_dir_key=lambda val: f"bm_{val}"),
        hyperparam(
            "--tokens-per-sample",
            tokens_per_sample,
            save_dir_key=lambda val: f"tps{val}",
        ),
        hyperparam(
            "--multilang-sampling-alpha",
            [0.3, 0.7, 1.0],
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
        hyperparam("--criterion", "cross_entropy"),
        hyperparam(
            "--share-decoder-input-output-embed", save_dir_key=lambda val: "share"
        ),
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam("--adam-betas", "(0.9, 0.98)", save_dir_key=lambda val: "b2_0.98"),
        hyperparam("--adam-eps", 1e-8, save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0, save_dir_key=lambda val: f"cl{val}"),
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

    if lang_to_shard_ratio_file is not None:
        grid += [hyperparam("--lang-to-offline-shard-ratio", lang_to_shard_ratio_file)]

    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams)
