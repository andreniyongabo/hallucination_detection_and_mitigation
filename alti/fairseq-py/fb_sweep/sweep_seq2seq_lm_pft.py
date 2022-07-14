#!/usr/bin/env python

import os

import sweep
from sweep import hyperparam


def get_grid(args):
    grid = []

    # for debugging: overfit the eval set
    # grid += [hyperparam("--train-subset", "eval")]

    # validation and checkpoint settings
    grid += [
        hyperparam("--no-epoch-checkpoints"),
        # hyperparam("--save-interval-updates", 100),
        # hyperparam("--validate-interval-updates", 100),
        # hyperparam("--no-save-optimizer-state"),
        # hyperparam("--no-save"),
        # hyperparam("--no-last-checkpoints"),
        # hyperparam("--no-best-checkpoints"),
        # hyperparam("--save-interval", 11),
        # hyperparam("--save-interval-updates", 100),
    ]
    if args.no_save:
        grid.append(hyperparam("--no-save"))
    else:
        grid.extend(
            [
                hyperparam("--best-checkpoint-metric", "accuracy"),
                hyperparam("--maximize-best-checkpoint-metric"),
            ]
        )

    # model settings
    grid += [
        hyperparam(
            "--arch", "transformer_lm"
        ),  # not used, we will load `--model-name` directly
        hyperparam("--user-dir", "examples/few_shot/finetune"),
        hyperparam("--task", "prompt_tuning"),
        hyperparam("--criterion", "prompt_tuning"),
        # NOTE the --report-accuracy option will forward negative examples, so
        # may increase memory pressure
        hyperparam("--report-accuracy"),
        hyperparam("--num-prefix-tokens", [0], save_dir_key=lambda val: f"pr{val}"),
        hyperparam("--max-train-samples-per-task", -1),
        # hyperparam("--prefix-init-method", "mean_embed_with_noise", save_dir_key=lambda val: f"init_{val}"),
        # hyperparam("--prefix-with-positional-embed", [False], binary_flag=True, save_dir_key=lambda _: "pr_pos_emb"),
        # hyperparam("--add-negative-ce-loss", [False, True], binary_flag=True, save_dir_key=lambda _: "neg_ce_loss"),
        # hyperparam("--loss-for-label-tokens-only", [False, True], binary_flag=True, save_dir_key=lambda _: "label_toks_only"),
        # By default we train on 80% of the training data and hold out 20% as a
        # "valid" set. Ideally one should use the "valid" accuracy for model
        # selection and then report results on "eval", which refers to the
        # evaluation dataset in the gpt3_eval.py code.
        hyperparam("--valid-subset", "valid,eval"),
        # You can optionally "cheat" by training on the full training set (100%)
        # and doing model selection on the eval data. This is not recommended.
        # hyperparam("--cheat", save_dir_key=lambda _: "cheat"),
        # TODO
        hyperparam("--downstream-task", "later", save_dir_key=lambda val: val),
        hyperparam("--sample-break-mode", "eos", save_dir_key=lambda val: f"sbm_{val}"),
        # hyperparam("--max-source-positions", 512, save_dir_key=lambda val: f"tok_{val}"),
        # hyperparam("--max-target-positions", 512, save_dir_key=lambda val: f"tok_{val}"),
        hyperparam("--tokens-per-sample", 768, save_dir_key=lambda val: f"tok_{val}"),
        hyperparam("--init-model-on-gpu"),
        # hyperparam("--num-prefix-tokens", [16, 32, 64, 128, 256], save_dir_key=lambda val: f"pr{val}"),
    ]

    # TODO
    if args.model_name == "125M":
        grid += [
            hyperparam(
                "--model-name", "125M_gpt3_setting", save_dir_key=lambda val: val
            ),
            hyperparam("--batch-size", [4], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [1], save_dir_key=lambda val: f"uf{val}"),
            # ebs= 4 (bs) * 1 (uf) * n (8) = 32
        ]
    elif args.model_name == "125M_gptz_reshard":
        grid += [
            hyperparam(
                "--model-name", "125M_gptz_reshard", save_dir_key=lambda val: val
            ),
            hyperparam("--batch-size", [4], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [1], save_dir_key=lambda val: f"uf{val}"),
            hyperparam("--use-sharded-state", save_dir_key=lambda val: f"uf{val}"),
            hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            hyperparam(
                "--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"
            ),
            # ebs= 4 (bs) * 1 (uf) * n (8) = 32
        ]
    elif args.model_name == "355M":
        grid += [
            hyperparam(
                "--model-name", "355M_gpt3_setting", save_dir_key=lambda val: val
            ),
            hyperparam("--batch-size", [4], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [16], save_dir_key=lambda val: f"uf{val}"),
            # hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            # hyperparam("--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"),
            # ebs= 4 (bs) * 1 (uf) * n (8) = 32
        ]
    elif args.model_name == "1.3B":
        # Srini ran on 128 GPUS, but this can run on 8
        grid += [
            hyperparam(
                "--model-name", "1.3B_gpt3_setting", save_dir_key=lambda val: val
            ),
            hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            hyperparam(
                "--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"
            ),
            hyperparam("--batch-size", [4], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [1], save_dir_key=lambda val: f"uf{val}"),
            # bs=1, ebs= 1 (bs) * 1 (uf) * n (8) = 8
        ]
    elif args.model_name == "2.7B":
        # Srini ran on 128 GPUS, but this can run on 8
        grid += [
            hyperparam(
                "--model-name", "2.7B_gpt3_setting", save_dir_key=lambda val: val
            ),
            hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            hyperparam(
                "--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"
            ),
            hyperparam("--batch-size", [1], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [4], save_dir_key=lambda val: f"uf{val}"),
            # ebs= 1 (bs) * 1 (uf) * n (128) = 128
        ]
    elif args.model_name == "6.7B":
        # Srini ran on 128 GPUS, but this can run on 8
        grid += [
            # Doesn't work yet. Runs OOM
            hyperparam(
                "--model-name", "6.7B_gpt3_setting", save_dir_key=lambda val: val
            ),
            hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            hyperparam(
                "--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"
            ),
            hyperparam("--batch-size", [1], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [4], save_dir_key=lambda val: f"uf{val}"),
            # ebs= 1 (bs) * 1 (uf) * n (128) = 128
        ]
    elif args.model_name == "13B":
        # Srini ran on 64 GPUS, but this can run on 8
        grid += [
            # Doesn't work yet. Runs OOM
            hyperparam(
                "--model-name", "13B_gpt3_setting", save_dir_key=lambda val: val
            ),
            hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            hyperparam(
                "--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"
            ),
            hyperparam("--batch-size", [1], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [8], save_dir_key=lambda val: f"uf{val}"),
            # ebs= 1 (bs) * 1 (uf) * n (128) = 128
        ]
    elif args.model_name == "15B":
        grid += [
            hyperparam("--model-name", "moe_15B", save_dir_key=lambda val: val),
            hyperparam("--batch-size", [1], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [1], save_dir_key=lambda val: f"uf{val}"),
            # n=4, g=8, bs=1, ebs= 1 (bs) * 1 (uf) * n (128) = 128
        ]
    elif args.model_name == "52B":
        grid += [
            hyperparam("--model-name", "moe_52B", save_dir_key=lambda val: val),
            hyperparam("--batch-size", [1], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [1], save_dir_key=lambda val: f"uf{val}"),
            # n=4, g=8, bs=1, ebs= 1 (bs) * 1 (uf) * n (128) = 128
        ]
    elif args.model_name == "207B":
        grid += [
            hyperparam("--model-name", "moe_207B", save_dir_key=lambda val: val),
            hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            hyperparam(
                "--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"
            ),
            hyperparam("--batch-size", [1], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [1], save_dir_key=lambda val: f"uf{val}"),
            # n=16, g=8, bs=1, ebs= 1 (bs) * 1 (uf) * n (128) = 128
        ]
    elif args.model_name == "523B":
        grid += [
            hyperparam("--model-name", "moe_523B", save_dir_key=lambda val: val),
            hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            hyperparam(
                "--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"
            ),
            hyperparam("--batch-size", [1], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [1], save_dir_key=lambda val: f"uf{val}"),
            # n=32, g=8, bs=1, ebs= 1 (bs) * 1 (uf) * n (128) = 128
        ]
    elif args.model_name == "1.1T":
        grid += [
            # Doesn't run yet. Runs OOM
            hyperparam("--model-name", "moe_1.1T", save_dir_key=lambda val: val),
            hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            hyperparam(
                "--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"
            ),
            hyperparam("--batch-size", [1], save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", [1], save_dir_key=lambda val: f"uf{val}"),
        ]
    elif args.model_name == "175B":
        grid += [
            hyperparam(
                "--model-name", "175B_last_reshard", save_dir_key=lambda val: val
            ),
            hyperparam("--checkpoint-activations", save_dir_key=lambda _: "chkact"),
            hyperparam(
                "--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"
            ),
            hyperparam("--batch-size", 1, save_dir_key=lambda val: f"bsz{val}"),
            hyperparam("--update-freq", 1, save_dir_key=lambda val: f"uf{val}"),
            hyperparam("--use-sharded-state", save_dir_key=lambda val: f"uf{val}"),
        ]

    grid += [
        hyperparam("--max-update", [None], save_dir_key=lambda val: f"mu{val}"),
        hyperparam("--required-batch-size-multiple", 1),
    ]

    # regularization
    grid += [
        hyperparam("--dropout", [0.1], save_dir_key=lambda val: f"dr{val}"),
        # --attention-dropout will be set to mirror --dropout in postprocess_args
        hyperparam("--attention-dropout", -1, save_dir_key=lambda val: f"atdr{val}"),
        hyperparam("--activation-dropout", 0.0, save_dir_key=lambda val: f"actdr{val}"),
        # hyperparam("--weight-decay", [0.0, 0.01], save_dir_key=lambda val: f"wd{val}"),
    ]

    # optimization settings
    grid += [
        hyperparam("--optimizer", "adam", save_dir_key=lambda val: val),
        hyperparam(
            "--adam-betas", "(0.9, 0.98)"
        ),  # , save_dir_key=lambda val: "beta98"),
        hyperparam("--adam-eps", 1e-6),  # , save_dir_key=lambda val: f"eps{val}"),
        hyperparam("--clip-norm", 0.0),  # , save_dir_key=lambda val: f"clip{val}"),
        hyperparam("--fp16-adam-stats", save_dir_key=lambda val: ""),
    ]

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        # hyperparam("--lr", [5e-3, 7e-3, 1e-2, 3e-2, 5e-2, 7e-2], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
        # hyperparam("--lr", [1e-4, 3e-4, 1e-5, 3e-5], save_dir_key=lambda val: f"lr{val}"),
        # this will be set to --max-update
        hyperparam("--total-num-update", [None]),
        # this will be set to 6% of --max-update
        hyperparam("--warmup-updates", [None], save_dir_key=lambda val: f"warm{val}"),
    ]
    # grid += [hyperparam("--fp16", save_dir_key=lambda val: "fp16")]
    grid += [hyperparam("--memory-efficient-fp16", save_dir_key=lambda val: "me_fp16")]

    # grid += [hyperparam("--seed", [0, 1, 2], save_dir_key=lambda val: f"s{val}")]

    grid += [
        # hyperparam("--reset-meters"),
        # TODO try not resetting optimizer
        # hyperparam("--reset-optimizer"),
        # hyperparam("--reset-lr-scheduler"),
        # hyperparam("--reset-dataloader"),
    ]

    # data loading settings
    grid += [
        hyperparam("--num-workers", 0),
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
        # hyperparam("--log-interval", 10 if not args.local else 1),
    ]
    if not args.freeze:
        grid.append(
            hyperparam(
                "--finetune-model-weights",
                binary_flag=True,
                save_dir_key=lambda _: "ft",
            )
        )
    return grid


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    assert config["--attention-dropout"].current_value == -1
    if args.task == "openbookqa":
        num_examples = 4957
        num_epochs = 100
    elif args.task == "piqa":
        num_examples = 12890
        num_epochs = 100
    elif args.task == "boolq":
        num_examples = 7541  # 80-20 split
        num_epochs = 100
    elif args.task == "mnlimatched":
        num_examples = 314162  # 80-20 split
        num_epochs = 6
    elif args.task == "naturalinstructions":
        num_examples = 314162  # 80-20 split
        num_epochs = 3
    elif args.task == "winogrande":
        num_examples = 32318  # 80-20 split
        num_epochs = 25
    elif args.task == "hellaswag":
        num_examples = 39905  # 80-20 split
        num_epochs = 20
    elif args.task == "storycloze":
        num_examples = 1871  # 80-20 split
        num_epochs = 20
    elif args.task == "snli":
        num_examples = 440122  # 80-20 split
        num_epochs = 10
    elif args.task == "natural_instruct_exp_train_10":
        num_examples = 10000
        num_epochs = 30
        valid_tasks = [
            "natural_instruct_exp_train_test__task114",
            "natural_instruct_exp_train_test__task322",
        ]
        valid_tasks_str = ",".join(["valid_" + task for task in valid_tasks])
        config["--valid-subset"].current_value = valid_tasks_str
        config["--max-train-samples-per-task"].current_value = 1000
    else:
        # num_examples = 64000 #  FLAN used 30k x 8192. I tried 282k x 32 and it was too much for the 355M model. 128K updates seemed good.
        num_examples = 128000  #  FLAN used 30k x 8192. I tried 282k x 32 and it was too much for the 355M model. 128K updates seemed good.
        num_epochs = 32
    config["--max-update"].current_value = (
        num_examples
        * num_epochs
        // (
            config["--batch-size"].current_value
            * config["--update-freq"].current_value
            * args.num_nodes
            * args.num_gpus
        )
    )
    config["--downstream-task"].current_value = args.task

    config["--attention-dropout"].current_value = config["--dropout"].current_value
    config["--total-num-update"].current_value = config["--max-update"].current_value
    # config["--validate-interval-updates"].current_value = config["--total-num-update"].current_value // 8 # Save 8 times
    # config["--save-interval-updates"].current_value = config["--total-num-update"].current_value // 8 # Save 8 times

    # warmup for 6% of updates
    config["--warmup-updates"].current_value = int(
        0.06 * config["--max-update"].current_value
    )

    # if we don't have prefix tokens, we must finetune the model weights
    # if config["--num-prefix-tokens"].current_value == 0:
    # assert(config["--finetune-model-weights"].current_value)


def add_args(parser):
    parser.add_argument("--model-name", "--m", default="355M", type=str)
    parser.add_argument("--task", type=str, help="boolq, cb, ...")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--lr", default=1e-4, type=float)

    pass


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams, add_extra_options_func=add_args)
