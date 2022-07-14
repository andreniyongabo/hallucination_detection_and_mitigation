#!/usr/bin/env python

import os

import fb_sweep.sweep as sweep
from fb_sweep.sweep import hyperparam
from fb_sweep.sweep.flan_constants import flan_clusters


def get_grid(args):
    grid = []

    def H(*args, **kwargs):
        grid.append(hyperparam(*args, **kwargs))

    # for debugging: overfit the eval set
    # grid += [hyperparam("--train-subset", "eval")]

    grid += [
        hyperparam("--no-epoch-checkpoints"),
        hyperparam("--save-interval-updates", -1),
        hyperparam("--validate-interval-updates", -1),
    ]
    if args.no_save:
        H("--no-save")
    elif not "flan" in args.task:
        H("--best-checkpoint-metric", "accuracy")
        H("--maximize-best-checkpoint-metric")

    max_train_samples_per_task = -1
    if "flan" in args.task:
        downstream_task, valid_subset_str = _flan_task_mappings(
            args.task, flan_clusters
        )
        grid += [
            hyperparam(
                "--example-proportional-sampling",
                3000,
                save_dir_key=lambda val: f"eps_{val}",
            ),
        ]
    elif args.task == "natural_instruct_exp_train_10":
        downstream_task = args.task
        valid_subset_str = (
            "valid_natural_instruct_exp_train_test__task114,valid_natural_instruct_exp_train_test__task322",
        )
        max_train_samples_per_task = 1000
    else:
        downstream_task = args.task
        # By default we train on 80% of the training data and hold out 20% as a
        # "valid" set. Ideally one should use the "valid" accuracy for model
        # selection and then report results on "eval", which refers to the
        # evaluation dataset in the gpt3_eval.py code.
        valid_subset_str = "valid,eval"

    grid.extend(
        [
            hyperparam("--downstream-task", downstream_task),
            hyperparam("--valid-subset", valid_subset_str),
        ]
    )
    # model settings
    grid += [
        # arch not used, we will load `--model-name` directly
        hyperparam("--arch", "transformer_lm"),
        hyperparam("--user-dir", "examples/few_shot/finetune"),
        hyperparam("--task", "prompt_tuning"),
        hyperparam("--criterion", "prompt_tuning"),
        # NOTE the --report-accuracy option will forward negative examples, so
        # may increase memory pressure
        hyperparam("--num-prefix-tokens", 0, save_dir_key=lambda val: f"pr{val}"),
        hyperparam("--max-train-samples-per-task", max_train_samples_per_task),
        hyperparam("--sample-break-mode", "eos", save_dir_key=lambda val: f"sbm_{val}"),
        # hyperparam("--max-source-positions", 512, save_dir_key=lambda val: f"tok_{val}"),
        # hyperparam("--max-target-positions", 512, save_dir_key=lambda val: f"tok_{val}"),
        hyperparam(
            "--tokens-per-sample", args.tps, save_dir_key=lambda val: f"tok_{val}"
        ),
        hyperparam("--init-model-on-gpu"),
        # hyperparam("--num-prefix-tokens", [16, 32, 64, 128, 256], save_dir_key=lambda val: f"pr{val}"),
        hyperparam("--ddp-backend", "fully_sharded", save_dir_key=lambda _: "ddpfs"),
    ]

    grid.append(hyperparam("--report-accuracy", save_dir_key=lambda _: "acc"))

    if args.cheat:
        # You can optionally "cheat" by training on the full training set (100%)
        # and doing model selection on the eval data. This is not recommended.
        H("--cheat", save_dir_key=lambda _: "cheat")

    if args.model_name == "125M":
        H("--model-name", "125M_gpt3_setting", save_dir_key=lambda val: val)
    elif args.model_name == "125M_gptz_reshard":
        # HACK
        H("--checkpoint-activations", binary_flag=True, save_dir_key=lambda _: "ckpt")
        # this model requires checkpoint activations to load
        grid += [
            hyperparam(
                "--model-name", "125M_gptz_reshard", save_dir_key=lambda val: val
            ),
            hyperparam("--use-sharded-state", save_dir_key=lambda val: f"uf{val}"),
        ]

    elif args.model_name == "355M":
        H("--model-name", "355M_gpt3_setting", save_dir_key=lambda val: val)
    elif args.model_name == "1.3B":
        # Srini ran on 128 GPUS, but this can run on 8
        grid += [
            hyperparam(
                "--model-name", "1.3B_gpt3_setting", save_dir_key=lambda val: val
            ),
            hyperparam("--batch-size", [4], save_dir_key=lambda val: f"bsz{val}"),
            # bs=1, ebs= 1 (bs) * 1 (uf) * n (8) = 8
        ]
    elif args.model_name == "2.7B":
        # Srini ran on 128 GPUS, but this can run on 8
        H("--model-name", "2.7B_gpt3_setting", save_dir_key=lambda val: val)
    elif args.model_name == "6.7B":
        # Srini ran on 128 GPUS, but this can run on 8
        H("--model-name", "6.7B_gpt3_setting", save_dir_key=lambda val: val)
    elif args.model_name == "13B":
        # Srini ran on 64 GPUS, but this can run on 8
        H("--model-name", "13B_gpt3_setting", save_dir_key=lambda val: val)
    elif args.model_name == "175B":
        grid += [
            hyperparam(
                "--model-name", "175B_last_reshard", save_dir_key=lambda val: val
            ),
            hyperparam("--use-sharded-state", save_dir_key=lambda val: f"uf{val}"),
        ]

    grid += [
        hyperparam("--max-update", [None], save_dir_key=lambda val: f"mu{val}"),
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
    if args.model_name == "175B":
        # optimization settings
        H("--adam-eps", 1e-8)
        H("--clip-norm", 1.0)
    else:
        H("--adam-eps", 1e-6)
        H("--clip-norm", 0.0)

    if not args.no_fp16_adam:
        H("--fp16-adam-stats")
        H("--optimizer", "adam", save_dir_key=lambda val: "fp16adam")
    else:
        H("--optimizer", "adam", save_dir_key=lambda val: "fp32adam")

    # lr scheduler
    grid += [
        hyperparam("--lr-scheduler", "polynomial_decay"),
        hyperparam("--lr", args.lr, save_dir_key=lambda val: f"lr{val}"),
        # hyperparam("--lr", [1e-4, 3e-4, 1e-5, 3e-5], save_dir_key=lambda val: f"lr{val}"),
        hyperparam("--total-num-update", [None]),
        # this will be set to 6% of --max-update
        hyperparam("--warmup-updates", [None], save_dir_key=lambda val: f"warm{val}"),
    ]
    H("--memory-efficient-fp16")

    # data loading settings
    grid += [
        hyperparam("--num-workers", 0),
    ]

    # logging settings
    grid += [
        hyperparam("--skip-invalid-size-inputs-valid-test"),
        hyperparam("--log-format", "json"),
        hyperparam("--log-interval", 10),
    ]
    if not args.freeze:
        H("--finetune-model-weights", binary_flag=True, save_dir_key=lambda _: "ft")

    if args.ckpt:
        H("--checkpoint-activations", binary_flag=True, save_dir_key=lambda _: "ckpt")

    return grid


def _flan_task_mappings(task, flan_clusters):
    def _flan_task_parameters(ignore_patterns=[]):
        clusters = [c for c in flan_clusters if c not in ignore_patterns]
        tasks = [t for cluster in clusters for t in flan_clusters[cluster]]
        valid_tasks = [flan_clusters[cluster][0] for cluster in clusters]
        valid_subset_str = ",".join(["valid_" + t for t in valid_tasks])
        downstream_task = ",".join(tasks)
        return downstream_task, valid_subset_str

    if task == "flan_minus_qa":
        downstream_task, valid_subset_str = _flan_task_parameters(["qa"])
    elif task == "flan_minus_nli_para":
        downstream_task, valid_subset_str = _flan_task_parameters(["nli", "paraphrase"])
    elif task == "flan_minus_sentiment":
        downstream_task, valid_subset_str = _flan_task_parameters(["nli", "sentiment"])
    elif task == "flan_minus_commonsense":
        downstream_task, valid_subset_str = _flan_task_parameters(
            ["commonsense", "mrc_with_commonsense"]
        )
    elif task == "flan":
        downstream_task, valid_subset_str = _flan_task_parameters([])
    elif task == "flan_debug":
        valid_subset_str = "valid_flan__copa_10templates"
        downstream_task = "flan__copa_10templates"
    else:  # Specify individual flan task with flan__ prefix
        valid_subset_str = f"valid_{task}"
        downstream_task = task
    return downstream_task, valid_subset_str


def postprocess_flan_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    num_examples = (
        args.steps * 64
    )  # steps (from FLAN paper) * examples per step (approximated from 8192 tokens per batch)
    config["--max-update"].current_value = num_examples // (
        config["--batch-size"].current_value
        * config["--update-freq"].current_value
        * args.num_nodes
        * args.num_gpus
    )

    config["--total-num-update"].current_value = config["--max-update"].current_value
    config["--validate-interval-updates"].current_value = (
        config["--total-num-update"].current_value // 8
    )  # Save 8 times
    config["--save-interval-updates"].current_value = (
        config["--total-num-update"].current_value // 8
    )  # Save 8 times

    # warmup for 3% of updates
    config["--warmup-updates"].current_value = int(
        args.warmup * config["--max-update"].current_value
    )


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    if "flan" in args.task:
        postprocess_flan_hyperparams(args, config)
        return
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

    else:
        print(f"task {args.task} not supported, guessing num examples/epochs")
        # num_examples = 64000 #  FLAN used 30k x 8192. I tried 282k x 32 and it was too much for the 355M model. 128K updates seemed good.
        num_examples = 128000  #  FLAN used 30k x 8192. I tried 282k x 32 and it was too much for the 355M model. 128K updates seemed good.
        num_epochs = 32

    if args.max_update is None:
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
    else:
        config["--max-update"].current_value = args.max_update

    config["--total-num-update"].current_value = config["--max-update"].current_value
    # warmup for 6% of updates
    config["--warmup-updates"].current_value = int(
        args.warmup * config["--max-update"].current_value
    )
    save_interval = config["--total-num-update"].current_value // 8

    config["--validate-interval-updates"].current_value = save_interval
    config["--save-interval-updates"].current_value = save_interval


def add_args(parser):
    parser.add_argument("--model-name", "--m", default="355M", type=str)
    parser.add_argument("--task", type=str, help="boolq, cb, ...")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--cheat", action="store_true")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--no-fp16-adam", action="store_true")
    parser.add_argument("--max-update", "--mu", type=int, default=None)
    parser.add_argument("--tps", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ckpt", "--checkpoint-activations", action="store_true")
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--uf", type=int, default=1)
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--warmup", type=float, default=0.03)
    parser.add_argument("--steps", type=int, default=60000)


if __name__ == "__main__":
    sweep.main(get_grid, postprocess_hyperparams, add_extra_options_func=add_args)
