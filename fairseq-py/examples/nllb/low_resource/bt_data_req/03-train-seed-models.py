#!/usr/bin/env python3
import subprocess
from os import path

DATA_DIR = "/checkpoint/jeanm/nllb/bt_data_req"

LANGS = ["asm", "zul", "amh", "tel", "urd", "vie", "ita", "fra", "deu"]
SEED_SIZES_K = [1, 5, 10, 25, 50]

PARTITION = "learnaccel"
NUM_NODES = 2
NUM_GPUS_PER_NODE = 8
MAX_UPDATE = 3000
TIME = 800

WEIGHT_DECAY = 0.0001

PARAMS = {
    "med_res": {
        "arch": "transformer_12_12_03",
        "lr": 0.001,
        "dropout": 0.3,
        "update_freq": 2,
        "lsmooth": 0.2,
    },
    "low_res": {
        "arch": "transformer_6_6",
        "lr": 0.001,
        "dropout": 0.3,
        "update_freq": 2,
        "lsmooth": 0.2,
    },
}


def get_args(src, tgt, seed_data, arch, lr, dropout, update_freq, lsmooth, seed):
    return [
        "python",
        "examples/nllb/low_resource/bt_data_req/sweep_1n.py",
        "-d",
        seed_data,
        "-p",
        f"{src}-{tgt}.{seed_size}k",
        "--checkpoints-dir",
        checkpoint_dir,
        "--partition",
        PARTITION,
        "--constraint",
        "volta32gb",
        "-t",
        "4",
        "-n",
        str(NUM_NODES),
        "-g",
        str(NUM_GPUS_PER_NODE),
        "--resume-failed",
        "--arch",
        arch,
        "--time",
        str(TIME),
        "--seed",
        str(seed),
        "--langs",
        f"{src},{tgt}",
        "--lang-pairs",
        f"{src}-{tgt}",
        "--ddp-backend",
        "c10d",
        "--dropout",
        str(dropout),
        "--label-smoothing",
        str(lsmooth),
        "--weight-decay",
        str(WEIGHT_DECAY),
        "--max-update",
        str(MAX_UPDATE),
        "--update-freq",
        str(update_freq),
        "--max-tokens",
        "6000",
        "--lr",
        str(lr),
        "--wandb-project",
        "bt_data_req",
        "--no-tensorboard",
        "--keep-last-epochs",
        "2",
        "--snapshot-code",
    ]


for lang in LANGS:
    for seed_size in SEED_SIZES_K:
        for seed in (2,):
            for params in PARAMS.values():
                sorted_pair = f"eng-{lang}" if lang > "eng" else f"{lang}-eng"
                seed_data = path.join(
                    DATA_DIR, f"data-bin/seed/{sorted_pair}.{seed_size}k"
                )
                checkpoint_dir = path.join(DATA_DIR, "checkpoints")
                subprocess.call(
                    get_args(
                        src="eng", tgt=lang, seed_data=seed_data, **params, seed=seed,
                    )
                )
                subprocess.call(
                    get_args(
                        src=lang, tgt="eng", seed_data=seed_data, **params, seed=seed,
                    )
                )
