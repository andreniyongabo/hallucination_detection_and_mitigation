#!/usr/bin/env python3
import subprocess
from os import path

DATA_DIR = "/checkpoint/jeanm/nllb/bt_data_req"

LANGS = ["asm", "zul", "amh", "tel", "urd", "vie", "ita", "fra", "deu"]
SEED_SIZES_K = [1, 5, 10, 25, 50]
MONO_SIZES_K = [150, 300]

PARTITION = "learnaccel"
NUM_TRIALS = 4
NUM_NODES = 2
NUM_GPUS_PER_NODE = 8
TIME = 1000

WEIGHT_DECAY = 0.0001

# One setup for medium resource (bitext >= 50k sents), one for low-resource.
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
        "lr": 0.002,
        "dropout": 0.3,
        "update_freq": 2,
        "lsmooth": 0.2,
    },
}


def get_args(
    src,
    tgt,
    prefix,
    seed_data,
    arch,
    lr,
    dropout,
    update_freq,
    lsmooth,
    seed,
    upsampling_factor,
):
    return [
        "python",
        "examples/nllb/low_resource/bt_data_req/sweep_1n.py",
        "-d",
        seed_data,
        "-p",
        prefix,
        "--checkpoints-dir",
        checkpoint_dir,
        "--partition",
        PARTITION,
        "--constraint",
        "volta32gb",
        "-t",
        str(NUM_TRIALS),
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
        "5000",
        "--update-freq",
        str(update_freq),
        "--max-tokens",
        "6000",
        "--lr",
        str(lr),
        "--upsample-primary",
        str(upsampling_factor),
        "--wandb-project",
        "bt_data_req-aug",
        "--no-tensorboard",
        "--keep-last-epochs",
        "2",
        "--snapshot-code",
    ]


for lang in LANGS:
    for mono_size in MONO_SIZES_K:
        for seed_size in SEED_SIZES_K:
            for params in PARAMS.items():
                # Figure out the number of sentences in the monolingual data.
                # Note: it's not simply `mono_size`, since sentences will have been
                # filtered out from the original size of the monolingual corpus.
                bt_data = f"data/bt/{lang}.{mono_size}k/{seed_size}k.eng"
                with open(path.join(DATA_DIR, bt_data), "rt",) as fin:
                    mono_lines = sum(1 for _ in fin)
                print(
                    f"data/bt/{lang}.{mono_size}k/{seed_size}k.eng has {mono_lines} lines"
                )
                # work out by how much to upsample the seed data compared to bt data
                upsampling_factor = max(1, int(round(mono_lines / (seed_size * 1000))))
                print(
                    f"Upsampling factor for {seed_size}k-seed model is {upsampling_factor}"
                )
                sorted_pair = f"eng-{lang}" if lang > "eng" else f"{lang}-eng"
                combined_data = path.join(
                    DATA_DIR, f"data-bin/seed+bt/{sorted_pair}.{seed_size}k+{mono_size}k",
                )
                checkpoint_dir = path.join(DATA_DIR, "checkpoints")
                subprocess.call(
                    get_args(
                        src="eng",
                        tgt=lang,
                        prefix=f"eng-{lang}.{seed_size}k+{mono_size}k",
                        seed_data=combined_data,
                        upsampling_factor=upsampling_factor,
                        **params,
                        seed=2,
                    )
                )
