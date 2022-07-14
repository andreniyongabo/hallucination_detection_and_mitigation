# -----------------------------------------------------------
# Script to binarize from a bped raw text file
# by launching mutliple jobs on slurm.
# -----

#!/usr/bin/env python  # noqa
import argparse
import os
import random
import subprocess

from glob import glob

random.seed(50)


def main(args):
    all_input_files = glob(os.path.join(args.datadir + "*.txt")) + glob(
        os.path.join(args.datadir + "*.shard*")
    )
    random.shuffle(all_input_files)

    slurmdir = os.path.abspath(args.slurmdir)
    if not os.path.exists(slurmdir):
        os.mkdir(slurmdir)

    job_id = 0
    for i in range(0, len(all_input_files), args.chunks):
        job_name = "roberta_{0}".format(job_id)

        input_files = all_input_files[i : i + args.chunks]

        job_file_name = os.path.join(slurmdir, "slurm_{0}.job".format(job_name))
        with open(job_file_name, "w") as fout:
            fout.write(
                """#!/bin/bash
#SBATCH --job-name={0}.job
#SBATCH --output={1}/{0}.log
#SBATCH --error={1}/{0}.stderr
#SBATCH --time=4320
#SBATCH --mem-per-cpu=6G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=dev,learnfair
""".format(
                    job_name, slurmdir
                )
            )

            for input_file in input_files:

                prefname = "trainpref" if "train" in input_file else "validpref"
                lang = input_file.split("/")[-1].split(".")[0]

                destdir_subfolder = ""
                if "valid" in input_file:
                    destdir_subfolder = "valid"
                else:
                    assert "train" in input_file
                    if "shard" in input_file:
                        destdir_subfolder = input_file.split("/")[-1].split(".")[-1]
                    else:
                        destdir_subfolder = "shard0"

                destdir = os.path.join(args.outdir, lang, destdir_subfolder)

                fout.write(
                    """
python ~/fairseq-py-2/fairseq_cli/preprocess.py --{0} {1} --only-source --srcdict {2} --destdir {3} --workers {4};
""".format(
                        prefname,
                        input_file,
                        args.dict,
                        destdir,
                        args.processes,
                    )
                )

        subprocess.call(["sbatch", job_file_name])
        job_id += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        default="/large_experiments/xlmg/data/ccnet_head/020919/bped/train_test/",
    )
    parser.add_argument(
        "--outdir", default="/large_experiments/xlmg/data/ccnet_head/020919/bin/"
    )
    parser.add_argument(
        "--dict", default="/large_experiments/xlmg/data/ccnet_head/020919/bin/dict.txt"
    )
    parser.add_argument("--processes", type=int, default=64)
    parser.add_argument("--slurmdir", default="slurmdir")
    parser.add_argument("--chunks", type=int, default=8)
    main(parser.parse_args())
