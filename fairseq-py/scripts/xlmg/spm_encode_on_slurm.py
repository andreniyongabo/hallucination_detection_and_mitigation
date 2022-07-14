# -----------------------------------------------------------
# SPM encode data by launching many jobs on SLURM.
# -----------------------------------------------------------

#!/usr/bin/env python  # noqa
import argparse
import os
import random
import subprocess

from glob import glob

random.seed(50)


def main(args):
    """
    Reads all the files from datadir directory and distributes them to multiple
    machines for processing via slurm.
    """
    all_input_files = glob(os.path.join(args.datadir + "*.txt")) + glob(
        os.path.join(args.datadir + "*.shard*")
    )
    random.shuffle(all_input_files)

    slurmdir = os.path.abspath(args.slurmdir)
    if not os.path.exists(slurmdir):
        os.mkdir(slurmdir)

    job_id = 0
    for i in range(0, len(all_input_files), args.chunks):
        job_name = "process_all_{0}".format(job_id)

        input_files = all_input_files[i : i + args.chunks]
        output_files = [
            x.replace(args.datadir, args.outdir)
            for x in all_input_files[i : i + args.chunks]
        ]

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
#SBATCH --partition=learnfair,XLM-G
""".format(
                    job_name, slurmdir
                )
            )

            for input_file, output_file in zip(input_files, output_files):
                fout.write(
                    """
python scripts/xlmg/spm_encode_multiprocess.py --inputs {0} --outputs {1} --model {2} --processes {3} --keep-empty;
""".format(
                        input_file,
                        output_file,
                        args.spm_model,
                        args.processes,
                    )
                )

        subprocess.call(["sbatch", job_file_name])
        job_id += 1


"""
SPM encode data by launching many jobs on SLURM.
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", default="/large_experiments/moe/cc100_xl_roberta/sharded/"
    )
    parser.add_argument(
        "--outdir", default="/large_experiments/moe/cc100_xl_roberta/spm_encoded/"
    )
    parser.add_argument(
        "--spm-model",
        default="/large_experiments/flores/namangoyal/cc100_combined/spm_256000.model",
    )
    parser.add_argument("--processes", type=int, default=64)
    parser.add_argument("--slurmdir", default="slurmdir")
    parser.add_argument(
        "--chunks",
        type=int,
        default=64,
        help="divide work into chunks / multiple machines",
    )
    main(parser.parse_args())
