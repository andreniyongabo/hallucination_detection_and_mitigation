import argparse
import os
import random
import subprocess

from glob import glob
random.seed(50)


def main(args):
    slurmdir = os.path.abspath(args.slurmdir)
    if not os.path.exists(slurmdir):
        os.mkdir(slurmdir)


    for direction in args.directions.split(','):
        tag = args.destdir.split('/')[-1]
        job_name = f"bin.{tag}"

        job_file_name = os.path.join(slurmdir, 'slurm_{0}.job'.format(job_name))
        with open(job_file_name, 'w') as fout:
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
#SBATCH --partition=learnfair
""".format(job_name, slurmdir)
            )

            fout.write(
f"""
python examples/wmt21/preprocessing_scripts/binarize.py \
    --destdir {args.destdir} \
    --src-spm-vocab {args.src_spm_vocab} \
    --tgt-spm-vocab {args.tgt_spm_vocab} \
    --direction {direction};
""")

        subprocess.call(['sbatch', job_file_name])


"""
SPM encode data by launching many jobs on SLURM.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    DIR = "/private/home/chau/wmt21/multilingual_bin/bilingual/en_ha.wmt.joined.32k"
    parser.add_argument('--destdir',
            default=DIR)
    parser.add_argument('--src-spm-vocab',
            default=f'{DIR}/sentencepiece.256000.vocab')
    parser.add_argument('--tgt-spm-vocab',
            default=f'{DIR}/sentencepiece.256000.vocab')
    parser.add_argument('--directions',
            default='en-ha')
    parser.add_argument('--processes', type=int, default=64)
    parser.add_argument('--slurmdir', default='slurmdir')
    args = parser.parse_args()
    main(args)

