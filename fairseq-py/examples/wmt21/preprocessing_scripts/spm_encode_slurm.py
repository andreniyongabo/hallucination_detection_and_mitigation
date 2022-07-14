import argparse
import os
import random
import subprocess

from glob import glob
random.seed(50)


def spm_on_slurm(all_input_files, spm_model, args, tag):
    slurmdir = os.path.abspath(args.slurmdir)
    if not os.path.exists(slurmdir):
        os.mkdir(slurmdir)
    for i in range(0, len(all_input_files), args.chunks):
        job_name=f"{tag}_{i}"

        input_files = all_input_files[i: i+args.chunks]
        output_files = [x.replace(args.datadir, args.outdir) for x in all_input_files[i: i+args.chunks]]

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

            for input_file, output_file in zip(input_files, output_files):
                fout.write(
"""
python examples/wmt21/preprocessing_scripts/spm_encode_multiprocess.py --inputs {0} --outputs {1} --model {2} --processes {3} --keep-empty;
""".format(
input_file,
output_file,
spm_model,
args.processes,
)
                )

        subprocess.call(['sbatch', job_file_name])


def main(args):
    src, tgt = args.direction.split('-')
    src_files = glob(os.path.join(args.datadir + f"/train.*.{src}")) + \
            glob(os.path.join(args.datadir + f"/valid.*.{src}")) + \
            glob(os.path.join(args.datadir + f"/test.*.{src}"))
    tgt_files = glob(os.path.join(args.datadir + f"/train.*.{tgt}")) + \
            glob(os.path.join(args.datadir + f"/valid.*.{tgt}")) + \
            glob(os.path.join(args.datadir + f"/test.*.{tgt}"))
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)
    tag = args.datadir.split('/')[-1]
    spm_on_slurm(src_files, args.src_spm_model, args, tag=f"{tag}.{src}")
    spm_on_slurm(tgt_files, args.tgt_spm_model, args, tag=f"{tag}.{tgt}")



"""
SPM encode data by launching many jobs on SLURM.
"""
if __name__ == '__main__':
    DIR = "/private/home/chau/wmt21/multilingual_bin/bilingual/en_ha.wmt.separated.32k"
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir',
            default=DIR)
    parser.add_argument('--direction',
            default='en-ha')
    parser.add_argument('--outdir',
            default=f'{DIR}/spm_outs')
    parser.add_argument('--src-spm-model',
            default=f'{DIR}/sentencepiece.en.32000.model')
    parser.add_argument('--tgt-spm-model',
            default=f'{DIR}/sentencepiece.ha.32000.model')
    parser.add_argument('--processes', type=int, default=64)
    parser.add_argument('--slurmdir', default='slurmdir')
    parser.add_argument('--chunks', type=int, default=1)
    main(parser.parse_args())
