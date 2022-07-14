import argparse
import os
import random
import subprocess
import sacremoses
import time
from glob import glob


random.seed(50)


def main(args):
    input_files = glob(f"{args.input_dir}/*")
    num_files = len(input_files)
    slurmdir = os.path.abspath(args.slurmdir)
    if not os.path.exists(slurmdir):
        os.mkdir(slurmdir)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    corpus_name = input_files[0].split('/')[-2]
    job_name = f'labse.{corpus_name}'
    job_file_name = os.path.join(slurmdir, f'{job_name}.job')
    with open(job_file_name, 'w') as fout:
        fout.write(
f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output={job_file_name}.%A_%a.out
#SBATCH --error={job_file_name}.%A_%a.err

#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=60g
#SBATCH --time=720
#SBATCH --constraint=volta32gb
#SBATCH --array=0-{num_files-1}

python examples/wmt21/iterative_mining/labse_encoder.py \\
        --input-file {args.input_dir}/${{SLURM_ARRAY_TASK_ID}}.preprocessed \\
        --output-file {args.output_dir}/${{SLURM_ARRAY_TASK_ID}}.labse_embeddings
""")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir',
        default='/large_experiments/mmt/wmt21/monolingual_preprocessed/ha/news.2020.ha.shuffled.deduped.lid.txt')
    parser.add_argument('--output-dir',
            default=f'/large_experiments/mmt/wmt21/labse_embeddings/monolingual/ha/news.2020.ha.shuffled.deduped.lid.txt')
    parser.add_argument('--slurmdir', default='labse_slurmdir')
    main(parser.parse_args())


