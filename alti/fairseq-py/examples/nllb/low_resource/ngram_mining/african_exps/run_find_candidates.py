import os
import glob
import subprocess

NUM_TGT_SHARDS = 1024
NGRAM_MAX_FREQ = 500
NGRAM_ORDER = 4
NUM_SCORE_SHARDS = 128
MAX_CONCURRENT_JOBS = 256


for lang in ["amh"]:
    src = lang
    tgt = "eng"
    paths = f"/large_experiments/mmt/data/backtranslation/african20/{src}-{tgt}/*.{tgt}"
    output_dir = f"/large_experiments/mmt/ngram_mining/african20/{src}-{tgt}/candidates"
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    num_src_shards = len(
        glob.glob(
            f"/large_experiments/mmt/ngram_mining/monolingual/{src}_index/african20/{NGRAM_ORDER}gram/shard*_combined.pbz2"
        )
    )
    print(f"Finding candidates from {num_src_shards} {lang} shards")
    for src_shard in range(num_src_shards):
        job_file = os.path.join(output_dir, f"find_candidates.{src_shard}.job")
        with open(job_file, "w") as job_out:
            print(
                f"""#!/bin/bash
#SBATCH --job-name=find_candidates.{src}-{tgt}.{src_shard}
#SBATCH --output={output_dir}/logs/find_candidates.{src}-{tgt}.{src_shard}.%A%a.out
#SBATCH --error={output_dir}/logs/find_candidates.{src}-{tgt}.{src_shard}.%A%a.err

#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=200g
#SBATCH --time=360
#SBATCH --constraint=volta16gb
#SBATCH --array=0-{NUM_TGT_SHARDS-1}%{MAX_CONCURRENT_JOBS}

python examples/nllb/low_resource/ngram_mining/find_candidates.py \
  --src-index-path /large_experiments/mmt/ngram_mining/monolingual/{src}_index/african20/{NGRAM_ORDER}gram/shard{src_shard}_combined.pbz2 \
  --tgt-index-path /large_experiments/mmt/ngram_mining/monolingual/{tgt}_index/{NGRAM_ORDER}gram/shard${{SLURM_ARRAY_TASK_ID}}_combined.pbz2 \
  --output-path {output_dir}/{src_shard}/candidates.{src_shard}.${{SLURM_ARRAY_TASK_ID}} \
  --num-shards {NUM_SCORE_SHARDS} \
  --ngram-max-freq {NGRAM_MAX_FREQ}
""",
                file=job_out,
            )
        subprocess.call(["sbatch", job_file])
