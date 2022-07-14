import os
import glob
import math
import subprocess

NUM_SHARDS = 128
NGRAM_ORDER = 4
SCORER = "bleu"
THRESHOLD = 30


for lang in ["amh"]:
    src = lang
    tgt = "eng"
    output_dir = f"/large_experiments/mmt/ngram_mining/african20/{src}-{tgt}/mined.{SCORER}/sharded_scores"
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    job_file = os.path.join(output_dir, f"score_candidates.job")
    with open(job_file, "w") as job_out:
        print(
            f"""#!/bin/bash
#SBATCH --job-name=score_candidates.{src}-{tgt}
#SBATCH --output={output_dir}/logs/score_candidates.{src}-{tgt}.%A%a.out
#SBATCH --error={output_dir}/logs/score_candidates.{src}-{tgt}.%A%a.err

#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=200g
#SBATCH --time=360
#SBATCH --constraint=volta16gb
#SBATCH --array=0-{NUM_SHARDS-1}

python examples/nllb/low_resource/ngram_mining/score_candidates.py \
  --step score \
  --candidates-path "/large_experiments/mmt/ngram_mining/african20/{src}-{tgt}/candidates/*/*/shard${{SLURM_ARRAY_TASK_ID}}.bz2" \
  --output-dir {output_dir}/shard${{SLURM_ARRAY_TASK_ID}} \
  --src-txt-dir /large_experiments/mmt/ngram_mining/monolingual/{src}_txt/african20 \
  --tgt-txt-dir /large_experiments/mmt/ngram_mining/monolingual/{tgt}_txt \
  --src {src} \
  --tgt {tgt} \
  --scorer {SCORER} \
  --threshold {THRESHOLD}
""",
            file=job_out,
        )
    subprocess.call(["sbatch", job_file])
