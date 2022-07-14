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
    output_dir = (
        f"/large_experiments/mmt/ngram_mining/african20/{src}-{tgt}/mined.{SCORER}"
    )
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    job_file = os.path.join(output_dir, f"merge_filter.{THRESHOLD}.job")
    with open(job_file, "w") as job_out:
        print(
            f"""#!/bin/bash
#SBATCH --job-name=merge_filter.{src}-{tgt}
#SBATCH --output={output_dir}/logs/merge_filter.{src}-{tgt}.%A%a.out
#SBATCH --error={output_dir}/logs/merge_filter.{src}-{tgt}.%A%a.err

#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=200g
#SBATCH --time=360
#SBATCH --constraint=volta16gb

python examples/nllb/low_resource/ngram_mining/score_candidates.py \
  --step merge_filter \
  --candidates-path "/large_experiments/mmt/ngram_mining/african20/amh-eng/mined.bleu/sharded_scores/shard*/scores.{src}-{tgt}" \
  --output-dir {output_dir} \
  --src-txt-dir /large_experiments/mmt/ngram_mining/monolingual/{src}_txt/african20 \
  --tgt-txt-dir /large_experiments/mmt/ngram_mining/monolingual/{tgt}_txt \
  --src {src} \
  --tgt {tgt} \
  --scorer {SCORER} \
""",
            file=job_out,
        )
    #subprocess.call(["sbatch", job_file])
