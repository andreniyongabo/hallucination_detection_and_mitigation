import os
import glob
import math
import subprocess


for lang in ["amh"]:
    src = lang
    tgt = "eng"
    paths = f"/large_experiments/mmt/data/backtranslation/african20/{src}-{tgt}/*.{tgt}"
    output_dir = f"/large_experiments/mmt/ngram_mining/monolingual/{src}_txt/african20"
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    num_shards = math.ceil(len(glob.glob(paths)) / 100)  # 10M sentences per shard
    print(f"Tokenizing {paths}. Splitting into {num_shards} shards")
    job_file = os.path.join(output_dir, "tokenize.job")
    with open(job_file, "w") as job_out:
        print(
            f"""#!/bin/bash
#SBATCH --job-name=tokenize.{src}-{tgt}
#SBATCH --output={output_dir}/logs/tokenize.{src}-{tgt}.%A%a.out
#SBATCH --error={output_dir}/logs/tokenize.{src}-{tgt}.%A%a.err

#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=20g
#SBATCH --time=360
#SBATCH --constraint=volta16gb
#SBATCH --array=0-{num_shards-1}
python examples/nllb/low_resource/ngram_mining/tokenizer.py \
  --paths "{paths}" \
  --output-dir {output_dir} \
  --num-shards {num_shards} \
  --shard-id ${{SLURM_ARRAY_TASK_ID}} \
  --is-translated \
  --tgt {tgt} \
  --src {src}
""",
            file=job_out,
        )
    subprocess.call(["sbatch", job_file])
