import os
import glob
import math
import subprocess

NGRAM_ORDER = 4

for lang in ["amh"]:
    src = lang
    tgt = "eng"
    paths = f"/large_experiments/mmt/ngram_mining/monolingual/{src}_txt/african20/*.tok"
    num_shards = len(glob.glob(paths))
    output_dir = f"/large_experiments/mmt/ngram_mining/monolingual/{src}_index/african20/{NGRAM_ORDER}gram"
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    job_file = os.path.join(output_dir, "create_index_map.job")
    with open(job_file, "w") as job_out:
        print(
            f"""#!/bin/bash
#SBATCH --job-name=create_index_reduce.{src}-{tgt}
#SBATCH --output={output_dir}/logs/create_index_reduce.{src}-{tgt}.%A%a.out
#SBATCH --error={output_dir}/logs/create_index_reduce.{src}-{tgt}.%A%a.err

#SBATCH --partition=learnaccel
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=20g
#SBATCH --time=360
#SBATCH --constraint=volta16gb
#SBATCH --array=0-{num_shards-1}
python examples/nllb/low_resource/ngram_mining/create_index.py \
  --input-path "{output_dir}/*/shard${{SLURM_ARRAY_TASK_ID}}.bin" \
  --step reduce \
  --output-path {output_dir}/shard${{SLURM_ARRAY_TASK_ID}}_combined.pbz2
""",
            file=job_out,
        )
    subprocess.call(["sbatch", job_file])
