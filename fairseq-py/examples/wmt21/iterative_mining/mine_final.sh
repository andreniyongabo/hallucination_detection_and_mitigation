#!/bin/bash
#SBATCH --job-name=labse_mine_slurm/en_zh
#SBATCH --output=labse_mine_slurm/en_zh.%A_%a.out
#SBATCH --error=labse_mine_slurm/en_zh.%A_%a.err

#SBATCH --partition=No_Language_Left_Behind
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=200g
#SBATCH --time=3000
#SBATCH --constraint=volta32gb

num_shards=64
src=zh
tgt=en
dim=768
k=4
threshold=1.08
min_count=500
out_dir=/large_experiments/mmt/wmt21/iterative_mining/mine_labse/en-zh


python examples/wmt21/iterative_mining/mine_sharded.py \
  --src-lang ${src} \
  --tgt-lang ${tgt} \
  --dim ${dim} \
  --neighborhood ${k} \
  --src-dir "/large_experiments/mmt/wmt21/labse_embeddings/monolingual/zh/news*" \
  --tgt-dir "/large_experiments/mmt/wmt21/labse_embeddings/monolingual/en/*" \
  --output ${out_dir} \
  --stdout ${out_dir}/stdout.final \
  --threshold ${threshold}  \
  --min-count ${min_count} \
