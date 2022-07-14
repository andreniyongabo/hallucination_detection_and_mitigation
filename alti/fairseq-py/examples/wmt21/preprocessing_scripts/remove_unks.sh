#!/bin/bash
#SBATCH --job-name=rm_unks
#SBATCH --output=rm_unks.en_zh.%A_%a.out
#SBATCH --error=rm_unks.en_zh.%A_%a.err

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=0
#SBATCH --mem=60g
#SBATCH --time=720
#SBATCH --array=1-3
shard_id=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
tgt=ha
python examples/wmt21/preprocessing_scripts/remove_unks.py \
  --input-prefix /large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.64_shards/bt_preprocess/sharded_bin/shard${shard_id}/train.sharded.en_${tgt} \
  --output-prefix /large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.64_shards/bt_preprocess/sharded_bin/shard${shard_id}/train.sharded.rm_unks.en_${tgt} \
  --direction en-${tgt} \
  --tgt-dict /large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.64_shards/dict.${tgt}.txt
