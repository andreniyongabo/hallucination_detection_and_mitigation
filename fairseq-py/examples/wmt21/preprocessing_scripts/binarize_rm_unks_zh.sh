#!/bin/bash
#SBATCH --job-name=bin
#SBATCH --output=bin.en_zh.%A_%a.out
#SBATCH --error=bin.en_zh.%A_%a.err

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=0
#SBATCH --mem=60g
#SBATCH --time=720
#SBATCH --array=0-63
src=en
tgt=zh
shard_id=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.64_shards
SHARD_DIR=$DATADIR/bt_preprocess/sharded_bin/shard${shard_id}
python fairseq_cli/preprocess.py \
  --source-lang ${src} \
  --target-lang ${tgt} \
  --trainpref ${SHARD_DIR}/train.sharded.rm_unks.${src}_${tgt} \
  --validpref ${DATADIR}/valid.spm.${src}_${tgt}\
  --srcdict ${DATADIR}/dict.${src}.txt \
  --tgtdict ${DATADIR}/dict.${tgt}.txt \
  --destdir ${SHARD_DIR} --workers 60 > ${SHARD_DIR}/preprocess.${src}-${tgt}.log
