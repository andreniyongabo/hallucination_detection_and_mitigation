#!/bin/bash

# Note: you will need to install fairseq, activate the conda env, load modules
# conda activate fairseq-20210318 (from https://fb.workplace.com/groups/fairseq/permalink/262715387865587/)
# module load anaconda3/2020.11 cudnn/v8.0.3.33-cuda.11.0 cuda/11.0 openmpi/4.1.0/cuda.11.0-gcc.9.3.0
# cd $FAIRSEQ_DIR
# git checkout <branch>
# pip install --editable .

FAIRSEQ_USER=${USER}
FAIRSEQ_DIR="${HOME}/Projects/fairseq-py"
PARTITION="devaccel"

# Languages
# langs="en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
# id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE"
langs="en_XX,fr_XX,ur_PK,zh_CN"

# Data
DATA_BIN="/large_experiments/moe/cc100_xl_roberta/final_bin"
DATA_SHARD="${DATA_BIN}/shard0"

# Checkpoint path
checkpoint_dir="/checkpoint/${FAIRSEQ_USER}/multilingual_moe_lm/"
# model_dir="${checkpoint_dir}/l1_64e_top2/l1_64e_top2.me_fp16.bm_none.tps1024.samplealpha0.7.nlangs_1.transformer_lm_gpt2_small.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf16.mu400000.s1.ngpu64/"
# model_path="${model_dir}/checkpoint_40_32000.pt"
model_dir="/large_experiments/moe/shru/moe_multi_lm/dense/dense.me_fp16.bm_none.tps1024.samplealpha0.2.nlangs_16.transformer_lm_gpt2_big_wide.dl12.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu100000.s1.ngpu64/"
model_path="${model_dir}/checkpoint_last.pt"
results_path="eval_ppl"

# Tokenization
sp_model="/large_experiments/xlmg/data/cc100_combined/v1/raw/unsharded/spm_256000.model"

# Inference hyperparameters
batch_size=1
world_size=8
model_overrides="{\\\"world_size\\\": ${world_size}, \\\"bpe\\\": \\\"sentencepiece\\\", \\\"sentencepiece_model\\\": \\\"${sp_model}\\\", \\\"moe_eval_capacity_token_fraction\\\": 0.05}"

SCRIPT="${FAIRSEQ_DIR}/fairseq_cli/eval_lm.py"
# Add --is-moe flag for MoE models
CMD="srun -u python3 ${SCRIPT} \    
    ${DATA_SHARD} \
  --langs "${langs}" \
  --task multilingual_language_modeling \
  --path ${model_path} \
  --memory-efficient-fp16 \
  --batch-size ${batch_size} \
  --distributed-world-size ${world_size} \
  --distributed-port 15187 \
  --tokens-per-sample 1024 \
  --gen-subset valid \
  --results-path ${results_path} \
  --model-overrides \"${model_overrides}\" \
  $@"

echo ${CMD}
bash -c "${CMD}" 
