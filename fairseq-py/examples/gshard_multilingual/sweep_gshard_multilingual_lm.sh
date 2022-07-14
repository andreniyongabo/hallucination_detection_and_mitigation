#!/bin/bash

FAIRSEQ_USER="${USER}"

FAIRSEQ_DIR="${HOME}/Projects/fairseq-py"
# Note: you will need to install fairseq, activate the conda env, load modules
# conda activate fairseq-20200821 (from https://fb.workplace.com/groups/fairseq/permalink/262715387865587/)
# module load anaconda3/2020.11 cudnn/v8.0.3.33-cuda.11.0 cuda/11.0 openmpi/4.1.0/cuda.11.0-gcc.9.3.0
# cd $FAIRSEQ_DIR
# git checkout <branch>
# pip install --editable .

partition="learnaccel"
num_trials=1
num_gpus_per_node=8
num_experts_per_gpu=1

SCRIPT="${FAIRSEQ_DIR}/fairseq_cli/train.py"
SWEEP_SCRIPT="${FAIRSEQ_DIR}/fb_sweep/sweep_gshard_multilingual_lm.py"

# Data
# DATA_BIN="/datasets01/cc100-bin/072820/250"
# DATA_BIN="/large_experiments/moe/cc100_xl_full/bin"
DATA_BIN="/large_experiments/moe/cc100_xl_roberta/final_bin"
NUM_DATA_SHARDS=64
DATA_DIR="${DATA_BIN}/shard0"
for i in $(seq 1 $(($NUM_DATA_SHARDS-1))); do
    DATA_DIR="${DATA_DIR}:${DATA_BIN}/shard${i}";
done

# Languages
declare -A dialects=(
    [1]="en_XX"\
    [4]="en_XX,fr_XX,ur_PK,zh_CN"\
    [16]="en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT"\
    [32]="en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE"\
    [64]="en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE,\
fa_IR,sk_SK,ms_MY,lv_LV,az_AZ,tl_XX,ja_XX,bn_IN,eu_ES,te_IN,mn_MN,si_LK,af_ZA,ne_NP,kn_IN,eo_EO,\
cy_GB,gu_IN,ps_AF,ky_KG,uz_UZ,hi_IN_rom,ga_IE,ur_PK_rom,sv_SE,bn_IN_rom,jv_ID,gd_GB,lo_LA,sa_IN,br_FR,my_MM"\
    [100]="en_XX,vi_VN,ru_RU,de_DE,fr_XX,es_XX,bg_BG,el_GR,ar_AR,tr_TR,th_TH,hi_IN,ur_PK,sw_KE,zh_CN,ht_HT,\
id_ID,ro_RO,pt_XX,hu_HU,pl_PL,it_IT,hr_HR,lt_LT,ca_ES,sl_SI,sq_AL,et_EE,sr_RS,ta_IN,mk_MK,qu_PE,\
fa_IR,sk_SK,ms_MY,lv_LV,az_AZ,tl_XX,ja_XX,bn_IN,eu_ES,te_IN,mn_MN,si_LK,af_ZA,ne_NP,kn_IN,eo_EO,\
cy_GB,gu_IN,ps_AF,ky_KG,uz_UZ,hi_IN_rom,ga_IE,ur_PK_rom,sv_SE,bn_IN_rom,jv_ID,gd_GB,lo_LA,sa_IN,br_FR,my_MM,\
no_XX,da_DK,fi_FI,ko_KR,nl_XX,he_IL,cs_CZ,is_IS,gl_ES,kk_KZ,ka_GE,hy_AM,la_VA,be_BY,ml_IN,zh_TW,\
mr_IN,am_ET,pa_IN,ku_TR,so_SO,ha_NG,my_MM_zaw,sd_PK,te_IN_rom,km_KH,or_IN,ta_IN_rom,fy_NL,mg_MG,bs_BA,xh_ZA,\
su_ID,om_KE,uk_UA,as_IN"
)

# Experiment control
wpb=$((  2 * 1024 * 1024 ))
batch_size=2
tokens_per_sample=1024
update_freq=$(( wpb / tokens_per_sample / batch_size / num_gpus_per_node / 8 ))

for num_langs in {32,}; do
  for num_nodes in {1,}; do
    num_experts=$(( num_experts_per_gpu * num_nodes * num_gpus_per_node ))
    type="l${num_langs}_${num_experts}e_top2"
    checkpoint_dir="/checkpoint/${FAIRSEQ_USER}/multilingual_moe_lm/${type}/"

    CMD="python3 ${SWEEP_SCRIPT} \
        --script ${SCRIPT} \
        --data ${DATA_DIR} \
        --prefix $type \
        --checkpoints-dir ${checkpoint_dir} \
        --langs ${dialects[${num_langs}]}\
        --partition ${partition} \
        --constraint volta32gb \
        --num-trials ${num_trials} \
        --num-nodes ${num_nodes} \
        --num-gpus ${num_gpus_per_node} \
        --local-experts ${num_experts_per_gpu} \
        --update-freq ${update_freq} \
        --mem 470G \
        --resume-failed \
        --no-wandb
    "
    echo ${CMD}
    ${CMD}
  done
done
