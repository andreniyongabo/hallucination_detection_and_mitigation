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
SWEEP_SCRIPT="${FAIRSEQ_DIR}/fb_sweep/sweep_dense_multilingual_lm.py"

# CC100 XL
data="cc100_xl_unilm"

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
su_ID,om_KE,uk_UA,as_IN"\
    [134]="af_ZA,am_ET,ar_AR,ar_AR_rom,as_IN,az_AZ,az_IR,be_BY,bg_BG,bm_ML,bn_IN,bn_IN_rom,br_FR,bs_BA,ca_ES,\
cb_IQ,ci_IT,cs_CZ,cx_PH,cy_GB,da_DK,de_DE,el_GR,en_XX,eo_EO,es_XX,et_EE,eu_ES,fa_IR,ff_NG,fi_FI,\
fr_XX,fy_NL,ga_IE,gd_GB,gl_ES,gn_PY,gu_IN,ha_NG,he_IL,hi_IN,hi_IN_rom,hr_HR,ht_HT,hu_HU,hy_AM,id_ID,\
ig_NG,is_IS,it_IT,iu_CA,ja_XX,jv_ID,ka_GE,kg_AO,kk_KZ,km_KH,kn_IN,ko_KR,ku_TR,ky_KG,la_VA,lg_UG,\
ln_CD,lo_LA,lt_LT,lv_LV,mg_MG,mk_MK,ml_IN,mn_MN,mr_IN,ms_MY,my_MM,my_MM_zaw,ne_NP,nl_XX,no_XX,ns_ZA,\
om_KE,or_IN,pa_IN,pl_PL,ps_AF,pt_XX,q3_CV,qa_MM,qd_MM,qf_CM,qh_PH,qi_PH_rom,qj_ML,ql_ML_rom,qm_AO,qp_AO,\
qq_KE,qu_PE,qw_KE,qx_KE,qy_KE,ro_RO,ru_RU,sa_IN,sd_PK,si_LK,sk_SK,sl_SI,so_SO,sq_AL,sr_RS,ss_SZ,\
su_ID,sv_SE,sw_KE,ta_IN,ta_IN_rom,te_IN,te_IN_rom,th_TH,ti_ET,tl_XX,tn_BW,tr_TR,uk_UA,ur_PK,ur_PK_rom,\
uz_UZ,vi_VN,wo_SN,xh_ZA,yo_NG,zh_CN,zh_TW,zu_ZA"
)

# Experiment control
wpb=$(( 1 * 1024 * 1024 ))
batch_size=2
tokens_per_sample=1024
update_freq=$(( wpb / tokens_per_sample / batch_size / num_gpus_per_node / 8 ))

for num_langs in {134,}; do
  for num_nodes in {8,}; do
    type="${data}_l${num_langs}_dense_top2"
    checkpoint_dir="/checkpoint/${FAIRSEQ_USER}/multilingual_dense_lm/${type}/"

    CMD="python3 ${SWEEP_SCRIPT} \
        --script ${SCRIPT} \
        --data ${data} \
        --prefix $type \
        --checkpoints-dir ${checkpoint_dir} \
        --langs ${dialects[${num_langs}]}\
        --partition ${partition} \
        --constraint volta32gb \
        --num-trials ${num_trials} \
        --num-nodes ${num_nodes} \
        --num-gpus ${num_gpus_per_node} \
        --update-freq ${update_freq} \
        --mem 470G \
        --resume-failed \
        --no-wandb \
    "
    echo ${CMD}
    ${CMD}
  done
done