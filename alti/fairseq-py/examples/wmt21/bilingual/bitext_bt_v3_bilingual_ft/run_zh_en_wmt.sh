# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=1
num_nodes=1
num_gpus_per_node=8
prefix="dense"
lang="zh"
data_type="wmt"
tag="${lang}_en.wmt-mined_${data_type}.v3.32k"

checkpoint_dir="/checkpoint/$USER/wmt21/bilingual_v3_blft.without2020.wmt/${tag}"

langs="en,${lang}"
lang_pairs="${lang}-en"
data_dir="/private/home/angelafan/wmt21/finetuning_data/final_data/without_2020/${data_type}/binarized_rev"

# dense vs moe config updates
if [ "$1" == "moe" ]; then
    is_moe=true
    prefix="moe"
else
    is_moe=""
fi
moe_param="--moe"

for seed in 1; do
  for arch in "transformer_12_12_8k_no_share"; do
    for lr in 0.0001; do
      python examples/wmt21/bilingual/sweep_1n.py \
          -d ${data_dir} -p "$prefix" \
          --checkpoints-dir ${checkpoint_dir} \
          --partition No_Language_Left_Behind --constraint volta32gb \
          -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
          --arch ${arch} \
          --time 3999 \
          --seed ${seed} \
          --langs $langs \
          --lang-pairs ${lang_pairs} \
          --ddp-backend no_c10d \
          --dropout 0.1 \
          --warmup-updates 1 \
          --label-smoothing 0.1 \
          --max-epoch 20 \
          --update-freq 1 \
          --max-tokens 2048 \
          --encoder-langtok src \
          --decoder-langtok \
          --lr ${lr} \
          ${is_moe:+"$moe_param"} \
          --tensorboard-logdir ${checkpoint_dir}/tensorboard \
          --finetune-from-model /checkpoint/chau/wmt21/bilingual_v3_blft.without2020.wmt/zh_en.wmt_minednews.v3.32k/dense.mfp16.fp16.uf1.entsrc.SPL_temperature.tmp5.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed1.no_c10d.det.mt2048.transformer.ELS12.DLS12.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8/checkpoint_best.pt \
          --snapshot-code
    done
  done
done




