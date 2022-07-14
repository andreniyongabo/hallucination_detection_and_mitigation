# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=1
num_nodes=1
num_gpus_per_node=8
prefix="dense"
lang="cs"
data_type="ft"
tag="${lang}_en.${data_type}.v3.32k"

checkpoint_dir="/checkpoint/$USER/wmt21/bilingual_v3_blft.without2020.wmt/${tag}"

langs="en,${lang}"
lang_pairs="${lang}-en"

data_dir="/private/home/angelafan/wmt21/finetuning_data/final_data/without_2020/wmt/binarized_rev"

# dense vs moe config updates
if [ "$1" == "moe" ]; then
    is_moe=true
    prefix="moe"
else
    is_moe=""
fi
moe_param="--moe"

for seed in 1; do
  for arch in "transformer_12_12_no_share"; do
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
          --finetune-from-model /checkpoint/chau/wmt21/bitext_bt_v3_bilingual/wmt_only_rev.bitext_bt.v3.64_shards.transformer_12_12.fp16.SPL_temperature.tmp5.adam.lr0.001.drop0.1.ls0.1.seed1.c10d.det.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32.cs_en/checkpoint_best.pt \
          --snapshot-code
    done
  done
done




