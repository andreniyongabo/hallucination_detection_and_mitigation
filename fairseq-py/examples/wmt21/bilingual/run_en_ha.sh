# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=4
num_nodes=4
num_gpus_per_node=8
prefix="dense"
lang="ha"


langs="en,${lang}"
lang_pairs="en-${lang}"

tag="en_${lang}.bitext_bt_st.32k"
data_prefix="/large_experiments/mmt/wmt21/bilingual_bin/self_training/en_ha.bitext_bt_st.128k/sharded_bin"
checkpoint_dir="/checkpoint/$USER/wmt21/bitext_bt_st/${tag}"

# dense vs moe config updates
if [ "$1" == "moe" ]; then
    is_moe=true
    prefix="moe"
else
    is_moe=""
fi
moe_param="--moe"

data_dir=""
num_shards=64
for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
do
  shard_id=$(printf "%03d" $shard_num)
  data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
done
data_dir=${data_dir:1}
for lr in 0.001; do
  for dropout in  0.3; do
    for lsmooth in  0.3; do
      for seed in 1 ; do
        #for arch in "transformer_6_6_wide" "transformer_6_6_wide_03"; do
        for arch in "transformer_6_6_wide_02" "transformer_6_6_wide_03"; do
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
              --ddp-backend c10d \
              --dropout ${dropout} \
              --label-smoothing ${lsmooth} \
              --max-update 100000 \
              --update-freq 4 \
              --max-tokens 4000 \
              --lr ${lr} \
              ${is_moe:+"$moe_param"} \
              --tensorboard-logdir ${checkpoint_dir}/tensorboard \
              --snapshot-code
        done
      done
    done
  done
done

#for num_shards in 1; do
#  data_dir=""
#  for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
#  do
#    shard_id=$(printf "%03d" $shard_num)
#    data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
#  done
#  data_dir=${data_dir:1}
#  for seed in 1 2 3; do
#    for arch in "transformer_4_4"; do
#      python examples/wmt21/bilingual/sweep_1n.py \
#          -d ${data_dir} -p "$prefix" \
#          --checkpoints-dir ${checkpoint_dir} \
#          --partition priority --constraint volta16gb \
#          -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
#          --arch ${arch} \
#          --time 3999 \
#          --seed ${seed} \
#          --langs $langs \
#          --lang-pairs ${lang_pairs} \
#          --ddp-backend c10d \
#          --dropout 0.3 \
#          --label-smoothing 0.3 \
#          --max-epoch 8 \
#          --update-freq 1 \
#          --max-tokens 3000 \
#          --lr ${lr} \
#          ${is_moe:+"$moe_param"} \
#          --tensorboard-logdir ${checkpoint_dir}/tensorboard \
#          --finetune-from-model /checkpoint/chau/wmt21/bilingual/en_ha.wmt_fb.joined.32k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.3.wd0.0.seed${seed}.c10d.det.mt3000.transformer.ELS4.DLS4.encffnx2048.decffnx2048.E512.H8.NBF.ATTDRP0.3.RELDRP0.3.ngpu32/checkpoint40.pt \
#          --snapshot-code
#    done
#  done
#done
