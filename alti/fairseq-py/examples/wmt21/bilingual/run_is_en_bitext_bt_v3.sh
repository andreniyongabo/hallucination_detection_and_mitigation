# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=4
num_nodes=4
num_gpus_per_node=8
prefix="dense"
lang="is"


langs="en,${lang}"
lang_pairs="${lang}-en"

tag="${lang}_en.bitext_bt.v3"
data_prefix="/large_experiments/mmt/wmt21/bt_multilingual_bin/${tag}/sharded_bin"
checkpoint_dir="/checkpoint/$USER/wmt21/bitext_bt_v3_bilingual/${tag}"

# dense vs moe config updates
if [ "$1" == "moe" ]; then
    is_moe=true
    prefix="moe"
else
    is_moe=""
fi
moe_param="--moe"

data_dir=""
num_shards=16
for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
do
  shard_id=$(printf "%03d" $shard_num)
  data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
done
data_dir=${data_dir:1}
for lr in 0.001; do
  for dropout in  0.2; do
    for lsmooth in 0.2; do
      for seed in 1; do
        for arch in "transformer_12_12_8k_03"; do
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
              --max-tokens 6000 \
              --lr ${lr} \
              ${is_moe:+"$moe_param"} \
              --tensorboard-logdir ${checkpoint_dir}/tensorboard \
              --snapshot-code
        done
      done
    done
  done
  done

