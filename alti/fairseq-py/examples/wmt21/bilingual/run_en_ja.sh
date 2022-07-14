# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=4
num_nodes=4
num_gpus_per_node=8
prefix="dense"
lang="ja"
data_type="wmt_mined"
tag="en_${lang}.${data_type}.joined.32k"

checkpoint_dir="/checkpoint/$USER/wmt21/bilingual/${tag}"

langs="en,${lang}"
lang_pairs="en-${lang}"

data_prefix="/private/home/chau/wmt21/multilingual_bin/bilingual_en_x/${tag}/sharded_bin"

# dense vs moe config updates
if [ "$1" == "moe" ]; then
    is_moe=true
    prefix="moe"
else
    is_moe=""
fi
moe_param="--moe"

for num_shards in 4; do
  data_dir=""
  for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
  do
    shard_id=$(printf "%03d" $shard_num)
    data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
  done
  data_dir=${data_dir:1}
  for seed in 1 2 3; do
    for arch in "transformer_12_12"; do
      for lr in 0.001; do
        python examples/wmt21/bilingual/sweep_1n.py \
            -d ${data_dir} -p "$prefix" \
            --checkpoints-dir ${checkpoint_dir} \
            --partition learnfair --constraint volta32gb \
            -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
            --arch ${arch} \
            --time 3999 \
            --seed ${seed} \
            --langs $langs \
            --lang-pairs ${lang_pairs} \
            --ddp-backend c10d \
            --dropout 0.1 \
            --label-smoothing 0.1 \
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
