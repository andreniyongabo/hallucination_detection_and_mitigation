# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=4
num_nodes=8
num_gpus_per_node=8
num_shards=64
prefix="dense"
tag="wmt_only.wmt_mined.joined.128k"


checkpoint_dir="/checkpoint/$USER/wmt21/zero3_${tag}"

langs="en,ha,is,ja,cs,ru,zh,de"
lang_pairs="en-ha,en-is,en-ja,en-cs,en-ru,en-zh,en-de"

data_prefix="/large_experiments/mmt/wmt_only/${tag}/sharded_bin"
data_dir=""
for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
do
  shard_id=$(printf "%03d" $shard_num)
  data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
done
data_dir=${data_dir:1}

# dense vs moe config updates
if [ "$1" == "moe" ]; then
    is_moe=true
    prefix="moe"
else
    is_moe=""
fi
moe_param="--moe"

for arch in "transformer_12_12" "transformer_24_24"; do
  for lr in 0.001; do
    python examples/wmt21/multilingual/sweep_1n.py \
        -d ${data_dir} -p "$prefix" \
        --checkpoints-dir ${checkpoint_dir} \
        --partition learnfair,No_Language_Left_Behind --constraint volta32gb \
        -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
        --arch ${arch} \
        --time 3999 \
        --langs $langs \
        --lang-pairs ${lang_pairs} \
        --ddp-backend fully_sharded \
        --max-update 300000 \
        --max-tokens 16000 \
        --lr ${lr} \
        ${is_moe:+"$moe_param"} \
        --tensorboard-logdir ${checkpoint_dir}/tensorboard \
        --snapshot-code 
    done
done
