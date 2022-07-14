# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=4
num_nodes=8
num_gpus_per_node=8
local_experts=1
num_shards=64
tag="wmt_no_cjk_1n.wmt_mined.joined.64k"
prefix="dense"

checkpoint_dir="/fsx/$USER/wmt21/zero3_${tag}"

langs="cs,de,en,km,pl,ps,ru,ta"
lang_pairs="en-cs,en-de,en-km,en-pl,en-ps,en-ru,en-ta"

data_prefix="/fsx/shru/data/${tag}/sharded_bin"
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

for arch in "transformer_12_12"; do
  for lr in 0.001 ; do
    python examples/wmt21/multilingual/sweep_1n.py \
        -d ${data_dir} -p "$prefix" \
        --checkpoints-dir ${checkpoint_dir} \
        -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
        --arch ${arch} \
        --mem 0 \
        --cpus-per-task 80 \
        --time 5999 \
        --langs $langs \
        --lang-pairs ${lang_pairs} \
        --ddp-backend fully_sharded \
        --max-update 300000 \
        --max-tokens 6000 \
        --lr ${lr} \
        ${is_moe:+"$moe_param"} \
	--moe-local-experts ${local_experts} \
        --tensorboard-logdir ${checkpoint_dir}/tensorboard \
        --snapshot-code 
    done
done
