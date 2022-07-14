# bash sweep_1n.sh
num_trials=1
num_nodes=8
num_gpus_per_node=8
num_shards=64
tag="wmt_no_cjk_1n.wmt_mined.joined.64k"
prefix="sparse"

checkpoint_dir="/checkpoint/$USER/wmt21/moe_small/zero3_${tag}"

langs="cs,de,en,km,pl,ps,ru,ta"
lang_pairs="en-cs,en-de,en-km,en-pl,en-ps,en-ru,en-ta"

data_prefix="/private/home/chau/wmt21/multilingual_bin/${tag}/sharded_bin"
data_dir=""
for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
do
  shard_id=$(printf "%03d" $shard_num)
  data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
done
data_dir=${data_dir:1}


for arch in "transformer_6_6"; do
  # for lr in 0.0002 0.0005 0.001; do
  #   python examples/wmt21/multilingual/moe/sweep_1n.py \
  #       -d ${data_dir} -p "$prefix" \
  #       --checkpoints-dir ${checkpoint_dir} \
  #       --partition learnfair --constraint volta32gb \
  #       -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
  #       --arch ${arch} \
  #       --time 3999 \
  #       --langs $langs \
  #       --lang-pairs ${lang_pairs} \
  #       --ddp-backend no_c10d \
  #       --max-update 300000 \
  #       --max-tokens 6000 \
  #       --lr ${lr} \
  #       --experts-per-gpu 1 \
  #       --tensorboard-logdir ${checkpoint_dir}/tensorboard \
  #       --snapshot-code 
  # done
  for experts in 2 4; do
    python examples/wmt21/multilingual/moe/sweep_1n.py \
        -d ${data_dir} -p "$prefix" \
        --checkpoints-dir ${checkpoint_dir} \
        --partition learnfair --constraint volta32gb \
        -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
        --arch ${arch} \
        --time 3999 \
        --langs $langs \
        --lang-pairs ${lang_pairs} \
        --ddp-backend no_c10d \
        --max-update 300000 \
        --max-tokens 6000 \
        --lr 0.0005 \
        --experts-per-gpu ${experts} \
        --tensorboard-logdir ${checkpoint_dir}/tensorboard \
        --snapshot-code 
  done
done
