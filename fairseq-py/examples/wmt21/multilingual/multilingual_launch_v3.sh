# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=1
num_nodes=32
num_gpus_per_node=8
num_shards=16
tag="wmt_only.bitext_bt.v3.16_shards"
# tag="wmt_no_cjk_1n.wmt_mined.joined.64k"
prefix="dense"

checkpoint_dir="/checkpoint/$USER/wmt21/${tag}"

langs="en,ha,is,ja,cs,ru,zh,de"
# langs="en,ha,is,ja,ps,km,ta,cs,ru,zh,de,pl"
# langs="en,ps,km,ta,cs,ru,de,pl"
# lang_pairs="en-ha,en-sw,en-is,en-da,en-no,en-sv,en-ja,en-ko,en-ps,en-fa,en-km,en-vi,en-th,en-lo,en-ta,en-ml,en-cs,en-ru,en-zh,en-de,en-pl"
# lang_pairs="en-ha,en-is,en-ja,en-ps,en-km,en-ta,en-cs,en-ru,en-zh,en-de,en-pl"
# lang_pairs="ha-en,is-en,ja-en,ps-en,km-en,ta-en,cs-en,ru-en,zh-en,de-en,pl-en"
# lang_pairs="en-cs,en-de,en-km,en-pl,en-ps,en-ru,en-ta"
lang_pairs="en-ha,en-is,en-ja,en-cs,en-ru,en-zh,en-de"

data_prefix="/large_experiments/mmt/wmt21/bt_multilingual_bin/${tag}/sharded_bin"
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

for arch in "transformer_48_48_wide" ; do
  for lr in 0.001; do
    python examples/wmt21/multilingual/sweep_1n.py \
        -d ${data_dir} -p "$prefix" \
        --checkpoints-dir ${checkpoint_dir} \
        --partition learnfair --comment "wmt21" --constraint volta32gb \
        -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
        --arch ${arch} \
        --time 4320 \
        --langs ${langs} \
        --lang-pairs ${lang_pairs} \
        --ddp-backend fully_sharded \
        --max-update 200000 \
        --max-tokens 4000 \
        --update-freq 1 \
        --save-interval-updates 10000 \
        --lr ${lr} \
        --encoder-langtok "src" \
        --decoder-langtok \
        --sampling-method 'concat' \
        --tensorboard-logdir ${checkpoint_dir}/tensorboard  
    done
done

# for arch in "transformer_24_24_wide"; do
#   for lr in 0.001; do
#     python /private/home/angelafan/wmt21/scripts/sweep_1n.py \
#         -d ${data_dir} -p "$prefix" \
#         --checkpoints-dir ${checkpoint_dir} \
#         --partition No_Language_Left_Behind --comment "wmt21" --constraint volta32gb \
#         -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
#         --arch ${arch} \
#         --time 4320 \
#         --langs ${langs} \
#         --lang-pairs ${lang_pairs} \
#         --ddp-backend fully_sharded \
#         --max-update 300000 \
#         --max-tokens 6000 \
#         --update-freq 1 \
#         --lr ${lr} \
#         --tensorboard-logdir ${checkpoint_dir}/tensorboard 
#     done
# done
