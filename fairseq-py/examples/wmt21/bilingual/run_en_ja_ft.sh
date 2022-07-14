# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=4
num_nodes=1
num_gpus_per_node=8
prefix="dense"
lang="ja"
data_type="ft"
tag="en_${lang}.${data_type}.joined.32k"

checkpoint_dir="/checkpoint/$USER/wmt21/bilingual/${tag}"

langs="en,${lang}"
lang_pairs="en-${lang}"

data_prefix="/private/home/chau/wmt21/multilingual_bin/finetuning/${tag}/sharded_bin"

# dense vs moe config updates
if [ "$1" == "moe" ]; then
    is_moe=true
    prefix="moe"
else
    is_moe=""
fi
moe_param="--moe"

for num_shards in 1; do
  data_dir=""
  for (( shard_num=0; shard_num<$num_shards; shard_num++ ))
  do
    shard_id=$(printf "%03d" $shard_num)
    data_dir=${data_dir}:${data_prefix}/shard${shard_id}/
  done
  data_dir=${data_dir:1}
  for seed in 1 2 3; do
    for arch in "transformer_12_12"; do
      for lr in 0.0002; do
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
            --max-epoch 20 \
            --update-freq 1 \
            --max-tokens 3000 \
            --lr ${lr} \
            ${is_moe:+"$moe_param"} \
            --tensorboard-logdir ${checkpoint_dir}/tensorboard \
            --finetune-from-model /checkpoint/chau/wmt21/bilingual/en_ja.wmt_mined.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed${seed}.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32/checkpoint95.pt \
            --snapshot-code
      done
    done
  done
done

