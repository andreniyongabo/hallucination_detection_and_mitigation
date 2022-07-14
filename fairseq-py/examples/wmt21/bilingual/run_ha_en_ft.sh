# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=4
num_nodes=4
num_gpus_per_node=8
prefix="dense"
lang="ha"


langs="en,ha,is,ja,cs,ru,zh,de"
lang_pairs="${lang}-en"

tag="ha_en.bitext_bt.v3"
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
mkdir -p ${checkpoint_dir}/tensorboard 
data_dir=${data_dir:1}
for lr in 0.00015; do
  for dropout in  0.3; do
    for lsmooth in 0.2; do
      for seed in 2; do
        for arch in "transformer_12_12"; do
          python examples/wmt21/bilingual/sweep_1n.py \
              -d ${data_dir} -p "$prefix" \
              --checkpoints-dir ${checkpoint_dir} \
              --partition No_Language_Left_Behind --constraint volta32gb \
              -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
              --arch ${arch} \
              --time 3999 \
              --warmup-updates 1 \
              --seed ${seed} \
              --langs $langs \
              --lang-pairs ${lang_pairs} \
              --ddp-backend c10d \
              --dropout ${dropout} \
              --label-smoothing ${lsmooth} \
              --max-update 10000 \
              --update-freq 8 \
              --finetune-from-model /checkpoint/chau/wmt21/wmt_only_rev.bitext_bt.v3.64_shards/dense.fp16.mfp16.uf4.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.c10d.det.mt4000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64/checkpoint_25_190000.pt \
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
