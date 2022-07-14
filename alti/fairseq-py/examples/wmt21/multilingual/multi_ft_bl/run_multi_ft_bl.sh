# bash sweep_multi.sh moe/dense en_to_many/many_to_en
num_trials=1
num_nodes=1
prefix="dense"
for lang in cs ha is ja ru zh; do
  tag="${lang}_en.ft.v3"

  checkpoint_dir="/checkpoint/$USER/wmt21/dense_blft.without2020.wmt/${tag}"

  langs="en,ha,is,ja,cs,ru,zh,de"
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

  for num_gpus_per_node in 8; do
    for arch in "transformer_24_24"; do
      for lr in 0.0001; do
        python examples/wmt21/multilingual/sweep_1n.py \
            -d ${data_dir} -p "$prefix" \
            --checkpoints-dir ${checkpoint_dir} \
            --partition No_Language_Left_Behind --constraint volta32gb \
            -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
            --arch ${arch} \
            --time 3999 \
            --seed 1 \
            --langs $langs \
            --lang-pairs ${lang_pairs} \
            --ddp-backend no_c10d \
            --dropout 0.1 \
            --warmup-updates 1 \
            --max-epoch 20 \
            --update-freq 1 \
            --max-tokens 2048 \
            --encoder-langtok src \
            --decoder-langtok \
            --lr ${lr} \
            ${is_moe:+"$moe_param"} \
            --tensorboard-logdir ${checkpoint_dir}/tensorboard \
            --finetune-from-model /checkpoint/chau/wmt21/wmt_only_rev.bitext_bt.v3.64_shards/dense.fp16.mfp16.uf4.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.c10d.det.mt4000.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128/checkpoint_27_100000.pt \
            --snapshot-code
      done
    done
  done
done
