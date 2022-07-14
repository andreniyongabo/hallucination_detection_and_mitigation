#!/bin/bash
num_trials=1
num_nodes=8
num_gpus_per_node=8

direction=${1:-"x_en"}
if [[ $direction == "x_en" ]]; then
  tag="african20.x_en.64k"
  lang_pairs="amh-eng,ful-eng,hau-eng,ibo-eng,kam-eng,lin-eng,lug-eng,luo-eng,nso-eng,nya-eng,orm-eng,sna-eng,som-eng,swh-eng,umb-eng,wol-eng,xho-eng,yor-eng,zul-eng"
else
  tag="african20.en_x.64k"
  lang_pairs="eng-amh,eng-ful,eng-hau,eng-ibo,eng-kam,eng-lin,eng-lug,eng-luo,eng-nso,eng-nya,eng-orm,eng-sna,eng-som,eng-swh,eng-umb,eng-wol,eng-xho,eng-yor,eng-zul"
fi
checkpoint_dir="/checkpoint/$USER/nllb/${tag}/checkpoints"
sweep_script=examples/nllb/low_resource/african_exps/sweep_1n.py
data_dir=/large_experiments/mmt/multilingual_bin/flores_african_en_20langs.64k/data_bin/shard000/
langs="eng,amh,ful,hau,ibo,kam,lin,lug,luo,nso,nya,orm,sna,som,swh,umb,wol,xho,yor,zul"


for dropout in 0.2; do
  for arch in "transformer_12_12" ; do
    for lr in 0.001; do
      python $sweep_script \
          -d ${data_dir} -p "$tag" \
          --checkpoints-dir ${checkpoint_dir} \
          --partition learnaccel --comment "$tag" --constraint volta32gb \
          -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
          --arch ${arch} \
          --time 4320 \
          --langs ${langs} \
          --lang-pairs ${lang_pairs} \
          --ddp-backend c10d \
          --max-update 30000 \
          --max-tokens 4000 \
          --update-freq 4 \
          --save-interval-updates 10000 \
          --lr ${lr} \
          --encoder-langtok "src" \
          --decoder-langtok \
          --sampling-method 'concat' \
          --no-tensorboard \
          --dropout ${dropout} \
          --wandb-project african20 \
          --keep-last-epochs 2
      done
  done

done
