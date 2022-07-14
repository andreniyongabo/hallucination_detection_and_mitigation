#!/bin/bash

#SBATCH --job-name=gen_en_de
#SBATCH --output=gen.out
#SBATCH --error=gen.err

#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=60g
#SBATCH --time=1320
#SBATCH --constraint=volta32gb

src=en
tgt=ru
datadir=/private/home/chau/wmt21/multilingual_bin/finetuning/en_${tgt}.ft.joined.32k
input_file=/private/home/chau/wmt21/multilingual_bin/finetuning/en_${tgt}.ft.joined.32k/test.en_${tgt}.en
for seed in 1 2 3; do
  #model_dir=/checkpoint/chau/wmt21/bilingual/en_ru.ft.joined.32k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.0002.clip0.0.drop0.1.wd0.0.seed${seed}.c10d.det.mt3000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
  model_dir=/checkpoint/chau/wmt21/bilingual/en_ru.wmt_mined.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
  model_dir2=/checkpoint/chau/wmt21/bilingual/en_ru.ft.joined.32k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.0002.clip0.0.drop0.1.wd0.0.seed2.c10d.det.mt3000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
  model_dir3=/checkpoint/chau/wmt21/bilingual/en_ru.ft.joined.32k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.0002.clip0.0.drop0.1.wd0.0.seed3.c10d.det.mt3000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
  buffer_size=1024
  batch_size=16
  #model="${model_dir}/checkpoint_best.pt:${model_dir2}/checkpoint_best.pt:${model_dir3}/checkpoint_best.pt"
  model="${model_dir}/checkpoint_last.pt"
  langs="en,${tgt}"
  lang_pairs=en-${tgt}
  spm=${datadir}/sentencepiece.32000.model

  cat ${input_file} |  python fairseq_cli/interactive.py ${datadir} --path ${model} --task translation_multi_simple_epoch --langs ${langs} --lang-pairs ${lang_pairs} --bpe 'sentencepiece' --sentencepiece-model ${spm} --buffer-size ${buffer_size} --batch-size ${batch_size} -s ${src} -t ${tgt} --decoder-langtok --encoder-langtok src  --beam 5  --fp16 > ${model_dir}/test.single.log
  cat ${model_dir}/test.single.log | grep -P "^D-" | cut -f3 > ${model_dir}/test.single.output
  sacrebleu -t wmt20 -l ${src}-${tgt} < ${model_dir}/test.single.output > ${model_dir}/test.single.results
done
