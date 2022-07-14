#!/bin/bash

#SBATCH --job-name=gen_bilingual
#SBATCH --output=gen_bilingual.out
#SBATCH --error=gen_bilingual.err

#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=60g
#SBATCH --time=1320
#SBATCH --constraint=volta32gb
MOSES=~edunov/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl

for split in "test" valid; do
  for tgt in cs; do
    src=en
    datadir=/private/home/chau/wmt21/multilingual_bin/finetuning/en_${tgt}.ft.joined.32k
    spm=${datadir}/sentencepiece.32000.model
    source_file=/private/home/chau/wmt21/multilingual_bin/finetuning/en_${tgt}.ft.joined.32k/${split}.en_${tgt}.en
    ref_file=/private/home/chau/wmt21/multilingual_bin/finetuning/en_${tgt}.ft.joined.32k/${split}.en_${tgt}.${tgt}
    #model_dir=/checkpoint/chau/wmt21/bilingual/en_de.wmt_mined.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.2.wd0.0.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
    model_dir=/checkpoint/chau/wmt21/bilingual/en_cs.wmt_mined.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.2.wd0.0.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
    #model_dir=/checkpoint/chau/wmt21/bilingual/en_ru.wmt_mined.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
    #model_dir=/checkpoint/chau/wmt21/bilingual/en_zh.wmt_mined.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
    model="${model_dir}/checkpoint_last.pt"
    buffer_size=1024
    batch_size=16
    langs="en,${tgt}"
    lang_pairs="en-${tgt}"

    cat ${source_file} | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${src} | python fairseq_cli/interactive.py ${datadir} --path ${model} --task translation_multi_simple_epoch --langs ${langs} --lang-pairs ${lang_pairs} --bpe 'sentencepiece' --sentencepiece-model ${spm} --buffer-size ${buffer_size} --batch-size ${batch_size} -s ${src} -t ${tgt} --decoder-langtok --encoder-langtok src  --beam 5  --fp16 > ${model_dir}/${split}.${src}_${tgt}.log
    cat ${model_dir}/${split}.${src}_${tgt}.log | grep -P "^D-" | cut -f3 > ${model_dir}/${split}.${src}_${tgt}.output

    sacrebleu -l ${src}-${tgt} ${ref_file} < ${model_dir}/${split}.${src}_${tgt}.output > ${model_dir}/${split}.${src}_${tgt}.results
  done
done
