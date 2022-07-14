#!/bin/bash

#SBATCH --job-name=gen_multilingual
#SBATCH --output=gen_multilingual.out
#SBATCH --error=gen_multilingual.err

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
  for tgt in cs de ru; do
    src=en
    datadir=/private/home/chau/wmt21/multilingual_bin/wmt_no_cjk_1n.wmt_mined.joined.64k
    spm=${datadir}/sentencepiece.64000.model
    source_file=/private/home/chau/wmt21/multilingual_bin/finetuning/en_${tgt}.ft.joined.32k/${split}.en_${tgt}.en
    ref_file=/private/home/chau/wmt21/multilingual_bin/finetuning/en_${tgt}.ft.joined.32k/${split}.en_${tgt}.${tgt}
    model_dir=/checkpoint/chau/wmt21/wmt_no_cjk_1n.wmt_mined.joined.64k/dense.fp16.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.c10d.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64
    model="${model_dir}/checkpoint_last.pt"
    buffer_size=1024
    batch_size=16
    langs="cs,de,en,km,pl,ps,ru,ta"
    lang_pairs="en-cs,en-de,en-km,en-pl,en-ps,en-ru,en-ta"

    cat ${source_file} | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${src} | python fairseq_cli/interactive.py ${datadir} --path ${model} --task translation_multi_simple_epoch --langs ${langs} --lang-pairs ${lang_pairs} --bpe 'sentencepiece' --sentencepiece-model ${spm} --buffer-size ${buffer_size} --batch-size ${batch_size} -s ${src} -t ${tgt} --decoder-langtok --encoder-langtok src  --beam 5  --fp16 > ${model_dir}/${split}.${src}_${tgt}.log
    cat ${model_dir}/${split}.${src}_${tgt}.log | grep -P "^D-" | cut -f3 > ${model_dir}/${split}.${src}_${tgt}.output

    sacrebleu -l ${src}-${tgt} ${ref_file} < ${model_dir}/${split}.${src}_${tgt}.output > ${model_dir}/${split}.${src}_${tgt}.results
  done
done

