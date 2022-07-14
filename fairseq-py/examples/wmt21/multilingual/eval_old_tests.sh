#!/bin/bash

#SBATCH --job-name=gen_en_cs
#SBATCH --output=gen.out
#SBATCH --error=gen.err

#SBATCH --partition=learnfair,No_Language_Left_Behind
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

MODEL_FOLDER=/checkpoint/chau/wmt21/wmt_only.wmt_mined.joined.128k_rev/wmt_only.wmt_mined.joined.128k_rev.transformer_12_12.0.1.0.001.0.1.4000.fp16.SPL_temperature.tmp5.adam.lr0.001.drop0.1.ls0.1.seed1234.shem.c10d.det.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64
datadir=/large_experiments/mmt/wmt_only/wmt_only.wmt_mined.joined.128k_rev
langs="en,ha,is,ja,cs,ru,zh,de"
lang_pairs="ha-en,is-en,ja-en,cs-en,ru-en,zh-en,de-en"
spm=${datadir}/sentencepiece.128000.model
tgt=en
mkdir -p ${MODEL_FOLDER}/prev_tests
for src in cs de ru; do
  mkdir -p ${MODEL_FOLDER}/prev_tests/${src}_${tgt}
    for split in wmt14 wmt15 wmt16 wmt17 wmt18 ; do
      input_file=/private/home/chau/wmt21/previous_tests/${src}_${tgt}/${split}.${src}_${tgt}.${src}
      for lenpen in 1.0; do
        for beam in 5; do
            out_dir=$MODEL_FOLDER/prev_tests/${src}_${tgt}
            buffer_size=1024
            batch_size=16
            model=$MODEL_FOLDER/checkpoint_45_80000.pt
            prefix=${split}.${src}_${tgt}.${lenpen}.${beam}

            cat ${input_file} |  sed 's/^/wmtdata newsdomain /' | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${src}  | python fairseq_cli/interactive.py ${datadir} --path ${model} --task translation_multi_simple_epoch --langs ${langs} --lang-pairs ${lang_pairs} --bpe 'sentencepiece' --sentencepiece-model ${spm} --buffer-size ${buffer_size} --batch-size ${batch_size} -s ${src} -t ${tgt} --decoder-langtok --encoder-langtok src  --beam ${beam} --lenpen ${lenpen} --fp16 > ${out_dir}/${prefix}.log
            cat ${out_dir}/${prefix}.log | grep -P "^D-" | cut -f3 > ${out_dir}/${prefix}.output
            sacrebleu -t ${split} -l ${src}-${tgt} < ${out_dir}/${prefix}.output > ${out_dir}/${prefix}.results
        done
      done
    done
  done


