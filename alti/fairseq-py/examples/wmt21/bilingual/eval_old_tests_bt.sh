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

tgt=en
for src in cs de ru; do
  MODEL_FOLDER=/checkpoint/chau/wmt21/bitext_bt/${src}_en.bitext_bt.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
  datadir=/private/home/chau/wmt21/multilingual_bin/bilingual/${src}_en.wmt_mined.joined.32k
  model=$MODEL_FOLDER/checkpoint_best.pt
  langs="en,${src}"
  lang_pairs=${src}-${tgt}
  out_dir=$MODEL_FOLDER/prev_tests
  mkdir -p $out_dir
    for split in wmt14 wmt15 wmt16 wmt17 wmt18 ; do
      input_file=/private/home/chau/wmt21/previous_tests/${src}_${tgt}/${split}.${src}_${tgt}.${src}
      for lenpen in 1.0; do
        for beam in 5; do
            buffer_size=1024
            batch_size=16
            spm=${datadir}/sentencepiece.32000.model
            prefix=${split}.${src}_${tgt}.${lenpen}.${beam}

            cat ${input_file} |  sed 's/^/wmtdata newsdomain /' | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${src}  | python fairseq_cli/interactive.py ${datadir} --path ${model} --task translation_multi_simple_epoch --langs ${langs} --lang-pairs ${lang_pairs} --bpe 'sentencepiece' --sentencepiece-model ${spm} --buffer-size ${buffer_size} --batch-size ${batch_size} -s ${src} -t ${tgt} --decoder-langtok --encoder-langtok src  --beam ${beam} --lenpen ${lenpen} --fp16 > ${out_dir}/${prefix}.log
            cat ${out_dir}/${prefix}.log | grep -P "^D-" | cut -f3 > ${out_dir}/${prefix}.output
            sacrebleu -t ${split} -l ${src}-${tgt} < ${out_dir}/${prefix}.output > ${out_dir}/${prefix}.results
        done
      done
    done
  done


