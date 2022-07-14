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

for src in de ; do
  for tag in ""; do
    data_type=wmt_mined
    tgt=en
    datadir=/private/home/chau/wmt21/multilingual_bin/bilingual/${src}_en.${data_type}.joined.32k
    input_file=${datadir}/valid.${src}_en.${src}
    for lenpen in 1.0; do
      for beam in 5; do
        for cpn in checkpoint5 checkpoint6 ; do
          out_dir=/checkpoint/chau/wmt21/bitext_bt_ft/de_en.ft.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.0002.clip0.0.drop0.2.wd0.0.seed1.c10d.det.mt3000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu2
          #out_dir=/checkpoint/chau/wmt21/bitext_bt/${src}_en.bitext_bt.32k/$model_d
          buffer_size=1024
          batch_size=16
          model=$out_dir/${cpn}.pt
          langs="en,${src}"
          lang_pairs=${src}-${tgt}
          spm=${datadir}/sentencepiece.32000.model
          prefix=valid.${cpn}.single.${lenpen}.${beam}

          cat ${input_file} | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${src}  | python fairseq_cli/interactive.py ${datadir} --path ${model} --task translation_multi_simple_epoch --langs ${langs} --lang-pairs ${lang_pairs} --bpe 'sentencepiece' --sentencepiece-model ${spm} --buffer-size ${buffer_size} --batch-size ${batch_size} -s ${src} -t ${tgt} --decoder-langtok --encoder-langtok src  --beam ${beam} --lenpen ${lenpen} --fp16 > ${out_dir}/${prefix}.log
          cat ${out_dir}/${prefix}.log | grep -P "^D-" | cut -f3 > ${out_dir}/${prefix}.output
          sacrebleu /private/home/chau/wmt21/dev_data/${src}_en/valid.${src}_${tgt}.${tgt} -l ${src}-${tgt} < ${out_dir}/${prefix}.output > ${out_dir}/${prefix}.results
        done
      done
    done
  done
done
