#!/bin/bash

#SBATCH --job-name=gen_ha_en
#SBATCH --output=gen.out
#SBATCH --error=gen.err

#SBATCH --partition=learnfair
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

src=ha
tgt=en
datadir=/large_experiments/mmt/wmt21/bilingual_bin/ha_en.wmt_fb.32k
for tag in "" ".mined_other"; do
  input_file=$datadir/valid${tag}.${src}_${tgt}.${src}
  for lenpen in 1.0; do
    for beam in 5; do
	for model_d in dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.3.wd0.0.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32 dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.3.wd0.0.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.3.RELDRP0.3.ngpu32; do
          out_dir=/checkpoint/$USER/wmt21/bitext_bt/ha_en.bitext_bt.32k/$model_d
          model=$out_dir/checkpoint_best.pt
          buffer_size=1024
          batch_size=16
          langs="en,${src}"
          lang_pairs=${src}-en
          spm=${datadir}/sentencepiece.32000.model
          prefix=valid${tag}.single.${lenpen}.${beam}

          cat ${input_file} | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${src}  | python fairseq_cli/interactive.py ${datadir} --path ${model} --task translation_multi_simple_epoch --langs ${langs} --lang-pairs ${lang_pairs} --bpe 'sentencepiece' --sentencepiece-model ${spm} --buffer-size ${buffer_size} --batch-size ${batch_size} -s ${src} -t ${tgt} --decoder-langtok --encoder-langtok src  --beam ${beam} --lenpen ${lenpen} --fp16 > ${out_dir}/${prefix}.log
          cat ${out_dir}/${prefix}.log | grep -P "^D-" | cut -f3 > ${out_dir}/${prefix}.output
          sacrebleu -l ${src}-${tgt} /private/home/chau/wmt21/dev_data/${src}_en/valid.${src}_${tgt}.${tgt} < ${out_dir}/${prefix}.output > ${out_dir}/${prefix}.results
        done
    done
  done
done
