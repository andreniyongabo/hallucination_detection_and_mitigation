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

for tgt in zh; do
  for domain_tag in "wmtdata newsdomain"; do
    tag_name="${domain_tag/ /_}"
    echo $tag_name
    for split in test ; do
      src=en
      datadir=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards
      input_file=/private/home/chau/wmt21/dev_data/en_${tgt}/${split}.${src}_${tgt}.${src}
      for lenpen in 1.0; do
        for beam in 5; do
            out_dir=/checkpoint/chau/wmt21/bitext_bt_v3_bilingual/wmt_only.bitext_bt.v3.16_shards.en-${tgt}.transformer_12_12.fp16.SPL_temperature.tmp5.adam.lr0.001.drop0.1.ls0.1.seed1234.c10d.det.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
            buffer_size=1024
            batch_size=16
            model=$out_dir/checkpoint_best.pt
            langs="en,${tgt}"
            lang_pairs=${src}-${tgt}
            spm=${datadir}/sentencepiece.128000.model
            prefix=${src}_${tgt}.${split}.${tag_name}.best.single.${lenpen}.${beam}

            cat ${input_file} | sed "s/^/${domain_tag} /" | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${src}  | python fairseq_cli/interactive.py ${datadir} --path ${model} --task translation_multi_simple_epoch --langs ${langs} --lang-pairs ${lang_pairs} --bpe 'sentencepiece' --sentencepiece-model ${spm} --buffer-size ${buffer_size} --batch-size ${batch_size} -s ${src} -t ${tgt} --decoder-langtok --encoder-langtok src  --beam ${beam} --lenpen ${lenpen} --fp16 > ${out_dir}/${prefix}.log
            cat ${out_dir}/${prefix}.log | grep -P "^D-" | cut -f3 > ${out_dir}/${prefix}.output
            sacrebleu /private/home/chau/wmt21/dev_data/${src}_${tgt}/${split}.${src}_${tgt}.${tgt} -l ${src}-${tgt} < ${out_dir}/${prefix}.output > ${out_dir}/${prefix}.test.results
        done
      done
    done
  done
done
