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

MODEL_FOLDER=/checkpoint/angelafan/wmt21/full_wmt.wmt_mined.joined.128k_rev/dense.fp16.uf2.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.no_c10d.det.mt1500.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
OUT_MODEL_FOLDER=/checkpoint/chau/wmt21/full_wmt.wmt_mined.joined.128k_rev/dense.fp16.uf2.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.no_c10d.det.mt1500.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
langs="en,ha,is,ja,ps,km,ta,cs,ru,zh,de,pl"
lang_pairs="en-ha,en-is,en-ja,en-ps,en-km,en-ta,en-cs,en-ru,en-zh,en-de,en-pl"
datadir=/large_experiments/mmt/full_wmt/full_wmt.wmt_mined.joined.128k_rev
spm=${datadir}/sentencepiece.128000.model
tgt=en
mkdir -p ${OUT_MODEL_FOLDER}/prev_tests
for src in cs de ru; do
  mkdir -p ${OUT_MODEL_FOLDER}/prev_tests/${src}_${tgt}
    for split in wmt14 wmt15 wmt16 wmt17 wmt18 ; do
      input_file=/private/home/chau/wmt21/previous_tests/${src}_${tgt}/${split}.${src}_${tgt}.${src}
      for lenpen in 1.0; do
        for beam in 5; do
            out_dir=$OUT_MODEL_FOLDER/prev_tests/${src}_${tgt}
            buffer_size=1024
            batch_size=16
            model=$MODEL_FOLDER/checkpoint_last.pt
            prefix=${split}.${src}_${tgt}.${lenpen}.${beam}

            cat ${input_file} |  sed 's/^/wmtdata newsdomain /' | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l ${src}  | python fairseq_cli/interactive.py ${datadir} --path ${model} --task translation_multi_simple_epoch --langs ${langs} --lang-pairs ${lang_pairs} --bpe 'sentencepiece' --sentencepiece-model ${spm} --buffer-size ${buffer_size} --batch-size ${batch_size} -s ${src} -t ${tgt} --decoder-langtok --encoder-langtok src  --beam ${beam} --lenpen ${lenpen} --fp16 > ${out_dir}/${prefix}.log
            cat ${out_dir}/${prefix}.log | grep -P "^D-" | cut -f3 > ${out_dir}/${prefix}.output
            sacrebleu -t ${split} -l ${src}-${tgt} < ${out_dir}/${prefix}.output > ${out_dir}/${prefix}.results
        done
      done
    done
  done


