#!/bin/bash

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate fairseq-20210318

SRC=ha
TRG=en
STEM=$HOME/data/paracrawl/corpus/paracrawl-2021-07-19.mono.hau.backtranslated

MODEL=/checkpoint/angelafan/wmt21/wmt_only.wmt_mined.joined.128k_rev_/private/home/angelafan/wmt21/finetuning_data/final_data/without_2020/wmt/binarized_rev_ft/dense.fp16.mfp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.00015.clip0.0.drop0.1.wd0.0.seed2.fully_sharded.det.mt2048.transformer.ELS24.DLS24.encffnx16384.decffnx16384.E2048.H32.NBF.ATTDRP0.1.RELDRP0.0.ngpu8/checkpoint_best-shard0.pt
SPM=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_rev.bitext_bt.v3.64_shards/sentencepiece.128000.model
DATA_DIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_rev.bitext_bt.v3.64_shards

cd $HOME/data/paracrawl

MOSES=/private/home/pkoehn/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl

xzcat $STEM.in.xz | split -l 50000 - $STEM.in.part

for part in `ls $STEM.in.part??`; do

  echo "cat $part | cut -f 2 | $REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | python /private/home/pkoehn/project/fairseq-internal/fairseq_cli/interactive.py \
      ${DATA_DIR} \
      --path ${MODEL} \
      --task translation_multi_simple_epoch \
      --langs "en,ha,is,ja,cs,ru,zh,de" \
      --lang-pairs "${SRC}-${TRG}" \
      --bpe "sentencepiece" \
      --sentencepiece-model ${SPM} \
      --buffer-size 1024 \
      --batch-size 16 -s $SRC -t $TRG \
      --decoder-langtok \
      --encoder-langtok src  \
      --beam 5 \
      --lenpen 1.2 \
      --fp16 > $part.out" > $part.sh

  chmod +x $part.sh

  sbatch \
      --error $part.err \
      --job-name ${SRC}-${TRG} \
      --gpus-per-node 1 --nodes 1 --cpus-per-task 1 \
      --time 24:00:00 --mem 50000 --no-requeue \
      --partition learnfair --ntasks-per-node 1  \
      --open-mode append --no-requeue \
      --wrap $part.sh
done
