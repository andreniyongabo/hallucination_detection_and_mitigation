CHECKPOINT_DIR=/checkpoint/chau/wmt21/wmt_only_rev.bitext_bt.v3.64_shards/dense.fp16.mfp16.uf4.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.c10d.det.mt4000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64
#CHECKPOINT_DIR=/checkpoint/chau/wmt21/wmt_only_rev.bitext_bt.v3.64_shards/dense.fp16.mfp16.uf4.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.c10d.det.mt4000.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
#CHECKPOINT_DIR=/checkpoint/chau/wmt21/wmt_only_rev.bitext_bt.v3.64_shards/dense.fp16.mfp16.uf1.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.fully_sharded.det.mt8000.transformer.ELS24.DLS24.encffnx16384.decffnx16384.E2048.H32.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_rev.bitext_bt.v3.64_shards

for cpn in checkpoint_25_190000; do
  for lang in cs de ; do
    python examples/wmt21/eval_multi_benchmark.py \
      --checkpoint-dir ${CHECKPOINT_DIR} \
      --out-dir ${CHECKPOINT_DIR}/multi_benchmark/${lang}-en  \
      --direction ${lang}-en \
      --langs "en,ha,is,ja,cs,ru,zh,de" \
      --checkpoint-name ${cpn}.pt \
      --datadir $DATADIR  \
      --spm $DATADIR/sentencepiece.128000.model  \
      --lenpen 1.0 \
      --partition No_Language_Left_Behind \
      --sentence-splitted \
      --beam 4 
  done
done
