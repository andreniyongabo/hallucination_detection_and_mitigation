#CHECKPOINT_DIR=/checkpoint/chau/wmt21/wmt_only.bitext_bt.v3.16_shards/dense.fp16.uf4.entsrc.SPL_concat.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.c10d.det.mt4000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64
#CHECKPOINT_DIR=/checkpoint/chau/wmt21/wmt_only.bitext_bt.v3.16_shards/dense.fp16.uf4.entsrc.SPL_concat.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.c10d.det.mt4000.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
CHECKPOINT_DIR=/checkpoint/chau/wmt21/wmt_only.bitext_bt.v3.16_shards/dense.fp16.mfp16.uf1.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.fully_sharded.det.mt8000.transformer.ELS24.DLS24.encffnx16384.decffnx16384.E2048.H32.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards
for cpn in checkpoint_13_90000-shard0.pt; do
  for lang in ha; do
    python examples/wmt21/eval_multi_benchmark.py \
      --checkpoint-dir $CHECKPOINT_DIR \
      --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
      --direction en-${lang} \
      --langs "en,ha,is,ja,cs,ru,zh,de" \
      --domain-tag "mineddata otherdomain" \
      --checkpoint-name ${cpn} \
      --datadir  /large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards \
      --spm $DATADIR/sentencepiece.128000.model  \
      --lenpen 1.0 \
      --partition No_Language_Left_Behind \
      --sentence-splitted \
      --beam 4 
  done
  for lang in cs de is ja ru zh; do
    python examples/wmt21/eval_multi_benchmark.py \
      --checkpoint-dir $CHECKPOINT_DIR \
      --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
      --direction en-${lang} \
      --langs "en,ha,is,ja,cs,ru,zh,de" \
      --domain-tag "wmtdata newsdomain" \
      --checkpoint-name ${cpn} \
      --datadir  /large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards \
      --spm $DATADIR/sentencepiece.128000.model  \
      --lenpen 1.0 \
      --partition No_Language_Left_Behind \
      --sentence-splitted \
      --beam 4 
  done
done
