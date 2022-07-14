lang=de
CHECKPOINT_DIR=/checkpoint/chau/wmt21/bitext_bt_ft/de_en.ft.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.0002.clip0.0.drop0.2.wd0.0.seed1.c10d.det.mt3000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu2
DATADIR=/large_experiments/mmt/wmt21/bilingual_bin/bitext_bt/${lang}_en.bitext_bt.32k
python examples/wmt21/eval_multi_benchmark.py \
  --checkpoint-dir $CHECKPOINT_DIR \
  --out-dir $CHECKPOINT_DIR/multi_benchmark/${lang}-en \
  --direction ${lang}-en \
  --langs "en,${lang}" \
  --domain-tag "wmtdata newsdomain" \
  --checkpoint-name checkpoint5.pt \
  --datadir  $DATADIR \
  --spm $DATADIR/sentencepiece.32000.model  \
  --lenpen 1.0 \
  --partition No_Language_Left_Behind \
  --local \
  --sentence-splitted \
  --beam 4 
