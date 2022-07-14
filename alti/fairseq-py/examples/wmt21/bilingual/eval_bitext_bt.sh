#for lang in cs de is ja ru zh; do
#  CHECKPOINT_DIR=/checkpoint/chau/wmt21/bitext_bt/en_x
#  DATADIR=/large_experiments/mmt/wmt21/bilingual_bin/bitext_bt/en_${lang}.bitext_bt.32k
#  python examples/wmt21/eval_multi_benchmark.py \
#    --checkpoint-dir $CHECKPOINT_DIR \
#    --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
#    --direction en-${lang} \
#    --langs "en,${lang}" \
#    --domain-tag "wmtdata newsdomain" \
#    --checkpoint-name en_${lang}_bt.pt \
#    --datadir  $DATADIR \
#    --spm $DATADIR/sentencepiece.32000.model  \
#    --lenpen 1.0 \
#    --partition No_Language_Left_Behind \
#    --beam 4 
#done

for lang in ha; do
  for dropout in 0.2 0.3; do
    CHECKPOINT_DIR=/checkpoint/chau/wmt21/bitext_bt/en_ha.bitext_bt.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.3.wd0.0.ls0.3.seed1.c10d.det.mt6000.usp1.transformer.ELS6.DLS6.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP${dropout}.RELDRP0.3.ngpu8
    DATADIR=/large_experiments/mmt/wmt21/bilingual_bin/bitext_bt/en_${lang}.bitext_bt.32k
    python examples/wmt21/eval_multi_benchmark.py \
      --checkpoint-dir $CHECKPOINT_DIR \
      --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
      --direction en-${lang} \
      --langs "en,${lang}" \
      --domain-tag "mineddata otherdomain" \
      --checkpoint-name checkpoint_best.pt \
      --datadir $DATADIR \
      --spm $DATADIR/sentencepiece.32000.model  \
      --lenpen 1.0 \
      --partition No_Language_Left_Behind \
      --local \
      --beam 4 
  done
done
