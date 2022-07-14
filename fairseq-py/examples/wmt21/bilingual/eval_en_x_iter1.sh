for lang in cs de; do
  CHECKPOINT_DIR=/checkpoint/chau/wmt21/bilingual/en_${lang}.wmt_mined.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.2.wd0.0.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
  DATADIR=~/wmt21/multilingual_bin/bilingual_en_x/en_${lang}.wmt_mined.joined.32k
  python examples/wmt21/eval_multi_benchmark.py \
    --checkpoint-dir $CHECKPOINT_DIR \
    --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
    --direction en-${lang} \
    --langs "en,${lang}" \
    --domain-tag "wmtdata newsdomain" \
    --checkpoint-name checkpoint_last.pt \
    --datadir  $DATADIR \
    --spm $DATADIR/sentencepiece.32000.model  \
    --lenpen 1.0 \
    --partition No_Language_Left_Behind \
    --local \
    --beam 4 
done

for lang in ja ru zh; do
  CHECKPOINT_DIR=/checkpoint/chau/wmt21/bilingual/en_${lang}.wmt_mined.joined.32k/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed1.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
  DATADIR=~/wmt21/multilingual_bin/bilingual_en_x/en_${lang}.wmt_mined.joined.32k
  python examples/wmt21/eval_multi_benchmark.py \
    --checkpoint-dir $CHECKPOINT_DIR \
    --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
    --direction en-${lang} \
    --langs "en,${lang}" \
    --domain-tag "wmtdata newsdomain" \
    --checkpoint-name checkpoint_last.pt \
    --datadir  $DATADIR \
    --spm $DATADIR/sentencepiece.32000.model  \
    --lenpen 1.0 \
    --partition No_Language_Left_Behind \
    --beam 4 
done
