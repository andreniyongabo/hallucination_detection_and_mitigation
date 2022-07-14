for lang in zh; do
  for drop in 0.1 0.2; do
    CHECKPOINT_DIR=/checkpoint/chau/wmt21/bitext_bt_v3_bilingual/wmt_only.bitext_bt.v3.16_shards.en-zh.transformer_12_12.fp16.SPL_temperature.tmp5.adam.lr0.001.drop${drop}.ls0.1.seed1234.c10d.det.transformer.ELS12.DLS12.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
    DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards
    python examples/wmt21/eval_multi_benchmark.py \
      --checkpoint-dir $CHECKPOINT_DIR \
      --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
      --direction en-${lang} \
      --langs "en,${lang}" \
      --domain-tag "wmtdata newsdomain" \
      --checkpoint-name checkpoint_best.pt \
      --datadir  $DATADIR \
      --spm $DATADIR/sentencepiece.128000.model  \
      --lenpen 1.0 \
      --partition No_Language_Left_Behind \
      --beam 4 
    done
done

