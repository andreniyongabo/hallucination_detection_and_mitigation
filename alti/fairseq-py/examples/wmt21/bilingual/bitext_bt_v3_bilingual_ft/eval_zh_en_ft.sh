for lang in zh; do
  for drop in 0.1 ; do
    #CHECKPOINT_DIR=/checkpoint/chau/wmt21/bilingual_v3_blft.without2020.wmt/zh_en.wmt.v3.32k/dense.mfp16.fp16.uf1.entsrc.SPL_temperature.tmp5.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed1.no_c10d.det.mt2048.transformer.ELS12.DLS12.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
    #CHECKPOINT_DIR=/checkpoint/chau/wmt21/bilingual_v3_blft.without2020.wmt/zh_en.wmt_minednews.v3.32k/dense.mfp16.fp16.uf1.entsrc.SPL_temperature.tmp5.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed1.no_c10d.det.mt2048.transformer.ELS12.DLS12.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
    CHECKPOINT_DIR=/checkpoint/chau/wmt21/bilingual_v3_blft.without2020.wmt/zh_en.wmt-mined_wmt.v3.32k/dense.mfp16.fp16.uf1.entsrc.SPL_temperature.tmp5.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed1.no_c10d.det.mt2048.transformer.ELS12.DLS12.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
    DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_rev.bitext_bt.v3.64_shards
    python examples/wmt21/eval_multi_benchmark.py \
      --checkpoint-dir $CHECKPOINT_DIR \
      --out-dir $CHECKPOINT_DIR/multi_benchmark/${lang}-en \
      --direction ${lang}-en \
      --langs "en,${lang}" \
      --domain-tag "wmtdata newsdomain" \
      --checkpoint-name checkpoint_best.pt \
      --datadir  $DATADIR \
      --spm $DATADIR/sentencepiece.128000.model  \
      --lenpen 1.0 \
      --partition No_Language_Left_Behind \
      --sentence-splitted \
      --beam 4 
    done
done

