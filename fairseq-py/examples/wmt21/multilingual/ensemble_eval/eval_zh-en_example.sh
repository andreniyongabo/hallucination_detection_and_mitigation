
DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_rev.bitext_bt.v3.64_shards

CHECKPOINT_MULTI_24=/checkpoint/chau/wmt21/dense_blft.without2020.wmt/zh_en.ft.v3/dense.fp16.mfp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt2048.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8/checkpoint_best.pt
CHECKPOINT_MULTI_12=/checkpoint/chau/wmt21/dense_blft.without2020.wmt/zh_en.ft.v3/dense.fp16.mfp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt2048.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8/checkpoint_best.pt
CHECKPOINT_BI=/checkpoint/chau/wmt21/bilingual_v3_blft.without2020.wmt/zh_en.wmt-mined_wmt.v3.32k/dense.mfp16.fp16.uf1.entsrc.SPL_temperature.tmp5.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed1.no_c10d.det.mt2048.transformer.ELS12.DLS12.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8/checkpoint_best.pt
CHECKPOINT_LIST=$CHECKPOINT_MULTI_24:$CHECKPOINT_MULTI_12:$CHECKPOINT_BI

python examples/wmt21/eval_multi_benchmark.py \
    --model $CHECKPOINT_LIST \
    --out-dir /checkpoint/jcross/eval_zh-en_ensemble_example_output \
    --direction zh-en \
    --langs "en,ha,is,ja,cs,ru,zh,de" \
    --domain-tag "wmtdata newsdomain" \
    --datadir  $DATADIR \
    --spm $DATADIR/sentencepiece.128000.model  \
    --lenpen 1.0 \
    --partition No_Language_Left_Behind \
    --test-set wmt20 \
    --beam 4