#for lang in cs de ; do
#    #CHECKPOINT_DIR=/checkpoint/chau/wmt21/dense_blft.without2020.wmt/${lang}_en.ft.v3/dense.fp16.mfp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt2048.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
#    CHECKPOINT_DIR=/checkpoint/chau/wmt21/dense_blft.without2020.wmt/${lang}_en.ft.v3/dense.fp16.mfp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt2048.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
#    DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_rev.bitext_bt.v3.64_shards
#    python examples/wmt21/eval_multi_benchmark.py \
#      --checkpoint-dir $CHECKPOINT_DIR \
#      --out-dir $CHECKPOINT_DIR/multi_benchmark/${lang}-en \
#      --direction ${lang}-en \
#      --langs "en,ha,is,ja,cs,ru,zh,de" \
#      --domain-tag "wmtdata newsdomain" \
#      --checkpoint-name checkpoint_best.pt \
#      --datadir  $DATADIR \
#      --spm $DATADIR/sentencepiece.128000.model  \
#      --lenpen 1.0 \
#      --partition No_Language_Left_Behind \
#      --sentence-splitted \
#      --beam 4 
#done
#for lang in ja ru zh; do
#    #CHECKPOINT_DIR=/checkpoint/chau/wmt21/dense_blft.without2020.wmt/${lang}_en.ft.v3/dense.fp16.mfp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt2048.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
#    CHECKPOINT_DIR=/checkpoint/chau/wmt21/dense_blft.without2020.wmt/${lang}_en.ft.v3/dense.fp16.mfp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt2048.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
#    DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_rev.bitext_bt.v3.64_shards
#    python examples/wmt21/eval_multi_benchmark.py \
#      --checkpoint-dir $CHECKPOINT_DIR \
#      --out-dir $CHECKPOINT_DIR/multi_benchmark/${lang}-en \
#      --direction ${lang}-en \
#      --langs "en,ha,is,ja,cs,ru,zh,de" \
#      --domain-tag "wmtdata newsdomain" \
#      --checkpoint-name checkpoint_best.pt \
#      --datadir  $DATADIR \
#      --spm $DATADIR/sentencepiece.128000.model  \
#      --lenpen 1.0 \
#      --partition No_Language_Left_Behind \
#      --test-set wmt20 \
#      --beam 4 
#done


for lang in ha is ; do
    #CHECKPOINT_DIR=/checkpoint/chau/wmt21/dense_blft.without2020.wmt/${lang}_en.ft.v3/dense.fp16.mfp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt2048.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
    CHECKPOINT_DIR=/checkpoint/chau/wmt21/dense_blft.without2020.wmt/${lang}_en.ft.v3/dense.fp16.mfp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt2048.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu8
    DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_rev.bitext_bt.v3.64_shards
    python examples/wmt21/eval_multi_benchmark.py \
      --checkpoint-dir $CHECKPOINT_DIR \
      --out-dir $CHECKPOINT_DIR/multi_benchmark/${lang}-en \
      --direction ${lang}-en \
      --langs "en,ha,is,ja,cs,ru,zh,de" \
      --domain-tag "wmtdata newsdomain" \
      --checkpoint-name checkpoint_best.pt \
      --datadir  $DATADIR \
      --spm $DATADIR/sentencepiece.128000.model  \
      --lenpen 1.0 \
      --partition No_Language_Left_Behind \
      --test-set newsdev2021 \
      --beam 4 
done


####
## En-X
####
DATADIR=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards
# CHECKPOINT_DIR=/checkpoint/jcross/wmt21/averaged/mlft_dense_12_12_en2x
# CHECKPOINT_DIR=/large_experiments/mmt/jcross/averaged/mlft_dense_12_12_en2x_last20
CHECKPOINT_NAME=checkpoint_averaged.pt

CHECKPOINT_DIR=/large_experiments/mmt/jcross/wmt21/zero3_wmt_only.bitext_bt.v3.16_shards/dec_fullffn_spec.fp16.mfp16.uf4.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.no_c10d.det.mt4000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64/
CHECKPOINT_NAME=checkpoint_best.pt

for lang in cs de zh ; do
    python examples/wmt21/eval_multi_benchmark.py \
     --checkpoint-dir $CHECKPOINT_DIR \
     --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
     --direction en-${lang} \
     --langs "en,ha,is,ja,cs,ru,zh,de" \
     --domain-tag "wmtdata newsdomain" \
     --checkpoint-name $CHECKPOINT_NAME \
     --datadir  $DATADIR \
     --spm $DATADIR/sentencepiece.128000.model  \
     --lenpen 1.0 \
     --partition No_Language_Left_Behind \
     --sentence-splitted \
     --beam 4
done
for lang in ja ru ; do
   python examples/wmt21/eval_multi_benchmark.py \
     --checkpoint-dir $CHECKPOINT_DIR \
     --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
     --direction en-${lang} \
     --langs "en,ha,is,ja,cs,ru,zh,de" \
     --domain-tag "wmtdata newsdomain" \
     --checkpoint-name $CHECKPOINT_NAME \
     --datadir  $DATADIR \
     --spm $DATADIR/sentencepiece.128000.model  \
     --lenpen 1.0 \
     --partition No_Language_Left_Behind \
     --test-set wmt20 \
     --beam 4
done
# for lang in ha is ; do
for lang in is ; do
   python examples/wmt21/eval_multi_benchmark.py \
     --checkpoint-dir $CHECKPOINT_DIR \
     --out-dir $CHECKPOINT_DIR/multi_benchmark/en-${lang} \
     --direction en-${lang} \
     --langs "en,ha,is,ja,cs,ru,zh,de" \
     --domain-tag "wmtdata newsdomain" \
     --checkpoint-name $CHECKPOINT_NAME \
     --datadir  $DATADIR \
     --spm $DATADIR/sentencepiece.128000.model  \
     --lenpen 1.0 \
     --partition No_Language_Left_Behind \
     --test-set newsdev2021 \
     --beam 4
done