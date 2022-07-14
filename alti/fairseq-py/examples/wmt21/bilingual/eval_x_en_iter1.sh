for lang in de ; do
  CHECKPOINT_DIR=/checkpoint/chau/wmt21/bilingual/x_en.ft.joined.32k
  DATADIR=~/wmt21/multilingual_bin/bilingual/${lang}_en.wmt_mined.joined.32k
  python examples/wmt21/eval_multi_benchmark.py \
    --checkpoint-dir $CHECKPOINT_DIR \
    --out-dir $CHECKPOINT_DIR/multi_benchmark/${lang}-en \
    --direction ${lang}-en \
    --langs "en,${lang}" \
    --domain-tag "wmtdata newsdomain" \
    --checkpoint-name ${lang}_en_seed1.pt \
    --datadir  $DATADIR \
    --spm $DATADIR/sentencepiece.32000.model  \
    --lenpen 1.0 \
    --partition No_Language_Left_Behind \
    --sentence-splitted \
    --local \
    --beam 4 
done
