BASE_DIR="/large_experiments/mmt/wmt21/bilingual_bin/self_training"
for lang in ha; do
  directory=$BASE_DIR/en_${lang}.st.128k
  spm=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards/sentencepiece.128000.vocab
  python examples/wmt21/preprocessing_scripts/binarize_slurm.py \
    --destdir ${directory} \
    --direction en-${lang} \
    --src-spm-vocab ${spm} \
    --tgt-spm-vocab ${spm}
done

