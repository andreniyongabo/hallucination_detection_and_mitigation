BASE_DIR="/large_experiments/mmt/wmt21/bilingual_bin/self_training"
for lang in ha; do
  directory=$BASE_DIR/en_${lang}.st.128k
  spm=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only.bitext_bt.v3.16_shards/sentencepiece.128000.model
  python examples/wmt21/preprocessing_scripts/spm_encode_slurm.py \
    --datadir ${directory} \
    --direction en-${lang} \
    --outdir ${directory}/spm_outs \
    --src-spm-model ${spm} \
    --tgt-spm-model ${spm} 
done
