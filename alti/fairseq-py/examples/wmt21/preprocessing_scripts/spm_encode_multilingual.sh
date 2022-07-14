directory="/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_1n.128k"
for lang in zh ja ha cs de is ru; do
  python examples/wmt21/preprocessing_scripts/spm_encode_slurm.py \
    --datadir ${directory} \
    --direction en-${lang} \
    --outdir ${directory}/spm_outs \
    --src-spm-model ${directory}/sentencepiece.128000.model \
    --tgt-spm-model ${directory}/sentencepiece.128000.model 
done
