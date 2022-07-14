SRC=en
TGT=ps
tag="en_${TGT}.wmt_mined.joined.32k"
DATA=/private/home/chau/wmt21/multilingual_bin/bilingual/${tag}
SPM=$DATA/sentencepiece.32000.model

langs="en,ps"
lang_pairs="en-ps"
gen_split='valid'
for CHECKPOINT_NAME in checkpoint_last ; do
  for lr in 0.001; do
    MODEL_FOLDER=/checkpoint/chau/wmt21/bilingual/en-ps.joined.transformer_12_12.0.001.0.1.fp16.SPL_temperature.tmp5.adam.lr0.001.drop0.1.ls0.1.seed2.c10d.det.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
    MODEL=$MODEL_FOLDER/${CHECKPOINT_NAME}.pt
    OUTDIR=${MODEL_FOLDER}/${SRC}-${TGT}_${CHECKPOINT_NAME}_${gen_split}
    mkdir -p $OUTDIR
    echo $OUTDIR
    echo "python fairseq_cli/generate.py \
      ${DATA} \
      --path ${MODEL} \
      --task translation_multi_simple_epoch \
      --langs "${langs}" \
      --lang-pairs "${lang_pairs}" \
      --source-lang ${SRC} --target-lang ${TGT} \
      --encoder-langtok "src" \
      --decoder-langtok \
      --gen-subset ${gen_split} \
      --beam 4 \
      --bpe 'sentencepiece' \
      --sentencepiece-model ${SPM} \
      --sacrebleu \
      --max-tokens 5000 \
      --max-sentences 20 | tee ${OUTDIR}/gen_best.out" > ${OUTDIR}/gen.sh

    sbatch \
      --error ${OUTDIR}/eval.err \
      --job-name ${SRC}-${TGT}.eval \
      --gpus-per-node 1 --nodes 1 --cpus-per-task 8 \
      --time 1000 --mem 50000 --no-requeue \
      --partition dev --ntasks-per-node 1  \
      --open-mode append --no-requeue \
      --wrap "srun sh ${OUTDIR}/gen.sh; srun sh examples/wmt21/evaluation.sh ${OUTDIR} ${TGT} > ${OUTDIR}/test_bleu_results"
  done
done

