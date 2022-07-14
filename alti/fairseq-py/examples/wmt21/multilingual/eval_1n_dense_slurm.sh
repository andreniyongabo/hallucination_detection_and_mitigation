tag="wmt_no_cjk_1n.wmt_mined.joined.64k"
DATA=/private/home/chau/wmt21/multilingual_bin/${tag}
SRC=en
SPM=$DATA/sentencepiece.64000.model

checkpoint_dir="/checkpoint/chau/wmt21/${tag}"
langs="cs,de,en,km,pl,ps,ru,ta"
lang_pairs="en-cs,en-de,en-km,en-pl,en-ps,en-ru,en-ta"
#CHECKPOINT_NAME=checkpoint_last
gen_split='valid'
for CHECKPOINT_NAME in checkpoint_last ; do
  for lr in 0.001 0.005; do
    MODEL_FOLDER=/checkpoint/chau/wmt21/wmt_no_cjk_1n.wmt_mined.joined.64k/dense.fp16.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr${lr}.clip0.0.drop0.1.wd0.0.ls0.1.seed2.c10d.det.transformer.ELS6.DLS6.encffnx2048.decffnx2048.E512.H8.NBF.ATTDRP0.1.RELDRP0.0.ngpu64
    MODEL=$MODEL_FOLDER/${CHECKPOINT_NAME}.pt
    for TGT in cs de km pl ps ru ta
    do
      OUTDIR=${MODEL_FOLDER}/${SRC}-${TGT}_${CHECKPOINT_NAME}_${gen_split}
      mkdir -p $OUTDIR
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
        --gpus 1 --nodes 1 --cpus-per-task 8 \
        --time 1000 --mem 50000 --no-requeue \
        --partition dev --ntasks-per-node 1  \
        --open-mode append --no-requeue \
        --wrap "srun sh ${OUTDIR}/gen.sh; srun sh examples/wmt21/evaluation.sh ${OUTDIR} ${SRC} ${TGT} > ${OUTDIR}/test_bleu_results"
    done
  done
done
