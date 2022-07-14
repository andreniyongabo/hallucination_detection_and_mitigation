SRC=en
TGT=ja
data_type=wmt_mined
tag="en_${TGT}.${data_type}.joined.32k"
DATA=/private/home/chau/wmt21/multilingual_bin/bilingual_en_x/${tag}
SPM=$DATA/sentencepiece.32000.model

langs="en,${TGT}"
lang_pairs="en-${TGT}"
gen_split='test'
# en-ru: checkpoint75
# en-cs: checkpoint80
# en-ha: checkpoint40
# en-is: checkpoint20
# en-ja: checkpoint95
# en-zh: checkpoint_last
# en-de: checkpoint50
for CHECKPOINT_NAME in checkpoint95; do
  for seed in 1 2 3 ; do
    MODEL_FOLDER=/checkpoint/chau/wmt21/bilingual/${tag}/dense.fp16.uf4.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed${seed}.c10d.det.mt6000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu32
    MODEL=$MODEL_FOLDER/${CHECKPOINT_NAME}.pt
    OUTDIR=${MODEL_FOLDER}/${SRC}-${TGT}_${CHECKPOINT_NAME}_${gen_split}_norm
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
      --wrap "srun sh ${OUTDIR}/gen.sh; srun sh examples/wmt21/evaluation.sh ${OUTDIR} ${SRC} ${TGT} > ${OUTDIR}/test_bleu_results"
  done
done


