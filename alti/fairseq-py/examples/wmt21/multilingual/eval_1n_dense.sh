tag="wmt_no_cjk_1n.joined.64k"
DATA=/private/home/chau/wmt21/multilingual_bin/${tag}
SRC=en
MODEL_FOLDER=/checkpoint/chau/wmt21/wmt_no_cjk_1n.wmt_mined.joined.64k/dense.fp16.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.c10d.det.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64
CHECKPOINT_NAME=checkpoint_last
MODEL=$MODEL_FOLDER/${CHECKPOINT_NAME}.pt
SPM=$DATA/sentencepiece.64000.model

checkpoint_dir="/checkpoint/chau/wmt21/${tag}"
langs="cs,de,en,km,pl,ps,ru,ta"
lang_pairs="en-cs,en-de,en-km,en-pl,en-ps,en-ru,en-ta"
for TGT in cs de en km pl ps ru ta
do
  OUTDIR=${MODEL_FOLDER}/${SRC}-${TGT}_${CHECKPOINT_NAME}
  mkdir -p $OUTDIR
  CUDA_VISIBLE_DEVICES=0 \
  python fairseq_cli/generate.py \
    ${DATA} \
    --path ${MODEL} \
    --task translation_multi_simple_epoch \
    --langs "${langs}" \
    --lang-pairs "${lang_pairs}" \
    --source-lang ${SRC} --target-lang ${TGT} \
    --encoder-langtok "src" \
    --decoder-langtok \
    --gen-subset valid \
    --bpe 'sentencepiece' \
    --sentencepiece-model ${SPM} \
    --beam 4 \
    --sacrebleu \
    --max-tokens 5000 \
    --max-sentences 20 | tee $OUTDIR/gen_best.out

  # postprocessing + evaluation
  bash examples/wmt21/evaluation.sh $OUTDIR ${SRC} ${TGT} > $OUTDIR/test_bleu_results
done
