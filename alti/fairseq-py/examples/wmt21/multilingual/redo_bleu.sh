MODEL_FOLDER=/checkpoint/chau/wmt21/zero3_wmt_no_cjk_1n.wmt_mined.joined.64k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.fully_sharded.det.mt6000.transformer.ELS48.DLS48.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
SRC=en
gen_split=valid
CHECKPOINT_NAME=checkpoint50-shard0
for TGT in cs de km pl ps ru ta; do 
  OUTDIR=${MODEL_FOLDER}/${SRC}-${TGT}_${CHECKPOINT_NAME}_${gen_split}
  bash examples/wmt21/evaluation.sh ${OUTDIR} ${TGT} > ${OUTDIR}/test_bleu_results
done
