SRC=de
TGT=en
DATA="/private/home/angelafan/wmt21/finetuning_data/final_data/without_2020/wmt/binarized_rev"
SPM=/large_experiments/mmt/wmt21/bt_multilingual_bin/wmt_only_rev.bitext_bt.v3.64_shards/sentencepiece.128000.model

langs="en,de"
lang_pairs="de-en"
gen_split='train'
# en-ru: checkpoint75
# en-cs: checkpoint80
# en-ha: checkpoint40
# en-is: checkpoint20
# en-ja: checkpoint
# en-zh: checkpoint
# en-de: checkpoint
for CHECKPOINT_NAME in checkpoint_best; do
    MODEL_FOLDER=/checkpoint/chau/wmt21/bilingual_v3_blft.without2020.wmt/de_en.ft.v3.32k/dense.mfp16.fp16.uf2.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu1.lr0.0001.clip0.0.drop0.1.wd0.0.seed1.no_c10d.det.mt2500.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu2
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
      --fp16 \
      --beam 4 \
      --bpe 'sentencepiece' \
      --sentencepiece-model ${SPM} \
      --sacrebleu \
      --max-tokens 5000 \
      --max-sentences 20 | tee ${OUTDIR}/gen_best.out" > ${OUTDIR}/gen.sh

    bash ${OUTDIR}/gen.sh
done


