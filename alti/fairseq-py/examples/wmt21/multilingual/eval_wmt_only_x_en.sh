#!/bin/bash
#SBATCH --partition=learnfair,No_Language_Left_Behind
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=60g
#SBATCH --time=3000
#SBATCH --constraint=volta32gb
# Usage:
# bash eval_slurm.sh aws moe en_to_many
# bash eval_slurm.sh fair dense many_to_en

tag="wmt_only_rev.bitext_bt.v3.64_shards"
MOSES=~edunov/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl

CLUSTER=${1:-"fair"}
MODEL_TYPE=${2:-"dense"}
DIRECTION=${3:-"many_to_en"}

if [ $CLUSTER == "aws" ] ; then
    DATA=/fsx/shru/data/${tag}
    mem=0
    # there is no other partition on AWS
    partition=learnfair
    constraint=""
else
    DATA=/large_experiments/mmt/wmt21/bt_multilingual_bin/${tag}
    mem="480G"
    # other partitions are available in the FAIR cluster
    partition=learnfair
    constraint="-C volta32gb"
fi

if [ $DIRECTION == "en_to_many" ] ; then
    SRC=en
else
    TGT=en
fi

SPM=$DATA/sentencepiece.128000.model
WS=8
langs="en,ha,is,ja,cs,ru,zh,de"
lang_pairs="ha-en,is-en,ja-en,cs-en,ru-en,zh-en,de-en"

#MODEL_FOLDER=/checkpoint/chau/wmt21/wmt_only_rev.bitext_bt.v3.64_shards/dense.fp16.mfp16.uf4.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.c10d.det.mt4000.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64
MODEL_FOLDER=/checkpoint/chau/wmt21/wmt_only_rev.bitext_bt.v3.64_shards/dense.fp16.mfp16.uf4.entsrc.SPL_concat.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.c10d.det.mt4000.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
OUT_MODEL_FOLDER=$MODEL_FOLDER
for CHECKPOINT_NAME in checkpoint_27_100000 ; do
  for lenpen in 8.0 16.0 32.0; do
    for gen_split in "test"; do
        MODEL=$MODEL_FOLDER/${CHECKPOINT_NAME}.pt
        for lang in ja; do
            if [ $DIRECTION == "en_to_many" ] ; then
                TGT=$lang
            else
                SRC=$lang
            fi
            input_file=$DATA/${gen_split}.${SRC}_${TGT}.${SRC}
            #input_file=/large_experiments/mmt/wmt21/bilingual_bin/ha_en.wmt_fb.32k/${gen_split}.${SRC}_${TGT}.${SRC}
            if [ ${MODEL_TYPE} == "moe" ] ; then
                MOE_PARAMS="--is-moe \
                --distributed-world-size ${WS} --distributed-port 15187 \
                --model-overrides \"{'world_size': ${WS}, 'moe_eval_capacity_token_fraction': 0.1 }\" "
                BSZ=16
                gpus=8
                cpus=80
            else
                MOE_PARAMS=""
                BSZ=50
                gpus=1
                cpus=8
            fi
            OUTDIR=${OUT_MODEL_FOLDER}/gen_output/${SRC}-${TGT}_${CHECKPOINT_NAME}_${gen_split}
            prefix="lenpen${lenpen}"
            mkdir -p $OUTDIR
            echo "cat ${input_file} | ${REPLACE_UNICODE_PUNCT} | ${NORM_PUNC} -l ${SRC}  |  python fairseq_cli/interactive.py \
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
                --lenpen ${lenpen} \
                --bpe 'sentencepiece' \
                --sentencepiece-model ${SPM} \
                --buffer-size 1024 \
                --sacrebleu \
                --fp16 \
                ${MOE_PARAMS} \
                --max-sentences $BSZ | tee ${OUTDIR}/gen_best.${prefix}.out

          cat ${OUTDIR}/gen_best.${prefix}.out | grep -P "^D-" | cut -f3 > ${OUTDIR}/gen_best.${prefix}.output
          sacrebleu -l ${SRC}-${TGT} /private/home/chau/wmt21/dev_data/${SRC}_${TGT}/${gen_split}.${SRC}_${TGT}.${TGT} < ${OUTDIR}/gen_best.${prefix}.output > ${OUTDIR}/bleu.${prefix}.results
                " > ${OUTDIR}/gen.sh

          bash ${OUTDIR}/gen.sh
        done
      done
    done
done


