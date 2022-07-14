# Usage:
# bash eval_slurm.sh aws moe en_to_many
# bash eval_slurm.sh fair dense many_to_en

tag="full_wmt.wmt_mined.joined.128k_rev"
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
    DATA=/large_experiments/mmt/full_wmt/${tag}
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
langs="en,ha,is,ja,ps,km,ta,cs,ru,zh,de,pl"
lang_pairs="en-ha,en-is,en-ja,en-ps,en-km,en-ta,en-cs,en-ru,en-zh,en-de,en-pl"

MODEL_FOLDER=/checkpoint/angelafan/wmt21/full_wmt.wmt_mined.joined.128k_rev/dense.fp16.uf2.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.no_c10d.det.mt1500.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
OUT_MODEL_FOLDER=/checkpoint/chau/wmt21/full_wmt.wmt_mined.joined.128k_rev/dense.fp16.uf2.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.no_c10d.det.mt1500.transformer.ELS24.DLS24.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
for CHECKPOINT_NAME in checkpoint_last ; do
    for gen_split in valid ; do
        MODEL=$MODEL_FOLDER/${CHECKPOINT_NAME}.pt
        for lang in ha; do
            if [ $DIRECTION == "en_to_many" ] ; then
                TGT=$lang
            else
                SRC=$lang
            fi
            #input_file=/large_experiments/mmt/wmt_only/wmt_only.wmt_mined.joined.128k_rev/${gen_split}.${SRC}_${TGT}.${SRC}
            input_file=/large_experiments/mmt/wmt21/bilingual_bin/ha_en.wmt_fb.32k/${gen_split}.${SRC}_${TGT}.${SRC}
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
                --bpe 'sentencepiece' \
                --sentencepiece-model ${SPM} \
                --buffer-size 1024 \
                --sacrebleu \
                --fp16 \
                ${MOE_PARAMS} \
                --max-sentences $BSZ | tee ${OUTDIR}/gen_best.out

          cat ${OUTDIR}/gen_best.out | grep -P "^D-" | cut -f3 > ${OUTDIR}/gen_best.output
          sacrebleu -l ${SRC}-${TGT} /private/home/chau/wmt21/dev_data/${SRC}_${TGT}/${gen_split}.${SRC}_${TGT}.${TGT} < ${OUTDIR}/gen_best.output > ${OUTDIR}/bleu.results
                " > ${OUTDIR}/gen.sh

            sbatch \
                --output ${OUTDIR}/eval.out \
                --error ${OUTDIR}/eval.err \
                --job-name ${SRC}-${TGT}.eval \
                --gpus-per-node $gpus --nodes 1 --cpus-per-task $cpus \
                --time 1000 --mem $mem \
                ${constraint} \
                --partition $partition \
                --ntasks-per-node 1  \
                --open-mode append --no-requeue \
                --wrap "srun sh ${OUTDIR}/gen.sh"
        done
    done
done


