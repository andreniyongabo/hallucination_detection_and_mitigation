# Usage:
# bash eval_slurm.sh aws moe en_to_many
# bash eval_slurm.sh fair dense many_to_en

tag="wmt_no_cjk_1n.wmt_mined.joined.64k"

CLUSTER=$1
MODEL_TYPE=$2
DIRECTION=$3

if [ $CLUSTER == "aws" ] ; then
    DATA=/fsx/shru/data/${tag}
    mem=0
    # there is no other partition on AWS
    partition=learnfair
    constraint=""
else
    DATA=/private/home/chau/wmt21/multilingual_bin/${tag}
    mem="480G"
    # other partitions are available in the FAIR cluster
    partition=learnfair,dev
    constraint="-C volta32gb"
fi

if [ $DIRECTION == "en_to_many" ] ; then
    SRC=en
else
    TGT=en
fi

SPM=$DATA/sentencepiece.64000.model
WS=8
langs="cs,de,en,km,pl,ps,ru,ta"
lang_pairs="en-cs,en-de,en-km,en-pl,en-ps,en-ru,en-ta"
# langs="cs,de,en"
# lang_pairs="en-cs,en-de"
#MODEL_FOLDER=/fsx/shru/wmt21/zero3_wmt_no_cjk_1n.wmt_mined.joined.64k/moe_v1.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.ilr1e-07.wu4000.lr0.001.clip0.0.drop0.1.wd0.0.seed2.fully_sharded.det.mt6000.experts64.glwt0.01.all.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64_converted
#MODEL_FOLDER=/checkpoint/shru/wmt21/wmt_no_cjk_1n.wmt_mined.joined.64k/dense.transformer_12_12.lr0.001.fp16.mpos256.SPL_temperature.tmp5.shem.adam.lr0.001.drop0.1.ls0.1.seed2.c10d.det.transformer.ELS12.DLS12.encffnx4096.decffnx4096.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu64
MODEL_FOLDER=/large_experiments/moe/shru/mmt/dense_9b_v2_16
MODEL_FOLDER=/checkpoint/chau/wmt21/zero3_wmt_no_cjk_1n.wmt_mined.joined.64k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.fully_sharded.det.mt6000.transformer.ELS48.DLS48.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
OUT_MODEL_FOLDER=MODEL_FOLDER
OUT_MODEL_FOLDER=/checkpoint/shru/wmt21/zero3_wmt_no_cjk_1n.wmt_mined.joined.64k/dense.fp16.uf1.entsrc.SPL_temperature.tmp5.shem.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr0.001.clip0.0.drop0.1.wd0.0.ls0.1.seed2.fully_sharded.det.mt6000.transformer.ELS48.DLS48.encffnx8192.decffnx8192.E1024.H16.NBF.ATTDRP0.1.RELDRP0.0.ngpu128
for CHECKPOINT_NAME in checkpoint35-shard0 ; do
    for gen_split in valid test ; do
        MODEL=$MODEL_FOLDER/${CHECKPOINT_NAME}.pt
        for LANG in cs de km pl ps ru ta  ; do
            if [ $DIRECTION == "en_to_many" ] ; then
                TGT=$LANG
            else
                SRC=$LANG
            fi
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
                --fp16 \
                ${MOE_PARAMS} \
                --max-sentences $BSZ | tee ${OUTDIR}/gen_best.out" > ${OUTDIR}/gen.sh

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
                --wrap "srun sh ${OUTDIR}/gen.sh; srun sh examples/wmt21/evaluation.sh ${OUTDIR} ${SRC} ${TGT} > ${OUTDIR}/test_bleu_results"
        done
    done
done
