#!/bin/bash

# Note: you will need to install fairseq, activate the conda env, load modules
# conda activate fairseq-20200821 (from https://fb.workplace.com/groups/fairseq/permalink/262715387865587/)
# module load cudnn/v7.6.5.32-cuda.10.1 cuda/10.1.243
# cd $FAIRSEQ
# git checkout <branch>
# pip install --editable .

FAIRSEQ=/private/home/shru/projects/fairseq-py-moe
SWEEP_NAME=billion_words_lm_training_v3.0
JOBSCRIPTS=/private/home/${USER}/scripts/moe_slurm/
mkdir -p ${JOBSCRIPTS}

SAVE_ROOT=/large_experiments/moe/${USER}/${SWEEP_NAME}


QUEUE=learnfair,moe
COMMENT=""
# TODO: change if needed
NODES=2
NTASKS_PER_NODE=8
GPUS_PER_TASK=1
CPUS_PER_TASK=$(( 8 * GPUS_PER_TASK ))
TIME=600
MEM="470G"

WORLD_SIZE=$(( NODES * NTASKS_PER_NODE ))
UF=1
NUM_EXPERTS=$(( WORLD_SIZE ))
DIST_PORT=12234
BUCKET_CAP_MB=200
VALID_INTERVAL=1000
SAVE_INT=2000
LAYERS=4
MAX_UPDATES=100000
BSZ=16

FFN=65536
CRITERION="moe_cross_entropy"
COMBINE_METHOD="sum"

LR=0.001

for FFN in 65536 ; do
for GATE_LOSS_WT in 0.005 ; do
for TRANSFORM in "none" ; do
for fp in "fp16" ; do
for gating in "fp32" ; do
for policy in "all"  ; do
for normbefore in "false" ; do
RUNNAME="ws${WORLD_SIZE}.${fp}.gating_${gating}.uf${UF}.ffn_${FFN}.e_${NUM_EXPERTS}.l_${LAYERS}.c_${CRITERION}.glwt_${GATE_LOSS_WT}.cm_${COMBINE_METHOD}.tr_${TRANSFORM}.bsz${BSZ}.lr${LR}.2ndexp${policy}.normbefore${normbefore}"
DATABIN=/private/home/shru/data/t2t_data/data-bin
SAVE=${SAVE_ROOT}.${RUNNAME}
mkdir -p ${SAVE}
JNAME=${SWEEP_NAME}.${RUNNAME}
SCRIPT=${JOBSCRIPTS}/run.${JNAME}.sh
SLURM=${JOBSCRIPTS}/run.${JNAME}.slrm
echo $SAVE

if [ "$fp" == "fp16" ] ; then
    fpstr=" --fp16 --fp16-no-flatten-grads "
else
    fpstr=" "
    # Note: FP32 training hangs after a while for unknown reasons
fi

if [ "$fp" == "fp16" ] && [ "$gating" == "fp32" ] ; then
    gating_str=" --moe-gating-use-fp32 "
else
    gating_str=" "
fi
if [ "$normbefore" == "true" ] ; then
    normbefore=" --moe-normalize-gate-prob-before-dropping "
else
    normbefore=" "
fi
expert_ffn_dim=$(( FFN / NUM_EXPERTS ))
COMMAND="NCCL_DEBUG=INFO python fairseq_cli/train.py ${DATABIN} \
    ${fpstr} \
    ${gating_str} \
    --task language_modeling --share-decoder-input-output-embed --sample-break-mode none \
    --ddp-backend=no_c10d --log-format simple --log-interval 50 \
    --skip-invalid-size-inputs-valid-test --validate-interval-updates ${VALID_INTERVAL} \
    --save-interval-updates ${SAVE_INT} --keep-interval-updates 1 --arch transformer_lm \
    --criterion ${CRITERION} --moe-gate-loss-wt ${GATE_LOSS_WT} --moe-gate-loss-combine-method ${COMBINE_METHOD} \
    --moe-gate-loss-transform ${TRANSFORM} \
    --lr-scheduler inverse_sqrt --warmup-init-lr 0.001 --lr ${LR} \
    --batch-size ${BSZ} --min-loss-scale 1e-10 --tokens-per-sample 256 --optimizer adafactor \
    --weight-decay 0.0 --decoder-attention-heads 4 \
    --decoder-layers ${LAYERS} --decoder-ffn-embed-dim ${FFN} --dropout 0.1 --attention-dropout 0.1 \
    --relu-dropout 0.1 --max-update ${MAX_UPDATES} --warmup-updates 10000 --update-freq ${UF} --clip-norm 0.0 --save-dir ${SAVE} \
    --moe-freq 2 --moe-expert-count ${NUM_EXPERTS} \
    --required-batch-size-multiple 16 \
    --batch-size-valid 2 \
    --distributed-world-size ${WORLD_SIZE} --distributed-port ${DIST_PORT} \
    --bucket-cap-mb ${BUCKET_CAP_MB} \
    ${normbefore} --moe-second-expert-policy ${policy} \
    --moe-expert-decoder-ffn-dim ${expert_ffn_dim} "

if [ $QUEUE == "dgx_a100" ] ; then
    CONSTRAINT=""
else
    CONSTRAINT="-C volta32gb"
fi

if [ $1 == "sbatch" ] ; then
    echo "#!/bin/sh" > ${SLURM}
    echo "#SBATCH --job-name=$JNAME" >> ${SLURM}
    echo "#SBATCH --output=${SAVE}/stdout.%j" >> ${SLURM}
    echo "#SBATCH --error=${SAVE}/stderr.%j" >> ${SLURM}
    echo "#SBATCH --signal=USR1" >> ${SLURM}
    echo "#SBATCH --partition=${QUEUE}" >> ${SLURM}
    echo "#SBATCH --comment=\"${COMMENT}\"" >> ${SLURM}
    echo "#SBATCH --nodes=${NODES} ${CONSTRAINT} " >> ${SLURM}
    echo "#SBATCH --ntasks-per-node=${NTASKS_PER_NODE}" >> ${SLURM}
    echo "#SBATCH --mem=${MEM}" >> ${SLURM}
    echo "#SBATCH --gpus-per-task=${GPUS_PER_TASK}" >> ${SLURM}
    echo "#SBATCH --cpus-per-task=${CPUS_PER_TASK}" >> ${SLURM}
    echo "#SBATCH --time=${TIME}" >> ${SLURM}
    echo "srun sh ${SCRIPT}" >> ${SLURM}

    echo "#!/bin/sh" > ${SCRIPT}
    echo "{ " >> ${SCRIPT}
    echo "echo ${SWEEP_NAME} ${RUNNAME} " >> ${SCRIPT}
    echo "cd $FAIRSEQ" >> ${SCRIPT}
    echo ${COMMAND} >> ${SCRIPT}
    echo "kill -9 \$\$" >> ${SCRIPT}
    echo "} & " >> ${SCRIPT}
    echo "child_pid=\$!" >> ${SCRIPT}
    echo "trap \"echo 'TERM Signal received';\" TERM" >> ${SCRIPT}
    echo "trap \"echo 'Signal received'; if [ \"\$SLURM_PROCID\" -eq \"0\" ]; then sbatch ${SLURM}; fi; kill -9 \$child_pid; \" USR1" >> ${SCRIPT}
    echo "while true; do     sleep 1; done" >> ${SCRIPT}

    sbatch ${SLURM}
else
    $COMMAND ;
fi

done
done
done
done
done
done
done