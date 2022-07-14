#!/bin/bash
#
# Train a single NMT system on previsouly binarized bitexts
# with default settings (which sould be reasonnable for 1-10M bietxts)
#
# NLLB, Holger Schwenk
#
# usage:  train.sh SRC-LANG TRG-LANG
#
# Directory structure with respect to $bdir
#  bitexts/     bitexts, e.g. eng-hau.bitextf.tsv.gz (alphabetical order)
#  bin/         binarized bitexts and FLORES  dev/test sets
#               (see script preproc-binarize-mined.sh)
#  models/      trained FAIRSEQ models

######################################
# Calling parameters
######################################

set -e
l1=$1
l2=$2
ID=$3
max_epoch=${4:-100}
nnodes=${5:-2}

arch=""  # default
enc_lyr=6; enc_heads=8; enc_edim=512;  enc_ffn=4096; dec_lyr=6; dec_heads=8; dec_edim=512; dec_ffn=4096      # 133M params
let enc_ffn=enc_heads*enc_edim
let dec_ffn=dec_heads*dec_edim

######################################
# *** START CONFIGURATION HERE ****
######################################

bdir="/private/home/pkoehn/experiment/mining/$ID"
FSEQ="$HOME/project/fairseq-internal"

######################################
# *** END CONFIGURATION HERE ****
######################################

# default directory structure and naming
DATADIR="$bdir/bin/"  # all binarized data is expected here
odir="$bdir/model.$l1-$l2.epoch$max_epoch/"

# submission parameters
QUEUE="learnfair"
#QUEUE="devaccel"
EXCLUDE=""
COMMENT=""

######################################
# UNUSED larger configs
#self.head_dim * num_heads == self.embed_dim,
# 12l 768h 12h
# 24l 1024h 16h
#enc_lyr=12; enc_heads=12; enc_edim=768; enc_ffn=4096; dec_lyr=12; dec_heads=12; dec_edim=768; dec_ffn=4096
#enc_lyr=24; enc_heads=16; enc_edim=1024; enc_ffn=4096; dec_lyr=25; dec_heads=16; dec_ffn=4096;dec_edim=1024; nnodes=4
#arch=".l${enc_lyr}h${enc_heads}x${enc_edim}-l${dec_lyr}h${dec_heads}x${dec_edim}.n$nnodes"
#arch=".l${enc_lyr}h${enc_heads}f${enc_ffn}-l${dec_lyr}h${dec_heads}f${dec_ffn}.n$nnodes"

ngpu=8
let sumgpu=$nnodes*$ngpu

mkdir -p $odir
echo "Checkpoints in $odir on $sumgpu GPUs"

script="$odir/train.sh"
cat > $script << EOF
#!/bin/bash

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh
conda activate fairseq-20210318

  #LD_LIBRARY_PATH="/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/amazon/efa/lib:$LD_LIBRARY_PATH"
  export FI_OFI_RXR_RX_COPY_UNEXP="1"
  export FI_OFI_RXR_RX_COPY_OOO="1"
  export FI_EFA_MR_CACHE_ENABLE="1"
  export FI_OFI_RXR_INLINE_MR_ENABLE="1"
  export NCCL_DEBUG="info",
  cd $odir
  echo -e "\nRUNNING code $FSEQ\n\n" >> $odir/train.log
  hostname >> $odir/train.log

  srun python $FSEQ/train.py $DATADIR \
    --distributed-port 9218 --distributed-world-size $sumgpu \
    --arch transformer --share-decoder-input-output-embed \
    --source-lang $l1 --target-lang $l2 \
    --encoder-embed-dim $enc_edim --decoder-embed-dim $dec_edim \
    --encoder-layers $enc_lyr --decoder-layers $dec_lyr \
    --encoder-ffn-embed-dim $enc_ffn --decoder-ffn-embed-dim $dec_ffn \
    --encoder-attention-heads $enc_heads --decoder-attention-heads $dec_heads \
    --dropout 0.3 --attention-dropout 0.2 --relu-dropout 0.2 \
    --encoder-normalize-before --decoder-normalize-before \
    --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-7 \
    --lr 1e-3 --stop-min-lr 1e-9 \
    --max-tokens 2000 --update-freq 4 \
    --no-progress-bar --log-interval 500 \
    --max-epoch $max_epoch --save-interval 1 \
    --save-dir ${odir} >> ${script//.sh/.log} 2>&1
#--keep-last-epochs 10
#--log-interval 100 --log-format simple
#--encoder-layers
#--decoder-layers
EOF

chmod 755 $script
sbatch -J "nmt.$l1-$l2$arch.$th$TH" \
    --partition=$QUEUE --comment "$COMMENT" --exclude="$EXCLUDE" \
    --nodes=$nnodes --gpus-per-node=$ngpu --cpus-per-task=2 \
    --ntasks-per-node=$ngpu --time=2880 \
    --mem 400G \
    ${script}
