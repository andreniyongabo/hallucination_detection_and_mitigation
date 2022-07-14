#!/bin/bash

. /public/apps/anaconda3/5.0.1/etc/profile.d/conda.sh

#module load cuda/10.1
#module load cudnn/v7.6.5.32-cuda.10.1
#module load anaconda3/5.0.1
conda activate laser

cd /private/home/pkoehn/experiment/wmt21-ha-en/filtering

python $LASER/source/mine_bitexts.py --src-lang en --trg-lang ha --output train-all.laser2 --mode score --retrieval max --margin ratio -k 4 --verbose --gpu --unify --src-embeddings train-all.laser2.en.emb --trg-embeddings train-all.laser2.ha.emb ../train-all.en ../train-all.ha.no-tag >& train-all.laser.log
