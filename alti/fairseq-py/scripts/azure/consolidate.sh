#!/bin/bash
export NCCL_IB_PCI_RELAXED_ORDERING=1
export UCX_IB_PCI_RELAXED_ORDERING=on
export NCCL_SOCKET_IFNAME=eth0
export UCX_NET_DEVICES=eth0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMPI_MCA_COLL_HCOLL_ENABLE=0
export OMPI_MCA_coll_hcoll_enable=0
export LD_LIBRARY_PATH=/shared/home/myleott/src/nccl/build/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/shared/home/myleott/src/nccl/build/lib/libnccl.so
export PATH=/shared/home/myleott/bin/azcopy_linux_amd64_10.12.1:$PATH
export AZCOPY_AUTO_LOGIN_TYPE=MSI
export NCCL_DEBUG="WARN"
RUN_DIR=175B_run12.55
prefix=$1
save_dir=$2
shift 2
SAS_TOKEN="sv=2020-08-04&ss=b&srt=sco&sp=rwdlactfx&se=2023-10-06T11:23:33Z&st=2021-10-06T03:23:33Z&spr=https&sig=s6aw4Ca4Ohbr7LQ%2BG9s58PEyYJsbXHjs%2Fc%2BuoTvzTUo%3D"
URL="https://fairacceleastus.blob.core.windows.net/susanz/2021-12-05/$RUN_DIR?$SAS_TOKEN"
DEST_URL="https://fairacceleastus.blob.core.windows.net/sshleifer/175B/?$SAS_TOKEN"
#prefix like "checkpoint_19_55000*.pt"
# check if path exists with readlink
#grep "https" $f | grep 62000
# 3750 updates per epoch
# 144 updates per hour
mkdir -p $RUN_DIR
if [ -f "$RUN_DIR/$prefix-model_part-7-shard68.pt" ]; then
    echo "Skipping Download"
else
    echo "Downloading $prefix"
    azcopy cp --include-pattern "$prefix*"  --recursive $URL .
fi

PYTHONPATH='.' python scripts/remove_opt_state.py $RUN_DIR/$prefix --nproc 62 --save-dir $save_dir --save-prefix $prefix
PYTHONPATH='.' python scripts/consolidate_fsdp_shards.py $save_dir/$prefix --new-arch-name transformer_lm_gpt
azcopy cp $save_dir/"$prefix"_consolidated.pt $DEST_URL

echo "On Fair cluster run: azcopy cp --include-pattern \"$prefix*.pt\" $DEST_URL /large_experiments/xlmg/models/sshleifer/175B/"
