#!/bin/bash
#SBATCH -t 00:03:00
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --mem=400GB

if [ $# -ne 1 ]; then
    echo "usage: $0 <SCRIPT_DIR>"
    exit 1
fi

# directory where _run_nccl_allreduce.sh is located
SCRIPT_DIR=$1

source /etc/profile.d/modules.sh
module load mpi/hpcx
export NCCL_DEBUG=info
export NCCL_TOPO_FILE=/opt/microsoft/ndv4-topo.xml
export NCCL_IB_PCI_RELAXED_ORDERING=1
export UCX_IB_PCI_RELAXED_ORDERING=on
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_SOCKET_IFNAME=eth0
export UCX_NET_DEVICES=eth0
export OMPI_MCA_COLL_HCOLL_ENABLE=0
export OMPI_MCA_coll_hcoll_enable=0
export LD_PRELOAD=$SCRIPT_DIR/bin/libnccl.so.2.10.3

export | grep SLURM

srun --cpu-bind=mask_cpu:ffffff000000,ffffff000000,ffffff,ffffff,ffffff000000000000000000,ffffff000000000000000000,ffffff000000000000,ffffff000000000000 --mpi=pmix --gpus-per-node=8 --ntasks-per-node=8 ${SCRIPT_DIR}/bin/all_reduce_perf -b 8 -f 4 -g 1 -e 8G -n 5
