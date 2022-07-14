#!/bin/bash
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node 8
module load openmpi

mpirun \
 -x FI_PROVIDER="efa" \
 -x NCCL_DEBUG=INFO \
 -x FI_EFA_USE_DEVICE_RDMA=1 -x NCCL_ALGO=ring -x RDMAV_FORK_SAFE=1 --mca pml ^cm \
 -x LD_PRELOAD=/usr/local/cuda/lib/libnccl.so.2.11.4 \
 -x LD_LIBRARY_PATH=/usr/local/cuda-11.3/efa/lib:/usr/local/cuda-11.3/lib:/usr/local/cuda-11.3/lib64:/usr/local/cuda-11.3:/opt/amazon/efa/lib:/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH \
 --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
 /usr/local/cuda-11.3/efa/test-cuda-11.3/all_reduce_perf -b 8 -e 8G -f 4 -g 1 -c 1 -n 5
