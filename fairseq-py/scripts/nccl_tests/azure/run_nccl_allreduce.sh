#!/bin/bash

if [ $# -lt 2 ]; then
    echo "usage: $0 <NUM_NODES> <OUTPUT_FILE> [SBATCH_PASSTHROUGH_ARGS...]"
    exit 1
fi

# number of nodes to run nccl tests on
NODES=$1

# where to save nccl test output (absolute path)
OUTPUT_FILE=$(readlink -f $2)

# remove first two arguments, so that all remaining arguments are sbatch passthrough
shift 2

touch $OUTPUT_FILE

if [ ! -e "$(dirname $0)/bin/libnccl.so.2.10.3" -o ! -e "$(dirname $0)/bin/all_reduce_perf" ]; then
    echo "missing libnccl and all_reduce_perf; please run $(dirname $0)/get_nccl_tests.sh first!"
    exit 1
fi

sbatch -o $OUTPUT_FILE --nodes $NODES "$@" $(dirname $0)/_run_nccl_allreduce.sh $(dirname $0)
