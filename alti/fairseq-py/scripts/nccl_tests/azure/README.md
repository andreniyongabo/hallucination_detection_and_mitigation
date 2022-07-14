# Running NCCL tests on Azure

## Fetching libnccl and `all_reduce_perf` from blob storage

First call `get_nccl_tests.sh` to fetch `libnccl` and the `all_reduce_perf`
binary from blob storage.

```bash
./scripts/nccl_tests/azure/get_nccl_tests.sh
```

Note that this should be called from the scheduler node, which does not require
authentication to access the `fairacceleastus` blob storage account. If you are
calling this script from another node or cluster that has not been configured
this way, you may need to modify `get_nccl_tests.sh` to include blob access
credentials in the query string.

## Running NCCL tests

Next call `run_nccl_allreduce.sh` to launch NCCL tests. The script takes two
arguments:
1) the number of nodes
2) the file path to save the results

For example, to run NCCL tests on 2 nodes (16 GPUs) and save the results in
`nccl_test_output.txt`:
```bash
./scripts/nccl_tests/azure/run_nccl_allreduce.sh 2 nccl_test_output.txt
```

Any additional arguments given to the script will be passed through to `sbatch`.
For example, one can run the tests on specific hosts by passing the `-w` option
provided by `sbatch`:
```bash
./scripts/nccl_tests/azure/run_nccl_allreduce.sh 2 nccl_test_output.txt -w hpc-pg0-[1-2]
```
