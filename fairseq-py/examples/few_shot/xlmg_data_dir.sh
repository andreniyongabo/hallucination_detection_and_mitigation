
DATA_BIN="/private/home/sshleifer/sharded_cc"
NUM_DATA_SHARDS=40
DATA_DIR="${DATA_BIN}/shard0"
for i in $(seq 1 $(($NUM_DATA_SHARDS-1)));
  do
    DATA_DIR="${DATA_DIR}:${DATA_BIN}/shard${i}";
  done
