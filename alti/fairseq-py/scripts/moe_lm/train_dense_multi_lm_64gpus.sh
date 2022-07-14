partition=learnfair
num_trials=1
num_nodes=8
num_gpus_per_node=8
type="dense"

#DATA_BIN="/datasets01/cc100-bin/072820/250"
DATA_BIN="/large_experiments/moe/cc100_xl/bin"
NUM_DATA_SHARDS=64
DATA_DIR="${DATA_BIN}/shard0"
for i in $(seq 1 $(($NUM_DATA_SHARDS-1))); do
    DATA_DIR="${DATA_DIR}:${DATA_BIN}/shard${i}";
done

script_name="sweep_dense_multilingual_lm.py"
checkpoint_dir="/checkpoint/$USER/moe_multi_lm/${type}/"

python fb_sweep/${script_name} -d ${DATA_DIR} -p $type \
    --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
    -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} \
    --resume-failed --time 3999 --mem 470G
