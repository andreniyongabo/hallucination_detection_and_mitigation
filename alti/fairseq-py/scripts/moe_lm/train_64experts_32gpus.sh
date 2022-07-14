partition=learnfair
num_trials=1
num_nodes=4
num_gpus_per_node=8
experts_per_node=2
num_experts=$(( experts_per_node * num_nodes * num_gpus_per_node ))
type="top2_${num_experts}e"
export AWS_SHARED_CREDENTIALS_FILE=~/.fairusers_aws/credentials
# export S3_DIR="s3://fairusersglobal/users/$USER/moe_lm"
# export DBG=1
script_name="sweep_gshard_lm_64experts_32gpus.py"
checkpoint_dir="/checkpoint/$USER/moe_lm/${type}/"

DATA_BIN=/checkpoint/sshleifer/data-bin/rc-bin-40
NUM_DATA_SHARDS=40
DATA_DIR="${DATA_BIN}/shard0"
for i in $(seq 1 $(($NUM_DATA_SHARDS-1)));
  do
    DATA_DIR="${DATA_DIR}:${DATA_BIN}/shard${i}";
  done
python fb_sweep/${script_name} -d ${DATA_DIR} -p $type \
    --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
    -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} \
    --resume-failed --time 3999 --mem 470G
