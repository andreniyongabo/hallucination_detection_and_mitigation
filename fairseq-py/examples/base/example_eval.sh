#!/bin/bash
srun python fairseq_cli/eval_lm.py \
/checkpoint/sshleifer/data-bin/rc-bin-40/shard0/ \
  --path $base_moe_16_gpu \
  --tokens-per-sample 512 \
  --gen-subset valid \
  --batch-size 1 \
  --fp16 \
  --distributed-port 15187 \
  --model-overrides "{'is_base_moe': True}" \
  --sp base_moe_16.json \
  $@
