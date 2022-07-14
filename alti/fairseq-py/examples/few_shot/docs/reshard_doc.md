- Written for `gshard` branch
- "run" means continue training or evaluate.

with `--use-sharded-state`, training and evaluation can load multiple such checkpoints per worker, so, if there is no "padding", you can run on any world size that your number of checkpoints is divisible by.
The easiest way to figure out if there is padding is to just try ^^.



If there is padding, you can "consolidate" to make 1 global checkpoint (and run without `--use-sharded-state`):

```bash
srun --gres=gpu:8 --partition=devaccel --nodes=1 --cpus-per-task 64 --ntasks-per-node 1 \
  --constraint volta32gb --gpus-per-node 8 --time="2-00:00:00" --pty /bin/zsh -l
# activate venv, load modules
model_path=ft_stream2/doc-block.gpt2.sbm_none.ckpt.tps_2048.175b.mu30000.bsz1.uf1.dr0.1.atdr0.1.actdr0.0.fp16adam.lr1e-05.ngpu512
python scripts/remove_opt_state.py $model_path/checkpoint_3_4000 --save-prefix checkpoint_3_4000_eval --nproc 16
python scripts/consolidate_fsdp_shards.py $model_path/checkpoint_3_4000_eval --new_arch_name transformer_lm_gpt
```

You only need to "reshard" if your consolidated checkpoint is bigger than ~50GB . Then you use:

```
python fairseq_cli/reshard.py $model-path $save-dir --target-world-size XX
```

(There is one bug in this script that I will fix hopefully today: (tracked here https://github.com/fairinternal/fairseq-py/issues/2894).
Then run **with** `--use-sharded-state`.
