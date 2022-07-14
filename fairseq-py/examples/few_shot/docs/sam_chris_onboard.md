

# Setup
(1) Ask Brian O'Horo
(2)
```
git clone git@github.com:fairinternal/fairseq-py.git
cd fairseq-py
git checkout gshard
```

- If no privs ask Brian O'Horo

(3) Conda Env
```bash
FAIRSEQ_ENV=/private/home/dianaml/.conda/envs/fairseq-20210318-py38
module purge && module load anaconda3/2020.11 cudnn/v8.0.3.33-cuda.11.0 cuda/11.0 fairusers_aws nvtop/1.0.0/gcc.7.3.0 openmpi/4.1.0/cuda.11.0-gcc.9.3.0 ripgrep/11.0.2 NCCL/2.8.3-1-cuda.11.0
conda env create --clone $FAIRSEQ_ENV -n fairseq-20210318-py38
pip install -e . (from fairseq-py)
```

(4) My ssh-config [here](https://gist.github.com/sshleifer/81369b940f471faed5253edc9df5ef00)

Other Resources:
- myle made some videos about fp16 training/FSDP that are probably good.
- Stephen Roller/Naman Goyal for Model Parallel stuff
- gh cli: https://cli.github.com/

### Aliases
```
cc () {
	source /etc/profile
	source ~/.zshrc
	module purge && module load anaconda3/2020.11 cudnn/v8.0.3.33-cuda.11.0 cuda/11.0 fairusers_aws nvtop/1.0.0/gcc.7.3.0 openmpi/4.1.0/cuda.11.0-gcc.9.3.0 ripgrep/11.0.2 NCCL/2.8.3-1-cuda.11.0
	module load
	module load
}
```
File Transfer: azcopy


### Srun workflow

```bash
srun --gres=gpu:8 --partition=devaccel --nodes=1 --cpus-per-task 64 --ntasks-per-node 1 --constraint volta32gb --gpus-per-node 8 --time="2-00:00:00" -x learnfair5031 --pty /bin/zsh -l
# In the resultant shell
module purge && module load anaconda3/2020.11 cudnn/v8.0.3.33-cuda.11.0 cuda/11.0 fairusers_aws nvtop/1.0.0/gcc.7.3.0 openmpi/4.1.0/cuda.11.0-gcc.9.3.0 ripgrep/11.0.2 NCCL/2.8.3-1-cuda.11.0
conda activate VENV
# run command
```





# Pre-Training Language Models


Intuition:
"Causal Language Modeling"
Feed language model a bunch of text and teach it to predict `words[x]` conditional on `words[x-seq_len: x-1]`. (`seq_len` typically 2048).


sweep scripts:
    `fb_sweep/xlmg/sweep_xlmg_en_lm.py` is GPT3/SOTA LM Config

run that with `--benchmark --local`, then use agg-results
```
agg () {
        ~/fairseq-py/fb_sweep/agg_results.py "$@"
}
alias sq="squeue -o "%.9i %.6D %80j %.8T %.10M  %.3P" -u $USER"
```

Artifacts: `train.log`, `train.stderr`, `checkpoint*.pt`
-  (I run custom python code in a jupyter notebook to collate my experiment results, but some people have other setups (e.g. Tensorboard)).

FSDP:
    - Shards the model state accross workers, makes all workloads faster.
    - You cannot train anything bigger than 1.3B without FSDP!
    - causes some downstream annoyance because it saves one checkpoint file per worker
    - lots of Glue code about combining/resharding these checkpoints to resume training/evaluate on different world sizes.
    - see [`examples/few_shot/reshard_doc.md`](examples/few_shot/reshard_doc.md)

Model Parallelism (not on gshard branch, on `gshard_combine_megatron_fsdp` branch) (Contact=Stephen Roller):
    Only speeds up 175B: not useful at smaller scale.


Mixture of Experts: (SKIP)



### 0 Shot Eval

TLDR:
Feed language model:
```
"Does Christopher Dewan work at Meta? Options: Yes No. No"
"Does Christopher Dewan work at Meta? Options: Yes No. Yes"
"{PROMPT}{CANDIDATES}{LABEL_WORD}"
```
and see which has higher probability.

```bash
pkill -f v4
m=$1
rd=ftstream_azure_results/ftstream_azure_results_v2_"$m"
mkdir -p $rd
shift
PYTHONPATH="."  python -m examples.few_shot.gpt3_eval --max-tokens 2048 \
    --results-dir $rd --batch-size 1 --nshot 0 --model-name $m \
    --tasks sst2 arcchallenge arceasy piqa flan__arc_easy.alltemplates flan__arc_challenge.alltemplates flan__sst2.alltemplates --user-dir examples/few_shot/finetune \
    "$@" | tee $rd/log.out
```



Some team members use `schedule_jobs_few_shot.py` which I hate. But launches many evals in parallel with submitit. Tougher to see errors, easier to complete huge workloads faster.



### Env

### CI/Tests

CPU Tests: .circleci/config.yml
GPU Tests: .github/workflow.yaml


(after srun)
```
CUDA_VISIBLE_DEVICES=0,1 pytest tests/gpu/ -sv -k fsdp --maxfail 1
```
