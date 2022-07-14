# Fine-tuning and Prefix Tuning

### Run CB locally
```bash
./fb_sweep/finetune_lm.py -t 1 --num-nodes 1 --num-gpus 1 \
  -p pt.124M.cb \
  --local
```

### Sweep all tasks
Modify `./fb_sweep/finetune_lm.py` and make `--downstream-task` an  array to sweep over different tasks.


### 0 Shot Eval of tuned model

```bash
eval_pft () {
  model=$(readlink -f $1)
  out_dir=$1.eval
  tasks=$2
  shift; shift
  mkdir -p $out_dir
  python -m examples.few_shot.gpt3_eval --model-name $model \
   --user-dir examples/few_shot/finetune \
   --nb-few-shot-samples-values 0 \
   --tasks $tasks \
   --results-dir $out_dir \
   --max-tokens 2048 "$@" | tee $out_dir/log.out
}
```
```bash
m=/checkpoint/myleott/2021-06-11/test_pt.124M.cb.transformer_lm_gpt.pr64.mt1024.mt1024.mt4.uf2.mu250.dr0.0.atdr0.0.actdr0.0.wd0.0.adam.beta9999.eps1e-08.lr3e-05.warm25.fp16.ngpu1/checkpoint_best.pt
eval_pft $m cb
```
FYI, you can pass a path or a model name in ``model_configs.py`` as the first argument.

### Run Many few shot evals
This evaluates every model on every task. You can get it done faster by only evaluating each model on the associated task.
This python code prints out many bash commands you can paste into your terminal.
It assumes it is being run from the checkpoints dir. It will likely need to be modified to reflect your filesystem.
It also depends on the prefix you used for various runs.
```python
from pathlib import Path
import re
from examples.few_shot.tasks import get_all_tasks
ckpt_dir='.'

tasks = get_all_tasks()
def get_task(dirname):
    for task in tasks:
        if task in dirname:
            return task
    raise ValueError(f'no tasks found in {dirname}')
prefix = 'srini*'
paths = list(Path(ckpt_dir).glob(f'{prefix}/checkpoint.best_loss*.pt'))
for p in paths:
    task = get_task(p.parent.name)
    print(f'eval_pft {str(p)} {task}')
```


#### Full fine-tuning

Add the `--finetune-model-weights` option in `./fb_sweep/sweep_seq2seq_lm_pft.py`.


### Supported Models

The following models have been tested:
- `125M_gpt3_setting`
- `355M_gpt3_setting`
- `1.3B_gpt3_setting`
- `175B`


### 175B Command (V0)
- @sshleifer
```bash
conda create --clone /private/home/sviyer/.conda/envs/fairseq-20210318 -n srini-env
source activate srini-env
module purge && module load anaconda3/2020.11 cudnn/v8.0.3.33-cuda.11.0 cuda/11.0 fairusers_aws nvtop/1.0.0/gcc.7.3.0 openmpi/4.1.0/cuda.11.0-gcc.9.3.0 ripgrep/11.0.2 NCCL/2.8.3-1-cuda.11.0

PYTHONPATH="." ./fb_sweep/finetune_lm.py -t 1 -n 32 -g 8 --constraint volta32gb \
  --checkpoints-dir $HOME/gshard_pft_v2  \
  --model 175B --task flan --resume-failed --partition xlmg
```

### MOE Support

Not currently tested.
