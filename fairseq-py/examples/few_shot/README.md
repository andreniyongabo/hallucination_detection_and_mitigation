# XL Generative Zero-Shot Learner

## Setup the environment

### Base setup
If you are using fairseq for the first time you will want to setup the proper environment. 

1. For best chance of success, copy the current environment from [this post](https://fb.workplace.com/groups/fairseq/permalink/262715387865587/)

After copying and activating the environment, you need to build fairseq ([2133](https://github.com/pytorch/fairseq/issues/2133#issuecomment-628923011)) using the following command:
```
python setup.py build_ext --inplace
```

2. Install [fairscale](https://github.com/facebookresearch/fairscale)

3. Install the extra requirements:
```bash
pip install -r examples/few_shot/extra_requirements.txt
```

### Advanced setup
This is not mandatory for evaluation and debugging so if you don't know what you are doing stop here. 

4. Install megatron

Clone from the v2.4 tag:
```bash 
git clone --depth=1 --branch v2.4 https://github.com/NVIDIA/Megatron-LM.git
```

Manually edit setup.py and replace `name=__package_name__` to `name="megatron"`. The default setup results in installing `megatron-lm` package but the code expects `megatron`.

Install megatron:
```bash
cd Megatron-LM
pip install -e .
```

Known issues: 
- Megatron does not seem to work on devfair with the fairseq-20210318 environment (`2021-10-13`). It fails with `RuntimeError: CUDA error: no kernel image is available for execution on the device`. Anyway, if you git it to work, please update this message with the proposed fix. 


## Example Usage
### CLI
```bash
$ python -m examples.few_shot.gpt3_eval --model-name 124M --tasks copa cb --nb-few-shot-samples-values 0 1 32 --num-trials 5 --train-sep "\n"

model_name=124M
Infering max tokens for model...
Setting max_tokens to 16384
task=copa
nb_few_shot_samples=0
100it [00:00, 838.46it/s]
results={'task': 'copa', 'nb_few_shot_samples': 0, 'scores': [64.0], 'mean': 64.0, 'std': 0.0, 'mean_confidence_interval': nan}

nb_few_shot_samples=1
100it [00:00, 680.98it/s]
results={'task': 'copa', 'nb_few_shot_samples': 1, 'scores': [62.0, 62.0, 65.0, 62.0, 60.0], 'mean': 62.2, 'std': 1.5999999999999999, 'mean_confidence_interval': 2.2211560841582387}

nb_few_shot_samples=32
OOM: max_tokens=16384 ==> max_tokens=8192
100it [00:02, 49.96it/s]
results={'task': 'copa', 'nb_few_shot_samples': 32, 'scores': [63.0, 65.0, 63.0, 64.0, 60.0], 'mean': 63.0, 'std': 1.6733200530681511, 'mean_confidence_interval': 2.3229406353851942}

task=cb
nb_few_shot_samples=0
56it [00:00, 190.88it/s]
results={'task': 'cb', 'nb_few_shot_samples': 0, 'scores': [42.857142857142854], 'mean': 42.857142857142854, 'std': 0.0, 'mean_confidence_interval': nan}

nb_few_shot_samples=1
56it [00:00, 133.04it/s]
results={'task': 'cb', 'nb_few_shot_samples': 1, 'scores': [41.07142857142857, 17.857142857142858, 41.07142857142857, 41.07142857142857, 12.5], 'mean': 30.71428571428571, 'std': 12.797480619406368, 'mean_confidence_interval': 17.76575121230725}

nb_few_shot_samples=32
56it [00:04, 12.43it/s]
results={'task': 'cb', 'nb_few_shot_samples': 32, 'scores': [42.857142857142854, 41.07142857142857, 41.07142857142857, 42.857142857142854, 41.07142857142857], 'mean': 41.78571428571428, 'std': 0.874817765279706, 'mean_confidence_interval': 1.2144417511754582}
```

### Python
```python
from examples.few_shot.gpt3_eval import run_evaluations_from_model_name

run_evaluations_from_model_name(model_name="124M", tasks=["copa", "cb"], nb_few_shot_samples_values=[0, 1, 32], num_trials=5, train_sep="\n")
```



### Testing

See `testing.md` to learn how to run quick (2 minute) unit tests on 2 v100s (or add them).

For longer benchmarks, the following command can be used to ensure that a 203,585,536 parameter model
achieves a reasonable perplexity and throughput in 7200 steps.

```bash
bash scripts/check_gshard_ppl_regression.sh
```
This takes about 80 minutes to run.

Tip: You can pass extra clargs to the script, as long as they are defined in `fb_sweep/benchmark_lm.py`. For example,
```bash
bash scripts/check_gshard_ppl_regression.sh --epg 1 --dropout 0.1
```
will train an MoE with 1 expert on each of the 8 GPUs (142 minute runtime) with dropout.


# Expected Results

(1) training performance (the last `train_inner` entry of each run):
```bash
./fb_sweep/agg_results.py "/checkpoint/$USER/regression_check*/train.log" --keep-cols cuda_gb_free,wps,ups,loss,ppl,num_updates --log-pattern "| train_inner |"
```

Expected results:
```
+--------------------------------------------------------------------+----------------+--------+-------+--------+-------+---------------+
| parent_path                                                        |   cuda_gb_free |    wps |   ups |   loss |   ppl |   num_updates |
+====================================================================+================+========+=======+========+=======+===============+
| regression_check_2021-08-03-15:36.adam.ful.dl12.d1024.bs8.ngpu8    |           10.2 | 188497 |  1.44 |   5.28 | 38.85 |          7200 |
+--------------------------------------------------------------------+----------------+--------+-------+--------+-------+---------------+
| regression_check_2021-08-03-15:37.8e.adam.ful.dl12.d1024.bs8.ngpu8 |            4.4 | 112606 |  0.86 |   4.93 | 28.73 |          7200 |
+--------------------------------------------------------------------+----------------+--------+-------+--------+-------+---------------+
```

(2) valid perplexity  (the last `train_inner` entry of each run):

```bash
./fb_sweep/agg_results.py "/checkpoint/$USER/regression_check*/train.log" --keep-cols ppl,num_updates --log-pattern valid
```

Expected results:

```
+--------------------------------------------------------------------+-------+---------------+
| parent_path                                                        |   ppl |   num_updates |
+====================================================================+=======+===============+
| regression_check_2021-08-03-15:36.adam.ful.dl12.d1024.bs8.ngpu8    | 33.88 |          7200 |
+--------------------------------------------------------------------+-------+---------------+
| regression_check_2021-08-03-15:37.8e.adam.ful.dl12.d1024.bs8.ngpu8 | 23.47 |          7200 |
+--------------------------------------------------------------------+-------+---------------+
```
