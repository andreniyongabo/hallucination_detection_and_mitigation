# GShard Multilingual Language Modeling Experiments

## Environment Setup

You will need to install fairseq, activate the [conda env](https://fb.workplace.com/groups/fairseq/permalink/262715387865587/) and load modules.
```
conda activate fairseq-20210318
module load anaconda3/2020.11 cudnn/v8.0.3.33-cuda.11.0 cuda/11.0 openmpi/4.1.0/cuda.11.0-gcc.9.3.0
```

## Training sweep
The sweep script supports training MoE multilingual checkpoints with different language sets and different number of experts. Adjust the hyperparameters in the script to start.
```
./examples/gshard_multilingual/sweep_gshard_multilingual_lm.sh
```

## Evaluate perplexity
The evaluation script evaluates a multilingual language model (dense or MoE) on a set of pre-specified target languages and output the validation ppl of each language and all languages combined.
```
./examples/gshard_multilignual/eval_gshard_multilingual_lm.sh
```

Example output:
```
{
    "valid": {
        "loss": 4.0976,
        "perplexity": 17.12,
        "r0_tps_step": 16040.7358,
        "ntok_total": 4086608,
        "gpu_step_seconds": 254.337,
        "wall_time": 129.6861,
        "wall_time_load": 89.8148,
        "wall_time_model": 39.8713
    },
    "valid_en_XX": {
        "loss": 4.5222,
        "perplexity": 22.9786,
        "r0_tps_step": 16372.9738,
        "ntok_total": 1047744,
        "gpu_step_seconds": 63.8789,
        "wall_time": 13.6644,
        "wall_time_load": 0.0098,
        "wall_time_model": 13.6546
    },
    "valid_fr_XX": {
        "loss": 3.9755,
        "perplexity": 15.7307,
        "r0_tps_step": 16358.2513,
        "ntok_total": 729043,
        "gpu_step_seconds": 45.4383,
        "wall_time": 11.0432,
        "wall_time_load": 0.0076,
        "wall_time_model": 11.0357
    }
}
```