# Evaluation with Prompting

We currently have an easy way to evaluate a model using in-context learning (prompting) for a specific task. 

## How to evaluate a model on a dataset

Every once in a while, we want to evaluate a model on a specific validation dataset (e.g. ThePILE) and see it’s perplexity. Here are the steps to do that (source (https://fb.workplace.com/groups/4059845974043806/permalink/4569149763113422)).

First you want to allocate the resources with salloc

```
salloc --gpus-per-node 8 -C volta32gb --nodes 1 --ntasks-per-node 1 --cpus-per-task 80 --time 4000 --mem 480G --partition learnfair
```

To run the eval with MoE models, you can use the following command

```
srun python -m fairseq_cli.eval_lm /large_experiments/xlmg/data/the_pile/v1/bin/ \
--path /checkpoint/halilakin/moe_lm/top2_64e/top2_64e.roberta.me_fp16.bm_none.tps1024.transformer_lm_gpt2_big_wide.dl12.moe_w0.01.all.share.adam.b2_0.98.eps1e-06.cl0.1.lr0.002.wu2000.dr0.0.atdr0.0.wd0.0.ms2.uf8.mu28000.s1.ngpu32/checkpoint_last.pt \
--gen-subset valid --task language_modeling --sample-break-mode none \
--batch-size 4 --fp16 --is-moe --distributed-world-size 8 \
--model-overrides "{'world_size': 8, 'moe_eval_capacity_token_fraction': 0.05 }"
```


## How to evaluate a model on downstream tasks

### You have gpus allocated

You would usually want to run a single model and few tasks for debugging purposes, and iterate over config, etc. 
If you want to run on your devfair and see some numbers in the output:

```
python -m examples.few_shot.gpt3_eval --model-name 2.7B_gpt3_setting \
--tasks copa boolq cb rte --nb-few-shot-samples-values 0 1 --results-dir experiments_group_out_dir
```

If you pass —results-dir the result metrics and potentially some metadata will be written in a .json file for each evaluation run (task, template, lang, num few-shot examples):

`experiments_group_out_dir/task.{task_name}_tmp.{template_name}_lang.{lang_name}_calib.{calibrator_name}_fs{nb_few_shot_samples}_results.json`

 
If you want to run experiments and export the predictions for analysis, you can pass extra params. The following exports the predictions in a json format in a given directory. 

```
out_dir="experiments_group_out_dir"
python -m examples.few_shot.gpt3_eval --model-name 2.7B_gpt3_setting \
 --tasks copa --nb-few-shot-samples-values 0 1 \
 --predictions-dump-dir $out_dir \
 --results-dir $out_dir \
 --add-prompt-to-meta \
 --add-positional-scores-to-meta \
 --add-prompt-tokens-to-meta \
 --add-calib-to-meta 
```

Moe 500B model
You need 16 x 8 gpus for this so you need to allocate resources:
```
salloc --nodes 16 --ntasks-per-node 8 --gpus-per-node 8 --partition learnfair --time 3-00:00:00 --mem-per-gpu 58G --cpus-per-task 8 -C volta32gb
```

When you are granted slurm allocation you can run the evaluation:
```
out_dir="experiments_moe_500B_300b_tokens"
moe_token_fraction=0.10
mkdir -p $out_dir
srun -e $out_dir/log.err -o $out_dir/log.out python -m examples.few_shot.gpt3_eval --model-name moe_500B_300b_tokens \
 --tasks copa --nb-few-shot-samples-values 0 \
 --moe-eval-capacity-token-fraction ${moe_token_fraction} \
 --predictions-dump-dir $out_dir \
 --results-dir $out_dir \
 --add-prompt-to-meta \
 --add-positional-scores-to-meta \
 --add-prompt-tokens-to-meta \
 --add-calib-to-meta 
```

### You want to run a model that requires more gpus

Allocate slurm resources for interactive use. 
Note: This is discouraged by some people in the FAIR cluster group but it is the way to go if you want to do iterations over the code etc and you need more than the 2 gpus you have on the devfair. Please, adjust the number of nodes and gpus per node according to your needs. The number of gpus = nodes x gpus-per-node need to be == to the world-size of your model.

```
salloc --nodes 1 --ntasks-per-node 1 --gpus-per-node 8 --partition XLM-G \
       --time 3-00:00:00 --mem-per-gpu 58G --cpus-per-task 8 -C volta32gb
# Resources will be allocated for you. 
# You need to run the commands above with 'srun' to run commands on the nodes allocated with slurm

srun python -m examples.few_shot.gpt3_eval --model-name 2.7B_gpt3_setting \
--tasks copa boolq cb rte --nb-few-shot-samples-values 0 1
```


## Run multiple evaluations on slurm

If you want to evaluate multiple model(s) across multiple tasks you can copy and do that with the following script:
examples/few_shot/scripts/schedule_jobs_few_shot.py (https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/scripts/schedule_jobs_few_shot.py) 

The aim of the script is to be able to select multiple models, for multiple tasks,  and execute them in a slurm execution pool. That is, we use submit to schedule the tasks in the slurm queue and dump the logs in a given directory. 

DO NOT USE THE SCRIPT INSIDE salloc!


## Choose or add a new model

The models are defined inexamples/few_shot/models.py (https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/models.py). Currently they are defined is several groups by data source. 
Here is an example how to define a new model:
```
"moe_500B_300b_tokens": { # This is the name to use in --model-name when running experiments
        "model_path": "/large_experiments/moe/namangoyal/checkpoints/moe_lms/enlm/moe_500b_24l_sqrt.me_fp16.bm_none.tps1024.transformer_lm_gpt2_bigger.dl24.demb2304.dffn9216.moe_w0.01.all.share.adam.b2_0.98.eps1e-08.cl0.0.lr0.0007.wu2000.dr0.0.atdr0.0.wd0.0.ms8.uf1.mu72000.s1.ngpu512/converted/checkpoint_3_72000.pt",
        "dict_path": "/private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/dict.txt",
        "extra_args": [ # These are additional arguments for the eval_lm.
                        # If you created a new model, you know what to do here
            "--batch-size", "1",
            "--distributed-world-size", "128", 
            "--distributed-port", "15187",
            "--is-moe", # This is used only for the moe models!
        ],
        "model_overrides": { # Model specific overrides 
            'world_size': 128, # Number of gpus. Should be == n_nodes x gpu_per_node
            "bpe": "gpt2",
            "moe_eval_capacity_token_fraction": 0.10,
        },
    },
```

## Choose a task

We currently have multiple tasks implemented in examples/few_shot/tasks.py (https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/tasks.py)
The list of tasks that are implemented currently are listed in thisdocuments. (https://docs.google.com/spreadsheets/d/1C0vhEx0t6c_rpxQog_UF9EJaZCsFHw9YwbidVFWKG4E/edit#gid=0)
The actual task names used in —tasks param are:
['copa', 'xcopa', 'hellaswag', 'storycloze', 'winograd', 'winogrande', 'piqa', 'arcchallenge', 'arceasy', 'openbookqa', 'commonsenseqa', 'exams', 'naturalquestions', 'triviaqa', 'webquestions', 'wic', 'boolq', 'cb', 'rte', 'wsc', 'record', 'multirc', 'snli', 'mnlimatched', 'mnlimismatched', 'anlir1', 'anlir2', 'anlir3', 'xnli', 'addition2digit', 'addition3digit', 'addition4digit', 'addition5digit', 'addition6digit', 'subtraction2digit', 'subtraction3digit', 'subtraction4digit', 'subtraction5digit', 'subtraction6digit', 'multiplication2digit', 'singledigit3ops', 'sumofdigits', 'cycledletters', 'anagrams1', 'anagrams2', 'symbolinsertion', 'reversedwords', 'wmt14fren', 'wmt14enfr', 'wmt16deen', 'wmt16ende', 'wmt16roen', 'wmt16enro', 'satanalogies', 'simplification', 'realtoxicityprompts', 'naturalinstructions']

Note: If you want to quickly run some experiments on some tasks, consider the model and execution_time column in [this sheet.](https://docs.google.com/spreadsheets/d/1Mnqc3PpARkka-ETmwslM4xSf2EwB0Fc-yW-lFQ4z6_U/edit#gid=763367053)


## Run experiments with the Openai api

To run experiments with the OpenAI api, you need to have an api key and the openai python library installed.
```
pip install openai
export OPENAI_API_KEY=your-api-key-here
```

You can use the following models as `--model-name`:
`openai_ada`, `openai_babbage`, `openai_curie`, `openai_davinci`

Use `--predictor-name CLMPromptingOpenaiApi`

Note: Please, when running experiments, note the [price per token: Davinci > Curie > Babbage > Ada](https://beta.openai.com/pricing)!

## Run experiments with HuggingFace

To run experiments with HuggingFace you will need the HuggingFace library installed.

```
pip install transformers
```

You can use the following models as `--model-name`:
`huggingface_gpt2`, `huggingface_gpt2-xl`, or `huggingface_EleutherAI=gpt-neo-2.7B`

Use `--predictor-name CLMPromptingHuggingFacePredictor`

## Run experiments with the Natural Instructions evaluation
We have implemented the prompting baselines for the Natural Instructions dataset which will be used as a basis for future explorations. 

```
salloc --nodes 1 --ntasks-per-node 1 --gpus-per-node 8 --partition XLM-G \
       --time 3-00:00:00 --mem-per-gpu 58G --cpus-per-task 8 -C volta32gb
# Resources will be allocated for you. 
# You need to run the commands above with 'srun' to run commands on the nodes allocated with slurm

srun python -m examples.few_shot.gpt3_eval --model-name 2.7B_gpt3_setting --train-sep '\n' 
--tasks natural_instructions --nb-few-shot-samples-values 0
```

## Description of metadata

When running ``examples.few_shot.gpt3_eval``, several extra fields will be emitted that relate to the perplexity of the candidates:
* `ppl_selected_candidate`: perplexity of the best candidate tokens (only the candidate tokens)
* `ppl_full_selected_candidate`: perplexity of the best candidate tokens (including the common prefix)
* `ppl_common_prefix`: perplexity of the common prefix tokens
* `ppl_candidates_full_prompt__neutral`: perplexity of all candidate tokens (incl.  common prefix) for prompts generated for label "neutral" (not only gold label)
* `ppl_candidates_full_prompt__entailment`: perplexity of all candidate tokens (incl. common prefix) for prompts generated for label "entailment" (not only gold label)
* `ppl_candidates_full_prompt__contradiction`: perplexity of all candidate tokens (incl. common prefix) for prompts generated for label "contradiction" (not only gold label)
* `ppl_candidates_full_prompt`: perplexity of all candidate tokens (incl. common prefix)
* `ppl_candidates`: perplexity of all candidate tokens (only the candidate tokens)
