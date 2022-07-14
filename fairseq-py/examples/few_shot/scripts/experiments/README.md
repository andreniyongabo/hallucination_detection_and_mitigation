# Organization and scheduling of multiple few-shot prompting evaluation jobs

Here we describe how to create new, use and update the existing few-shot prompting eval scheduling scripts.
The aim of having specific few-shot scheduling scripts is to allow configuring and scheduling slurm experiments, in a similar way to how the current sweep scripts work but with configuration options tied around the few-shot prompting evaluation.
The benefits are:

* Anyone can easily re-run existing experiments when new model or template are added.
* Easily define new task-specific and model-specific configurations for few-shot learning.
* Change the configurations in the python code, so the changes are tracked.
* Create new experiments script easily by copy-pasting existing scripts.
* Easily execute multiple model and task configurations locally (bash, srun) or on slurm.
* Do not re-run configurations that are already executed.
* Well working configurations of models and tasks can be defined in the base script and reused.

An example for such script is the [RAI eval](https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/docs/EVAL_RAI_FAIRNESS_AND_INCLUSION.md) that contains tasks for measuring stereotypical bias and will be executed for each new model that we train in order to compare it to previous models.
A list of existing executable configurations are:

* [schedule_jobs_few_shot_english.py](https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py) - Few-shot eval for English tasks.
* [schedule_jobs_rai_fairness_and_inclusion.py](https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/scripts/experiments/schedule_jobs_rai_fairness_and_inclusion.py) - RAI evaluation tasks.
* [schedule_jobs_few_shot_multilingual.py](https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py) - Multilingual tasks evaluation experiments.
* [schedule_jobs_few_shot_baselines_random.py](https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/scripts/experiments/schedule_jobs_few_shot_baselines_random.py) - Random baselines for multiple multichoice tasks.
* [schedule_jobs_few_shot_calibration.py](https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/scripts/experiments/schedule_jobs_few_shot_calibration.py) - Base evaluation configurations for tasks that support calibration.
* [schedule_jobs_few_shot_ppl_explore.py](https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py) - Experiments used for various experimetns with PPL-based metrics.

Look at the scripts for detailed information and example commands to multiple evaluations.
Here is an example of how to run the standard evaluations:

## Some standard configurations

* RAI eval:

```python
# --local will run all configurations sequentially in the terminal -- remove to run on slurm
# --dry-run will execute the script to verify that everything is correct and output directories can be created. 
 
RAI_RESULTS_DIR=/large_experiments/xlmg/results/fairness-and-inclusion/
python examples/few_shot/scripts/experiments/schedule_jobs_rai_fairness_and_inclusion.py \
-o $RAI_RESULTS_DIR -m gpt3_setting -t all --local --dry-run 
```

* Eval on diverse English benchmarks (BLIMP linguistic tasks, some SuperGLUE tasks, LM, Commonsense, ScienceQA)
```# --local will run all configurations sequentially in the terminal -- remove to run on slurm
    # --dry-run will execute the script to verify that everything is correct and output directories can be created. 
     
    RAI_RESULTS_DIR=/large_experiments/xlmg/results/fairness-and-inclusion/
    python examples/few_shot/scripts/experiments/schedule_jobs_rai_fairness_and_inclusion.py \
    -o $RAI_RESULTS_DIR -m gpt3_setting -t all --local --dry-run 
```

## How to define a new script

Currently there are several scripts that can be used as examples -- see above!

1. Copy (or update) an existing script from examples/few_shot/scripts/experiments
2. Update with new task/models settings that you want to experiment with: 
3. Define the execution restrictions and parameters for model configs 
4. Define groups of model settings to execute multiple experiments easily 
5. Define the task configurations that you want to execute 
6. Define groups of tasks to be able to execute multiple experiments easily 
7. Update the description of the experiment to reflect your experiment details and help others understand what the current script does. 
8. Add an example command-line with the new settings.

### 2. Update an a script with your setting

The base functionality of a jobs scheduling script is in the base [examples/few_shot/scripts/experiments/schedule_jobs_few_shot.py](https://github.com/fairinternal/fairseq-py/blob/gshard-crosslingual-prompting/examples/few_shot/scripts/experiments/examples/few_shot/scripts/experiments/schedule_jobs_few_shot.py) so if you have doubts of how something works, read the functions doc string there.

### Define the execution restrictions and parameters for models and organize into groups

We can define model configs and groups of configurations


```python
    # model settingsavailable_model_settings = default_model_settings.copy()  # inherit the settings from schedule_jobs_few_shot.py# Update with new settings that you want to useavailable_model_settings.update({
        "dense_lang16": # reference key - used for naming the model setting -- we might have different settings for the same model, depending on the 
            ("dense_lang16", 
                {}, # job_params - if you pass "combine_tasks": True, all tasks will be evaluated using a single call. # This made sense for huge models when the model weights loading took a lot of time. Now this will be empty for most cases. It is also suitable only for tasks which settings are similar
                {"train_sep": "\n"}, #  custom params - corresponds to the gpt3_eval argument variables1, # nodes - slurm param - overrides `--slurm-nodes` for jobs exdecuted with the current model1, # gpus_per_node - slurm param - overrides `--gpus-per-node`1, # ntasks_per_node - slurm param - overrides `--ntasks-per-node`8, # cpus_per_task - slurm param - overrides `--ntasks-per-node` 3, # max_parralel jobs for the current pool. The tasks for each model are executed in a common pool.
            ),
        
        # reference key:  (model_name, job_params, custom params, nodes, gpus_per_node, ntasks_per_node, cpus_per_task, max parallel jobs in this pool)"moe_128exp_lang16": ("moe_128exp_lang16", {}, {"train_sep": "\n"}, 2, 8, 1, 8, 3),
    })

    # defining groups alows scheduling jobs for multiple models with meaningful namemodel_run_groups = default_model_run_groups.copy() # inherit the settings from schedule_jobs_few_shot.pymodel_run_groups.update({
        "dense_vs_moe_lang16": ["dense_lang16", "moe_128exp_lang16"] # this will schedule experiments for both models
    })
```

### Define tasks settings and groups of task settings

The configuration is similar to how model settings are organized.


```python
    # task settingsavailable_tasks_settings = default_tasks_settings.copy()  # Inherit default tasks from schedule_jobs_few_shot.pyavailable_tasks_settings.update({
        # Override or define task settings"pawsx_default": # task setting key used for execution
            ("pawsx", # tasks run name - used in the name of the directory where the results are saved. The results of settings with the same value will be saved in the same directory.
            ["pawsx"], # list of tasks to be evaluated in this run
            {"pawsx_template": ["pawsx_default"]} # evaluation params - corresponds to the gpt3_eval argument variables
            ),
        # ...."pawsx_googletranslate_calib": ("pawsx", ["pawsx"], {"pawsx_template": ["pawsx_googletranslate"], 
                                                            "calibrator_name": "average_option",
                                                            "pawsx_calibration_options": ["sentence1:: | sentence2::"], }),
    })

    # Task groups#task_run_groups = default_task_run_groups.copy() # inherit groupstask_run_groups = {
        "multilingual_multichoice": ["xcopa", "xnli", "exams", "pawsx"],
        "translation": ['wmt14fren', 'wmt14enfr', 'wmt16deen', 'wmt16ende', 'wmt16roen', 'wmt16enro'],
        "pawsx_default_vs_gt_calib": ["pawsx_default_calib", "pawsx_googletranslate_calib"]
    }
```

The settings shown above are from the multilingual evaluation script and will allow us to several experiments with the following command:


```bash
python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py \
-t pawsx_default_vs_gt_calib -m dense_vs_moe_lang16 --nshot 0
```

### How does the scheduling/execution work?

The command line params `args`, and the task settings and groups, model settings and groups that we defined above are used to schedule jobs with the following function `schedule_experiment_jobs`. See the signature and doc string below to learn how conflicting arguments are handled.


```python
def schedule_experiment_jobs(args, 
                            task_run_groups, available_tasks_settings, 
                            model_run_groups, available_model_settings, 
                            custom_base_run_args = {}, 
                            custom_override_run_args = {}):
    """
    This method runs multiple experiments with slurms or locally by calling `run_evaluations_from_model_name(**run_args)` for each model_setting and task_setting.
    The variable `run_args` contains input arguments which are built as dictionary and passed to the function. 
    For each model_setting and task_setting we init `run_args` with the arguments passed from the `args` and update them with the 
    configuration properties first from model_setting then from task_setting.
    The variable `args` contains `models` and `tasks` arguments that specify lists of keys for model and task settings groups. 
    For each key we seek for model or task settings keys (available_[model/task]_settings) or groups of keys ([model/task]_run_groups) to use for the condifurations.

    The order of forming run_args is: 
        0) `run_args` default values are set from the input `args`
        -> 1) `run_args` is updated with `custom_base_run_args` (empty by default)
        -> 2) `run_args` is updated using the model_setting params
        -> 3) for each task_setting a clone of `run_args` is updated with the task_setting params
        -> 4) for each task_setting a clone of `run_args` is updated with the task_setting params
        -> 1) `run_args` is updated with `custom_override_run_args` (empty by default)
        
        -> run_args are passed to `run_evaluations_from_model_name`

    Args:
        args: Parsed command line arguments - these are used to update the .
        task_run_groups: Task groups. A dictionary with list of task_settings keys.
        available_tasks_settings: Available task settings. Dictionary with named task settings/configurations
        model_run_groups: Model groups. A dictionary with list of model_settings keys.
        available_model_settings: Available model settings. Dictionary with named model settings/configurations
        custom_base_run_args: Override some of the run params manually. These are updated before the model and task params.
    """
```

### Command line arguments

The scripts support multiple arguments that cover aspects of the few-shot evaluation framework and slurm parameters (Use -h to see the available parameters for particular scripts).


```
# Use -h to see the available parameters
> python examples/few_shot/scripts/experiments/schedule_jobs_few_shot.py -h

usage: schedule_jobs_few_shot.py [-h] [--local] [--dry-run] [--override-completed] [--slurm-partition SLURM_PARTITION] [--scoring SCORING] [--predictor-name PREDICTOR_NAME]
                                 [--nb-few-shot-samples-values NB_FEW_SHOT_SAMPLES_VALUES [NB_FEW_SHOT_SAMPLES_VALUES ...]] [--num-trials N] [-o OUTPUT] [-t TASKS [TASKS ...]]
                                 [-m MODELS [MODELS ...]]

Schedule few-shot jobs.

optional arguments:
  -h, --help            show this help message and exit
  --local               Run locally (not on slurm).
  --dry-run             Dry run.
  --override-completed  Overrides results that are available.
  --slurm-partition SLURM_PARTITION
                        Slurm setting: partition.
  --scoring SCORING     Scoring setting
  --predictor-name PREDICTOR_NAME
                        Predictor name
  --nb-few-shot-samples-values NB_FEW_SHOT_SAMPLES_VALUES [NB_FEW_SHOT_SAMPLES_VALUES ...], --nshot NB_FEW_SHOT_SAMPLES_VALUES [NB_FEW_SHOT_SAMPLES_VALUES ...]
                        subsample K examples from the training set for one-shot or few-shot learning
  --num-trials N        when --nb-few-shot-samples is provided, repeat experiments N times and report all results (few-shot experiments can have a high variance).
  -o OUTPUT, --output OUTPUT
                        The parent output directory for the jobs in the current run!
  -t TASKS [TASKS ...], --tasks TASKS [TASKS ...]
                        List of individual tasks or groups of tasks. Available groups are:`rai`; Available single task settings are: <here you will see a list of all available task configurations>
  -m MODELS [MODELS ...], --models MODELS [MODELS ...]
                        List of individual tasks or groups of tasks. Available groups are:`all`,`gpt3_setting`,`moe`; Available single model settings
                        are:`1.3B_gpt3_setting`,`2.7B_gpt3_setting`,`6.7B_gpt3_setting_1024ctx`,`125M_gpt3_setting`,`355M_gpt3_setting`,`moe_1.1T`,`moe_523B`,`moe_207B`
```

### Override command-line arguments

We can override the default values for the command line arguments in order to set default values for your experiments.
Here is an example from [schedule_jobs_few_shot_multilingual.py](https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/scripts/experiments/schedule_jobs_few_shot_multilingual.py).


```python
    # override defaultsUSER = os.getenv("USER")
    arg_modify_default(parser, "output", f"/checkpoint/{USER}/few_shot/multilingual")
    arg_modify_default(parser, "scoring", "mean")
    arg_modify_default(parser, "nb_few_shot_samples_values", [0, 32])
    arg_modify_default(parser, "num_trials", 10)
```

