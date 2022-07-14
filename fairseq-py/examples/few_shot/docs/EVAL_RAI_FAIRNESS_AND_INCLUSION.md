# RAI Fairness & Inclusion Evaluation
Our aim is to easily execute consistent experiments for all fairness benchmarks and available internal models.

## Run evaluations
The [schedule_jobs_rai_fairness_and_inclusion.py](examples/few_shot/scripts/experiments/schedule_jobs_rai_fairness_and_inclusion.py) script contains the implemented runnable benchmarks for all models that we care about.

See available args:
```
python examples/few_shot/scripts/experiments/schedule_jobs_rai_fairness_and_inclusion.py -h
```

### When to run the evaluations? 
We want to run the script only when new models and new tasks are added.


### Run dense models on slurm
Running the following command will schedule slurm jobs for all `gpt3_setting` models and all tasks that do not yet have produced results in the output directory:
```
RAI_RESULTS_DIR=/large_experiments/xlmg/results/fairness-and-inclusion/
python examples/few_shot/scripts/experiments/schedule_jobs_rai_fairness_and_inclusion.py \
-o $RAI_RESULTS_DIR -m gpt3_setting -t all
```
Note: The script will try to schedule the jobs on the XLM-G partition, including separate jobs with 256 gpus for the MoE models which can take time. 


### Run inside salloc
If you want to run the experiments with the large MoE models soon, then it is better to allocate resources and run locally:

Allocate enough resources for the large MoE models
```
salloc --nodes 32 --gpus-per-node 8 --partition XLM-G --time 3-00:00:00 \
--mem-per-gpu 58G --cpus-per-task 8 -C volta32gb
```

Run the script locally for all moe models:
```
RAI_RESULTS_DIR=/large_experiments/xlmg/results/fairness-and-inclusion/
srun -e rai_moe.err -o rai_moe.out python examples/few_shot/scripts/experiments/schedule_jobs_rai_fairness_and_inclusion.py \
-o $RAI_RESULTS_DIR -m moe -t all --local
```

## Add new models and tasks
To add new models and implemented tasks, update the 
`available_model_settings` and `available_task_settings` dictionaries in the script!

## Gather results
Run the [collect_results.py](examples/few_shot/docs/DISPLAY_RESULTS.md) script with the results output dir.
```
RAI_RESULTS_DIR=/large_experiments/xlmg/results/fairness-and-inclusion/
python examples/few_shot/scripts/collect_results.py -i $RAI_RESULTS_DIR -v preferred_metrics_mean -o ${RAI_RESULTS_DIR}/results.tsv
```



