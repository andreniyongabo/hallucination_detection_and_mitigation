# How to collect and display results

## Basic usage
The current results from few-shot prompting evaluations are exported to a single `{setting}_results.json` file for each (task, model, template, n_few_shot) setting. The evaluation script also has the option to export single predictions with meta fields to `{setting}_predictions.json`.

The results can be aggregated using the following script:
```
python examples/few_shot/scripts/collect_results.py \
 -i "/checkpoint/tbmihaylov/few_shot/2021-05-26-paper-experiments/*/" "/large_experiments/xlmg/models/sshleifer/few_shot_results/" \
 -o all_results.tsv -t superglue sciq --view by_run_params_simple
```

The script has the following arguments:
```
usage: collect_results.py [-h] [-i INPUT_DIRS [INPUT_DIRS ...]] [-o OUTPUT]
                          [-v {raw,preferred_metrics_mean,preferred_metrics,by_run_params_simple,by_run_params_simple_ext}]
                          [-t TASKS [TASKS ...]]
                          [--recalc-metrics-from-positional-scores]
                          [--calculate-ensemble] [--debug] [--sep SEP]
                          [--overwrite-output] [--watch]
                          [--watch-interval WATCH_INTERVAL]
                          [--wandb-project WANDB_PROJECT]
                          [--wandb-run-name WANDB_RUN_NAME]
                          [--resume-wandb-run-id RESUME_WANDB_RUN_ID]
                          [--wandb-log-preferred-metrics]

Aggregate and display results from multiple result directories.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_DIRS [INPUT_DIRS ...], --input-dirs INPUT_DIRS [INPUT_DIRS ...]
                        List of directories with results. Can include * to
                        expand multiple dirs. By default the script looks at
                        the given directory and subdirectories +1 level.
  -o OUTPUT, --output OUTPUT
                        The path to an output csv/tsv file. Raw results file
                        OUTPUT.raw.jsonl is also generated!
  -v {raw,preferred_metrics_mean,preferred_metrics,by_run_params_simple,by_run_params_simple_ext}, --view {raw,preferred_metrics_mean,preferred_metrics,by_run_params_simple,by_run_params_simple_ext}
                        The view for the output. Options are: `raw` - Raw
                        results with all generated
                        fields.;`preferred_metrics_mean` - Task, model, and
                        preferred metrics with mean only;`preferred_metrics` -
                        Base run columns and preferred
                        metrics.;`by_run_params_simple` - Results grouped by
                        run settings. Only preferred_metric with std is
                        displayed.;`by_run_params_simple_ext` - Results
                        grouped by run settings, incuding train_sep. Only
                        preferred_metric with std is displayed.
  -t TASKS [TASKS ...], --tasks TASKS [TASKS ...]
                        List of individual tasks to include or groups of
                        tasks. Currently the following groups are available:
                        `superglue`,`sciq`,`rai`,`natural_instructions`
  --recalc-metrics-from-positional-scores, -r
                        [SLOW] Recalculates accuracy scores from predictions.
                        This requires the predictions files to have the
                        positional scores expoerted.
  --calculate-ensemble  Calculates ensemble score from multiple prediction
                        files when available
  --debug               Output detailed error message when processing files.
  --sep SEP             Output file separator.
  --overwrite-output    Overwrite the existing output file. If not specified
                        new results are appended.
  --watch               Watches for new
  --watch-interval WATCH_INTERVAL
                        The interval in seconds to check for new results.
  --wandb-project WANDB_PROJECT
                        Name of the wandb project where metrics will be logged. WandB will create this project for you if it doesn't already exist. If unspecified, wandb logging will be disabled.
  --wandb-run-name WANDB_RUN_NAME
                        A short descriptive name for the WandB logging run. This name will show up in the UI.                                                                                      
  --resume-wandb-run-id RESUME_WANDB_RUN_ID
                        WandB projects can have multiple runs. If you want to resume a previous WandB run, specify the run id here. If unspecified, WandB will create a new run automatically.     
  --wandb-log-preferred-metrics
                        Log only preferred metrics to WandB. If unspecified, all metrics are logged to WandB, which makes it difficult to visualize dashboards.
  ```


### Ensemble accuracy
When multiple predictions files are available for some of the runs, ensemble accuracy is calculated for multi-choice tasks. The ensemble is calculated using sum of the normalized scores for multiple candidate choices. The following results are available:
`accuracy__raw_ensemble` - This is accuracy calculated from the unnormalized `score_raw` from the prediction candidates `meta`.
`accuracy_run_ensemble` - This is accuracy calculated from the `score` (`score` is the normalized score when calibration is used. otherwise it is equal to `score_raw`)

### Recalculated accuracy scores from positional_scores
`--recalc-metrics-from-positional-scores` or `-r` calculates several additional accuracy scores if positional_scores are available in the predictions file:
To show this we will compute the mean of the model performance. For each setting we scored the model with several metrics including:
- `accuracy_mean_full::mean`, # mean(positional_scores) - recalculated from predictions.json
- `accuracy_mean::mean`, # mean(positional_scores[common_prefix_end:]) - recalculated from predictions.json
- `accuracy_mean_full_calib::mean`, # mean_full(prompt) - mean_full(calib)
- `accuracy_mean_calib::mean`, # mean_suffix(prompt) - mean_suffix(calib)
- `accuracy_sum::mean`, # sum(positional_scores) - recalculated from predictions.json
- `accuracy_sum_charnorm::mean`, # sum(positional_scores)/len(prompt) - recalculated from predictions.json
- `accuracy_sum_calib::mean`, # sum(prompt["positional_scores"]) - sum(calib_prompt["positional_scores"]) - recalculated from predictions.json
- `accuracy_sum_charnorm_calib::mean` # sum_charnorm(prompt) - sum_charnorm(calib_prompt)
- `accuracy_run::mean`, # re-calculated on the original score from the run (whatever it is)

## Custom views
Because the script extracts multiple metrics and fields for different tasks, in some cases we might want to create custom views. 
The scripts generates a raw results file OUTPUT.raw.jsonl. This can be loaded later and processed to create custom views.

Here is an example [notebook for reading the .raw.jsonl and creating custom views](https://github.com/fairinternal/fairseq-py/blob/gshard/examples/few_shot/scripts/display_results_custom_view.ipynb)

## Logging evaluation metrics to Weights and Biases
`collect_results.py` script now supports logging evaluation metrics to Weights and Biases (WandB). We use fairwandb.org
(an internally hosted WandB instance) to log metrics.

### Basic Usage
```
python examples/few_shot/scripts/collect_results.py <your other arguments> --wandb-project xlmg_test_1 --wandb-run-name some_awesome_wandb_run_name
```
This will create a new run named some_awesome_wandb_run_name in the xlmg_test_1 WandB project. Metrics collected by the
collect_results.py script will be logged to `some_awesome_wandb_run_name` run.

### Logging only preferred metrics
Adding the --wandb-log-preferred-metrics flag will only log the preferred metrics per task, as compared to logging all metrics.
This is especially useful in the intermittent model evaluation flow, where we are getting new metrics for checkpoints being 
trained, and would like to compare metrics across checkpoints.

### Continuing a previous run
There are instances where we might want to continue a previously cancelled run. For instance, to add metrics for newly trained
checkpoints to the same dashboard. In these situations, we can use the --resume-wandb-run-id flag, and specify the run id
we want to extend.

The run id can be seen in the WandB URL. For instance, if the run link is
https://fairwandb.org/xlm-g/xlmg_test_1/runs/a33fk74g?workspace=user-punitkoura , the run id would be a33fk74g

Example:
```
python examples/few_shot/scripts/collect_results.py -i  "/checkpoint/victorialin/few_shot/dense_7.5B_lang30_new_cc100_xl_unigram_en_tasks/" -o ~/all_results.tsv --view by_run_params_simple --wandb-project xlmg_test_1 --wandb-run-name testing_naming_runs --resume-wandb-run-id 3vpyks2k
```
