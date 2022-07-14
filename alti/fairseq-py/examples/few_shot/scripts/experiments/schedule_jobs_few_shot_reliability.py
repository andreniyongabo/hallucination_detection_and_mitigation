# fmt: off
import argparse 
import sys
from examples.few_shot.scripts.experiments.schedule_jobs_few_shot import *
import copy
import re
from examples.few_shot.model_configs import get_model_checkpoint_names, get_model_names
from examples.few_shot.tasks import get_tasks_by_group

if __name__ == "__main__":
    """
    This script submits evaluation jobs that run evaluations with the .
    
    Setting:
        - Scoring - `mean`, since it has good justifications, and it was used by gpt3. 
        - nshot - 0
    
    Commands:
        - Run       
            # Run 13B model for all the reliability tests for a single task in the NIE sentiment cluster      
            PYTHONPATH=. python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_reliability.py -o /checkpoint/$USER/reliability_benchmark_results --slurm-partition xlmg -m 13B_gpt3_setting -t nie_reliability_nienoextemplate_task195

            # Run flan_minus_sentiment_13B for all the reliability tests for a single task in the NIE sentiment cluster
            PYTHONPATH=. python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_reliability.py -o /checkpoint/$USER/reliability_benchmark_results --slurm-partition xlmg -m flan_minus_sentiment_13B -t nie_reliability_flantemplate_task195 --user-dir examples/few_shot/finetune

            # Run T0pp model for all the reliability tests for a single task in the NIE sentiment cluster
            PYTHONPATH=. python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_reliability.py -o /checkpoint/$USER/reliability_benchmark_results --slurm-partition xlmg -m huggingface_bigscience=T0pp -t nie_reliability_nienoextemplate_task195 --predictor-name clmpromptinghuggingface

            # Run OpenAI-Instruct model for all the reliability tests for a single task in the NIE sentiment cluster (Note this needs OpenAI API Key)
            PYTHONPATH=. python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_reliability.py -o /checkpoint/$USER/reliability_benchmark_results --slurm-partition xlmg -m openai_curie-instruct-beta-v2 -t nie_reliability_nienoextemplate_task195 --predictor-name clmpromptingopenaiapi --local
    
    """

    # task settings    
    available_tasks_settings = default_tasks_settings.copy()
    task_run_groups = default_task_run_groups.copy()

    # NIE Reliability Tests Settings
    nie_reliability_tasks = []
    nie_reliability_sentiment_tasks = ["task195", "task196", "task284", "task363", 
        "task420", "task421", "task422", "task423", "task475", "task476", "task477",  
        "task478", "task517", "task518", "task746", "task819", "task823", "task833", "task923"
    ]
    nie_reliability_tasks += nie_reliability_sentiment_tasks
    nie_reliability_commonsense_tasks = ["task073", "task116", "task291", "task295", "task403",
    "task827", "task828", "task1135",
    ]
    nie_reliability_tasks += nie_reliability_commonsense_tasks
    all_tasks = get_all_tasks()
    nie_reliability_task_settings = {}
    for task in nie_reliability_tasks:
        # All tasks + their eval protocols tasks with FULL NIE template
        common_tasks = [t for t in all_tasks if task in t and "nie_reliability_benchmark__" in t]
        nie_reliability_task_settings["nie_reliability_"+task] = ["nie_reliability_"+task, common_tasks, {}]

        # All tasks + their eval protocols tasks with FLAN template
        eval_params = {t+"_template": "nienoexample_to_flan" for t in common_tasks}
        nie_reliability_task_settings["nie_reliability_flantemplate_"+task] = [
            "nie_reliability_flantemplate_"+task, 
            common_tasks, 
            eval_params
        ]

        # All tasks + their eval protocols tasks with NIE no example template
        eval_params = {t+"_template": "nienoexample" for t in common_tasks}
        nie_reliability_task_settings["nie_reliability_nienoextemplate_"+task] = [
            "nie_reliability_nienoextemplate_"+task, 
            common_tasks, 
            eval_params
        ]

        # Take only original tasks with nieonlyexamples template
        common_tasks = [t for t in all_tasks if task in t and "original" in t and "nie_reliability_benchmark__" in t]
        eval_params = {t+"_template": "nieonlyexamples" for t in common_tasks}
        nie_reliability_task_settings["nie_reliability_nieonlyextemplate_"+"original_"+task] = [
            "nie_reliability_nieonlyextemplate_"+"original_"+task, 
            common_tasks, 
            eval_params
        ]

    available_tasks_settings.update(nie_reliability_task_settings)
    
    # model settings
    available_model_settings, model_run_groups = get_extended_default_model_settings_and_groups()
    available_model_settings.update(
        {
            "flan_minus_sentiment_13B": ModelSchedulingSetting("flan_minus_sentiment_13B", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048, "distributed_port": 15188}, 1, 2, 1, 8, 3),
            "flan_minus_sentiment_6.7B": ModelSchedulingSetting("flan_minus_sentiment_6.7B", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048, "distributed_port": 15188}, 1, 2, 1, 8, 3),
        }
    )
            
   
    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs for english benchmarks")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)
    
    # override defaults
    USER = os.getenv("USER")
    arg_modify_default(parser, "output", f"/checkpoint/$USER/reliability_benchmark_results/")
    arg_modify_default(parser, "scoring", "mean")
    arg_modify_default(parser, "nb_few_shot_samples_values", [0])
    arg_modify_default(parser, "num_trials", 5)

    # set default dir for results
    args = parser.parse_args()

    #args.dry_run = True
    #args.local = True

    print("Arguments:")
    print(args)
       
    # schedule jobs -- see the function documenation to learn the order of param updates.
    schedule_experiment_jobs(args, 
                             task_run_groups, available_tasks_settings, 
                             model_run_groups, available_model_settings,
                             custom_base_run_args = {}, 
                             custom_override_run_args = {}
                            )

    print_display_results_command(args)
    sys.exit(0) # sometimes srun hangs and calling sys.exit(0) exits properly
