# fmt: off
import argparse 
import sys
from examples.few_shot.scripts.experiments.schedule_jobs_few_shot import *

if __name__ == "__main__":
    """
    The current script executes/schedules experiments on several fairness benchmarks and models. 
    
    Selected tasks: Currently we have implemented the following tasks:
        - RealToxicityPrompts 
        - Stereoset
        - CrowsPairs
    
    Setting:
        - Scoring - `mean`

    Commands:
        Note: The commands below end with --dry-run. Do not copy this part to actually run the experiments.

        - Debug locally with dense gpt3_setting 
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t stereoset -m gpt3_setting --local --dry-run
        
        - Run dense gpt3_setting on all tasks on slurn
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t all -m gpt3_setting --dry-run
    
        - Run on salloc with moe 
            # allocate 32 nodes with 8 gpus each -- takes some time 
            salloc --nodes 32 --gpus-per-node 8 --partition XLM-G --time 3-00:00:00 --mem-per-gpu 58G --cpus-per-task 8 -C volta32gb
            
            # remove the --dry-run for full expderiment
            srun -e moe_log.err -o moe_log.out python examples/few_shot/scripts/experiments/schedule_jobs_rai_fairness_and_inclusion.py -t all -m moe --local --dry-run
    """

    # task settings
    # available_tasks_settings = default_tasks_settings.copy()
    available_tasks_settings = {
        # tasks group : str, tasks list List[str], evaluation params: Dict[param_name, param_value] -- these are the same gpt3_eval input params
        "stereoset": ("stereoset", ["stereoset"], {}),
        "crowspairs": ("crowspairs", ["crowspairs"], {}), 
        "realtoxicityprompts": ("realtoxicityprompts", ["realtoxicityprompts"], {}),
    }

    # task_run_groups = default_task_run_groups.copy()
    task_run_groups = {
        "rai": list(available_tasks_settings.keys())
    }
    
    # model settings
    available_model_settings = default_model_settings.copy()
    model_run_groups = default_model_run_groups.copy()
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs.")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)
    
    # override defaults
    arg_modify_default(parser, "output", "/large_experiments/xlmg/results/fairness-and-inclusion/")
    arg_modify_default(parser, "scoring", "mean")
    arg_modify_default(parser, "tasks", ["rai"])
    arg_modify_default(parser, "models", ["gpt3_setting"])
    arg_modify_default(parser, "nb_few_shot_samples_values", [0])
    arg_modify_default(parser, "num_trials", 1)

    # set default dir for results
    args = parser.parse_args()

    # args.dry_run = True
    # args.local = True

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
    