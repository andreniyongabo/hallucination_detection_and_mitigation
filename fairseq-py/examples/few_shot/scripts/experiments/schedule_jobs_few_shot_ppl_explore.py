# fmt: off
import argparse 
import sys
from examples.few_shot.scripts.experiments.schedule_jobs_few_shot import *
import copy
import re
from examples.few_shot.tasks import get_tasks_by_group

if __name__ == "__main__":
    """
    The current script aims at executing experiments that will be used\
        to explore the prompt PPL to target performance. 
    
    Selected tasks: 
        - SuperGLUE - popular tasks
        - QA tasks:
            - ARC Challenge (w/ calib), OpenbookQA (w/ calib) which show improvement with calibration
            - ARC Easy which is a similar but easier task
        - LM tasks
            - Winogrande, piqa, hellaswag that has shown by gpt3 to offer emerging few-shot, also PPL matching the gpt3 models. 
    
    Setting:
        - Scoring - `mean`, since it has good justifications, and it was used by gpt3. 
                    We currently don't have an good reason to use 'sum' except that it works well. 
        - nshot - 0, 1, 32. The initial experiments with 0-shot has shown strong correlation and want to see if it works for > 0 shot. 
        - Trials - 10 trials are used to reduce variance.

    Commands:
        - Run 
            # run locally
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t copa -m moe --nshot 0 --local --dry-run

            # 1.3B_gpt3_setting_checkpoints
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t default -m 1.3B_gpt3_setting_checkpoints --nshot 0 -o /checkpoint/$USER/few_shot/2021-07-23-ppl-eval-checkpoints --slurm-partition learnaccel --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t default -m 1.3B_gpt3_setting_checkpoints --nshot 1 -o /checkpoint/$USER/few_shot/2021-07-23-ppl-eval-checkpoints --slurm-partition learnaccel --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t blimp_all -m gpt3_setting --nshot 0 -o /checkpoint/$USER/few_shot/2021-06-14-ppl-eval --slurm-partition learnaccel --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t default -m gpt3_setting --nshot 4 -o /checkpoint/$USER/few_shot/2021-06-14-ppl-eval --slurm-partition learnaccel --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t storycloze -m gpt3_setting --nshot 0 -o /checkpoint/$USER/few_shot/2021-06-14-ppl-eval --slurm-partition learnaccel --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t default -m 1.3B_gpt3_setting_checkpoints --nshot 32 -o /checkpoint/$USER/few_shot/2021-07-23-ppl-eval-checkpoints --slurm-partition learnaccel --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t default -m 1.3B_gpt3_setting_checkpoints --nshot 4 -o /checkpoint/$USER/few_shot/2021-07-23-ppl-eval-checkpoints --slurm-partition learnaccel
            
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t blimp_all -m moe_15B_checkpoints --nshot 0 -o /checkpoint/$USER/few_shot/2021-09-28-ppl-eval --slurm-partition learnaccel --local --dry-run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t stable -m moe_15B_checkpoints --nshot 0 -o /checkpoint/$USER/few_shot/2021-09-28-ppl-eval --slurm-partition learnaccel --local --dry-run
    
        - Run on salloc with moe
        
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t good_correlation -m moe --nshot 0 --local --dry-run
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t good_correlation -m moe_1.1T --nshot 0 --local
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t arceasy -m moe_207B --nshot 0 --local
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t arceasy -m moe_523B --nshot 0 --local
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_ppl_explore.py -t arceasy -m moe_1.1T --nshot 0 --local

    """

    # task settings
    available_tasks_settings = default_tasks_settings.copy()
    available_tasks_settings = {
            "copa": ("copa", ["copa"], {}),
            "cb": ("cb", ["cb"], {"cb_template": "cb"}),
            "rte": ("rte", ["rte"], {}),
            "wic": ("wic", ['wic'], {}),
            "wsc": ("wsc", ['wsc'], {}),
            "storycloze_en": ("storycloze", ['storycloze'], {"storycloze_language": ["en"]}),
            "openbookqa": ("openbookqa", ["openbookqa"], {"calibrator_name": "average_option",
                           "openbookqa_calibration_options": ["question::"], "openbookqa_template": ['arc_old'],},),
            "boolq": ("boolq", ["boolq"], {}),
            "piqa": ("piqa", ['piqa'], {}),
            "winogrande": ("winogrande", ['winogrande'], {}),
            "arceasy": ("arceasy", ["arceasy"], {}),
            "arcchallenge": ("arcchallenge", ["arcchallenge"], {"calibrator_name": "average_option",
                             "arcchallenge_calibration_options": ["question::"], "arcchallenge_template": ['arc_old'],},),
            "hellaswag": ("hellaswag", ['hellaswag'], {}),
            "blimp_all": ("blimp_all", get_tasks_by_group("blimp"), {}),
    }

    task_run_groups = default_task_run_groups.copy()
    task_run_groups = {
        "good_correlation": ["copa", "openbookqa", "winogrande"],
        "lm": ["arceasy", "arcchallenge", "winogrande", "piqa"],
        "default": ["storycloze_en", "blimp_all", "copa", "cb", "rte", "wic", "wsc", "openbookqa", "openbookqa", "boolq", "piqa", "winogrande", "arceasy", "arcchallenge", "hellaswag"],
        "stable": ["storycloze_en", "blimp_all", "openbookqa", "piqa", "winogrande", "arceasy", "arcchallenge", "hellaswag"],
    }
    
    # model settings
    available_model_settings, model_run_groups = get_extended_default_model_settings_and_groups()

    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs for ppl-based experiments")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)
    
    # override defaults
    USER = os.getenv("USER")
    arg_modify_default(parser, "output", f"/checkpoint/{USER}/few_shot/2021-06-30-ppl-eval-moe")
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
