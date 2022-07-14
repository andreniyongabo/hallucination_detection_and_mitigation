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
    This script submit evaluation jobs that reproduce the zero and few-shot evaluation on the English tasks reported in the XLM-G paper draft.
    TODO:
        Add more tasks as more are included in the paper.
    
    Selected tasks: 
        - QA tasks:
            - ARC Challenge (w/ calib), OpenbookQA (w/ calib) which show improvement with calibration
            - ARC Easy which is a similar but easier task
        - LM tasks
            - Winogrande, PIQA, Hellaswag, StoryCloze that have shown by gpt3 to offer emerging few-shot, also PPL matching the gpt3 models. 
        - BLIMP tasks
            - All tasks from the BLIMP benchmark
    
    Setting:
        - Scoring - `mean`, since it has good justifications, and it was used by gpt3. 
        - nshot - 0, 1, 32. The initial experiments with 0-shot has shown strong correlation and want to see if it works for > 0 shot. 
        - Trials - 5 trials are used to reduce variance.

    Commands:
        - Run             
            # moe_1.1T 
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_xlmg_en_gpt3_eval.py -t paper_draft -m moe_1.1T --nshot 0 -o /checkpoint/$USER/few_shot/xlmg_en_gpt3_eval --slurm-partition xlmg

            # 6.7B_gpt3_setting_checkpoints
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_xlmg_en_gpt3_eval.py -t paper_draft -m 6.7B_gpt3_setting_checkpoints --nshot 0 -o /checkpoint/$USER/few_shot/xlmg_en_gpt3_eval --slurm-partition devaccel,learnaccel --local --dry-run

            # 6.7B gpt3 setting
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_xlmg_en_gpt3_eval.py -t paper_draft -m 6.7B_gpt3_setting --nshot 0 -o /checkpoint/$USER/few_shot/xlmg_en_gpt3_eval --slurm-partition learnaccel --local --dry-run
    
        - Run on salloc with moe
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_xlmg_en_gpt3_eval.py -t paper_draft -m moe -o /checkpoint/$USER/few_shot/xlmg_en_gpt3_eval --nshot 0 --local --dry-run
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_xlmg_en_gpt3_eval.py -t arceasy -m moe_207B -o /checkpoint/$USER/few_shot/xlmg_en_gpt3_eval --nshot 0 --local
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_xlmg_en_gpt3_eval.py -t arceasy -m moe_523B -o /checkpoint/$USER/few_shot/xlmg_en_gpt3_eval --nshot 0 --local
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_xlmg_en_gpt3_eval.py -t arceasy -m moe_1.1T -o /checkpoint/$USER/few_shot/xlmg_en_gpt3_eval  --nshot 0 --local

    """

    # task settings    
    available_tasks_settings = default_tasks_settings.copy()
    available_tasks_settings = {
            "arceasy": ("arceasy", ["arceasy"], {}),
            "arcchallenge": (
                "arcchallenge", 
                ["arcchallenge"], 
                {
                    "calibrator_name": "average_option",
                    "arcchallenge_calibration_options": ["question::"], 
                    "arcchallenge_template": ['arc_old'],
                }
            ),
            "copa": ("copa", ["copa"], {}),
            "record": ("record", ["record"], {}),
            "openbookqa__unconditional": (
                "openbookqa__unconditional", 
                ["openbookqa"], 
                {
                    "scoring": "unconditional-norm",
                    "openbookqa_eval_set": ["test"]
                }
            ),
            "piqa": ("piqa", ['piqa'], {}),
            "winogrande__suffix": (
                "winogrande__suffix", 
                ['winogrande'], 
                {
                    "scoring":"suffix"
                }
            ),
            "hellaswag": ("hellaswag", ['hellaswag'], {}),
            "storycloze__en": (
                "storycloze", 
                ['storycloze'], 
                {
                    "storycloze_languages": ["en"]
                }
            ),
            "blimp_all": ("blimp_all", get_tasks_by_group("blimp"), {}),       
    }

    task_run_groups = default_task_run_groups.copy()
    task_run_groups = {
        "paper_draft": [
            "storycloze__en", 
            "openbookqa__unconditional", 
            "piqa", 
            "winogrande__suffix", 
            "arceasy", 
            "arcchallenge", 
            "hellaswag"
        ],
    }
    
    # model settings
    available_model_settings, model_run_groups = get_extended_default_model_settings_and_groups()        
   
    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs for english benchmarks")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)
    
    # override defaults
    USER = os.getenv("USER")
    arg_modify_default(parser, "output", f"/checkpoint/{USER}/few_shot/xlmg_en_gpt3_eval")
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
