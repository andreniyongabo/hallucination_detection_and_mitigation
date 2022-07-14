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
    The current script aims at executing experiments that will be used
    for English-only tasks.
    
    Selected tasks: 
        - SuperGLUE - popular tasks
        - QA tasks:
            - ARC Challenge (w/ calib), OpenbookQA (w/ calib) which show improvement with calibration
            - ARC Easy which is a similar but easier task
        - LM tasks
            - Winogrande, piqa, hellaswag that has shown by gpt3 to offer emerging few-shot, also PPL matching the gpt3 models. 
        - BLIMP tasks
            - All tasks from the BLIMP benchmark
    
    Setting:
        - Scoring - `mean`, since it has good justifications, and it was used by gpt3. 
        - nshot - 0, 1, 32. The initial experiments with 0-shot has shown strong correlation and want to see if it works for > 0 shot. 
        - Trials - 10 trials are used to reduce variance.

    Commands:
        - Run 
            # run debug locallys
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t copa -m moe --nshot 0 --local --dry-run
            
            # test that all multi-choice run
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t test_multichoice test_lama test_mlama diagnosis -m 125M_gpt3_setting --nshot 0  -o /checkpoint/$USER/few_shot/debug_multichoice --override-completed --local --dry-run            
            
            # 1.3B_gpt3_setting_checkpoints
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t default -m 1.3B_gpt3_setting_checkpoints --nshot 0 -o /checkpoint/$USER/few_shot/english-eval-checkpoints --slurm-partition learnaccel --local --dry-run

            # 1.3B gpt3 setting
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t default -m 1.3B_gpt3_setting --nshot 0 -o /checkpoint/$USER/few_shot/english-eval --slurm-partition learnaccel --local --dry-run
    
        - Run on salloc with moe
        
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t default -m moe -o /checkpoint/$USER/few_shot/english-eval --nshot 0 --local --dry-run
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t good_correlation -m moe_1.1T -o /checkpoint/$USER/few_shot/english-eval  --nshot 0 --local
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t arceasy -m moe_207B -o /checkpoint/$USER/few_shot/english-eval --nshot 0 --local
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t arceasy -m moe_523B -o /checkpoint/$USER/few_shot/english-eval --nshot 0 --local
            srun -e moe_debug.err -o moe_debug.out python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_english.py -t arceasy -m moe_1.1T -o /checkpoint/$USER/few_shot/english-eval  --nshot 0 --local

    """

    # task settings
    multi_choice_tasks = ["blimp_all", "hellaswag", "storycloze", "xcopa", "winograd", "winogrande", "commonsenseqa", "piqa", "arcchallenge", "arceasy", "openbookqa", "exams", "boolq", "cb", "copa", "rte", "wic", "wsc", "multirc", "record", "mnlimatched", "mnlimismatched", "xnli", "stereoset", "crowspairs"]
    
    available_tasks_settings = default_tasks_settings.copy()
    available_tasks_settings = {
            "copa": ("copa", ["copa"], {}),
            "cb": ("cb", ["cb"], {"cb_template": "cb"}),
            "rte": ("rte", ["rte"], {}),
            "wic": ("wic", ['wic'], {}),
            "wsc": ("wsc", ['wsc'], {}),
            "openbookqa": ("openbookqa", ["openbookqa"], {
                "calibrator_name": "average_option", 
                "openbookqa_calibration_options": ["question::"], 
                "openbookqa_template": ['arc_old'],
                }),
            "boolq": ("boolq", ["boolq"], {}),
            "piqa": ("piqa", ['piqa'], {}),
            "winogrande": ("winogrande", ['winogrande'], {}),
            "arceasy": ("arceasy", ["arceasy"], {}),
            "arcchallenge": ("arcchallenge", ["arcchallenge"], {"calibrator_name": "average_option",
                "arcchallenge_calibration_options": ["question::"], "arcchallenge_template": ['arc_old'],},),
            "hellaswag": ("hellaswag", ['hellaswag'], {}),
            "blimp_all": ("blimp_all", get_tasks_by_group("blimp"), {}),
            "test_multichoice": ("test_tasks", multi_choice_tasks, {"n_eval_samples": 5}),
            "test_lama": ("test_tasks", "lama", {"n_eval_samples": 5, "max_cands": 2}),
            "test_mlama": ("test_tasks", "mlama", {"n_eval_samples": 5, "max_cands": 2}),
            "diagnosis": ("test_tasks", "diagnosis", {"n_eval_samples": 5}),
    }

    task_run_groups = default_task_run_groups.copy()
    task_run_groups = {
        "test_tasks": ["test_multichoice", "test_lama", "test_mlama", "diagnosis"],
        "good_correlation": ["copa", "openbookqa", "winogrande"],
        "lm": ["arceasy", "arcchallenge", "winogrande", "piqa"],
        "default": ["blimp_all", "copa", "cb", "rte", "wic", "wsc", "openbookqa", "boolq", "piqa", "winogrande", "arceasy", "arcchallenge", "hellaswag"],
    }
    
    # model settings
    available_model_settings = default_model_settings.copy()
    
    model_run_groups = default_model_run_groups.copy()

    # Copy base model configs for all it's checkpoints
    model_1_3B_gpt3_setting_checkpoints = {mcpt: copy_model_setting(available_model_settings["1.3B_gpt3_setting"], mcpt) 
                                                      for mcpt in get_model_checkpoint_names("1.3B_gpt3_setting")}
    available_model_settings.update(model_1_3B_gpt3_setting_checkpoints)       
    model_run_groups["1.3B_gpt3_setting_checkpoints"] = model_1_3B_gpt3_setting_checkpoints.keys()              

   
    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs for english benchmarks")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)
    
    # override defaults
    USER = os.getenv("USER")
    arg_modify_default(parser, "output", f"/checkpoint/{USER}/few_shot/english-eval")
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
