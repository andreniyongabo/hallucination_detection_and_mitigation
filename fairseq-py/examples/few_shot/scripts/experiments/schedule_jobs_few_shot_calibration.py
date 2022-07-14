# fmt: off
import argparse 
import sys
from examples.few_shot.scripts.experiments.schedule_jobs_few_shot import *

if __name__ == "__main__":
    """
    The current script aims at collecting the different calibration settings.
    
    Selected tasks: 
        - QA tasks:
            - OpenBookQA
            - ARC Challenge (w/ calib), OpenbookQA (w/ calib) which show improvement with calibration
    
    Models: 
        - in the default configuration we will experiment with gpt3_setting models.
    Setting:
        - Scoring - `mean`
        - nshot - 0 - by default we will run experiments wiht 0 shot
        - Trials - 10 trials are used to reduce variance.

    Commands:
        - Run locally or slurm (remove the --local for slurm)
            
            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_calibration.py -t qa_multiple -m 1.3B_gpt3_setting --nshot 0 --local --dry-run

            python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_calibration.py -t qa_gpt3_mean -m 1.3B_gpt3_setting --nshot 0 --local --dry-run
    """

    # task settings
    #available_tasks_settings = default_tasks_settings.copy()
    available_tasks_settings = {
            # Default calibrations with empty fields
            "openbookqa": ("openbookqa", ["openbookqa"], {"calibrator_name": "average_option",
                "openbookqa_calibration_options": ["question::"], "openbookqa_template": ['arc_old'],},),
            "arcchallenge": ("arcchallenge", ["arcchallenge"], {"calibrator_name": "average_option",
                "arcchallenge_calibration_options": ["question::"], "arcchallenge_template": ['arc_old'],},),
            "arceasy_calib": ("arceasy_calib", ["arceasy"], {"calibrator_name": "average_option",
                    "arceasy_calibration_options": ["question::"], "arceasy_template": ['arc_old'],},),
            "copa_calib": ("copa_calib", ["copa"], {
                "calibrator_name": "average_option",
                "copa_calibration_options": ["premise::"],}),
            "boolq_calib": ("boolq_calib", ["boolq"], {
                "calibrator_name": "average_option",
                "boolq_calibration_options": ["question::"],}),
            "cb_calib": ("cb_calib", ["cb"], {
                "calibrator_name": "average_option",
                "cb_calibration_options": ["premise::"],}),
            "rte_calib": ("rte_calib", ["rte"], {
                "calibrator_name": "average_option",
                "rte_calibration_options": ["premise::"],}),
            "multirc_calib": ("multirc_calib", ['multirc'], {"calibrator_name": "average_option",
                                "multirc_calibration_options": ["question::", "paragraph::", "paragraph::|question::"]}),
            "exams_calib": ("exams_calib", ['exams'], {"calibrator_name": "average_option",
                                "exams_calibration_options": ["question::"]}),
            "xcopa_calib": ("xcopa_calib",['xcopa'], {"calibrator_name": "average_option",
                                 "xcopa_calibration_options": ["premise::"],}),   
            
            # Calibration of qa tasks with multiple templates
            "openbookqa_multiple_templates": ("openbookqa", ["openbookqa"], {"calibrator_name": "average_option",
                "openbookqa_calibration_options": ["question::"], "openbookqa_template": ['arc_old', 'arc', 'arc_no_choice_lowercase', 'arc_struct_1', 'arc_struct_2', 'arc_capitalize_choice', 'arc_calib_format_1', 'arc_calib_format_2', 'arc_calib_format_3', 'arc_descr_1', 'arc_descr_2', 'arc_descr_3', 'arc_choice_format_1', 'arc_choice_format_2', 'arc_choice_format_3',],},),
            "arceasy_multiple_templates": ("arceasy", ["openbookqa"], {"calibrator_name": "average_option",
                "arceasy_calibration_options": ["question::"], "arceasy_template": ['arc_old', 'arc', 'arc_no_choice_lowercase', 'arc_struct_1', 'arc_struct_2', 'arc_capitalize_choice', 'arc_calib_format_1', 'arc_calib_format_2', 'arc_calib_format_3', 'arc_descr_1', 'arc_descr_2', 'arc_descr_3', 'arc_choice_format_1', 'arc_choice_format_2', 'arc_choice_format_3',],},),
            
            # QA GPT3 templates, sum and sum scoring scoring
            # mean scoring
            "openbookqa_gpt3_mean": ("openbookqa_mean", ["openbookqa"], {"calibrator_name": "average_option",
                "openbookqa_calibration_options": ["question::"], 
                "openbookqa_template": ['arcchallenge_gpt3', 'openbookqa_gpt3_calib_v1', 'openbookqa_gpt3_calib_v2', 'openbookqa_gpt3_calib_empty'],
                "scoring": "mean"},),
            
            "arcchallenge_gpt3_mean": ("arcchallenge_mean", ["arcchallenge"], {"calibrator_name": "average_option",
                "arcchallenge_calibration_options": ["question::"], 
                "arcchallenge_template": ['arcchallenge_gpt3'],
                "scoring": "mean"},),
            
            
    }

    task_run_groups = default_task_run_groups.copy()
    task_run_groups = {
        "qa_multiple": ["openbookqa_multiple_templates", "arceasy_multiple_templates"],
        "qa_gpt3_mean": ["openbookqa_gpt3_mean", "arcchallenge_gpt3_mean"],
    }
    
    # model settings
    available_model_settings = default_model_settings.copy()
    model_run_groups = default_model_run_groups.copy()
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule calibration experiments.")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)
    
    # override defaults
    USER = os.getenv("USER")
    arg_modify_default(parser, "output", f"/checkpoint/{USER}/few_shot/2021-07-calibration")
    #arg_modify_default(parser, "scoring", "sum")
    arg_modify_default(parser, "tasks", ["openbookqa_multiple_templates"])
    arg_modify_default(parser, "nb_few_shot_samples_values", [0, 32])
    arg_modify_default(parser, "num_trials", 10)

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
