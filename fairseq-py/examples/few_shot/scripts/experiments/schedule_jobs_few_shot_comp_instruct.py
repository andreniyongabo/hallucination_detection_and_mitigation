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
            # Run for all dense models      
            PYTHONPATH=. python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_comp_instruct.py -o /checkpoint/$USER/compositional_benchmark_results/cic_v1_1/ --slurm-partition xlmg -m 13B_gpt3_setting 6.7B_gpt3_setting 2.7B_gpt3_setting 1.3B_gpt3_setting 125M_gpt3_setting -t cic_v1_1 --slurm-array-parallelism 4
            
            # random
            PYTHONPATH=. python examples/few_shot/scripts/experiments/schedule_jobs_few_shot_comp_instruct.py -o /checkpoint/$USER/compositional_benchmark_results/cic_v1_1/ --slurm-partition xlmg -m random -t cic_v1_1 --slurm-array-parallelism 4 --predictor-name random

    """

    # task settings    
    available_tasks_settings = default_tasks_settings.copy()
    task_run_groups = {}
    
    # Adding various number of steps for synthetic compositional tasks
    available_tasks_settings.update(
        {
            "cic_random_labels_steps2": [
                "cic_random_labels_steps2", 
                ["cic_random_labels"], 
                {"cic_random_labels_eval_set": ["wiki40b-val-fact-composition-2"]}
            ],
            "cic_random_labels_steps3": [
                "cic_random_labels_steps3", 
                ["cic_random_labels"], 
                {"cic_random_labels_eval_set": ["wiki40b-val-fact-composition-3"]}
            ],
            "cic_random_labels_steps2_negatedtemplate": [
                "cic_random_labels_steps2_negatedtemplate", 
                ["cic_random_labels"], 
                {"cic_random_labels_template": "compositional_instructions_classification_custom_v2_negated", "cic_random_labels_eval_set": ["wiki40b-val-fact-composition-2"]}
            ],
            "cic_random_labels_steps3_negatedtemplate": [
                "cic_random_labels_steps3_negatedtemplate", 
                ["cic_random_labels"], 
                {"cic_random_labels_template": "compositional_instructions_classification_custom_v2_negated", "cic_random_labels_eval_set": ["wiki40b-val-fact-composition-3"]}
            ],
            "cic_random_labels_steps4": [
                "cic_random_labels_steps4", 
                ["cic_random_labels"], 
                {"cic_random_labels_eval_set": ["wiki40b-val-fact-composition-4"]}
            ],
            "cic_random_labels_steps5": [
                "cic_random_labels_steps5", 
                ["cic_random_labels"], 
                {"cic_random_labels_eval_set": ["wiki40b-val-fact-composition-5"]}
            ],
            "cic_random_labels_steps4_negatedtemplate": [
                "cic_random_labels_steps4_negatedtemplate", 
                ["cic_random_labels"], 
                {"cic_random_labels_template": "compositional_instructions_classification_custom_v2_negated", "cic_random_labels_eval_set": ["wiki40b-val-fact-composition-4"]}
            ],
            "cic_random_labels_steps5_negatedtemplate": [
                "cic_random_labels_steps5_negatedtemplate", 
                ["cic_random_labels"], 
                {"cic_random_labels_template": "compositional_instructions_classification_custom_v2_negated", "cic_random_labels_eval_set": ["wiki40b-val-fact-composition-5"]}
            ],

        }
    )

    # default template
    available_tasks_settings.update(
    {
        "cic_v1_1_comp_subtasks_steps2": [
                "cic_v1_1_comp_subtasks_steps2", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-2"]}
            ],
        "cic_v1_1_comp_subtasks_steps3": [
                "cic_v1_1_comp_subtasks_steps3", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-3"]}
            ],
        "cic_v1_1_comp_subtasks_steps4": [
                "cic_v1_1_comp_subtasks_steps4", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-4"]}
            ],
        "cic_v1_1_comp_subtasks_steps5": [
                "cic_v1_1_comp_subtasks_steps5", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-5"]}
            ],
    })
    
    # Template v4
    curr_template = "cic_v4_simple"
    available_tasks_settings.update(
    {
        "cic_v1_1_comp_subtasks_steps2_tmpl_v4": [
                "cic_v1_1_comp_subtasks_steps2", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-2"],
                 "cic_v1_1_comp_subtasks_template": [curr_template]}
            ],
        "cic_v1_1_comp_subtasks_steps3_tmpl_v4": [
                "cic_v1_1_comp_subtasks_steps3", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-3"],
                 "cic_v1_1_comp_subtasks_template": [curr_template]}
            ],
        "cic_v1_1_comp_subtasks_steps4_tmpl_v4": [
                "cic_v1_1_comp_subtasks_steps4", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-4"],
                 "cic_v1_1_comp_subtasks_template": [curr_template]}
            ],
        "cic_v1_1_comp_subtasks_steps5_tmpl_v4": [
                "cic_v1_1_comp_subtasks_steps5", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-5"],
                 "cic_v1_1_comp_subtasks_template": [curr_template]}
            ],
        
        # flan template
        "cic_v1_1_flantmp_comp_subtasks_steps2_tmplv4": [
                "cic_v1_1_comp_subtasks_steps2", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-2"], "cic_v1_1_comp_subtasks_template": "cic_v4_simple_flanformat"}
            ],
        "cic_v1_1_flantmp_comp_subtasks_steps3_tmplv4": [
                "cic_v1_1_comp_subtasks_steps3", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-3"], "cic_v1_1_comp_subtasks_template": "cic_v4_simple_flanformat"}
            ],
        "cic_v1_1_flantmp_comp_subtasks_steps4_tmplv4": [
                "cic_v1_1_comp_subtasks_steps4", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-4"], "cic_v1_1_comp_subtasks_template": "cic_v4_simple_flanformat"}
            ],
        "cic_v1_1_flantmp_comp_subtasks_steps5_tmplv4": [
                "cic_v1_1_comp_subtasks_steps5", 
                ["cic_v1_1_comp_subtasks"], 
                {"cic_v1_1_comp_subtasks_eval_set": ["wiki40b-val-fact-500-docs-comp-5"], "cic_v1_1_comp_subtasks_template": "cic_v4_simple_flanformat"}
            ],
    
    })
    
    curr_template = "cic_v4_simple_negated"
    curr_scheduling_setting_names = []
    for compi in range(2, 6):
        curr_scheduling_setting_name = f"cic_v1_1_comp_subtasks_steps{compi}_tmpl_v4_neg"
        available_tasks_settings.update(
        {
            curr_scheduling_setting_name: [
                    f"cic_v1_1_comp_subtasks_steps{compi}", 
                    ["cic_v1_1_comp_subtasks"], 
                    {"cic_v1_1_comp_subtasks_eval_set": [f"wiki40b-val-fact-500-docs-comp-{compi}"],
                    "cic_v1_1_comp_subtasks_template": [curr_template]}
                ],
        })
        curr_scheduling_setting_names.append(curr_scheduling_setting_name)
    
    # cic_v4_simple_flanformat_negated
    curr_template = "cic_v4_simple_flanformat_negated"
    curr_scheduling_setting_names = []
    for compi in range(2, 6):
        curr_scheduling_setting_name = f"cic_v1_1_flan_comp_subtasks_steps{compi}_tmpl_v4_neg"
        available_tasks_settings.update(
        {
            curr_scheduling_setting_name: [
                    f"cic_v1_1_comp_subtasks_steps{compi}", 
                    ["cic_v1_1_comp_subtasks"], 
                    {"cic_v1_1_comp_subtasks_eval_set": [f"wiki40b-val-fact-500-docs-comp-{compi}"],
                    "cic_v1_1_comp_subtasks_template": [curr_template]}
                ],
        })
        curr_scheduling_setting_names.append(curr_scheduling_setting_name)
    task_run_groups["cic_v1_1_tmpl_v4_flantmp_neg"] = curr_scheduling_setting_names
    
    task_run_groups.update({
        "cic_v1": [
            "cic_random_labels_steps2", "cic_random_labels_steps2_negatedtemplate", 
            "cic_random_labels_steps3", "cic_random_labels_steps3_negatedtemplate", 
            "cic_random_labels_steps4", "cic_random_labels_steps4_negatedtemplate", 
            "cic_random_labels_steps5", "cic_random_labels_steps5_negatedtemplate"
        ],
        "cic_v1_1": [
            "cic_v1_1_comp_subtasks_steps2",
            "cic_v1_1_comp_subtasks_steps3",
            "cic_v1_1_comp_subtasks_steps4",
            "cic_v1_1_comp_subtasks_steps5",
        ],
        "cic_v1_1_tmpl_v4": [
            "cic_v1_1_comp_subtasks_steps2_tmpl_v4",
            "cic_v1_1_comp_subtasks_steps3_tmpl_v4",
            "cic_v1_1_comp_subtasks_steps4_tmpl_v4",
            "cic_v1_1_comp_subtasks_steps5_tmpl_v4",
        ],
        "cic_v1_1_tmpl_v4_negated": [
            "cic_v1_1_comp_subtasks_steps2_tmpl_v4_neg",
            "cic_v1_1_comp_subtasks_steps3_tmpl_v4_neg",
            "cic_v1_1_comp_subtasks_steps4_tmpl_v4_neg",
            "cic_v1_1_comp_subtasks_steps5_tmpl_v4_neg",
        ],
        "cic_v1_1_tmpl_v4_flantmp": [
            "cic_v1_1_flantmp_comp_subtasks_steps2_tmplv4",
            "cic_v1_1_flantmp_comp_subtasks_steps3_tmplv4",
            "cic_v1_1_flantmp_comp_subtasks_steps4_tmplv4",
            "cic_v1_1_flantmp_comp_subtasks_steps5_tmplv4",
        ]
    })
    
    # model settings
    available_model_settings, model_run_groups = get_extended_default_model_settings_and_groups()        
   
    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs for english benchmarks")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)
    
    # override defaults
    USER = os.getenv("USER")
    arg_modify_default(parser, "output", f"/checkpoint/$USER/compositional_benchmark_results/")
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
    total_jobs, jobs_status = schedule_experiment_jobs(args,
                             task_run_groups, available_tasks_settings,
                             model_run_groups, available_model_settings,
                             custom_base_run_args = {},
                             custom_override_run_args = {}
                            )

    update_jobs_status(jobs_status)

    jobs_current_status_cnt = dict(Counter([x.get("current_status", None) for x in jobs_status]))
    logging.info(f"Submission status: {jobs_current_status_cnt}")

    # Log job status
    out_dir = args.output
    job_log_path = os.path.join(out_dir, "scheduled_jobs_log.jsonl")
    append_submitted_jobs_info_to_file(job_log_path, jobs_status)
    logging.info(f"Submitted jobs info logged to {job_log_path}")

    print_display_results_command(args)
    sys.exit(0) # sometimes srun hangs and calling sys.exit(0) exits properly
