

import argparse
import json

from examples.few_shot.scripts.collect_results import get_wandb_res_log_key
from examples.few_shot.tasks import get_all_tasks, get_tasks_by_group
from examples.few_shot.tasks_organization import get_task_display_groups, get_tasks_to_groups_mapping

def iterate_jsonl_file(input_file, 
                       fields=None
                       ):
    with open(input_file, mode="r") as f_in:
        for line in f_in:
            line = line.strip()
            item = json.loads(line)
            if fields is not None:
                item = {fld: item[fld] for fld in fields}
            
            yield item

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs.")

    parser.add_argument(
        "-f",
        "--results-file",
        default="/large_experiments/xlmg/results/intermediate_eval/1.3B_gpt3_setting_kitchen_sink_23/results.tsv.raw.jsonl",
        help="The path to the results file!",
    )

    parser.add_argument(
        "--wandb-value-fields",
        default=["_metric_val"],
        nargs="+",
        help="List of wandb fields to group with" ,
    )
    args = parser.parse_args()

    # get tasks and wandb results keys keys -> {"diagnosisbrand": ["diagnosisbrand_1_shot", "diagnosisbrand_32_shot"]}
    task_with_wnadb_log_key = [(item["task"], get_wandb_res_log_key(item)) for item in iterate_jsonl_file(input_file=args.results_file, fields=["task", "nb_few_shot_samples"])]
    task_with_wnadb_log_key = list(set(task_with_wnadb_log_key))
    wandb_keys_by_task = {}
    for task, wandb_res_key in task_with_wnadb_log_key:
        if task not in wandb_keys_by_task:
            wandb_keys_by_task[task] = []
        wandb_keys_by_task[task].append(wandb_res_key)

    # get task groups
    tasks_by_group = get_task_display_groups()
    
    # generate sum experessions
    for group, tasks in tasks_by_group.items():
        # expand group with variations of the keys
        # for example diagnosis -> diagnosis_1_shot
        groups_wandb = {}
        for task in tasks:
            if task not in wandb_keys_by_task:
                continue

            # generate subgroups for each key suffix:
            #  diagnosisone_1_shot -> 1_shot -> diagnosis_all_1shot
            wandb_task_keys = sorted(wandb_keys_by_task[task])
            for wandb_key in wandb_task_keys:
                key_suffix = wandb_key.lstrip(task).strip()
                
                # add the new wandb group
                wandb_group = group + key_suffix
                if wandb_group not in groups_wandb:
                    groups_wandb[wandb_group] = []
                groups_wandb[wandb_group].append(wandb_key)

       
        for gr_wandb, gr_wandb_keys in groups_wandb.items():
            for fld in args.wandb_value_fields:
                expression = "+".join([f"${{{x}/{fld}}}" for x in gr_wandb_keys])
                print(f"Chart title: {gr_wandb}/{fld}")
                print(f"Y-axis expression:\n ({expression})/{len(gr_wandb_keys)}")
                print()

        
        print()



    
    

    