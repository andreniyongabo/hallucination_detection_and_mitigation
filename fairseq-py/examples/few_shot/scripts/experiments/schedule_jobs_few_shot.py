# fmt: off
import argparse
import copy
import datetime
import glob
import json
import logging
import os
import random
import re
import subprocess
import sys
from abc import ABC
from collections import Counter
from typing import Any, Dict, List, NamedTuple

import submitit
from fb_sweep.sweep.slurm import get_random_port

from examples.few_shot.gpt3_eval import run_evaluations, run_evaluations_from_model_name
from examples.few_shot.model_configs import get_model_checkpoint_names, get_model_names, gptz_sharded_config
from examples.few_shot.scripts.checkpoint_helpers import (
    get_checkpoint_id_from_filename,
    get_checkpoint_ids_from_text,
)
from examples.few_shot.tasks import get_all_tasks

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


UNAVAILABLE_CHECKPOINT_PATH_DEFAULT_VALUE = "checkpoint_file_is_not_available"
JOB_STATUS_CURRENT_SUBMIT = "submit"
JOB_STATUS_CURRENT_SKIP = "skip"
JOB_STATUS_CURRENT_EXECUTE_LOCALLY = "execute_locally"

def copy_model_setting(model_setting, new_model_name):
    new_setting = copy.deepcopy(list(model_setting))
    new_setting[0] = new_model_name

    return new_setting


RESULT_FILE_STATUS_COMPLETED = "completed"
RESULT_FILE_STATUS_SUBMITTED = "submitted"
RESULT_FILE_STATUS_MISSING = "missing"

def get_result_file_status(res_file_name):
    if os.path.exists(res_file_name):
        return RESULT_FILE_STATUS_COMPLETED
    elif os.path.exists(res_file_name+ ".submitted"):
        return RESULT_FILE_STATUS_SUBMITTED
    else:
        return RESULT_FILE_STATUS_MISSING


class ModelSchedulingSetting(NamedTuple):
    model_name:str
    scheduling_params: Dict[str, Any] = {}  # These params specify how to aggregate the tasks.
                                            # "combine_tasks":True - all task settings will be squashed in 1 job.
                                            #     - Useful for large slurm allocations such as MoE which requires 8 nodes and can take time
    eval_params: Dict[str, Any] = {}  # These are the parameters that will be passed to the eval code (similar to gpt3_eval.py)
    slurm_nodes: int = 1  # Slurm: Number of nodes. Defaults to 1.
    slurm_gpus_per_node: int = 1  # Slurm: Gpus per node. Defaults to 1.
    slurm_ntasks_per_node: int = 1  # Slurm: Number of tasks per node. Defaults to 1.
    slurm_cpus_per_task: int = 8  # Slurm: Cpus per task. Defaults to 8.
    slurm_array_parallelism: int = 3  # Slurm: The maximum number of concurrent tasks created in the pool. Defaults to 3.


class ModelCheckpointInfoWithSchedulingSetting(NamedTuple):
    id:int
    checkpoint_path:str
    scheduling_setting_key: str
    scheduling_setting: ModelSchedulingSetting  # Slurm: The maximum number of concurrent tasks created in the pool. Defaults to 3.

default_model_settings = {
        # model_name, job_params, custom params, nodes, gpus_per_node, ntasks_per_node, cpus_per_task, max parallel jobs in this pool

        # dummy models
        "majority": ModelSchedulingSetting("majority", {}, {"predictor_name": "majority", "train_sep": "\n", "replace_newline_with_eos": True }, 1, 1, 1, 8, 3),
        "random": ModelSchedulingSetting("random", {}, {"predictor_name": "random", "train_sep":"\n", "replace_newline_with_eos": True }, 1, 1, 1, 8, 3),

        # dense
        "1.3B_gpt3_setting": ModelSchedulingSetting(**{
            "model_name": "1.3B_gpt3_setting",
            "scheduling_params": {},
            "eval_params": {
                "train_sep": "\n",
                "replace_newline_with_eos": True
            },
            "slurm_nodes": 1,
            "slurm_gpus_per_node": 2,
            "slurm_ntasks_per_node": 1,
            "slurm_cpus_per_task": 8,
            "slurm_array_parallelism": 3
        }),
        "2.7B_gpt3_setting": ModelSchedulingSetting("2.7B_gpt3_setting", {}, {"train_sep":"\n", "replace_newline_with_eos": True }, 1, 1, 1, 8, 3),
        "flan_minus_nli_para_2.7B": ModelSchedulingSetting("flan_minus_nli_para_2.7B", {}, {"train_sep":"\n", "replace_newline_with_eos": True }, 1, 1, 1, 8, 3),
        "flan_minus_nli_para_6.7B": ModelSchedulingSetting("flan_minus_nli_para_6.7B", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048}, 1, 2, 1, 8, 3),
        "flan_minus_nli_para_13B": ModelSchedulingSetting("flan_minus_nli_para_13B", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048}, 1, 2, 1, 8, 3),
        "flan_minus_qa_2.7B": ModelSchedulingSetting("flan_minus_qa_2.7B", {}, {"train_sep":"\n", "replace_newline_with_eos": True }, 1, 1, 1, 8, 3),
        "flan_minus_qa_6.7B": ModelSchedulingSetting("flan_minus_qa_6.7B", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048}, 1, 2, 1, 8, 3),
        "flan_minus_qa_13B": ModelSchedulingSetting("flan_minus_qa_13B", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048}, 1, 2, 1, 8, 3),
        "flan_minus_qa_13B_v2": ModelSchedulingSetting("flan_minus_qa_13B_v2", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048}, 1, 2, 1, 8, 3),
        "flan_minus_sentiment_2.7B": ModelSchedulingSetting("flan_minus_sentiment_2.7B", {}, {"train_sep":"\n", "replace_newline_with_eos": True}, 1, 1, 1, 8, 3),
        "flan_minus_sentiment_6.7B": ModelSchedulingSetting("flan_minus_sentiment_6.7B", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048}, 1, 2, 1, 8, 3),
        "flan_minus_sentiment_13B": ModelSchedulingSetting("flan_minus_sentiment_13B", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048, "max_positions": 1024}, 1, 2, 1, 8, 3),
        "6.7B_gpt3_setting_1024ctx": ModelSchedulingSetting("6.7B_gpt3_setting_1024ctx", {}, {"train_sep":"\n", "replace_newline_with_eos": True}, 1, 2, 1, 8, 3),
        "6.7B_gpt3_setting": ModelSchedulingSetting("6.7B_gpt3_setting", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048}, 1, 2, 1, 8, 3),
        "125M_gpt3_setting": ModelSchedulingSetting("125M_gpt3_setting", {}, {"train_sep":"\n", "replace_newline_with_eos": True }, 1, 1, 1, 8, 3),
        "355M_gpt3_setting": ModelSchedulingSetting("355M_gpt3_setting", {}, {"train_sep":"\n", "replace_newline_with_eos": True }, 1, 1, 1, 8, 3),
        "13B_gpt3_setting": ModelSchedulingSetting("13B_gpt3_setting", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 1024, "distributed_port": 15188}, 1, 4, 1, 8, 3),

        # Models fine-tuned with flan.
        "flan_minus_nli_para_13B": ModelSchedulingSetting("flan_minus_nli_para_13B", {}, {"user_dir": "examples/few_shot/finetune", "train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 1024, "distributed_port": 15188}, 1, 4, 1, 8, 3),
        "flan_minus_sentiment_13B": ModelSchedulingSetting("flan_minus_sentiment_13B", {}, {"user_dir": "examples/few_shot/finetune", "train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 1024, "distributed_port": 15188}, 1, 4, 1, 8, 3),
        "flan_minus_commonsense_13B": ModelSchedulingSetting("flan_minus_commonsense_13B", {}, {"user_dir": "examples/few_shot/finetune", "train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 1024, "distributed_port": 15188}, 1, 4, 1, 8, 3),
        
        # moe
        "moe_1.1T_2048ctx": ModelSchedulingSetting("moe_1.1T_2048ctx", {"combine_tasks": False}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875, "max_tokens": 2048, "distributed_port": 15187}, 32, 8, 1, 8, 2),
        "moe_1.1T": ModelSchedulingSetting("moe_1.1T", {"combine_tasks": False}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875, "max_tokens": 1024, "distributed_port": 15187}, 32, 8, 1, 8, 2),
        "moe_523B": ModelSchedulingSetting("moe_523B", {"combine_tasks": False}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875, "distributed_port": 15187}, 16, 8, 1, 8, 1),
        "moe_207B": ModelSchedulingSetting("moe_207B", {"combine_tasks": False}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875, "distributed_port": 15187}, 16, 8, 1, 8, 1),
        "moe_52B": ModelSchedulingSetting("moe_52B", {"combine_tasks": False}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.046875, "distributed_port": 15187}, 2, 8, 1, 8, 1),
        "moe_15B": ModelSchedulingSetting("moe_15B", {"combine_tasks": False}, {"train_sep": "\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.25, "distributed_port": 15187}, 1, 8, 1, 8, 1),

        #175B
        "175B_gpt3_setting__last": ModelSchedulingSetting("175B_gpt3_setting__last", {"combine_tasks": False}, { "fsdp": True, "max_tokens": 512, "model_configs": {"175B_gpt3_setting__last": gptz_sharded_config(  # adding this for backward compatibility with existing experiments and results
        "/large_experiments/xlmg/models/sshleifer/175B/reshard.pt"
    ),}}, 4, 8, 1, 64, 8),
        "175B_gpt3_setting__step00135000": ModelSchedulingSetting("175B_gpt3_setting__step00135000", {"combine_tasks": False}, { "fsdp": True, "max_tokens": 512}, 4, 8, 1, 64, 8),
        
        # multilingual - dense
        "dense_564M_lang30_new_cc100_xl_unigram__step119209": ModelSchedulingSetting("dense_564M_lang30_new_cc100_xl_unigram__step119209", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048 }, 1, 1, 1, 8, 3),
        "dense_1.7B_lang30_new_cc100_xl_unigram__step58000": ModelSchedulingSetting("dense_1.7B_lang30_new_cc100_xl_unigram__step58000", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048 }, 1, 1, 1, 8, 3),
        "dense_2.9B_lang30_new_cc100_xl_unigram__step59604": ModelSchedulingSetting("dense_2.9B_lang30_new_cc100_xl_unigram__step59604", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048 }, 1, 1, 1, 8, 3),
        "dense_7.5B_lang30_new_cc100_xl_unigram__step00065000": ModelSchedulingSetting("dense_7.5B_lang30_new_cc100_xl_unigram__step00065000", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "max_tokens": 2048 }, 1, 1, 1, 8, 8),

        # multilingual - moe
        "moe_200B_lang30_new_cc100_xl_unigram__step00048000": ModelSchedulingSetting("moe_200B_lang30_new_cc100_xl_unigram__step00048000", {}, {"train_sep":"\n", "replace_newline_with_eos": True, "moe_eval_capacity_token_fraction": 0.015625, "max_tokens": 2048, "distributed_port": 15187}, 4, 8, 1, 8, 3),

        # gpt3 models
        "openai_curie": ModelSchedulingSetting("openai_curie", {},{"predictor_name": "CLMPromptingOpenaiApi".lower()}, 1, 1, 1, 8, 3),
        "openai_curie-instruct-beta-v2": ModelSchedulingSetting("openai_curie-instruct-beta-v2", {},{"predictor_name": "CLMPromptingOpenaiApi".lower()}, 1, 1, 1, 8, 3),
        "openai_davinci-instruct-beta-v3": ModelSchedulingSetting("openai_davinci-instruct-beta-v3", {"predictor_name": "CLMPromptingOpenaiApi".lower()},{}, 1, 1, 1, 8, 3),
        
        # T0 models
        "huggingface_bigscience=T0pp": ModelSchedulingSetting("huggingface_bigscience=T0pp", {},{"predictor_name": "CLMPromptingHuggingFace".lower(), "train_sep":"\n", "replace_newline_with_eos": True,  "max_tokens": 1024}, 1, 2, 1, 8, 3),
        "huggingface_bigscience=T0_3B": ModelSchedulingSetting("huggingface_bigscience=T0_3B", {},{"predictor_name": "CLMPromptingHuggingFace".lower(),  "train_sep":"\n", "replace_newline_with_eos": True,  "max_tokens": 2048}, 1, 1, 1, 8, 3),
}

default_model_run_groups = {
        "all": list(default_model_settings.keys()),
        "gpt3_setting": [x for x in list(default_model_settings.keys()) if "_gpt3_setting" in x],
        "moe": ["moe_207B", "moe_523B", "moe_1.1T", "moe_52B"],
    }


def get_extended_default_model_settings_and_groups():
    available_model_settings = default_model_settings.copy()

    model_run_groups = default_model_run_groups.copy()

    # Copy base model configs for all of it's checkpoints
    # 6.7B_gpt3_setting
    model_6_7B_gpt3_setting_checkpoints = {mcpt: copy_model_setting(available_model_settings["6.7B_gpt3_setting"], mcpt)
                                                      for mcpt in get_model_checkpoint_names("6.7B_gpt3_setting")}
    available_model_settings.update(model_6_7B_gpt3_setting_checkpoints)
    model_run_groups["6.7B_gpt3_setting_checkpoints"] = model_6_7B_gpt3_setting_checkpoints.keys()

    # 1.3B_gpt3_setting
    model_1_3B_gpt3_setting_checkpoints = {mcpt: copy_model_setting(available_model_settings["1.3B_gpt3_setting"], mcpt)
                                                      for mcpt in get_model_checkpoint_names("1.3B_gpt3_setting")}
    available_model_settings.update(model_1_3B_gpt3_setting_checkpoints)
    model_run_groups["1.3B_gpt3_setting_checkpoints"] = model_1_3B_gpt3_setting_checkpoints.keys()

    # moe_52B
    model_moe_52B_checkpoints = {mcpt: copy_model_setting(available_model_settings["moe_52B"], mcpt)
                                                      for mcpt in get_model_checkpoint_names("moe_52B")}
    available_model_settings.update(model_moe_52B_checkpoints)
    model_run_groups["moe_52B_checkpoints"] = model_moe_52B_checkpoints.keys()

    # moe_15B
    model_moe_15B_checkpoints = {mcpt: copy_model_setting(available_model_settings["moe_15B"], mcpt)
                                                      for mcpt in get_model_checkpoint_names("moe_15B")}
    available_model_settings.update(model_moe_15B_checkpoints)
    model_run_groups["moe_15B_checkpoints"] = model_moe_15B_checkpoints.keys()

    # 1.3B_gpt3_setting
    model_7_5B_lang30_setting_checkpoints = {mcpt: copy_model_setting(available_model_settings["dense_7.5B_lang30_new_cc100_xl_unigram__step00065000"], mcpt)
                                                    for mcpt in get_model_checkpoint_names("dense_7.5B_lang30_new_cc100_xl_unigram")}
    available_model_settings.update(model_7_5B_lang30_setting_checkpoints)
    model_run_groups["dense_7.5B_lang30_new_cc100_xl_unigram_cpts"] = model_7_5B_lang30_setting_checkpoints.keys()

    # multilingual moe_200B
    model_moe_200B_lang30_setting_checkpoints = {mcpt: copy_model_setting(available_model_settings["moe_200B_lang30_new_cc100_xl_unigram__step00048000"], mcpt)
                                                    for mcpt in get_model_checkpoint_names("moe_200B_lang30_new_cc100_xl_unigram")}
    available_model_settings.update(model_moe_200B_lang30_setting_checkpoints)
    model_run_groups["model_moe_200B_lang30_setting_checkpoints"] = model_moe_200B_lang30_setting_checkpoints.keys()

    return available_model_settings, model_run_groups


class TaskSchedulingSetting(NamedTuple):
    job_name_prefix: str  #  This is used as a prefix for the job or results saving sub-directory created for this task setting.
    tasks: List[str]  # This is the list of tasks that we want to eval. Same as the gpt3_eval --tasks.
    eval_params: Dict[str, Any]  # These are the eval argument names used in gpt3_eval.


default_tasks_settings = {x: TaskSchedulingSetting(x, [x], {}) for x in get_all_tasks()}
default_tasks_settings.update(
{
    # task run key: (task run name - used for results directory prefix, tasks list List[str], evaluation params: Dict[param_name, param_value] -- these are the same gpt3_eval input params)
    "intereval_blimp_all": TaskSchedulingSetting(**{"job_name_prefix":"blimp_all", "tasks":["blimp"], "eval_params": {}}),
    "intereval_diagnosis_0shot": ["diagnosis_0shot", ["diagnosis"], {"scoring": "sum", "nb_few_shot_samples_values": [0]}],
    "intereval_diagnosis_1shot": ["diagnosis_1shot", ["diagnosis"], {"scoring": "sum", "nb_few_shot_samples_values": [1]}],
    "intereval_diagnosis_32shot": ["diagnosis_32shot", ["diagnosis"], {"scoring": "sum", "nb_few_shot_samples_values": [32]}],
    "intereval_storycloze": ["storycloze", ['storycloze'], {"storycloze_languages": ["en"]}],
    "intereval_openbookqa": ["openbookqa", ["openbookqa"], {"calibrator_name": "average_option",
                           "openbookqa_calibration_options": ["question::"], "openbookqa_template": ['arc_old'],},],
    "intereval_arcchallenge": ["arcchallenge", ["arcchallenge"], {"calibrator_name": "average_option",
                             "arcchallenge_calibration_options": ["question::"], "arcchallenge_template": ['arc_old'],},],
    "intereval_sciqa_combined": ["sciqa", ["arcchallenge", "arceasy", "openbookqa"], {"calibrator_name": "average_option",
                             "arcchallenge_calibration_options": ["question::"], "arcchallenge_template": ['arc_old'],
                             "openbookqa_calibration_options": ["question::"], "openbookqa_template": ['arc_old'],
                             },],
    "intereval_winogrande_storycloze_and_piqa": TaskSchedulingSetting(**{"job_name_prefix":"blimp_all", "tasks":["winogrande", "storycloze", "piqa"], "eval_params": {"storycloze_languages": ["en"]}}),

    "flan_cb": ["flan_cb", ["flan__cb.alltemplates"], {"scoring": "sum", "nb_few_shot_samples_values": [0]}],
    "flan_nli": ["flan_nli", ["flan__rte.alltemplates", "flan__anli_r1.alltemplates", "flan__anli_r2.alltemplates", "flan__anli_r3.alltemplates"], {"scoring": "mean", "nb_few_shot_samples_values": [0]}],
    "flan_qa": ["flan_qa", ["flan__arc_easy.alltemplates", "flan__arc_challenge.alltemplates"], {"scoring": "mean", "nb_few_shot_samples_values": [0]}],
    "flan_sentiment": ["flan_sentiment", ["flan__yelp_polarity_reviews.alltemplates", "flan__sst2.alltemplates", "flan__sentiment140.alltemplates"], {"nb_few_shot_samples_values": [0]}],
})

default_task_run_groups = {
    "rai": list(default_tasks_settings.keys()),
    "intermediate_eval_v1": ["intereval_blimp_all", 
                             "intereval_storycloze","hellaswag", "piqa", "winogrande", # LM, commonsense
                             "arceasy", "intereval_arcchallenge",  "intereval_openbookqa", # science qa
                             "intereval_diagnosis_0shot", "intereval_diagnosis_1shot", "intereval_diagnosis_32shot", # diagnostic tasks
                             ],
    "intermediate_eval_175B": ["intereval_winogrande_storycloze_and_piqa", "intereval_sciqa_combined", "intereval_blimp_all","hellaswag"],
}


def get_settings(setting_options, setting_groups, available_settings):
    """Selects settings to execute based on setting groups and available seeings

    Args:
        setting_options (List[str]): The list of setting group keys or available setting keys.
        setting_groups (Dict[str, Any]): Dictionary of settings groups
        available_settings (Dict[str, Any]): Dictionary of named settings

    Raises:
        ValueError: [description]

    Returns:
        List[Any]: List of settings to execute
    """

    settings = []
    for settin_option in setting_options:
        settings_names = setting_groups.get(settin_option,
                    [settin_option] if settin_option in available_settings else None) # Search in setting keys and if not found, assume it is a group
        if settings_names is None:
            available_settings_display = str(sorted(list(available_settings.keys())))
            raise ValueError(f"Option {settin_option} is not available in the groups or single settings! Available settings:{available_settings_display}")

        for setting_name_key in settings_names:
            current_setting = list(available_settings[setting_name_key])
            current_setting = [setting_name_key] + current_setting
            settings.append(current_setting)

    return settings


def get_reproducibility_info(prefix: str) -> Dict[str, Any]:
    """
    Collect some reproducibility information including the git commit, timestamp, user.

    Args:
        prefix (str): The prefix in front of

    Returns:
        Dict[str, any]: The reproducibility fields.
    """
    repro_info = {}

    # get git commit version
    git_commit = "unknown"
    try:
        git_commit = subprocess.check_output("git log | head -n 1", shell=True, encoding="utf-8")
        git_commit = git_commit.rstrip()
    except Exception as e:
        logging.exception(f"Could not get git_commit!", e)

    repro_info[prefix + "git_commit"] = git_commit

    # date and time timestamp
    schedule_time_stamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
    repro_info[prefix + "timestamp"] = schedule_time_stamp

    # log user
    scheduled_by = "unknown"
    try:
        scheduled_by = os.getenv("USER")
    except Exception as e_user_var_read:
        logging.exception(f"Could not read $USER variable!", e_user_var_read)

    repro_info[prefix + "user"] = scheduled_by

    return repro_info


def get_schedule_reproducibility_info():
    return get_reproducibility_info(prefix="schedule_")


def write_text_to_file(file_name, text):
    with open(file_name, mode="w") as f_out:
        f_out.write(text)


def get_expected_results_files(folder, run_args):
    # dry-run to get the expected results files!
    dry_run_args = copy.deepcopy(run_args)
    dry_run_args.update({"dry_run": True,
                            "return_result_files_list": True})

    tasks_list = dry_run_args["tasks"]
    del dry_run_args["tasks"]  # delete since it is an explicit argument
    _, expected_results_files = run_evaluations(None, tasks_list, **dry_run_args)

    expected_results_files = [os.path.join(folder, os.path.basename(x)) for x in expected_results_files]

    return expected_results_files


def get_expected_results_files_with_status(run_args):
    expected_results_files = get_expected_results_files(run_args)
    expected_results_files_with_status = get_expected_results_files_status(expected_results_files)

    return expected_results_files_with_status


def get_expected_results_files_status(expected_results_files):
    expected_results_files_with_status = [(x, get_result_file_status(x)) for x in expected_results_files]

    return expected_results_files_with_status


def get_expected_results_cnt_by_status(expected_results_files):
    expected_results_files_with_status = get_expected_results_files_status(expected_results_files)
    expected_results_cnt_by_status = dict(Counter([x[1] for x in expected_results_files_with_status]))

    return expected_results_cnt_by_status


def get_jobs_results_status(jobs_status:List[Dict[str, any]]):
    current_jobs_results_status = {}
    for job_info in jobs_status:
        job_result_files_status = job_info["results_cnt_by_status"]
        for stat_key, stat_cnt in job_result_files_status.items():
            if stat_key not in current_jobs_results_status:
                current_jobs_results_status[stat_key] = 0

            current_jobs_results_status[stat_key] += stat_cnt
    return current_jobs_results_status


def schedule_experiment_jobs(args,
                            task_run_groups, available_tasks_settings,
                            model_run_groups, available_model_settings,
                            custom_base_run_args = {},
                            custom_override_run_args = {},
                            silent=False,
                            dry_run=False,
                            override_completed=False,
                            ):
    """
    This method runs multiple experiments with slurms or locally by calling `run_evaluations_from_model_name(**run_args)` for each model_setting and task_setting.
    The variable `run_args` contains input arguments which are built as dictionary and passed to the function.
    For each model_setting and task_setting we init `run_args` with the arguments passed from the `args` and update them with the
    configuration properties first from model_setting then from task_setting.
    The variable `args` contains `models` and `tasks` arguments that specify lists of keys for model and task settings groups.
    For each key we seek for model or task settings keys (available_[model/task]_settings) or groups of keys ([model/task]_run_groups) to use for the condifurations.
    The order of forming run_args is:
        0) `run_args` default values are set from the input `args`
        -> 1) `run_args` is updated with `custom_base_run_args` (empty by default)
        -> 2) `run_args` is updated using the model_setting params
        -> 3) for each task_setting a clone of `run_args` is updated with the task_setting params
        -> 4) for each task_setting a clone of `run_args` is updated with the task_setting params
        -> 1) `run_args` is updated with `custom_override_run_args` (empty by default)

        -> run_args are passed to `run_evaluations_from_model_name`

    Args:
        args: Parsed command line arguments - these are used to update the .
        task_run_groups: Task groups. A dictionary with list of task_settings keys.
        available_tasks_settings: Available task settings. Dictionary with named task settings/configurations
        model_run_groups: Model groups. A dictionary with list of model_settings keys.
        available_model_settings: Available model settings. Dictionary with named model settings/configurations
        custom_base_run_args: Override some of the run params manually. These are updated before the model and task params.

    Returns:
        Tuple(int, List[Dict[str, any]]): Returns the total number of jobs attempted to be logged
                                            and jobs that are being submitted to slurm.
                                            When dry_run == True the returned jobs are not actually submitted but "would have been"!
                                            The returned jobs can be used to determine if anything must be executed.
    """
    schedule_log = {}

    repro_info = get_schedule_reproducibility_info()
    dry_run = dry_run or args.dry_run  # This is needed for:
                                       #   1) being able to pass the args.dry_run as args (backward compatibility with existing scripts)
                                       #   2) Use dry_run input param for checking scheduling status, etc.

    # Update the output dir.
    out_dir = args.output

    if args.recalc_metrics_from_predictions:
        # I am sure that there is a better way to implement this with the argparser but I don't have time to learn it right now! Please, refactor!
        assert args.local, "recalc_metrics_from_predictions:True must be executed locally with --local! Since the model is not loaded, it does not make sense to allocate resources!"
        custom_override_run_args["predictor_name"] = "clmprompting" # TO DO Fix this when we implement recalc for openaiapi and other predictors

    dump_predictions = True

    scoring_setting = args.scoring
    slurm_partition = args.slurm_partition # "learnfair", "XLM-G"
    predictor_name = args.predictor_name
    execute_locally = args.local

    override_completed = override_completed or (args.override_completed)
    skip_if_executed_before = not override_completed

    num_trials = args.num_trials

    if not dry_run:
        # This will fail if the env is not okay -- cuda is missing, modules are not loaded, etc.
        check_env_f = submitit.helpers.CommandFunction(["python", "examples/few_shot/scripts/check_environment.py"])
        logging.info(check_env_f())

    # Select task and model settings
    task_settings_to_execute = get_settings(args.tasks, task_run_groups, available_tasks_settings)
    model_settings = get_settings(args.models, model_run_groups, available_model_settings)

    few_shot_samples = args.nb_few_shot_samples_values
    n_eval_samples = args.n_eval_samples

    total_jobs = 0
    submitted_jobs_to_slurm_all = []
    for model_setting in model_settings:
        (
            model_schedule_setting_key,
            model_name,
            job_params,
            model_args,
            nodes,
            gpus_per_node,
            ntasks_per_node,
            cpus_per_task,
            slurm_array_parallelism
        ) = model_setting

        if "distributed_port" in model_args or gpus_per_node > 1:
            # Making sure no two jobs use same port if 
            # they got assigned to same node by chance.
            model_args["distributed_port"] = str(get_random_port())
            
        job_name_global = f"{model_name}_few_shot"

        slurm_additional_parameters = {}
        if args.slurm_mail_user is not None:
            slurm_additional_parameters["mail-user"] = args.slurm_mail_user
            slurm_additional_parameters["mail-type"] = args.slurm_mail_type

        current_slurm_array_parallelism = slurm_array_parallelism
        if args.slurm_array_parallelism is not None:
            current_slurm_array_parallelism = args.slurm_array_parallelism
            print(f"slurm_array_parallelism={current_slurm_array_parallelism} is set!")

        executor = submitit.AutoExecutor(folder=out_dir)
        executor.update_parameters(
                slurm_array_parallelism=current_slurm_array_parallelism,
                timeout_min=int(60*24*1),
                slurm_constraint="volta32gb",
                slurm_partition=slurm_partition,
                slurm_nodes=nodes,
                slurm_ntasks_per_node=ntasks_per_node,
                slurm_cpus_per_task=cpus_per_task,
                slurm_gpus_per_node=gpus_per_node,
                slurm_job_name=job_name_global,
                slurm_mem_per_gpu="58G"
            )

        if len(slurm_additional_parameters) > 0:
            executor.update_parameters(slurm_additional_parameters=slurm_additional_parameters)

        current_model_task_settings_to_execute = task_settings_to_execute
        combine_tasks = job_params.get("combine_tasks", False)
        if combine_tasks and len(current_model_task_settings_to_execute) > 1:
            task_group = "combined_task_run"
            combined_task_names = []
            combined_custom_args = {}
            for tasks, task_names, custom_args in current_model_task_settings_to_execute:
                combined_task_names.extend(task_names)
                combined_custom_args.update(custom_args)

            current_model_task_settings_to_execute = [(task_group, combined_task_names, combined_custom_args)]

        def schedule_or_run_job(job_name, run_args, executor=None):
            if execute_locally:
                print(f"Executing locally {job_name}")
                args_without_model_name = {k: v for k,v in run_args.items() if k != "model_name"}
                if not dry_run:
                    run_evaluations_from_model_name(run_args["model_name"], **args_without_model_name)
                else:
                    print("Dry run! Nothing was executed!")
            else:
                print(f"Submitting {job_name} with tasks "+",".join(run_args["tasks"][:]))
                if not dry_run:
                    job = executor.submit(run_evaluations_from_model_name, **run_args)
                    print(job_name)
                    return 1
            return 0

        def schedule_or_run_few_shot_job(run_args, tasks_group, executor=None):
            fs = "_".join([str(x) for x in run_args["nb_few_shot_samples_values"]])
            #unique_suffix = "_" + datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f")
            unique_suffix = "_current"

            few_shot_job_name = (
                f'{tasks_group}_{run_args["model_name"]}_fs.{fs}'
                + f'_t{run_args["num_trials"]}_smpl.{run_args.get("n_eval_samples","All")}{unique_suffix}'
            )
            folder = os.path.join(out_dir, few_shot_job_name)
            os.makedirs(folder, exist_ok=True)

            if skip_if_executed_before:
                # dry-run to get the expected results files!
                dry_run_args = copy.deepcopy(run_args)
                dry_run_args.update({"dry_run": True, "return_result_files_list": True})

                tasks_list = dry_run_args["tasks"]
                del dry_run_args["tasks"]
                _, expected_results_files = run_evaluations(None, tasks_list, **dry_run_args)

                expected_results_files = [os.path.join(folder, os.path.basename(x)) for x in expected_results_files]
                missing_files = [x for x in expected_results_files if not os.path.exists(x)]

                # check if results files are available
                if len(missing_files) == 0:
                    print(f"--- Skipping {few_shot_job_name} -- has results: {len(expected_results_files)} results files")
                    return 0

            if dump_predictions:
                run_args.update(
                    {"predictions_dump_dir": folder,
                    "results_dir": folder,
                    "add_prompt_to_meta": True,
                    "add_positional_scores_to_meta": True,
                    "add_prompt_tokens_to_meta": True,
                    "add_calib_meta": True}
                )

            job_cnt = 0
            if args.split_few_shot_trials:
                num_trials = run_args['num_trials']
                for trial_seed in range(num_trials):
                    run_args['trial_seed'] = trial_seed
                    trial_job_name = f'_t{num_trials}_seed{trial_seed}'.join(few_shot_job_name.split(f'_t{num_trials}'))
                    job_cnt += schedule_or_run_job(trial_job_name, run_args, executor)
            else:
                job_cnt += schedule_or_run_job(few_shot_job_name, run_args, executor)

            return job_cnt

        def schedule_or_run_batch_jobs(executor=None):
            jobs_cnt = 0
            submitted_jobs_to_slurm = []
            for current_task_schedule_setting in current_model_task_settings_to_execute:
                jobs_cnt += 1  # attempted jobs

                (task_schedule_setting_key, # This is the unique setting key
                tasks_group, # This is the task directory predix
                task_names,
                custom_args) = current_task_schedule_setting

                run_args = {
                    "tasks": task_names,
                    "model_name": model_name,
                    "repro_info": repro_info,
                    "nb_few_shot_samples_values": few_shot_samples,
                    "num_trials": num_trials,
                    "scoring": scoring_setting,
                    "uniform_sampling": False,
                    "predictor_name": predictor_name,
                    "max_positions": None,
                    "skip_completed": skip_if_executed_before
                }
                # if args.user_dir:
                #     run_args["user_dir"] = os.path.abspath(args.user_dir)

                if n_eval_samples > 0:
                    run_args["n_eval_samples"] = n_eval_samples

                run_args["user_dir"] = args.user_dir

                run_args.update(custom_base_run_args)
                run_args.update(model_args)
                run_args.update(custom_args)
                run_args.update(custom_override_run_args)

                fs = "_".join([str(x) for x in run_args["nb_few_shot_samples_values"]])
                unique_suffix = "_current"

                job_name = (
                    f'{tasks_group}_{run_args["model_name"]}_fs.{fs}'
                    + f'_t{run_args["num_trials"]}_smpl.{run_args.get("n_eval_samples","All")}{unique_suffix}'
                )

                # The initial incentive of using task_group was to group different task runs in the same directory
                # for easier results collection and to skip actual eval combinations (task, model setting) being executed
                # multiple times from different configurations.
                # In contrast, for bulk scheduling where we want to keep track of submitted (not only executed results)
                # it is better to keep track of unique scheduling job names.
                job_unique_name = (
                    f'{task_schedule_setting_key}_{model_schedule_setting_key}_fs.{fs}'
                    + f'_t{run_args["num_trials"]}_smpl.{run_args.get("n_eval_samples","All")}{unique_suffix}'
                )
                run_args.update({"executed_by_job_name": job_unique_name})

                # Create output directory if does not exist
                folder = os.path.join(out_dir, job_name)
                os.makedirs(folder, exist_ok=True)

                job_info = {
                    "job_settings_name": job_unique_name,
                    "model_setting_key": model_schedule_setting_key,
                    "task_setting_key": task_schedule_setting_key,
                    "out_dir": out_dir,
                    "timestamp": datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S-%f"),
                    "results_dir": folder,
                }

                run_args.update({"predictions_dump_dir": folder,
                    "results_dir": folder,})
                expected_results_files = get_expected_results_files(folder, run_args)
                expected_results_files_with_status = get_expected_results_files_status(expected_results_files)
                expected_results_cnt_by_status = dict(Counter([x[1] for x in expected_results_files_with_status]))
                job_info["results_cnt_by_status"] = expected_results_cnt_by_status

                # Dump all predictions by default since the predictions files are useful when necessary to debug, analyze, regenerate fields, count examples.
                run_args.update(
                    { "add_prompt_to_meta": True,
                    "add_positional_scores_to_meta": True,
                    "add_prompt_tokens_to_meta": True,
                    "add_calib_meta": True}
                )

                missing_results_files = [x[0] for x in expected_results_files_with_status if x[1] == RESULT_FILE_STATUS_MISSING]
                completed_results_files = [x[0] for x in expected_results_files_with_status if x[1] == RESULT_FILE_STATUS_COMPLETED]
                if args.recalc_metrics_from_predictions:
                    if len(completed_results_files) == 0: # skip if not executed before
                        continue
                elif skip_if_executed_before:
                    # check if results files are available
                    if len(missing_results_files) == 0:
                        job_info["current_status"] = JOB_STATUS_CURRENT_SKIP
                        submitted_jobs_to_slurm.append(job_info)

                        if not silent:
                            logging.info(f"--- Skipping {job_name} - results files are already completed or pending!")
                        continue

                if execute_locally:
                    logging.info(f"Executing locally {job_name}")
                    if not dry_run:
                        args_without_model_name = {k: v for k,v in run_args.items() if k != "model_name"}
                        if args.recalc_metrics_from_predictions:
                            run_tasks = args_without_model_name["tasks"]
                            args_without_model_name_and_tasks = {k: v for k,v in run_args.items() if k != "tasks"}
                            args_without_model_name_and_tasks["recalc_metrics_from_predictions"] = True
                            run_evaluations(None, run_tasks, **args_without_model_name_and_tasks)
                        else:
                            run_evaluations_from_model_name(run_args["model_name"], **args_without_model_name)
                        # update results_status
                    else:
                        if not silent:
                            logging.info("Dry run! Nothing was executed!")

                    job_info["current_status"] = JOB_STATUS_CURRENT_EXECUTE_LOCALLY
                    job_info["results_cnt_by_status"] = get_expected_results_cnt_by_status(expected_results_files)

                    submitted_jobs_to_slurm.append(job_info)
                else:
                    if not silent:
                        logging.info(f"Submitting {job_unique_name}")

                    if not dry_run:
                        job = executor.submit(run_evaluations_from_model_name, **run_args)

                        for missing_result_file in missing_results_files:
                            # Log the expected file names so a second pass does not try to submit the same tasks.
                            # In case that we want to resubmit the job, we need to manually deleted the .submitted files (same with .json results files)
                            result_file_flagged_as_submitted = missing_result_file + ".submitted"
                            write_text_to_file(result_file_flagged_as_submitted, json.dumps(job_info))

                        job_info["slurm_job"] = job
                        if not silent:
                            logging.info(f"{job_unique_name} submitted!")

                    job_info["current_status"] = JOB_STATUS_CURRENT_SUBMIT
                    job_info["results_cnt_by_status"] = get_expected_results_cnt_by_status(expected_results_files)

                    submitted_jobs_to_slurm.append(job_info)

            return jobs_cnt, submitted_jobs_to_slurm

        if execute_locally:
            schedule_or_run_batch_jobs()
        else:
            if dry_run:
                attempted_jobs_cnt, submitted_to_slurm_for_current_model = schedule_or_run_batch_jobs(None)
            else:
                with executor.batch():
                    attempted_jobs_cnt, submitted_to_slurm_for_current_model = schedule_or_run_batch_jobs(executor)
            submitted_jobs_to_slurm_all.extend(submitted_to_slurm_for_current_model)
            total_jobs += attempted_jobs_cnt

    if not execute_locally:
        if not silent:
            jobs_current_status_cnt = dict(Counter([x.get("current_status", None) for x in submitted_jobs_to_slurm_all]))
            logging.info(("Dry run! " if dry_run else "") + f"Submission status: {jobs_current_status_cnt}")
    return total_jobs, submitted_jobs_to_slurm_all


def add_base_arguments(parser):
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally (not on slurm).",
    )

    parser.add_argument(
        "--recalc-metrics-from-predictions",
        action="store_true",
        help="This will recalculate metrics from predictions without scheduling anything. It will only work if results.json are available to avoid trying to execute when prediction files are not presented.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run.",
    )

    parser.add_argument(
        "--override-completed",
        action="store_true",
        help="Overrides results that are available.",
    )

    parser.add_argument(
        "--split-few-shot-experiments",
        action="store_true",
        help="If set, submit concurrent jobs for different number of few-shot samples"
    )

    parser.add_argument(
        "--split-jobs-by-eval-lang",
        action="store_true",
        help="If set, submit concurrent jobs for each evaluation language"
    )

    parser.add_argument(
        "--split-few-shot-trials",
        action="store_true",
        help="If set, "
    )

    parser.add_argument(
        "--slurm-array-parallelism",
        type=int,
        default=None,
        help="Slurm setting: The slurm array parallelism. This overrides the model setting.",
    )

    parser.add_argument(
        "--slurm-partition",
        default=f"learnaccel",
        help="Slurm setting: partition.",
    )

    parser.add_argument(
        "--slurm-mail-user",
        default=None,
        help="Slurm setting: Mail address to send the notifications to.",
    )

    parser.add_argument(
        "--slurm-mail-type",
        default="FAIL",  # Bulk schedule could produce MANY e-mails so better log the critical by default!
        choices=['NONE', 'BEGIN', 'END', 'FAIL', 'REQUEUE', 'ALL', 'INVALID_DEPEND', 'STAGE_OUT', 'TIME_LIMIT', 'TIME_LIMIT_90', 'TIME_LIMIT_80', 'TIME_LIMIT_50'],
        help='Slrum setting: The type of e-mails to be send. See https://slurm.schedmd.com/sbatch.html for detailed description'
    )

    parser.add_argument(
        "--scoring",
        default=f"mean",
        help="Scoring setting",
    )

    parser.add_argument(
        "--predictor-name",
        default=f"clmprompting",
        help="Predictor name",
    )

    parser.add_argument(
        "--nb-few-shot-samples-values",
        "--nshot", # alias
        type=int,
        default=[0],
        nargs="+",
        help="subsample K examples from the training set for one-shot or "
        "few-shot learning",
    )

    parser.add_argument(
        "--slurm-parallel",
        type=int,
        default=0,
        metavar="N",
        help="Number of samples to use for the evaluation. Use 0 for full set. Use a smaller number for debug purposes.",
    )

    parser.add_argument(
        "--n-eval-samples",
        type=int,
        default=0,
        metavar="N",
        help="Number of samples to use for the evaluation. Use 0 for full set. Use a smaller number for debug purposes.",
    )

    parser.add_argument(
        "--num-trials",
        type=int,
        default=5,
        metavar="N",
        help="when --nb-few-shot-samples is provided, repeat experiments N "
        "times and report all results (few-shot experiments can have a high variance).",
    )

    parser.add_argument(
        "--user-dir",
        default=None,
        help="path to user-defined tasks/criterions/etc."
    )


def add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings):
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="The parent output directory for the jobs in the current run!",
    )

    parser.add_argument(
        "-t",
        "--tasks",
        default=[],
        nargs="+",
        help="List of individual tasks or groups of tasks. Available groups are:" + ",".join([f"`{k}`" for k,f in task_run_groups.items()]) + "; Available single task settings are:" + ",".join([f"`{k}`" for k,f in available_tasks_settings.items()]) ,
    )

    parser.add_argument(
        "-m",
        "--models",
        default=[],
        nargs="+",
        help="List of individual tasks or groups of tasks. Available groups are:" + ",".join([f"`{k}`" for k,f in model_run_groups.items()]) + "; Available single model settings are:" + ",".join([f"`{k}`" for k,f in available_model_settings.items()]) ,
    )

def arg_modify_default(parser, dest, default):
    for action in parser._actions:
        if action.dest == dest:
            action.default = default
            return
    else:
        raise AssertionError('argument {} not found'.format(dest))


def print_display_results_command(args):
    logging.info("")
    logging.info("When experiments are done, aggregate the results with:")
    display_results_cmd = f"python examples/few_shot/scripts/collect_results.py -i {args.output} -o {args.output}/results.tsv -v preferred_metrics_mean"
    logging.info(display_results_cmd)
    logging.info("")
    logging.info("Please, also add the experiment to the XLM-G Results log at https://fburl.com/wm9i093h :")
    logging.info(f"{args.output}\t[description]\t{os.getenv('USER')}\t\t[copy command here]\t{display_results_cmd}")

def get_checkpoint_files_in_dir(model_checkpoint_directory, checkpoint_search_pattern):
    # Search in the local dir -- some checkpoints are directly in the dir (usually consolidated)!
    checkpoint_files = glob.glob(model_checkpoint_directory.rstrip("/") + checkpoint_search_pattern)  # checkpoint_6_50000-shard0.pt
    if len(checkpoint_files) == 0:
        # some checkpoints are in subdirectories
        checkpoint_files = glob.glob(model_checkpoint_directory.rstrip("/") + "/**" + checkpoint_search_pattern)  # checkpoint_6_50000-shard0.pt

    return checkpoint_files


class ModelCheckpointsEvalConfiguration(ABC):
    """This class is used for configuring model checkpoints evaluation.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    model_name_base: str  # The basename for the model
    model_checkpoints_directory: str  # Directory with model checkpoints
    checkpoint_id_filter: str  # Filter for model checkpoint ids
    model_config_base: Dict[str, Any]  #  This is the standard configuration from model_configs.py
    model_scheduling_setting_base: Dict[str, Any]  # This is the config used for ModelSchedulingSetting

    @property
    def checkpoint_ids(self):
        checkpoint_filter = self.checkpoint_id_filter
        expected_ids = set(get_checkpoint_ids_from_text(checkpoint_filter))
        expected_ids = list(sorted(expected_ids))

        return expected_ids

    def get_available_checkpoint_files(self):
        # find all checkpoints for the model that are available locally
        model_checkpoint_directory = self.model_checkpoints_directory

        # Note that this will only work for checkpoints with shard0.pt pattern!
        # Might not be compatible with moe or other checkpoints.
        # Search in subdirectories - In case that no checkpoints are found, we also look at subdirectories.

        # consolidated
        checkpoint_files = get_checkpoint_files_in_dir(model_checkpoint_directory, "/**/checkpoint_*_*consolidated.pt")  # consolidated - preferred
        if len(checkpoint_files) == 0:
            checkpoint_files = get_checkpoint_files_in_dir(model_checkpoint_directory, "/checkpoint_*_*consolidated.pt")  # consolidated - preferred

        # sharded if no consolidated are found
        if len(checkpoint_files) == 0:
            checkpoint_files = get_checkpoint_files_in_dir(model_checkpoint_directory, "/**/checkpoint_*_*-shard0.pt") # sharded
        if len(checkpoint_files) == 0:
            checkpoint_files = get_checkpoint_files_in_dir(model_checkpoint_directory, "/checkpoint_*_*-shard0.pt") # sharded

        return checkpoint_files

    def get_checkpoints_id_to_file(self, available_only):
        """Find all allowed checkpoint files.

        Returns:
            Dict[int, str]: Mapping between checkpoint_id and checkpoint_file
        """

        availtable_checkpoints_files = self.get_available_checkpoint_files()
        availtable_checkpoints_files_id_to_file_map = {get_checkpoint_id_from_filename(checkpoint_file): checkpoint_file for checkpoint_file in availtable_checkpoints_files}

        checkpoint_id_to_file_map = {cpt_id: availtable_checkpoints_files_id_to_file_map.get(cpt_id, UNAVAILABLE_CHECKPOINT_PATH_DEFAULT_VALUE)
                                    for cpt_id in self.checkpoint_ids}

        if available_only:
            # We are filtering explicitly for better readablity
            checkpoint_id_to_file_map = {cpt_id: cpt_path for cpt_id, cpt_path in checkpoint_id_to_file_map.items() if cpt_path != UNAVAILABLE_CHECKPOINT_PATH_DEFAULT_VALUE}

        return checkpoint_id_to_file_map

    def get_model_checkpoints_with_scheduling_settings(self, checkpoint_ids=None, available_only=False, start_with_last=True):
        """Get all available model checkpoint scheduling configurations

        Returns:
            Dict[str, ModelSchedulingSetting]: Dictionary with key model_name and model scheduling settings.
        """
        model_scheduling_settings = []
        checkpoints_id_to_file = self.get_checkpoints_id_to_file(available_only=available_only)
        checkpoints_id_to_file = sorted([(cpt_id, cpt_path) for cpt_id, cpt_path in checkpoints_id_to_file.items()], key=lambda x: x[0], reverse=start_with_last)
        for checkpoint_id, checkpoint_path in checkpoints_id_to_file:
            if checkpoint_ids is not None and checkpoint_id not in checkpoint_ids:
                continue

            model_name_checkpoint = f"{self.model_name_base}__step{checkpoint_id:08d}"
            model_config_checkpoint = copy.deepcopy(self.model_config_base)
            model_config_checkpoint["model_path"] = checkpoint_path

            # create the model scheduling settings
            checkpoint_scheduling_setting_kwargs = copy.deepcopy(self.model_scheduling_setting_base)
            checkpoint_scheduling_setting_kwargs["model_name"] = model_name_checkpoint
            checkpoint_scheduling_setting_kwargs["eval_params"].update({
                "model_configs": {model_name_checkpoint: model_config_checkpoint}
            })

            checkpoint_scheduling_setting = ModelSchedulingSetting(**checkpoint_scheduling_setting_kwargs)
            model_checkpoint_with_sched_setting = ModelCheckpointInfoWithSchedulingSetting(checkpoint_id, checkpoint_path, model_name_checkpoint, checkpoint_scheduling_setting)
            model_scheduling_settings.append(model_checkpoint_with_sched_setting)

        return model_scheduling_settings


def update_jobs_status(submitted_jobs_to_slurm):
    """Updates the status of submitted jobs

    Args:
        submitted_jobs_to_slurm ([type]): The job info to update from slurm
    """
    for job_info in submitted_jobs_to_slurm:
        job_settings_name = job_info["job_settings_name"]
        slurm_job = job_info.get("slurm_job", None)
        if slurm_job is None:
            continue
        try:
            job_info["job_id"] = slurm_job.job_id
            job_info["state"] = slurm_job.state
            job_info["paths.result_pickle"] = str(slurm_job.paths.result_pickle)
            job_info["paths.stderr"] = str(slurm_job.paths.stderr)
            job_info["paths.stdout"] = str(slurm_job.paths.stdout)
            job_info["paths.submission_file"] = str(slurm_job.paths.submission_file)
            job_info["paths.submitted_pickle"] = str(slurm_job.paths.submitted_pickle)
            job_info["paths.result_pickle"] = str(slurm_job.paths.result_pickle)
            job_info["last_check"] = datetime.fromtimestamp(slurm_job._last_status_check).strftime("%Y-%m-%d-%H-%M-%S-%f")
        except:
            pass


def append_submitted_jobs_info_to_file(job_log_path: str, submitted_jobs_to_slurm:List[Dict[str, any]], update_jobs_state: bool=False):
    """Save jobs info to file.

    Args:
        job_log_path (str): The path to log
        submitted_jobs_to_slurm (List[Dict[str, any]]): List of jobs info
        update_jobs_state (bool, optional): If true, the slurm state will be updated. Defaults to False.
    """

    if update_jobs_state:
        update_jobs_status(submitted_jobs_to_slurm)

    with open(job_log_path, mode="a+") as f_jobs_log:
        for job_info in submitted_jobs_to_slurm:
            if job_info["current_status"] != "submit":
                continue

            job_info_serializable = {k:v for k,v in job_info.items() if k != "slurm_job"}
            f_jobs_log.write(json.dumps(job_info_serializable))
            f_jobs_log.write("\n")


def get_jobs_submissions_status_cnt(jobs_status: List[Any]):
    jobs_current_submission_status_cnt = dict(Counter([x.get("current_status", None) for x in jobs_status]))

    return jobs_current_submission_status_cnt

if __name__ == "__main__":
    current_user = os.getenv("USER")

    # task settings
    available_tasks_settings = default_tasks_settings.copy()
    task_run_groups = default_task_run_groups.copy()

    # model settings
    available_model_settings = default_model_settings.copy()
    model_run_groups = default_model_run_groups.copy()

    # parse arguments
    parser = argparse.ArgumentParser(description="Schedule few-shot jobs.")
    add_base_arguments(parser)
    add_run_arguments(parser, task_run_groups, available_tasks_settings, model_run_groups, available_model_settings)

    # set default dir for results
    args = parser.parse_args()

    logging.info("Arguments:")
    logging.info(args)

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
