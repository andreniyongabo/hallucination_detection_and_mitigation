import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Union

from examples.few_shot.scripts.checkpoint_helpers import (
    delete_checkpoint_with_shards,
    read_json_config_file_and_populate_env_vars,
)
from examples.few_shot.scripts.collect_results import (
    WandbSimpleConfig,
    add_wandb_arguments,
    get_random_wandb_id,
    get_wandb_config,
    read_json_file,
    run_results_collection_and_attempt_loging_to_wandb,
    validate_wandb_login,
)
from examples.few_shot.scripts.download_checkpoints import (
    download_checkpoints_from_azure_blob,
)

# from examples.few_shot.scripts.experiments.schedule_jobs_few_shot import *
import copy
from examples.few_shot.scripts.experiments.schedule_jobs_few_shot import (
    JOB_STATUS_CURRENT_SUBMIT,
    RESULT_FILE_STATUS_COMPLETED,
    RESULT_FILE_STATUS_MISSING,
    RESULT_FILE_STATUS_SUBMITTED,
    UNAVAILABLE_CHECKPOINT_PATH_DEFAULT_VALUE,
    ModelCheckpointsEvalConfiguration,
    add_base_arguments,
    add_run_arguments,
    append_submitted_jobs_info_to_file,
    arg_modify_default,
    get_extended_default_model_settings_and_groups,
    get_jobs_results_status,
    get_jobs_submissions_status_cnt,
    schedule_experiment_jobs,
    update_jobs_status,
    default_tasks_settings,
    default_task_run_groups,
)
from examples.few_shot.tasks import get_tasks_by_group
from pathlib import Path

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

"""
    The current script aims at executing experiments from multiple checkpoints of the same model.

    Selected tasks:
        - intermediate_eval_v1

    Setting:
        - Scoring - `mean`, since it has good justifications, and it was used by gpt3.
                    We currently don't have an good reason to use 'sum' except that it works well.

    Commands:
        - Run
            # example with config with local checkpoints
            config_file=examples/few_shot/scripts/experiments/configs/1.3B_gpt3_setting_old.json
            out_dir=/large_experiments/xlmg/results/intermediate_eval/1.3B_gpt3_setting_old
            python examples/few_shot/scripts/experiments/schedule_jobs_intermittent_checkpoint_eval.py -t intermediate_eval_v1 \
                -o ${out_dir} --checkpoints-eval-config ${config_file} \
                --log-to-wandb --wandb-project gpt-z --wandb-logging-entity xlm-g --overwrite-collected-results

            # example with Azure checkpoints
            config_file=examples/few_shot/scripts/experiments/configs/1.3B_gpt3_setting_kitchen_sink_23.json
            out_dir=/large_experiments/xlmg/results/intermediate_eval/1.3B_gpt3_setting_kitchen_sink_23
            python examples/few_shot/scripts/experiments/schedule_jobs_intermittent_checkpoint_eval.py -t intermediate_eval_v1 \
                -o ${out_dir} --checkpoints-eval-config ${config_file} \
                --log-to-wandb --wandb-project gpt-z --wandb-logging-entity xlm-g \
                --download-checkpoints --delete-checkpoints-when-eval-is-successful --overwrite-collected-results
            
    """


def download_model_checkpoints_and_schedule_evaluation_jobs(
    checkpoints_eval_config: Union[str, Dict[str, Any]],
    download_checkpoints: bool,
    delete_checkpoints_when_eval_is_successful: bool,
    args,
):
    if isinstance(checkpoints_eval_config, str):
        checkpoints_eval_config_json = read_json_file(checkpoints_eval_config)
    elif isinstance(checkpoints_eval_config, dict):
        checkpoints_eval_config_json = checkpoints_eval_config
    else:
        raise ValueError(
            "checkpoints_eval_config must be either str (file_path) or Dict object but it is {type(checkpoints_eval_config)}!"
        )

    model_checkpoint_eval_configuration = ModelCheckpointsEvalConfiguration(
        **checkpoints_eval_config_json
    )

    download_checkpoints_enabled = download_checkpoints

    # get all settings
    model_checkpoint_scheduling_settings = model_checkpoint_eval_configuration.get_model_checkpoints_with_scheduling_settings(
        available_only=not download_checkpoints_enabled
    )

    if len(model_checkpoint_scheduling_settings) == 0:
        logger.warning(
            "No checkpoints found in {checkpoints_eval_config_json['model_checkpoints_directory']} and checkpoints download is disabled!"
        )
        logger.info("To enable checkpoints download:")
        if (
            "azure_copy_setting" not in checkpoints_eval_config_json
        ):  # TO DO: Fix this when AWS support is added
            logger.info("- Add `azure_copy_setting` in {checkpoints_eval_config}")
        logger.info(" - Use `--download-checkpoints`")
        sys.exit(0)

    available_checkpoint_ids_on_azure = []
    azure_blob_url = None  # this will be resolved later from the config
    # We iterate over all possible models checkpoints
    for (
        checkpoint_id,
        checkpoint_path,
        model_key,
        curr_model_setting,
    ) in model_checkpoint_scheduling_settings:
        logger.info(f"Checking for {model_key}")
        curr_available_model_settings = {model_key: curr_model_setting}
        model_keys_to_execute = sorted(list(curr_available_model_settings.keys()))
        args.models = model_keys_to_execute  # we add only the current model

        def schedule_jobs_current(
            args,
            task_run_groups,
            available_tasks_settings,
            model_run_groups,
            curr_available_model_settings,
            custom_base_run_args={},
            custom_override_run_args={},
            dry_run=False,
        ):
            """Execute the job scheduling.

            Args:
                dry_run (bool, optional): When it is a dry run, the results and jobs status is only returned. No jobs are scheduled. Defaults to False.

            Returns:
                [type]: [description]
            """

            if not dry_run:
                # validate that we do not send models without path
                for m, msetting in curr_available_model_settings.items():
                    if "model_configs" in msetting.eval_params:
                        for m_cfg_name, model_config in msetting.eval_params[
                            "model_configs"
                        ].items():
                            model_path = model_config["model_path"]
                            assert os.path.exists(
                                model_path
                            ), f"`model_path`:{model_path} for model {m_cfg_name} in scheduling setting {m}. Full model config:\n{model_config} "

            potential_total_jobs, jobs_status = schedule_experiment_jobs(
                args,
                task_run_groups,
                available_tasks_settings,
                model_run_groups,
                curr_available_model_settings,
                custom_base_run_args=custom_base_run_args,
                custom_override_run_args=custom_override_run_args,
                silent=True,
                dry_run=dry_run,
            )

            # Get submission status
            jobs_current_submission_status_cnt = get_jobs_submissions_status_cnt(
                jobs_status
            )

            # Note that the results files for each job are much more!
            current_jobs_results_status = get_jobs_results_status(jobs_status)

            return (
                jobs_status,
                jobs_current_submission_status_cnt,
                current_jobs_results_status,
            )

        # DRY RUN - CHECK HOW MANY JOBS CAN BE SUBMITTED
        # This dry run is necessary to see if there are
        #  - Any missing results files
        #  - Any jobs need to be executed
        (
            jobs_status,
            jobs_current_submission_status_cnt,
            current_jobs_results_status,
        ) = schedule_jobs_current(
            args,
            task_run_groups,
            available_tasks_settings,
            model_run_groups,
            curr_available_model_settings,
            custom_base_run_args={},
            custom_override_run_args={},
            dry_run=True,
        )

        jobs_that_need_submission_cnt = jobs_current_submission_status_cnt.get(
            "submit", 0
        )
        if (
            jobs_that_need_submission_cnt > 0
            and checkpoint_path == UNAVAILABLE_CHECKPOINT_PATH_DEFAULT_VALUE
        ):
            # Make sure that the checkpoint is available
            # Download if not available
            logger.warning(
                f"The checkpoint path for step {checkpoint_id} is not available locally!"
            )
            if "azure_copy_setting" not in checkpoints_eval_config_json:
                raise Exception(
                    "`azure_copy_setting` is not in the checkpoints eval config ({checkpoints_eval_config})!"
                    "Fix the azure_copy_setting of exclude the checkpoint id from the `checkpoints_id_filter`!"
                )

            # Do not query azure if we already have the checkpoints.
            if len(available_checkpoint_ids_on_azure) > 0:
                if checkpoint_id not in available_checkpoint_ids_on_azure:
                    logger.info(
                        f"Current checkpoint_id={checkpoint_id} is not in the available (max is {max(available_checkpoint_ids_on_azure)}) checkpoints at {azure_blob_url}. They are:\n {available_checkpoint_ids_on_azure}"
                    )
                    logger.info(" Skip...")
                    continue

            # Note that the download_checkpoints_from_azure_blob expects the parent directory!
            checkpoints_parent_dir = Path(
                checkpoints_eval_config_json["model_checkpoints_directory"]
            ).parent
            logger.info(f"Attempting to download checkpoint from Azure!")
            try:
                (
                    available_checkpoint_ids_on_azure,
                    azure_blob_url,
                    _,
                ) = download_checkpoints_from_azure_blob(
                    config_file_or_json=checkpoints_eval_config_json,
                    output_path=checkpoints_parent_dir,
                    checkpoint_ids=[checkpoint_id],
                    append_model_name_to_out_path=True,
                    silent=True,
                )
            except FileNotFoundError as fe:
                logger.exception(f"Error: {fe}")
                if fe.filename == "azcopy":
                    logger.info(
                        f"- Suggested fix: Make sure that {fe.filename} is in your path PATH=$PATH:path/to/{fe.filename}/dir"
                    )
                raise fe

                # TO DO: Better error logging
            except Exception as e:
                logger.exception(f"Download attempt failed!", e)
                raise e
                # continue

            if len(available_checkpoint_ids_on_azure) == 0:
                logger.info("No known checkpoints are available yet!\n")
                continue
            else:
                logger.info(
                    f"Available checkpoints at {azure_blob_url} are:\n {available_checkpoint_ids_on_azure}"
                )

                # checkpoint is not yet available
                if checkpoint_id not in available_checkpoint_ids_on_azure:
                    logger.info(
                        f"Current checkpoint_id={checkpoint_id} is not in the available (max is {max(available_checkpoint_ids_on_azure)}) checkpoints at {azure_blob_url}. They are:\n {available_checkpoint_ids_on_azure}"
                    )
                    logger.info(" Skip...")
                    continue

                # reload the model
                reloaded_checkpoint_with_scheduling_settings = model_checkpoint_eval_configuration.get_model_checkpoints_with_scheduling_settings(
                    checkpoint_ids=[checkpoint_id], available_only=False
                )
                (
                    checkpoint_id,
                    checkpoint_path,
                    model_key,
                    curr_model_setting,
                ) = reloaded_checkpoint_with_scheduling_settings[0]

                max_available_checkpoint_id = max(available_checkpoint_ids_on_azure)
                # Make sure that the path exists
                if not os.path.exists(checkpoint_path):
                    logger.warning(
                        f"{model_key} does not have a valid checkpoint - {checkpoint_path} does not exist! Max available on Azure is {max_available_checkpoint_id} > {checkpoint_id}!"
                    )
                    continue

                # set the new model_settings
                curr_available_model_settings = {model_key: curr_model_setting}

        res_files_with_status = sum(
            [res_cnt for _, res_cnt in current_jobs_results_status.items()]
        )
        assert res_files_with_status > 0, "Expected results files should be > 0!"
        results_files_that_need_completion = current_jobs_results_status.get(
            RESULT_FILE_STATUS_SUBMITTED, 0
        ) + current_jobs_results_status.get(RESULT_FILE_STATUS_MISSING, 0)
        if results_files_that_need_completion == 0:
            completed_results_cnt = current_jobs_results_status.get(
                RESULT_FILE_STATUS_COMPLETED, 0
            )
            logger.info(f"All {completed_results_cnt} expected results are completed!")
            if (
                checkpoint_path != UNAVAILABLE_CHECKPOINT_PATH_DEFAULT_VALUE
                and delete_checkpoints_when_eval_is_successful
            ):
                # delete checkpoints files.
                checkpoint_base = os.path.basename(checkpoint_path).replace(
                    "-shard0.pt", ""
                )
                logger.info(f"Deleting {checkpoint_base} to save space..")
                delete_checkpoint_with_shards(checkpoint_path)
                logger.info(f"{checkpoint_base} was deleted!")

            continue

        # Execute the jobs
        (
            jobs_status,
            jobs_current_submission_status_cnt,
            current_jobs_results_status,
        ) = schedule_jobs_current(
            args,
            task_run_groups,
            available_tasks_settings,
            model_run_groups,
            curr_available_model_settings,
            custom_base_run_args={},
            custom_override_run_args={},
            dry_run=False,
        )

        logger.info(f"jobs submission status: {jobs_current_submission_status_cnt}")
        logger.info(f"results status: {current_jobs_results_status}")

        if jobs_current_submission_status_cnt.get(JOB_STATUS_CURRENT_SUBMIT, 0) > 0:
            # Log jobs status to a json file.
            update_jobs_status(jobs_status)
            out_dir = args.output
            job_log_path = os.path.join(
                out_dir,
                os.path.basename(args.checkpoints_eval_config) + ".jobs_log.jsonl",
            )
            append_submitted_jobs_info_to_file(job_log_path, jobs_status)
            logger.info(f"Submitted jobs info is logged to {job_log_path}")


if __name__ == "__main__":
    # task settings
    available_tasks_settings = default_tasks_settings.copy()
    task_run_groups = default_task_run_groups.copy()

    # model settings
    (
        available_model_settings,
        model_run_groups,
    ) = get_extended_default_model_settings_and_groups()

    # parse arguments
    parser = argparse.ArgumentParser(description="Intermediate checkpoint evaluation")

    # register arguments
    add_base_arguments(parser)
    add_run_arguments(
        parser,
        task_run_groups,
        available_tasks_settings,
        model_run_groups,
        available_model_settings,
    )
    add_wandb_arguments(parser)
    parser.add_argument(
        "--log-to-wandb",
        action="store_true",
        help="Logs to wandb",
    )

    parser.add_argument(
        "--checkpoints-eval-config",
        help="The configuration file",
        required=True,
    )

    parser.add_argument(
        "--delete-checkpoints-when-eval-is-successful",
        action="store_true",
        help="Delete checkpoints when evaluation is successful. Be careful. Use for models that are auotmatically synced!",
    )

    parser.add_argument(
        "--download-checkpoints",
        action="store_true",
        help="Delete checkpoints when evaluation is successful. Be careful. Use for models that are auotmatically synced!",
    )

    parser.add_argument(
        "--no-watch",
        action="store_true",
        help="Watches for new checkpoints and results.",
    )

    parser.add_argument(
        "--overwrite-collected-results",
        action="store_true",
        help="Overwrite collected results. This will collect the results again and will resubmit them to wandb.",
    )

    parser.add_argument(
        "--watch-interval",
        type=int,
        help="The interval in seconds to check for new checkpoints.",
        default=60,
    )

    arg_modify_default(parser, "tasks", ["intermediate_eval_v1"])
    arg_modify_default(parser, "scoring", "mean")
    arg_modify_default(parser, "num_trials", 5)

    debug = False
    if debug:
        arg_modify_default(
            parser,
            "output",
            f"/large_experiments/xlmg/results/debug_intermittent_eval",
        )
        arg_modify_default(
            parser,
            "checkpoints_eval_config",
            f"/private/home/tbmihaylov/fairseq-xlmg/examples/few_shot/scripts/experiments/configs/1.3B_gpt3_setting_kitchen_sink_14_debug.json",
        )
        arg_modify_default(parser, "tasks", ["arceasy"])
        arg_modify_default(parser, "n_eval_samples", 10)

    # set default dir for results
    args = parser.parse_args()

    # args.dry_run = True
    # args.local = True

    logger.info("Arguments:")
    logger.info(args)

    checkpoints_eval_config = args.checkpoints_eval_config
    download_checkpoints = args.download_checkpoints
    delete_checkpoints_when_eval_is_successful = (
        args.delete_checkpoints_when_eval_is_successful
    )

    watch_interval = args.watch_interval

    if not args.no_watch:
        logger.info(f"Watching for new checkpoints every {watch_interval} sec")

    # make sure that the target directory exists
    logger.info(f"output dir:{args.output}")
    os.makedirs(args.output, exist_ok=True)

    # read config
    checkpoints_eval_config_json = read_json_config_file_and_populate_env_vars(
        checkpoints_eval_config
    )

    # copy config to output dir for easy reference when looking at results
    checkpoints_eval_config_target_dir = os.path.join(
        args.output, os.path.basename(checkpoints_eval_config)
    )
    model_name_base = checkpoints_eval_config_json["model_name_base"]
    model_name_base_short = checkpoints_eval_config_json.get(
        "model_name_base_short", model_name_base
    )

    # Init wandb
    wandb_config = None
    if args.log_to_wandb:
        if not args.wandb_project:
            raise ValueError(
                "Log to wandb is enabled but --wandb-project is not specified!"
            )

        wandb_config_file = checkpoints_eval_config_target_dir + ".wandb.json"
        wandb_config = get_wandb_config(
            wandb_config_file,
            wandb_project=args.wandb_project,
            wandb_run_name=f"{model_name_base_short}"
            if not args.wandb_run_name
            else args.wandb_run_name,
            wandb_run_id=args.wandb_run_id,
            wandb_logging_entity=args.wandb_logging_entity,
            create_or_update_file_on_new_id=True,
        )

    wandb_config_dict = vars(wandb_config) if wandb_config is not None else {}

    # output results
    results_input_dirs = [args.output]
    results_out_file = os.path.join(args.output, f"results.tsv")
    if args.overwrite_collected_results:
        for res_out_file in [results_out_file, results_out_file + ".raw.jsonl"]:
            if os.path.exists(res_out_file):
                os.remove(res_out_file)

    loop_i = 0
    all_results = []
    while True:
        # collect results
        logger.info("")
        logger.info(f"Checking {results_input_dirs} for results...")
        new_results, all_results = run_results_collection_and_attempt_loging_to_wandb(
            input_dirs=results_input_dirs,
            output=results_out_file,
            existing_results=all_results,
            loop_i=loop_i,
            overwrite_output=False,
            **wandb_config_dict,
        )
        logger.info("Results check is done!")

        # schedule jobs
        logger.info("")
        logger.info("Checking for new checkpoints!")
        download_model_checkpoints_and_schedule_evaluation_jobs(
            checkpoints_eval_config_json,
            download_checkpoints,
            delete_checkpoints_when_eval_is_successful,
            args,
        )

        logging.info(f"Sleeping for {watch_interval}...")
        logging.info("")
        time.sleep(watch_interval)

        if args.no_watch:
            break

    sys.exit(0)  # sometimes srun hangs and calling sys.exit(0) exits properly
