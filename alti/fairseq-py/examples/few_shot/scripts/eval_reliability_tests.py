import argparse
import copy
import datetime
import glob
import json
import logging
import os
import pathlib
import random
import re
import string
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy
from collect_results import get_results_files
from pandas import DataFrame

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def calculate_accuracy(preds: List[float]) -> float:
    """
    Simple accuracy
    """
    return np.mean(np.array(preds)) * 100


def calculate_accuracy_pearson(
    preds: List[float], original_preds: List[float]
) -> Tuple[float, float]:
    """
    Calculate both accuracy and pearson
    """
    return (
        np.mean(np.array(preds)) * 100,
        scipy.stats.pearsonr(np.array(preds), np.array(original_preds))[0],
    )


def calculate_accuracy_failure_rate(
    preds: List[float], original_preds: List[float]
) -> Tuple[float, float]:
    """
    Calculate both accuracys and failure rate (the percentage of predictions changed)
    """
    pred_arr = np.array(preds)
    acc = np.mean(preds) * 100
    change_rate = (preds != np.array(original_preds)).mean() * 100
    return acc, change_rate


EVALUATION_FUNCTIONS = {
    "original": calculate_accuracy,
    "label_translation": calculate_accuracy_failure_rate,
    "paraphrasing": calculate_accuracy_failure_rate,
    "skill_definition": calculate_accuracy_failure_rate,
    "distraction": calculate_accuracy_failure_rate,
    "ordering": calculate_accuracy_failure_rate,
    "negation": calculate_accuracy_failure_rate,
}


def softmax(scores):
    norm_scores = np.exp(scores)
    norm_scores = norm_scores / np.sum(norm_scores)
    return norm_scores


def get_predictions_gold_probability_for_results_file(
    results_file: str,
    soft_scores: bool,
    normalize_scores=True,  # normalize_scores
    ensemble: str = None,
) -> Union[List[List[float]], List[float]]:
    """
       Read predictions scores from the predictions files of a results file.
       If there are multiple predicitons files for different seeds, all are read.

    if ensemble == "majority":
        assert soft_scores == False, "soft_scores must be False when ensemble is majority"
    Args:
        results_file (str): The results file that corresponds to the predictions.
                            This is used to infer the predictions files which are usually in the same directory but with different suffix.
        soft_scores (bool): Wether to return the score per example of only the binary score - 1 if correct, 0 otherwise.
        ensemble (str): If we want to ensemble the results (`mean` or `majority`). If ensemble is not None, the results is list of floats, otherwise it returns list of list of floats.
        normalize_scores (bool, optional): Uses softmax to convert the scores to probabilities. Defaults to True since most predictions report the unnormalized score.
    Raises:
        FileNotFoundError: If the predictions file corresponding to the results is not found.
    Returns:
        Union[List[List[float]], List[float]]: If ensemble=True, the function returns a single list of results, otherwise it returns list of list of floats.
    """

    assert results_file.endswith("_results.json")

    predictions_scores_by_seed = []
    gold_ids = []
    for seed in range(0, 1000):
        predictions_file = results_file.replace(
            "_results.json", f"_seed{seed}_predictions.jsonl"
        )
        if not os.path.exists(predictions_file):
            if seed == 0:
                raise FileNotFoundError(
                    f"Predictions file {predictions_file} is not found! It should be located together with the results file {results_file}. "
                    " Make sure that you ran the evaluation with output predictions by setting the results_dir of the evaluation script!"
                )
            else:
                continue

        current_scores = []
        gold_ids = []
        with open(predictions_file) as f_pred:
            for line in f_pred:
                item = json.loads(line.strip())

                scores = []
                gold_id = -1
                for cand_id, cand_name_pred in enumerate(item["candidates"]):
                    cand_name, cand = cand_name_pred
                    score = cand["score"]
                    if cand["gold"]:
                        gold_id = cand_id
                    scores.append(score)

                assert gold_id > -1
                gold_ids.append(gold_id)

                scores = np.array(scores)
                if normalize_scores:
                    scores = softmax(scores)

                if ensemble == "majority":
                    # We make the scores the discrete vote for majority
                    scores_one_hot = np.zeros(scores.shape)
                    scores_one_hot[np.argmax(scores)] = 1.0

                current_scores.append(scores)
        predictions_scores_by_seed.append(current_scores)

    # ensemble
    if len(predictions_scores_by_seed) > 1:  # List of lists of nd.array(scores)
        if ensemble in ["mean", "majority"]:
            # For majority we already made the predictions discrete so the majority is just the mean

            num_prediction_runs = len(predictions_scores_by_seed)
            predictions_scores_by_seed_array = np.array(predictions_scores_by_seed)
            predictions_scores_ensemble = (
                np.sum(predictions_scores_by_seed_array, axis=0) / num_prediction_runs
            )
            predictions_scores_ensemble = list(predictions_scores_ensemble)
            predictions_scores_ensemble = [
                np.array(x) for x in predictions_scores_ensemble
            ]  # List[array] is expected

            predictions_scores_by_seed = [predictions_scores_ensemble]
        elif ensemble is not None:
            raise ValueError("Unsupported ensemble: {ensemble}!")

    gold_scores_by_run = []
    for run_scores in predictions_scores_by_seed:
        if not soft_scores:
            one_hot_scores = []
            for scores in run_scores:
                one_hot_score = np.zeros(scores.shape)
                one_hot_score[np.argmax(scores)] = 1.0
                one_hot_scores.append(one_hot_score)
            run_scores = one_hot_scores

        gold_scores = [scores[gold_id] for scores, gold_id in zip(run_scores, gold_ids)]
        gold_scores_by_run.append(gold_scores)

    return gold_scores_by_run


def parse_model_and_task_details(results_file: str):
    """
    Parse model and task details from the results file path

    Args:
        results_file (str): Path to the results file in string format

    Returns:
        Tuple: Tuple of cluster, eval_protocol, task, model_name
    """
    with open(results_file, "r") as f:
        data = json.load(f)
    model_name = data["model_name"]
    task = data["task"]
    cluster_name, eval_protocol_name, task_name = task.split("__")[-1].split("--")

    return (cluster_name, eval_protocol_name, task_name, model_name)


def get_predictions_from_files(result_files: List[str]) -> List[Dict]:
    """
    Collect the prediction scores from each result file.

    Args:
        result_files (List[str]): List of files

    Returns:
        List[Dict]: results
    """
    results = []
    for res_file in result_files:
        pred_scores = get_predictions_gold_probability_for_results_file(
            res_file,
            soft_scores=False,
        )
        (
            cluster_name,
            eval_protocol_name,
            task_name,
            model_name,
        ) = parse_model_and_task_details(res_file)
        res = {
            "cluster": cluster_name,
            "eval_protocol": eval_protocol_name,
            "task": task_name,
            "model_name": model_name,
            "prediction_scores": pred_scores[
                0
            ],  # single List as no seeds are considered
        }
        results.append(res)

    return results


def filter_results(
    results: List[Dict],
    tasks: Optional[List[str]],
    eval_protocols: Optional[List[str]],
    model_names: Optional[List[str]],
    existing_results: Optional[List[Dict]],
) -> List[Dict]:
    """
    Filter the results list based on tasks, models, and eval_protocols

    Args:
        results (List[Dict]): List of results
        tasks (List[str]): List of tasks or clusters to be only included in the filtered results
        eval_protocols (List[str]): List of evaluation protocols to be only included in the filtered results
        model_names (List[str]): List of model names to be only included in the filtered results
        existing_results (List[Dict]): List of results that are already present in the output results path
    Returns:
        filtered_results (List[Dict]): filtered results based on input conditions
    """
    filtered_results = []
    # Filter cluster, model pair from existing results
    existing_cluster_model_pairs = {}
    for res in existing_results:
        if not (res["cluster"], res["model_name"]) in existing_cluster_model_pairs:
            existing_cluster_model_pairs[(res["cluster"], res["model_name"])] = True
    for res in results:
        flag = True
        if model_names is not None:
            if res["model_name"] in model_names:
                flag = True
            else:
                flag = False
        if tasks is not None:
            if res["task"] in tasks or res["cluster"] in tasks:
                flag = True
            else:
                flag = False
        if eval_protocols is not None:
            if (
                res["eval_protocol"] == "original"
                or res["eval_protocol"] in eval_protocols
            ):
                flag = True
            else:
                flag = False
        # Check existing results
        if (res["cluster"], res["model_name"]) in existing_cluster_model_pairs:
            flag = False
        if flag:
            filtered_results.append(res)

    assert len(filtered_results) != 0, "Filter conditions lead to zero results!"

    return filtered_results


def calculate_per_task_benchmark_scores(results):
    """
    Calculate scores for benchmarking.
    """
    # convert into dictionary format
    results_pred_dict = {}
    for res in results:
        results_pred_dict[
            (res["cluster"], res["eval_protocol"], res["task"], res["model_name"])
        ] = res["prediction_scores"]

    results_scores = {}
    for details, preds in results_pred_dict.items():
        if details[1] == "original":  # eval protocol
            results_scores[details] = EVALUATION_FUNCTIONS["original"](
                preds
            )  # end evaluation may not be always accuracy
            continue
        original_details = (details[0], "original", details[2], details[3])
        assert original_details in results_pred_dict
        if not EVALUATION_FUNCTIONS[details[1]].__annotations__["return"] is float:
            eval_score = EVALUATION_FUNCTIONS[details[1]](
                preds, results_pred_dict[original_details]
            )
        else:
            eval_score = EVALUATION_FUNCTIONS[details[1]](preds)
        results_scores[details] = eval_score

    return results_scores


def calculate_per_cluster_benchmark_scores(results):
    """
    Calculate scores for benchmarking.
    """
    # convert into dictionary format
    results_pred_dict = {}
    for res in results:
        results_pred_dict[
            (res["cluster"], res["eval_protocol"], res["task"], res["model_name"])
        ] = copy.deepcopy(res["prediction_scores"])

    results_pred_per_cluster = {}
    results_pred_original = {}
    for details, preds in results_pred_dict.items():
        cluster, eval_protocol, task, model_name = details
        new_details = (cluster, eval_protocol, model_name)
        if eval_protocol == "original":
            if new_details in results_pred_per_cluster:
                results_pred_per_cluster[new_details] += preds
            else:
                results_pred_per_cluster[new_details] = copy.deepcopy(preds)
            continue

        if new_details in results_pred_per_cluster:
            results_pred_per_cluster[new_details] += preds
            results_pred_original[
                (cluster, f"{eval_protocol}-original", model_name)
            ] += results_pred_dict[(cluster, "original", task, model_name)]
        else:
            results_pred_per_cluster[new_details] = copy.deepcopy(preds)
            results_pred_original[
                (cluster, f"{eval_protocol}-original", model_name)
            ] = copy.deepcopy(
                results_pred_dict[(cluster, "original", task, model_name)]
            )

    results_scores = {}
    for details, preds in results_pred_per_cluster.items():
        cluster, protocol, model_name = details
        if protocol == "original":  # eval protocol
            results_scores[details] = EVALUATION_FUNCTIONS["original"](
                preds
            )  # end evaluation may not be always accuracy
            continue
        if not EVALUATION_FUNCTIONS[details[1]].__annotations__["return"] is float:
            eval_score = EVALUATION_FUNCTIONS[details[1]](
                preds,
                results_pred_original[(cluster, f"{protocol}-original", model_name)],
            )
        else:
            eval_score = EVALUATION_FUNCTIONS[details[1]](preds)

        results_scores[details] = eval_score

    return results_scores


def print_and_save_per_model_benchmark_scores(
    result, output_path, overwrite_results=False
):
    """
    Print benchmark scores at task level in pretty way,
    and also save to output path
    """
    if len(list(result.keys())[0]) == 4:
        task_or_cluster_index = 2
        protocol_index = 1
        model_index = 3
        result_level = "task"
    else:
        task_or_cluster_index = 0
        protocol_index = 1
        model_index = 2
        result_level = "cluster"

    task_to_cluster = {}
    if result_level == "task":
        for details in result.keys():
            task_to_cluster[details[2]] = details[0]

    all_tasks_or_clusters = set([k[task_or_cluster_index] for k in list(result.keys())])
    all_models = set([k[model_index] for k in list(result.keys())])
    all_eval_protocols = list(EVALUATION_FUNCTIONS.keys())
    all_acc_only_eval_protocols = [
        k
        for k, v in EVALUATION_FUNCTIONS.items()
        if v.__annotations__["return"] is float
    ]
    output_results = []
    for model in all_models:
        for task in all_tasks_or_clusters:
            logger.info(f"Evaluation scores for {result_level}={task} on model={model}")
            tmp = {result_level: task, "model_name": model}
            for protocol in all_eval_protocols:
                if result_level == "task":
                    details = (task_to_cluster[task], protocol, task, model)
                else:
                    details = (task, protocol, model)
                score = result.get(details, None)
                logger.info(f"\t {protocol}={score}")
                if protocol not in all_acc_only_eval_protocols:
                    if type(score) is tuple:
                        tmp[protocol + "_acc"] = round(score[0], 1)
                        tmp[protocol + "_failure_rate"] = round(score[1], 3)
                    else:
                        assert score is None
                        tmp[protocol + "_acc"] = ""
                        tmp[protocol + "_failure_rate"] = ""
                else:
                    if score is None:
                        tmp[protocol] = ""
                    else:
                        tmp[protocol] = round(score, 1)
            output_results.append(tmp)

    if result_level == "cluster":
        # Currently keeping track of cluster-level results only to
        # decide what has been already evaluated. #TODO: missing task-level tracking
        output_json_file = output_path + ".raw.jsonl"
        logger.info(f"Saving the raw results to {output_json_file}")
        with open(output_json_file, "w" if overwrite_results else "a") as f_json:
            for res in output_results:
                f_json.write(json.dumps(res))
                f_json.write("\n")

    logger.info(f"Saving the {result_level}-level benchmark scores to {output_path}")
    df = DataFrame(output_results)
    df.to_csv(output_path, sep="\t", mode="w" if overwrite_results else "a")


def compute_reliability_benchmark_scores(args):
    """
    Computing the Reliabiity Benchmark scores based on all the results files
    """

    # build results filters
    result_files = get_results_files(args.input_dirs)

    logger.info("{} result files found!".format(len(result_files)))
    if len(result_files) == 0:
        sys.exit(1)

    # Collect results results
    output_json_file = args.output.replace(".tsv", "_cluster.tsv") + ".raw.jsonl"
    append_results = (not args.overwrite_output) and os.path.exists(output_json_file)

    existing_results = []
    if append_results:
        # Further filter results based on what clusters already got evaluated
        logger.info(f"Reading existing results from {output_json_file}.")
        with open(output_json_file, mode="r") as f_json:
            for line in f_json:
                line = line.strip()
                res_item = json.loads(line)
                existing_results.append(res_item)

    results = get_predictions_from_files(result_files)

    results = filter_results(
        results,
        tasks=None if len(args.tasks) == 0 else args.tasks,
        eval_protocols=None if len(args.eval_protocols) == 0 else args.eval_protocols,
        model_names=None if len(args.model_names) == 0 else args.model_names,
        existing_results=existing_results,
    )

    logger.info("{} results found after filtering!".format(len(results)))

    per_task_benchmark_scores = calculate_per_task_benchmark_scores(results)

    logger.info("Task level benchmark scores...")
    print_and_save_per_model_benchmark_scores(
        per_task_benchmark_scores,
        output_path=args.output,
        overwrite_results=(not append_results),
    )

    per_cluster_benchmark_scores = calculate_per_cluster_benchmark_scores(results)

    logger.info("Cluster level benchmark scores...")
    print_and_save_per_model_benchmark_scores(
        per_cluster_benchmark_scores,
        output_path=args.output.replace(".tsv", "_cluster.tsv"),
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Obtain the evaluation scores based on prediction files"
    )

    parser.add_argument(
        "-i",
        "--input-dirs",
        default=[
            "/private/home/rpasunuru/reliability_benchmark_results/*/*_results.json",  # for debug purposes
        ],
        nargs="+",
        help="List of directories with results. Can include * to expand multiple dirs. By default the script looks at the given directory and subdirectories +1 level.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="results.tsv",
        help="The path to an output tsv file. Cluster-level results are generated at OUTPUT_cluster.tsv",
    )

    parser.add_argument(
        "-t",
        "--tasks",
        default=[],
        nargs="+",
        help="List of individual tasks to include or groups of tasks (clusters).",
    )

    parser.add_argument(
        "-e",
        "--eval-protocols",
        default=[],
        nargs="+",
        help="List of protocols to evaluate",
    )

    parser.add_argument(
        "-m",
        "--model-names",
        default=[],
        nargs="+",
        help="List of model names to perform the evaluation.",
    )

    parser.add_argument(
        "--sep",
        default="\t",
        help="Output file separator.",
    )

    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite the existing output file. If not specified new results are appended.",
    )

    args = parser.parse_args()

    compute_reliability_benchmark_scores(args)
