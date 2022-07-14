import argparse
import logging
import pandas as pd
from pandas import DataFrame
import glob
import json
import numpy as np
from typing import Any, List, NamedTuple, Tuple, Dict, Optional, Union
import scipy.stats
import sys
from examples.few_shot.tasks import get_all_tasks
import time
import datetime
import pathlib
import re
from examples.few_shot.tasks_organization import old_tasks_settings_default_eval_sets
import os
import copy
import wandb
import random
import string
from collections import Counter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

WANDB_LOGGING_ENTITY = 'xlm-g'


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)): 
            return None

        return json.JSONEncoder.default(self, obj)


# Filter the rows that we want to evaluate
def filter_by(df, constraints):
    """Filter MultiIndex by sublevels.
       source: https://stackoverflow.com/questions/25224545/filtering-multiple-items-in-a-multi-index-python-panda-dataframe
    """
    constraints = {k: v if isinstance(v, list) else [v] for k,v in constraints.items() if v is not None}
    if len(constraints) == 0:
        return df
    
    indexer = [constraints[name] if name in constraints else slice(None)
               for name in df.index.names]
    return df.loc[tuple(indexer)] if len(df.shape) == 1 else df.loc[tuple(indexer),]

    
def timestamp_to_nice_datetime_str(timestamp):
    dt = datetime.datetime.fromtimestamp(timestamp)
    dt_str = dt.strftime("%Y-%m-%d-%H-%M-%S-%f")
    return dt_str


def read_jsonl_file(file_path):
    items = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            item = json.loads(line)
            items.append(item)

    return items

def read_json_file(file_path):
    res_json = None
    with open(file_path) as f:
        res_json = json.load(f)
    return res_json


def append_score(dict_obj, key_name, value):
    if key_name not in dict_obj:
        dict_obj[key_name] = []
    dict_obj[key_name].append(value)


def ppl(positional_scores):
    return np.exp(np.mean(np.negative(positional_scores)))


def softmax(scores):
    norm_scores = np.exp(scores)
    norm_scores = norm_scores / np.sum(norm_scores)
    return norm_scores

def print_results_readable(result_row, key_re = "*"):
    result_row = [(k,v) for k,v in result_row.items() if re.match(key_re, k)]
    result_row = sorted(result_row, key=lambda x: x[0])
    max_len = max([len(x[0]) for x in result_row])
    for key, val in result_row:
        if isinstance(val, dict) and "mean" in val:
            val = round(val["mean"], 4)
        print(f"{key.rjust(max_len)} = {val}")


def get_common_prefix_limit(item, reverse=False, meta_field="prompt_tokens"):
    common_prefix_limit = 0

    if meta_field not in item["candidates"][0][1]["meta"]:
        return common_prefix_limit

    cand_prompts_tokens = [
        lbl_cnd[1]["meta"][meta_field]
        for cnd_id, lbl_cnd in enumerate(item["candidates"])
    ]
    if reverse:
        cand_prompts_tokens = [list(reversed(x)) for x in cand_prompts_tokens]

    common_prefix_max = min([len(x) for x in cand_prompts_tokens])
    while True:
        cand_token_at_idx = [
            prompt[common_prefix_limit] for prompt in cand_prompts_tokens
        ]

        if not all([x == cand_token_at_idx[0] for x in cand_token_at_idx]):
            break

        if common_prefix_limit >= common_prefix_max - 1:
            break

        common_prefix_limit += 1
    return common_prefix_limit


def sum_full(**kwargs):
    positional_scores = kwargs["positional_scores"]

    return np.sum(positional_scores)

def mean_full(**kwargs):
    positional_scores = kwargs["positional_scores"]

    return np.mean(positional_scores)

def mean_suffix(**kwargs):
    start_token = kwargs["prefix_end"]
    positional_scores = kwargs["positional_scores"][start_token:]
    
    return np.mean(positional_scores)

def sum_charnorm(**kwargs):
    prompt_char_len = len(kwargs["prompt"])
    positional_scores = kwargs["positional_scores"]
    score = np.sum(positional_scores)/prompt_char_len
    
    return score


def calculate_values_from_predictions_file(
    pred_file_path,
    recalc_scores = True,
    prepare_scores_for_ensemble=True,
    calc_score_contrast = False,
    score_funcs=[("sum", sum_full), mean_full, ("mean", mean_suffix), sum_charnorm],
    use_only_suffix_for_scoring=False,
    calibration=True,
    return_per_item_values=True,
    return_items = False,
    max_items = 0,
):
    """Calculates some scores form a prediction file

    Args:
        pred_file_path ([type]):  The path to the prediciton file
        recalc_scores (bool, optional): If true, the scores are recalculated from positional_scores. This is slow! Defaults to True.
        prepare_scores_for_ensemble (bool, optional): Group existing scores for ensemble calculation. This is slow! Defaults to True.
        calc_score_contrast (bool, optional): Calculates the sum of the score for positive and negative examples and their difference. Defaults to False.
        score_funcs (list, optional): List of scoring functions to apply. Defaults to [("sum", sum_full), mean_full, ("mean", mean_suffix), sum_charnorm].
        use_only_suffix_for_scoring (bool, optional): Only the suffix after the common prefix is used for scoring. Defaults to False.
        calibration (bool, optional): Use calibration when available when calculating ppl and accuracy. Defaults to True.
        return_per_item_values (bool, optional): Returns a dictionary of new values with the scores per item. This way an aggregation can be applied. Defaults to True.
        return_items (bool, optional): Returns a list of items as loaded by the prediction file. This can be used for debugging examples. Defaults to False.
        max_items (int, optional): Maximum items to read. This is used for debug Defaults to 0.

    Returns:
        [type]: [description]
    """
    correct = {x: [] for x in range(len(score_funcs))}
    correct_calib = {x: [] for x in range(len(score_funcs))}

    per_item_values = {}
    if return_items:
        items = []

    res_values, agg_values = {}, {}
    with open(pred_file_path) as f_in:
        for line_id, line in enumerate(f_in):
            if max_items > 0 and line_id >= max_items:
                break

            if not calc_score_contrast and not recalc_scores and not prepare_scores_for_ensemble:
                continue

            line = line.strip()

            item = json.loads(line)
            if return_items:
                items.append(item)

            # get gold answer - this is used for ensemble and calulating accuracies below
            cnd_gold = [cnd["gold"] for lbl, cnd in item["candidates"]]
            answer = np.argmax(np.array(cnd_gold))
            append_score(per_item_values, "gold_id", answer)

            if "meta" not in item["candidates"][0][1] or item["candidates"][0][1]["meta"] is None:
                continue

            if "score_raw" in item["candidates"][0][1]["meta"]:
                if calc_score_contrast or prepare_scores_for_ensemble: 
                    scores_raw = np.array(
                        [cnd["meta"]["score_raw"] for lbl, cnd in item["candidates"]]
                    )
                    append_score(per_item_values, "score_raw", scores_raw)

                if calc_score_contrast:
                    # add the contrast between positive and negative
                    score_raw_gold_correct_mean = np.mean([x[1] for x in zip(cnd_gold, scores_raw) if x[0] ])
                    score_raw_gold_incorrect_mean = np.mean([x[1] for x in zip(cnd_gold, scores_raw) if not x[0]])
                    
                    append_score(per_item_values, "score_raw_gold_correct_mean", score_raw_gold_correct_mean)
                    append_score(per_item_values, "score_raw_gold_incorrect_mean", score_raw_gold_incorrect_mean)
                    append_score(per_item_values, "score_raw_gold_correct_diff_incorrect", score_raw_gold_correct_mean - score_raw_gold_incorrect_mean)

            # score from the run - this can be different than score_raw if calibration is used
            scores_run = np.array(
                [cnd["score"] for lbl, cnd in item["candidates"]]
            )
            append_score(per_item_values, "score", scores_run)

            if not recalc_scores or ("prompt_tokens" not in item["candidates"][0][1]["meta"]) or (
                "positional_scores" not in item["candidates"][0][1]["meta"]
            ):
                continue

            common_prefix_end = get_common_prefix_limit(item)
            start_token = 0 if not use_only_suffix_for_scoring else common_prefix_end
            end_token = 0  # if not use_only_suffix_for_scoring else (get_common_prefix_limit(item, reverse=True))

            # calc predictions
            cands_info = [
                {
                    "positional_scores": np.array(cnd["meta"]["positional_scores"]),
                    "prompt": item["candidates"][0][1]["meta"].get("prompt", "*" * len(cnd["meta"]["positional_scores"])),
                    "prefix_end": common_prefix_end
                }
                for lbl, cnd in item["candidates"]
            ]

            # calculate number of tokens
            tokens_cnt = sum([
                len(cnd["meta"]["positional_scores"]) for lbl, cnd in item["candidates"]
            ])
            cands_cnt = len(item["candidates"])
            tokens_cnt_with_common_prefix = tokens_cnt - (cands_cnt - 1) * common_prefix_end
            tokens_per_cand = float(tokens_cnt) / cands_cnt

            append_score(
                    per_item_values, "prompt_cands_cnt", cands_cnt
                )
            append_score(
                    per_item_values, "prompt_tokens_full_cnt", tokens_cnt
            )
            append_score(
                    per_item_values, "prompt_tokens_common_prefix_calc_cnt", tokens_cnt_with_common_prefix
            )

            # check if prediction is correct
            calib_token_scores = None
            if calibration and "calib_metas" in item["candidates"][0][1]["meta"]:
                calib_token_scores = [
                    np.array(cnd["meta"]["calib_metas"][0]["positional_scores"])
                    for lbl, cnd in item["candidates"]
                ]

                cands_calib_info = [
                    {
                        "positional_scores": np.array(cnd["meta"]["calib_metas"][0]["positional_scores"]),
                        "prompt": cnd["meta"]["calib_metas"][0].get("prompt", "*" * len(cnd["meta"]["positional_scores"])),
                        "prefix_end": 0
                    }
                    for lbl, cnd in item["candidates"]
                ]
                calib_ppl = [ppl(ps) for ps in calib_token_scores]
                append_score(
                    per_item_values, "ppl_calib_prompt_gold", calib_ppl[answer]
                )
                append_score(
                    per_item_values,
                    "ppl_calib_prompts_cands_mean",
                    np.mean(calib_ppl),
                )
                append_score(
                    per_item_values,
                    "ppl_calib_prompts_cands_std",
                    np.std(calib_ppl),
                )
                append_score(
                    per_item_values,
                    "ppl_calib_prompts_cands_max",
                    np.max(calib_ppl),
                )
                append_score(
                    per_item_values,
                    "ppl_calib_prompts_cands_min",
                    np.min(calib_ppl),
                )

            for sc_id, score_func in enumerate(score_funcs):
                if isinstance(score_func, tuple):
                    score_func_name, score_func = score_func
                else:
                    score_func_name = score_func.__name__
                
                cnd_scores = np.array([score_func(**x) for x in cands_info])
                if calibration and calib_token_scores is not None:
                    cnd_calib_scores = (
                        np.array([score_func(**x) for x in cands_calib_info])
                    )
                    cnd_calibrated = cnd_scores - cnd_calib_scores
                    calib_selected = np.argmax(cnd_calibrated)

                    # add accuracy
                    val_key = "accuracy_{0}_calib".format(score_func_name)
                    append_score(per_item_values, val_key, calib_selected == answer)

                selected = np.argmax(cnd_scores)
                val_key = "accuracy_{0}".format(score_func_name)
                append_score(per_item_values, val_key, selected == answer)

        agg_values["eval_examples_cnt"] = line_id + 1


    # aggregate the results
    for key_name, val in per_item_values.items():
        if key_name.startswith("ppl_"):
            agg_values[key_name + "_mean"] = np.mean(val) if len(val) > 0 else 0.0
            agg_values[key_name + "_std"] = np.std(val) if len(val) > 0 else 0.0
        elif key_name.startswith("accuracy_"):
            agg_values[key_name] = 100 * np.mean(val) if len(val) > 0 else 0.0
        elif key_name.startswith("prompt_"):
            agg_values[key_name] = np.sum(val) if len(val) > 0 else 0.0
            agg_values[key_name + "_mean"] = np.sum(val) if len(val) > 0 else 0.0

    return_vals = [agg_values]
    if return_per_item_values:
        return_vals.append(per_item_values)
    if return_items:
        return_vals.append(items)

    return return_vals[0] if len(return_vals) == 1 else tuple(return_vals)


def aggregate_mean(scores, normalize_func=softmax):
    aggr_score = None
    for sc in scores:
        if aggr_score is None:
            aggr_score = normalize_func(sc) if normalize_func is not None else sc
        else:
            aggr_score += normalize_func(sc) if normalize_func is not None else sc
    aggr_score = aggr_score / len(scores)

    return aggr_score


def ensemble_scores(per_file_scores, scores_field="score_raw", strategy="mean"):
    assert strategy in ["mean"]
    accumulated_scores = []

    scores_raw = [x[scores_field] for x in per_file_scores]
    for i in range(len(scores_raw)):
        for j in range(len(scores_raw[0])):
            if len(accumulated_scores) < j + 1:
                accumulated_scores.append([scores_raw[i][j]])
            else:
                accumulated_scores[j].append(scores_raw[i][j])

    ensembled_scores = [aggregate_mean(x) for x in accumulated_scores]

    return ensembled_scores


def calc_accuracy(per_file_scores, scores_field="score_raw", gold_field="gold_id"):
    scores_raw = [x[scores_field] for x in per_file_scores]
    gold_answers = [x[gold_field] for x in per_file_scores]

    accuracies = []
    for i in range(len(scores_raw)):
        gold = gold_answers[i]
        predicted = [np.argmax(x) for x in scores_raw[i]]

        acc = 100 * np.mean([float(x == y) for x, y in zip(predicted, gold)])
        accuracies.append(acc)

    return accuracies


def get_predictions_gold_probability_for_results_file(results_file:str, 
        soft_scores:bool,  
        normalize_scores=True,  # normalize_scores
        ensemble:str=None
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
        predictions_file = results_file.replace("_results.json", f"_seed{seed}_predictions.jsonl")
        if not os.path.exists(predictions_file):
            if seed == 0:
                raise FileNotFoundError(f"Predictions file {predictions_file} is not found! It should be located together with the results file {results_file}. "
                                        " Make sure that you ran the evaluation with output predictions by setting the results_dir of the evaluation script!")
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
            predictions_scores_ensemble = np.sum(predictions_scores_by_seed_array, axis=0) / num_prediction_runs
            predictions_scores_ensemble = list(predictions_scores_ensemble)
            predictions_scores_ensemble = [np.array(x) for x in predictions_scores_ensemble]  # List[array] is expected

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

    
def get_mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    a = 1.0 * data
    a = a[~np.isnan(a)]
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


def aggregate_metrics(metric2scores):
    result_row = {}
    for metric, scores in metric2scores.items():
        scores_safe = [x if x is not None else np.nan for x in scores]
        result_row[metric] = {
            "scores": scores,
            "mean": np.mean(scores_safe),
            "std": np.std(scores_safe),
            "mean_confidence_interval": get_mean_confidence_interval(scores_safe),
        }

    return result_row


def flat_results_values(json_dict):
    flat_json = {}
    sep = "::"
    for k, v in json_dict.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat_json[sep.join([k, kk])] = vv
        else:
            flat_json[k] = v
        if k == 'accuracy':
            for i in range(len(json_dict[k]["scores"])):
                flat_json[f'{k}_{i}'] = json_dict[k]['scores'][i]
    return flat_json


# The first metric that is found in this mapping is used for display
preferred_metrics_by_task = {"multirc" : [
    "dataset_F1",
    "question_F1",
                                        "question_P",
                                        "question_R",
                                        "dataset_P",
                                        "dataset_R",],
                        "copa": ["accuracy", "accuracy_sum"],
                        "cb": ["accuracy",
                               "macro_F1",
                               "accuracy",
                               "macro_P",
                               "macro_R",
                               "micro_F1",
                               "micro_P",
                               "micro_R",],
                        "boolq": ["accuracy"],
                        "rte": ["accuracy"],
                        "xcopa": ["accuracy","accuracy_sum", ],
                        "arceasy": ["accuracy","accuracy_sum", ],
                        "arcchallenge": ["accuracy", "accuracy_sum_calib", ],
                        "openbookqa": ["accuracy", "accuracy_sum_calib",],
                        "commonsenseqa": ["accuracy","accuracy_sum"],
                        "exams": [ "accuracy","accuracy_mean"],
                        "naturalquestions": ["exact-match"],
                        "triviaqa": ["exact-match"],
                        "webquestions": ["exact-match"],
                        "wmt14enfr": ["bleu"],
                        "wmt14fren": ["bleu"],
                        "wmt16deen": ["bleu"],
                        "wmt16ende": ["bleu"],
                        "wmt16enro": ["bleu"],
                        "wmt16roen": ["bleu"],
                        "other":["accuracy"],
                        "stereoset":
                            [
                                "macro_icat_overall",
                                "macro_icat_religion",
                                "macro_icat_race",
                                "macro_icat_profession",
                                "macro_icat_gender",
                                "lms_overall",
                                "lms_religion",
                                "lms_race",
                                "lms_profession",
                                "lms_gender",
                                "ss_overall",
                                "ss_religion",
                                "ss_race",
                                "ss_profession",
                                "ss_gender",
                            ],
                        "realtoxicityprompts":[
                            "rtp_prompt_osm",
                            "rtp_prompt_olc",
                            "rtp_generation_osm",
                            "rtp_generation_olc",
                            "rtp_combined_osm",
                            "rtp_combined_olc",
                        ],
                        "crowspairs": ['overall_stereotype_score', 'age_anti-stereotype_score', 'age_metric_score', 'age_neutral_percent', 'age_num_neutral', 'age_stereotype_score', 'age_total_samples', 'disability_anti-stereotype_score', 'disability_metric_score', 'disability_neutral_percent', 'disability_num_neutral', 'disability_stereotype_score', 'disability_total_samples', 'execution_time', 'gender_anti-stereotype_score', 'gender_metric_score', 'gender_neutral_percent', 'gender_num_neutral', 'gender_stereotype_score', 'gender_total_samples', 'nationality_anti-stereotype_score', 'nationality_metric_score', 'nationality_neutral_percent', 'nationality_num_neutral', 'nationality_stereotype_score', 'nationality_total_samples', 'overall_anti-stereotype_score', 'overall_metric_score', 'overall_neutral_percent', 'overall_num_neutral', 'overall_total_samples', 'physical-appearance_anti-stereotype_score', 'physical-appearance_metric_score', 'physical-appearance_neutral_percent', 'physical-appearance_num_neutral', 'physical-appearance_stereotype_score', 'physical-appearance_total_samples', 'ppl_candidates', 'ppl_candidates_full_prompt', 'ppl_candidates_full_prompt__sent_less', 'ppl_candidates_full_prompt__sent_more', 'ppl_common_prefix', 'ppl_full_selected_candidate', 'ppl_selected_candidate', 'race-color_anti-stereotype_score', 'race-color_metric_score', 'race-color_neutral_percent', 'race-color_num_neutral', 'race-color_stereotype_score', 'race-color_total_samples', 'religion_anti-stereotype_score', 'religion_metric_score', 'religion_neutral_percent', 'religion_num_neutral', 'religion_stereotype_score', 'religion_total_samples', 'sexual-orientation_anti-stereotype_score', 'sexual-orientation_metric_score', 'sexual-orientation_neutral_percent', 'sexual-orientation_num_neutral', 'sexual-orientation_stereotype_score', 'sexual-orientation_total_samples', 'socioeconomic_anti-stereotype_score', 'socioeconomic_metric_score', 'socioeconomic_neutral_percent', 'socioeconomic_num_neutral', 'socioeconomic_stereotype_score', 'socioeconomic_total_samples'],
                        "lama_conceptnet": ["precision@1", "precision@10", "mrr"],
                        "lama_googlere": ["precision@1", "precision@10", "mrr"],
                        "lama_squad": ["precision@1", "precision@10", "mrr"],
                        "lama_trex": ["precision@1", "precision@10", "mrr"],
                        "mlama_googlere": ["precision@1"],
                        "mlama_trex": ["precision@1"],
                        "pawsx": ["accuracy", "auc_pr"],
                        "gluediag": ["r3__lexical*", "r3__predicate*", "r3__logic*", "r3__knowledge*"],
                        "xnli": ["accuracy",],
                        "compositional_instructions_classification": [
                            "accuracy_comp_hard",
                            "accuracy_full_task_only", 
                            "accuracy_subtasks_micro",
                            "accuracy_comp_soft",],
                        "cic_v1_1_comp_subtasks": [
                            "accuracy_comp_hard",
                            "accuracy_full_task_only", 
                            "accuracy_subtasks_micro",
                            "accuracy_comp_soft",],
                       }


def view_by_run_params_preferred_metric(df):
    value_cols = ["_metric_val", "_metric_val_std"]
    cols = ['model_name']
    index_cols = ["task", "eval_set", "language", "train_set", "train_lang", "template", "nb_few_shot_samples", "_metric", "calibration", "run_params::scoring"]
    pt = pd.pivot_table(df, values=value_cols, index=index_cols,
                    columns=cols, aggfunc=np.mean)
    pt = pt.swaplevel(0, 1, axis=1).sort_index(axis=1)

    return pt

def view_by_run_params_preferred_metric_ext(df):
    value_cols = ["_metric_val", "_metric_val_std"]
    cols = ['model_name']
    index_cols = ["task", "eval_set", "language", "train_set", "train_lang", "template", "nb_few_shot_samples", "_metric", "calibration", "run_params::scoring", "run_params::train_sep", "run_params::max_tokens"]
    pt = pd.pivot_table(df, values=value_cols, index=index_cols,
                    columns=cols, aggfunc=np.mean)
    pt = pt.swaplevel(0, 1, axis=1).sort_index(axis=1)

    return pt

def view_by_task_preferred_metric(df):
    pt = pd.pivot_table(df, values=["_metric_val", "_metric_val_std"], index=["task", "eval_set", "language", "train_set", "train_lang", "template", "nb_few_shot_samples", "_metric", "calibration", "run_params::scoring"],
                    columns=['model_name',
                            ], aggfunc=np.mean)

    return pt

def view_preferred_metrics_mean(df):
    run_columns = ["task", "eval_set", "language", "train_set", "train_lang", "template", "nb_few_shot_samples", "calibration", "run_params::scoring", "model_name"]
    suffixes = ["::mean"]

    return view_preferred_metrics(df, run_columns, suffixes)


def view_preferred_metrics(df, run_columns = ["task", "eval_set", "language", "train_set", "train_lang", "template", "nb_few_shot_samples", "calibration", "run_params::scoring", "model_name", "execution_time::mean"], suffixes = ["::mean", "::std"]):
    curr_tasks = list(set(df["task"]))
    all_columns = set(df.columns)

    # get all preferred metrics
    all_preferred_metrics = []
    for t in curr_tasks:
        t_metrics = preferred_metrics_by_task.get(t, [])

        for m in t_metrics:
            # If metric is a pattern to look for
            if "*" in m:
            #TODO can be made better with regex
                m = m.replace("*","")
                matching_m = []
                for col in all_columns:
                    if m in col:
                        matching_m.append(col.split("::")[0])
                # Remove repetitions
                m = set(matching_m)
            else:
                m = [m]

            for suff in suffixes:
                m_col =  [mc + suff for mc in m]
                for mc in m_col:
                    if mc in all_columns:
                        all_preferred_metrics.append(mc)

    all_preferred_metrics = list(set(all_preferred_metrics))
    all_preferred_metrics.sort()
    pt = df[run_columns + all_preferred_metrics]

    return pt


def view_multilingual_paper_main_results(
        df, 
        run_columns=[
            "task", 
            "eval_set", 
            "eval_examples_cnt::mean", 
            "language", 
            "train_set", 
            "train_lang", 
            "template", 
            "nb_few_shot_samples", 
            "nb_trunc_few_shot_samples::mean", 
            "calibration", 
            "run_params::scoring", 
            "model_name"
        ], suffixes=["::mean", "::std"]
        ):
    curr_tasks = list(set(df["task"]))
    all_columns = set(df.columns)

    # get all preferred metrics
    all_preferred_metrics = []
    for t in curr_tasks:
        t_metrics = preferred_metrics_by_task.get(t, [])

        for m in t_metrics:
            # If metric is a pattern to look for
            if "*" in m:
            #TODO can be made better with regex
                m = m.replace("*","")
                matching_m = []
                for col in all_columns:
                    if m in col:
                        matching_m.append(col.split("::")[0])
                # Remove repetitions
                m = set(matching_m)
            else:
                m = [m]

            for suff in suffixes:
                m_col =  [mc + suff for mc in m]
                for mc in m_col:
                    if mc in all_columns:
                        all_preferred_metrics.append(mc)
                        # TODO: it's better to read number of trials from the result data frame; now we hard code it to 5.
                        num_trials = 5
                        for i in range(num_trials):
                            if f"{mc.split('::')[0]}_{i}" in all_columns:
                                all_preferred_metrics.append(f"{mc.split('::')[0]}_{i}")

    all_preferred_metrics = list(set(all_preferred_metrics))
    all_preferred_metrics.sort()
    pt = df[run_columns + all_preferred_metrics]

    return pt



display_views = {
    # key: (view function, description)
    "raw": (lambda x: x, "Raw results with all generated fields."),
    "preferred_metrics_mean": (view_preferred_metrics_mean, "Task, model, and preferred metrics with mean only"),
    "preferred_metrics": (view_preferred_metrics, "Base run columns and preferred metrics."),
    "by_run_params_simple": (view_by_run_params_preferred_metric, "Results grouped by run settings. Only preferred_metric with std is displayed."),
    "by_run_params_simple_ext": (view_by_run_params_preferred_metric_ext, "Results grouped by run settings, incuding train_sep. Only preferred_metric with std is displayed."),
    "multilingual_paper_main_results": (view_multilingual_paper_main_results, "Main result table of our multilingual few-shot learning paper.")
}

task_groups = {
    "superglue": ["boolq", "copa", "cb", "rte", "wic", "wsc", "record", "multirc"],
    "sciq": ['arcchallenge', 'arceasy', 'openbookqa', 'commonsenseqa', 'exams'],
    "rai": ['realtoxicityprompts', 'stereoset'],
    "natural_instructions": [x for x in get_all_tasks() if x.startswith("naturalinstructions__")]
}

def get_tasks_to_use(tasks):
    if tasks[0] == "all":
        return None
    else:
        tasks_to_use = []
        for task in tasks:
            if task in task_groups:
                tasks_to_use.extend(task_groups[task])
            else:
                tasks_to_use.append(task)

        return tasks_to_use

def postprocess_flat_fields(result_data_flat):
    """Postprocess the results fields
    Args:
        result_data_flat ([type]): Dictionary with flat results fields
    """
    # map old model names to the new one
    if result_data_flat["model_name"] == "openai_api_2.7B_ada":
        result_data_flat["model_name"] = "openai_ada"
    elif result_data_flat["model_name"] == "dense_6.7B":
        result_data_flat["model_name"] = "6.7B_gpt3_setting_1024ctx"
    elif result_data_flat["model_name"] == "moe_500B_300b_tokens":
        result_data_flat["model_name"] = "moe_523B"

    # fix train sep display
    if "run_params::train_sep" not in result_data_flat:
        result_data_flat["run_params::train_sep"] = " "
    result_data_flat["run_params::train_sep"] = result_data_flat["run_params::train_sep"].replace("\n", "\\n")

    # add default values for old results files
    if "eval_set" not in result_data_flat:
        curr_lang = result_data_flat["language"]
        # this is an old results file
        old_task_settings = old_tasks_settings_default_eval_sets.get(result_data_flat["task"], {"eval_set": "unk", "train_set": "unk", "valid_set": None},)
        old_task_settings.update({
            "train_lang": None if old_task_settings["train_set"] is None else curr_lang,
            "valid_lang": None if old_task_settings["valid_set"] is None else curr_lang
        })
        result_data_flat.update(old_task_settings)

def read_results_file(file_path):
    res_json = None
    with open(file_path) as f:
        res_json = json.load(f)
    return res_json


def np_encoder(object):
    """
    Make numpy types serializable.
    """
    if isinstance(object, np.generic):
        return object.item()


def get_results_files(input_dirs, filter_suffix="/*_results.json"):
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]
    full_input_dirs = []
    for idr in input_dirs:
        if idr.endswith(".txt") and os.path.isfile(idr):
            # we support selecting .txt files with list of relevant dirs as newlines
            with open(idr, mode="r") as dl_f:
                for line in dl_f:
                    full_input_dirs.append(line.strip())
        else:
            full_input_dirs.append(idr)

    result_files = []
    for input_dir in full_input_dirs:
        curr_dir_res_files = get_results_files_for_single_filter(input_dir, filter_suffix)
        if len(curr_dir_res_files) == 0:
            curr_dir_res_files = get_results_files_for_single_filter(input_dir.rstrip("/") + "/*", filter_suffix)

        if len(curr_dir_res_files) == 0:
            logger.warning(f"{input_dir} does not have any results files at depth (+0/1)!")

        result_files.extend(curr_dir_res_files)
    return result_files


def get_results_files_for_single_filter(input_dir, filter_suffix="/*_results.json"):
    result_files_filter = (
        input_dir
        if input_dir.endswith("results.json")
        else input_dir.rstrip("/") + filter_suffix
    )

    curr_dir_res_files = []
    for results_file in glob.glob(result_files_filter, recursive="**" in result_files_filter):
        curr_dir_res_files.append(results_file)

    return curr_dir_res_files


def add_preferred_metrics(results, preferred_metrics_by_task):
    # select preferred metric
    for res_data in results:
        task_name = res_data["task"]
        task_pref_metrics = preferred_metrics_by_task.get(task_name, preferred_metrics_by_task.get("other"))

        # Try to get the task preferred metrics
        # This is done to allow minimal display fields
        for tm in task_pref_metrics:
            mean_metric = f"{tm}::mean"
            std_metric = f"{tm}::std"

            if mean_metric not in res_data:
                continue

            val_mean = res_data.get(mean_metric, None)
            val_std = res_data.get(std_metric, None)

            res_data["_metric"] = tm
            res_data["_metric_val"] = val_mean
            res_data["_metric_val_std"] = val_std

            break  # select only the first found metric


def get_prediction_files(res_file):
    pred_files = []
    for pred_file in glob.glob(
        res_file.replace("_results.json", "*_predictions.jsonl")
    ):
        pred_files.append(pred_file)

    return pred_files


def get_checkpoint_step_form_modelname(model_name):
    """Gets the step from model name with step info.
    Some of the models are representing checkpoints and their config name is 
    something like "1.3B_gpt3_seting__step00012345" (See model_config.py)

    Args:
        model_name (str): Model name

    Returns:
        int: -1 if no step number is found, the step number otherwise.
    """
    step_search = re.search("__step(\d+)", model_name)
    if step_search:
        return int(step_search.group(1))
    else:
        return -1


def collect_results_from_files(result_files, recalc_from_positional_scores, existing_results=None, calc_ensemble=False):
    """Collects results and postprocess.

    Args:
        result_files ([type]): List of files.
        recalc_from_positional_scores: Should recalc metrics from predicitons files. This might be slow for long files.
        existing_results (List[Dict]): List of available results
        calc_ensemble (bool): calculates ensemble accuracy score if more than one prediction files are available. 

    Returns:
        List[Dict]: New results.
    """
    existing_results_keys = {x["results_file"] for x in existing_results if "results_file" in x}
    results = []
    for f_id, res_file in enumerate(result_files):
        if res_file in existing_results_keys:
            # logging.info(f"Skipping {f_id+1}/{len(result_files)} - {res_file} - already processed")    
            continue
        logging.info(f"Processing {f_id+1}/{len(result_files)} - {res_file}")

        try:
            result_data = read_json_file(res_file)
        except Exception as e:
            logger.exception(f"Error reading {res_file}:", e)
            logger.warning(f"Skipping...")
            continue
        result_data["results_file"] = res_file
    
        # get file creation time -- to be able to sort by recency
        file_info = pathlib.Path(res_file)
        file_stats = file_info.stat()
        result_data["results_file_ctime"] = timestamp_to_nice_datetime_str(file_stats.st_ctime)
        result_data["results_file_mtime"] = timestamp_to_nice_datetime_str(file_stats.st_mtime)

        # additional columns
        result_data["calibration"] = (len(result_data.get("calibration_options", [])) > 0)
        result_data["checkpoint_steps"] = get_checkpoint_step_form_modelname(result_data["model_name"])
        # fix problem with candidates ppl
        result_data = {k:v for k,v in result_data.items() if "(" not in k}

        # get predictions files
        pred_files = get_prediction_files(res_file)

        if "eval_examples_cnt::mean" not in result_data and len(pred_files) > 0:
            with open(pred_files[0]) as f_pred:
                preds_cnt = len([1 for _ in f_pred])
                result_data["eval_examples_cnt::mean"] = preds_cnt

            
        use_pred_files = recalc_from_positional_scores or calc_ensemble
        if use_pred_files and len(pred_files) > 0:
            per_file_per_item_values = []
            keep_per_item_fields = ["gold_id", "score_raw", "score"]

            pred_file_metric_values = {}

            for pf_id, pred_file in enumerate(pred_files):
                logging.info(f"    {pf_id+1}/{len(pred_files)} - {pred_file}")

                (
                    calculated_values,
                    per_item_values,
                ) = calculate_values_from_predictions_file(pred_file, recalc_scores=recalc_from_positional_scores, prepare_scores_for_ensemble=calc_ensemble)

                # collect per file calculated values
                for k, v in calculated_values.items():
                    append_score(pred_file_metric_values, k, v)

                # collect scores for ensemble
                per_item_values_selected = {
                    fld: per_item_values[fld] for fld in keep_per_item_fields
                }
                per_file_per_item_values.append(per_item_values_selected)

            aggr_pred_file_metric_values = flat_results_values(
                aggregate_metrics(pred_file_metric_values)
            )
            result_data.update(aggr_pred_file_metric_values)

            # calc accuracy from predictions file
            for score_field, field_display in [("score_raw", "_raw"), ("score", "run")]:
                # mean
                accuracy_curr = aggregate_metrics({f"accuracy_{field_display}": calc_accuracy(per_file_per_item_values, score_field, "gold_id")})
                result_data.update(accuracy_curr)

                # ensemble
                if len(per_file_per_item_values) > 1:
                    # more than one run for the same task

                    ensembled_scores = ensemble_scores(
                        per_file_per_item_values, score_field
                    )
                    ensembled_predictions = [np.argmax(x) for x in ensembled_scores]
                    gold_ids = per_file_per_item_values[0]["gold_id"]

                    result_data[f"accuracy_{field_display}_ensemble"] = np.mean(
                        [float(x == y) for x, y in zip(ensembled_predictions, gold_ids)]
                    )

        result_data_flat = flat_results_values(result_data)
        postprocess_flat_fields(result_data_flat)
        results.append(result_data_flat)

    add_preferred_metrics(results, preferred_metrics_by_task)

    return results


def print_json_structure(json_dict, depth=0,
                         print_field_value=True, field_max_value=-1):
    """Print the structure of a json object for inspection"""
    if isinstance(json_dict, dict):
        for k,v in json_dict.items():
            display_value = None
            if print_field_value:
                if isinstance(v, str):
                    display_value = str(v)
                    if field_max_value > 0 and len(display_value) > field_max_value:
                        display_value = display_value[:field_max_value] + "..."
                elif isinstance(v, int) or isinstance(v, float):
                    display_value = v

                if display_value is not None:
                    display_value = f" = '{display_value}'"
                else:
                    display_value = ""

            if isinstance(v, list):
                display_value = " ({0})".format(len(v))
            print("  " * depth + f"{k}: {type(v).__name__}{display_value}")
            print_json_structure(v, depth=depth+1)
    elif isinstance(json_dict, list):
        if len(json_dict) > 0:
            print_json_structure({"item[0]": json_dict[0]}, depth=depth+1, print_field_value=print_field_value, field_max_value=field_max_value)


def get_first_candidates_prompts(prediction):
    printed_strs = []

    for cand, cand_info in prediction["candidates"]:
        printed_strs.append(f"### Prompt for cand `{cand}`:")
        printed_strs.append(cand_info["meta"]["prompt"])
        
        printed_strs.append("#" * 10)

        if "calib_metas" in cand_info["meta"]:
            printed_strs.append(f"### Calibrations prompts for cand `{cand}`:")
            for calib_id, calib_meta in enumerate(cand_info["meta"]["calib_metas"]):
                printed_strs.append(f"## calib option {calib_id} prompt:")
                printed_strs.append(calib_meta["prompt"])
        
        break  # print the first only

    return "\n".join(printed_strs)


def get_first_pred_prompts_example(file_path):
    if file_path.endswith("_results.json"):
        file_path = file_path.replace("_results.json", "_seed0_predictions.jsonl")
    
    with open(file_path) as f_pred:
        for line in f_pred:
            item = json.loads(line.strip())
            
            return get_first_candidates_prompts(item)



def load_results(results_json_file, task_to_groups=None, 
                 filter_item=None,
                 postprocess=None
                ):
    
    # This will copy the task results and add them as "group" results. 
    expand_task_groups = task_to_groups is not None and len(task_to_groups) > 0
  
    results = []
    expanded_results = []

    current_tasks_to_groups = {}
    with open(results_json_file) as f_res:
        for line in f_res:
            line = line.strip()
            item = json.loads(line)
            for k in list(item.keys()):
                if k.startswith("ppl_candidates_full_prompt__") or k.startswith("ppl_calib_candidates_full_prompt__"):
                    del item[k]

            current_task_name = item["task"]

            if postprocess is not None:
                postprocess(item)

            if "checkpoint_steps" not in item: 
                item["checkpoint_steps"] = get_checkpoint_step_form_modelname(item["model_name"])

            if filter_item is not None and filter_item(item):
                continue
            
            if 'eval_examples_cnt' not in item:
                item['eval_examples_cnt'] = int(item.get('eval_examples_cnt::mean', item.get('run_params::n_eval_samples',0)))

            if "macro_F1::scores" in item:
                item["macro_F1::max"] = max(item["macro_F1::scores"])
                item["macro_F1::min"] = max(item["macro_F1::scores"])
            if "accuracy::scores" in item:
                item["accuracy::max"] = max(item["accuracy::scores"])
                item["accuracy::min"] = max(item["accuracy::scores"])
            results.append(item)

            
            if expand_task_groups:
                task_groups = task_to_groups.get(current_task_name, [])
                for tg in task_groups:
                    group_result = copy.deepcopy(item)
                    group_result["task"] = tg
                    group_result["template"] = "*"
                    expanded_results.append(group_result)
                if current_task_name not in current_tasks_to_groups:
                    current_tasks_to_groups[current_task_name] = task_groups
    
    return results, expanded_results, current_tasks_to_groups 


def run_few_shot_results_aggregation(args, existing_results=[]):
    """
    Aggregate results from multiple few-shot learning trials and save 
    """
    # read existing results
    output_json_file = args.output + ".raw.jsonl"
    if existing_results is None or len(existing_results) == 0:
        existing_results = []
        logger.info(f"Reading existing results from {output_json_file}.")
        with open(output_json_file, mode="r") as f_json:
            for line in f_json:
                line = line.strip()
                res_item = json.loads(line)
                existing_results.append(res_item)

    # TODO: we hard code number of trials to 5; it is better to read this from the experiment log
    num_trials = 5
    for results in existing_results:
        # TODO: the preferred metrics may not always be accuracy; a more general implementation
        #   would be to select the preferred metrics
        results_file = results['results_file']      
        with open(results_file) as f:
            results = json.load(f)
        if results["nb_few_shot_samples"] > 0 and \
                (('accuracy' not in results or len(results['accuracy']['scores']) < num_trials)):
            print(f"Recalculating few shot results for {results_file}")
            with open(results_file) as f:
                result_data = json.load(f)
            pred_file_metric_values, per_file_per_item_values = {}, []
            keep_per_item_fields = ["gold_id", "score_raw", "score"]

            for in_jsonl in get_prediction_files(results_file):
                (
                    calculated_values,
                    per_item_values,
                )  = calculate_values_from_predictions_file(in_jsonl)

                # collect per file calculated values
                for k, v in calculated_values.items():
                    append_score(pred_file_metric_values, k, v)

                # collect scores for ensemble
                per_item_values_selected = {
                    fld: per_item_values[fld] for fld in keep_per_item_fields
                }
                per_file_per_item_values.append(per_item_values_selected)
                print(calculated_values)

            aggr_pred_file_metric_values = flat_results_values(
                aggregate_metrics(pred_file_metric_values)
            )
            result_data.update(aggr_pred_file_metric_values)

            # calc accuracy from predictions file
            for score_field, field_display in [("score_raw", "_raw"), ("score", "run")]:
                # nean
                accuracy_curr = aggregate_metrics({
                    f"accuracy_{field_display}": calc_accuracy(per_file_per_item_values, score_field, "gold_id")
                })
                result_data.update(accuracy_curr)
                # collect scores for ensemble
                per_item_values_selected = {
                    fld: per_item_values[fld] for fld in keep_per_item_fields
                }
                per_file_per_item_values.append(per_item_values_selected)

                # ensemble
                if len(per_file_per_item_values) > 1:
                    # more than one run for the same task

                    ensembled_scores = ensemble_scores(
                        per_file_per_item_values, score_field
                    )
                    ensembled_predictions = [np.argmax(x) for x in ensembled_scores]
                    gold_ids = per_file_per_item_values[0]["gold_id"]

                    result_data[f"accuracy_{field_display}_ensemble"] = np.mean(
                        [float(x == y) for x, y in zip(ensembled_predictions, gold_ids)]
                    )

            for key in ['scores', 'mean', 'std', 'mean_confidence_interval']:
                result_data['accuracy'][key] = result_data[f"accuracy_{result_data['run_params']['scoring']}::{key}"]
            print(f"Saving to {results_file}")
            with open(results_file, 'w') as o_f:
                json.dump(result_data, o_f, cls=NumpyEncoder)
            print()


def run_results_collection(input_dirs,
            output,
            overwrite_output=False,
            recalc_metrics_from_positional_scores=False,
            calculate_ensemble=False,
            view=None,
            sep="\t",
            existing_results=[], loop_i=0) -> Tuple[List[Dict], List[Dict]]:
    """Running the results collection.
    Args:
        input_dirs (Union[str, List[str]]): List of directories to search the results at.
        output (str): The output .tsv file with results. A file {output}.raw.jsonl is also written with the structured aggregated results.. 
        overwrite_output (bool, optional): Overwrite the output files if exist. Defaults to False.
        recalc_metrics_from_positional_scores (bool, optional): Recalculate metrics from the predictions files. See `run_results_collection`. Defaults to False.
        calculate_ensemble (bool, optional): Calculates ensenble from multiple prediction files.. Defaults to False.
        view (str, optional): The name of the csv view. See `display_views` . Defaults to None.
        sep (str, optional): CSV separator. Defaults to "\t".
        existing_results (List[Any]): List of existing results. These are used to determine which results are already loaded.
        loop_i (int, optional): If we run this in a loop, loop_id is > 0. Defaults to 0.
    Returns:
        new_results (List[Dict[str, any]]): New results collected during this round.
        all_results (List[Dict[str, any]]): existing_results appended to new_results.
    """
    view_key = view
    recalc_from_positional_scores = recalc_metrics_from_positional_scores
    calculate_ensemble = calculate_ensemble

    # build results filters
    result_files = get_results_files(input_dirs)

    logger.info("{} result files found!".format(len(result_files)))
    if len(result_files) == 0:
        logger.info(f"No results found!")
        return ([], existing_results)

    # Collect results results
    output_json_file = output + ".raw.jsonl"
    
    append_results = (not overwrite_output or loop_i > 0) and os.path.exists(output_json_file)
    if append_results:
        # read existing results
        if existing_results is None or len(existing_results) == 0:
            existing_results = []
            logger.info(f"Reading existing results from {output_json_file}")
            with open(output_json_file, mode="r") as f_json:
                for line in f_json:
                    line = line.strip()
                    res_item = json.loads(line)
                    existing_results.append(res_item)

    append_results = append_results and len(existing_results) > 0 
    new_results = collect_results_from_files(
        result_files, 
        recalc_from_positional_scores, 
        existing_results=existing_results, 
        calc_ensemble=calculate_ensemble
    )

    if len(new_results) > 0:
        logger.info(f"Saving {len(new_results)} new results to {output_json_file}")
        # display results
        if append_results:
            with open(output_json_file, mode="a") as f_json:
                for res in new_results:
                    f_json.write(json.dumps(res, default=np_encoder))
                    f_json.write("\n")
        else:
            with open(output_json_file, mode="w") as f_json:
                for res in new_results:
                    f_json.write(json.dumps(res, default=np_encoder))
                    f_json.write("\n")

        logger.info(f"Results raw jsons saved to {output_json_file}.")
    else:
        logger.info(f"No new results!")

    results = existing_results
    if len(new_results) > 0:
        # Export to table
        results += new_results
        if len(results) > 10000:
            logger.warn(f"Skipping tsv export - {len(results)} are too many to save to a single tsv.")
            return (new_results, results)
        
        logger.info(f"Saving {len(new_results)} new results to tsv - {output}")
        results_df = DataFrame.from_records(results)
        results_df = results_df[[x for x in results_df.columns if not x.endswith("::scores")]]

        # Format view
        view_func = display_views[view_key][0]
        view_df = view_func(results_df)
        view_df.to_csv(output, sep=sep)
        logger.info(f"Results formatted with `{view_key}` view exported to {output}.")

    return (new_results, results)


def _filter_keys_for_loggin_to_wandb(
    result_key: str,
    task_name: str,
    only_log_preferred_metrics: bool,
) -> bool:
    if only_log_preferred_metrics:
        return (
            result_key in preferred_metrics_by_task.get(task_name, [])
            or
            result_key in ["_metric_val", "_metric_val_std"]
            or
            result_key.endswith('checkpoint_steps')
        ) 
    else:
        return not result_key.startswith('ppl_candidates_full_prompt')


def validate_wandb_login(wandb_project:str,
    wandb_run_name:str,
    wandb_run_id:Optional[str]=None,
    wandb_logging_entity:Optional[str]=None,
):
    wandb_run = wandb.init(
        project=wandb_project,
        entity=wandb_logging_entity,
        id=wandb_run_id,
        name=wandb_run_name,
        reinit=True,
    )

    return wandb_run.id


def get_wandb_res_log_key(result):
    task_name = result['task']
    nshots = result.get("nb_few_shot_samples", 0)
    if nshots > 0:
        return f"{task_name}_{nshots}_shot"

    return task_name


def attempt_to_log_results_to_wandb(
    results: List[Dict], 
    wandb_project: Optional[str],
    wandb_run_id: str,
    wandb_run_name: Optional[str],
    only_log_preferred_metrics: bool,
    wandb_logging_entity=WANDB_LOGGING_ENTITY,
) -> None:
    """
    Log results to WandB project. If the wandb_project is not specified, ignore logging
    Args:
        results (List): Results to be logged to WandB
        wandb_project (str): WandB project name where metrics will be populated
        wandb_run_id (str): The WandB run identifier for logging metrics (useful for resuming runs)
        wandb_run_name (str): A human readable name for the run
        only_log_preferred_metrics (bool): If set, only log preferred metrics. Otherwise, all metrics
            are logged
    """
    if len(results) == 0:
        return 
    if wandb_project is not None:
        logger.info(f"Logging {'preferred' if only_log_preferred_metrics else 'all'} "
                     f"metrics to WandB project {wandb_project}")
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_logging_entity,
            id=wandb_run_id,
            name=wandb_run_name,
            reinit=True,
        )
    else:
        logger.info(f"Logging to WandB project is disabled")
        # Early return. Logging to WandB is disabled
        return

    wandb_log_key_fields = [
        "task",
    ]

    task_list = set([get_wandb_res_log_key(result) for result in results])

    with wandb_run:
        # Define a custom X axis for all metrics. By default, this is the checkpoint_steps
        logger.info(f"Logging {len(results)} results to WandB project {wandb_project}, run id {wandb_run.id}")
        for task in task_list:
            wandb_run.define_metric(f"{task}/checkpoint_steps")
            wandb_run.define_metric(f"{task}/*", step_metric=f"{task}/checkpoint_steps")

        for result in results:
            wandb_log_key_prefix = get_wandb_res_log_key(result)
            wandb_log_dict = {}
            for result_key, result_value in result.items():
                if (
                    result_key not in wandb_log_key_fields 
                    and _filter_keys_for_loggin_to_wandb(
                        result_key,
                        result["task"],
                        only_log_preferred_metrics,
                    )
                ):
                    wandb_log_key = "/".join([wandb_log_key_prefix, result_key])
                    wandb_log_dict[wandb_log_key] = result_value

                    if result_key == 'checkpoint_steps':
                        # Log checkpoint_steps separately as well, as a standalone key
                        wandb_log_key = result_key
                        wandb_log_dict[wandb_log_key] = result_value

            wandb_run.log(wandb_log_dict)

    return task_list


def get_random_wandb_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=8))

def add_wandb_arguments(parser):

    # WandB logging arguments
    parser.add_argument(
        "--wandb-project",
        required=False,
        type=str,
        help="Name of the wandb project where metrics will be logged. "
        "WandB will create this project for you if it doesn't already exist. "
        "If unspecified, wandb logging will be disabled.",
    )

    parser.add_argument(
        "--wandb-run-name",
        required=False,
        type=str,
        help="A short descriptive name for the WandB logging run. This name "
        "will show up in the UI.",
    )

    parser.add_argument(
        "--wandb-logging-entity",
        required=False,
        type=str,
        default=None,
        help="Wandb logging entity",
    )

    parser.add_argument(
        "--wandb-run-id",
        "--resume-wandb-run-id",
        required=False,
        type=str,
        default=None,
        help="WandB projects can have multiple runs. If you want to resume a "
        "previous WandB run, specify the run id here. If unspecified, WandB will "
        "create a new run automatically.",
    )

    parser.add_argument(
        '--wandb-log-preferred-metrics',
        action="store_true",
        help="Log only preferred metrics to WandB. If unspecified, all metrics are "
        "logged to WandB, which makes it difficult to visualize dashboards.",
    )


def collect_results_arg_parser():
    parser = argparse.ArgumentParser(description="Aggregate and display results from multiple result directories.")
    parser.add_argument(
        "-i",
        "--input-dirs",
        default=[
            "/checkpoint/tbmihaylov/few_shot/2021-06-30-ppl-eval-moe/*/*openbook*_results.json",  # for debug purposes
            # "/large_experiments/xlmg/models/sshleifer/few_shot_results/*/"
        ],
        nargs="+",
        help="List of directories with results. Can include * to expand multiple dirs. By default the script looks at the given directory and subdirectories +1 level.",
    )

    parser.add_argument(
        "-o",
        "--output",
        default="results.tsv",
        help="The path to an output csv/tsv file. Raw results file OUTPUT.raw.jsonl is also generated!",
    )

    parser.add_argument(
        "-v",
        "--view",
        default="preferred_metrics",
        choices=list(display_views.keys()),
        help="The view for the output. Options are: " + ";".join([f"`{k}` - {f[1]}" for k,f in display_views.items()]) ,
    )

    parser.add_argument(
        "-t",
        "--tasks",
        default=["all"],
        nargs="+",
        help="List of individual tasks to include or groups of tasks. Currently the following groups are available: " + ",".join([f"`{k}`" for k,f in task_groups.items()]) ,
    )

    parser.add_argument(
        "--recalc-metrics-from-positional-scores",
        "-r",
        action="store_true",
        help="[SLOW] Recalculates accuracy scores from predictions. This requires the predictions files to have the positional scores exported. ",
    ) 

    parser.add_argument(
        "--calculate-ensemble",
        action="store_true",
        help="Calculates ensemble score from multiple prediction files when available",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Output detailed error message when processing files.",
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

    parser.add_argument(
        "--overwrite-formatted-result-table",
        action="store_true",
        help="Overwrite the existing formatted table. By default, only write to the formatted table when there are new results.",
    )

    parser.add_argument(
        "--aggregate-few-shot-trial-results",
        action="store_true",
        help="Aggregate multiple trial results for few-shot learning experiments"
    )

    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watches for new ",
    )

    parser.add_argument(
        "--watch-interval",
        default=30,
        type=int,
        help="The interval in seconds to check for new results.",
    )

    add_wandb_arguments(parser)

    return parser


def run_results_collection_and_attempt_loging_to_wandb(
    input_dirs: Union[str, List[str]],
    output: str,
    existing_results:List[Dict[str, Any]], 
    overwrite_output:bool=False,
    recalc_metrics_from_positional_scores:bool = False,
    calculate_ensemble:bool = False,
    csv_view:str=None,
    csv_sep:str="\t",
    wandb_project:str=None,
    wandb_run_id:str=None,
    wandb_run_name:str=None,
    wandb_only_log_preferred_metrics:bool=True,
    wandb_logging_entity:str=None,
    loop_i:int=0,
):
    """Collect results and log to wandb if the wandb settings are provided.

    Args:
        input_dirs (Union[str, List[str]]): List of directories to search the results at.
        output (str): The output .tsv file with results. A file {output}.raw.jsonl is also written with the structured aggregated results.. 
        existing_results (List[Any]): List of existing results. These are used to determine which results are already loaded.
        overwrite_output (bool, optional): Overwrite the output files if exist. Defaults to False.
        recalc_metrics_from_positional_scores (bool, optional): Recalculate metrics from the predictions files. See `run_results_collection`. Defaults to False.
        calculate_ensemble (bool, optional): Calculates ensenble from multiple prediction files.. Defaults to False.
        csv_view (str, optional): The name of the csv view. See `display_views` . Defaults to None.
        csv_sep (str, optional): CSV separator. Defaults to "\t".
        wandb_project (str, optional): See `attempt_to_log_results_to_wandb`. Defaults to None.
        wandb_run_id (str, optional): See `attempt_to_log_results_to_wandb`. Defaults to None.
        wandb_run_name (str, optional): See `attempt_to_log_results_to_wandb`. Defaults to None.
        wandb_only_log_preferred_metrics (bool, optional): See `attempt_to_log_results_to_wandb`. Defaults to True.
        wandb_logging_entity (str, optional): See `attempt_to_log_results_to_wandb`. Defaults to None.
        loop_i (int, optional): If we run this in a loop, loop_id is > 0. Defaults to 0.. Defaults to 0.

    Returns:
        [type]: [description]
    """
    new_results, all_results = run_results_collection(
        input_dirs=input_dirs,
        output=output,
        overwrite_output=overwrite_output,
        recalc_metrics_from_positional_scores=recalc_metrics_from_positional_scores,
        calculate_ensemble=calculate_ensemble,
        view=csv_view,
        sep=csv_sep,
        existing_results=existing_results, 
        loop_i=loop_i,        
    )

    attempt_to_log_results_to_wandb(
        results=new_results,
        wandb_project=wandb_project,
        wandb_run_id=wandb_run_id,
        wandb_run_name=wandb_run_name,
        only_log_preferred_metrics=wandb_only_log_preferred_metrics,
        wandb_logging_entity=wandb_logging_entity
    )

    return new_results, all_results


def get_wandb_config(
    wandb_config_file, 
    wandb_project,
    wandb_run_name,
    wandb_run_id,
    wandb_logging_entity,
    create_or_update_file_on_new_id=True):

    if os.path.exists(wandb_config_file):
        wandb_config_json = read_json_file(wandb_config_file)
        wandb_config = WandbSimpleConfig(**wandb_config_json)
    else:
        wandb_config = WandbSimpleConfig(
            wandb_project=wandb_project,
            wandb_run_name=wandb_run_name,
            wandb_run_id=wandb_run_id,
            wandb_logging_entity=wandb_logging_entity,
        )
    
    wandb_run_id = validate_wandb_login(wandb_config.wandb_project,
        wandb_run_name=wandb_config.wandb_run_name,
        wandb_run_id=wandb_config.wandb_run_id,
        wandb_logging_entity=wandb_config.wandb_logging_entity,
    )

    if wandb_run_id is None:
        wandb_run_id = get_random_wandb_id()

    if wandb_run_id != wandb_config.wandb_run_id:
        wandb_config.wandb_run_id = wandb_run_id
        if create_or_update_file_on_new_id:
            wandb_config_dict = vars(wandb_config)
            json_str = json.dumps(wandb_config_dict, indent=4)
            with open(wandb_config_file, mode="w") as f_wandb_cfg:
                f_wandb_cfg.write(json_str)
    
    return wandb_config

class WandbSimpleConfig(object):
    wandb_project:str
    wandb_run_name:str
    wandb_run_id:Optional[str]=None
    wandb_logging_entity:Optional[str]=None

    def __init__(self, **kwargs) -> None:
        super().__init__()
        for fld, val in kwargs.items():
            setattr(self, fld, val)


def collect_results_main(sys_argv):
    parser = collect_results_arg_parser()
    args = parser.parse_args(sys_argv[1:])
    
    logger.info(f"output tsv: {args.output}")
    logger.info(f"output jsonl: {args.output}.raw.jsonl")

    if args.aggregate_few_shot_trial_results:
        run_few_shot_results_aggregation(args)
        exit(0)
        
    if args.watch:
        logger.info(f"Watching for new results every {args.watch_interval} seconds.")
    
    if args.wandb_project:
        wandb_run_id=validate_wandb_login(args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            wandb_run_id=args.wandb_run_id,
            wandb_logging_entity=args.wandb_logging_entity,
        )

        if wandb_run_id is None:
            wandb_run_id = get_random_wandb_id()
    
    loop_i = 0
    all_results = []
    while True:
        # This will terminate after the first iteration if not args.watch!

        new_results, all_results = run_results_collection_and_attempt_loging_to_wandb(
            input_dirs=args.input_dirs,
            output=args.output,
            existing_results=all_results, 
            loop_i=loop_i,
            overwrite_output=args.overwrite_output,
            recalc_metrics_from_positional_scores=args.recalc_metrics_from_positional_scores,
            calculate_ensemble=args.calculate_ensemble,
            csv_view=args.view,
            csv_sep=args.sep,
            wandb_project=args.wandb_project,  # wandb settings
            wandb_run_id=args.wandb_run_id,
            wandb_run_name=args.wandb_run_name,
            wandb_only_log_preferred_metrics=args.wandb_log_preferred_metrics,
            wandb_logging_entity=args.wandb_logging_entity
        )

        if not args.watch:
            break
        
        logger.info(f"Sleep {args.watch_interval}s ...")
        time.sleep(args.watch_interval)
        loop_i += 1
    
if __name__ == "__main__":
    collect_results_main(sys.argv)
    