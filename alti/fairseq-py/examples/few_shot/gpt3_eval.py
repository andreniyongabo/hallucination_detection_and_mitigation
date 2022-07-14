#!/usr/bin/env python

import argparse
import collections
import copy
import datetime
import json
import os
import time

import numpy as np
import scipy.stats
import torch

from examples.few_shot.metrics import GoldAnswerPPLMetric
from examples.few_shot.models import load_lm_and_run_func
from examples.few_shot.predictors import (
    Prediction,
    PromptingPredictor,
    ScoredCandidate,
    get_calibrator_class_by_name,
    get_predictor_class_by_name,
)
from examples.few_shot.tasks import (
    get_all_tasks,
    get_task_class_by_name,
    get_task_class_custom_init_params,
    get_task_eval_attributes,
    get_tasks_by_group,
    init_task_with_custom_args,
    is_task_group,
)
from examples.few_shot.templates import get_template_class_by_name
from fairseq import utils
from fairseq.utils import print_r0
from fb_sweep.sweep.slurm import get_random_port


def cli_main():
    """Example usage:
    python -m examples.few_shot.gpt3_eval --model-name 125M_gpt3_setting --tasks copa --nb-few-shot-samples-values 0
    python -m examples.few_shot.gpt3_eval --model-name 125M_gpt3_setting --tasks xcopa --xcopa-languages vi et --nb-few-shot-samples-values 0
    python -m examples.few_shot.gpt3_eval --model-name 125M_gpt3_setting --tasks copa cb --nb-few-shot-samples-values 0 1 32

    python -m examples.few_shot.gpt3_eval --model-name 125M_gpt3_setting --tasks processtext --nb-few-shot-samples-values 0 --eval-set dummy.txt
    """
    parser = get_argparser()

    args = parser.parse_args()
    args.train_sep = args.train_sep.replace(
        "\\n", "\n"
    )  # Newlines are escaped by argparse
    utils.import_user_module(args)
    run_evaluations_from_model_name(**vars(args))


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--tasks", default=None, nargs="+")
    parser.add_argument("--predictor-name", default="clmprompting")
    parser.add_argument(
        "--calibrator-name",
        default=None,
        help="Leave empty for no calibration. Set to 'average_option' for calibration with averaging the calibration samples results.",
    )
    parser.add_argument(
        "--predictions-dump-dir",
        default=None,
        type=str,
        help="If set, the predictions are exported to this directory in a jsonl file.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        type=str,
        help="Directory to export the results .json file. One file per evaluation!",
    )
    parser.add_argument("--user-dir", help="path to user-defined tasks/criterions/etc.")
    parser.add_argument(
        "--add-prompt-to-meta",
        action="store_true",
        help="Adds the formulated prompt to the prediction meta",
    )
    parser.add_argument(
        "--add-calib-to-meta",
        action="store_true",
        help="Adds the calibration meta to each candidate. If calibration is enabled!",
    )
    parser.add_argument(
        "--add-positional-scores-to-meta",
        action="store_true",
        help="Adds the positional scores to the prediciton meta",
    )
    parser.add_argument(
        "--add-prompt-tokens-to-meta",
        action="store_true",
        help="Adds prompt tokens to the prediciton meta.",
    )

    parser.add_argument(
        "--scoring",
        default="sum",
        choices=["mean", "sum", "suffix", "unconditional-norm"],
    )
    parser.add_argument(
        "--n-eval-samples",
        "--n",  # alias
        type=int,
        default=None,
        metavar="N",
        help="Evaluate on the first N samples only",
    )
    parser.add_argument("--confusion-matrix", action="store_true")

    parser.add_argument(
        "--batch-size", type=int, default=1, help="inference batch size"
    )

    parser.add_argument(
        "--nb-few-shot-samples-values",
        "--nshot",  # alias
        type=int,
        default=None,
        nargs="+",
        help="subsample K examples from the training set for one-shot or "
        "few-shot learning",
    )

    parser.add_argument(
        "--uniform-sampling",
        action="store_true",
        help="take the same number of candidates per class when sampling",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        metavar="N",
        help="beam size for generative tasks",
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
        "--moe-eval-capacity-token-fraction",
        "--cap",  # alias
        type=float,
        # default=0.1,
        help="Sets the moe_eval_capacity_token_fraction of MoE model. This limits the max fraction of tokens in the batch to be assigned to a single expert. Only applied when is_moe is set for the model",
    )

    parser.add_argument(
        "--distributed-port",
        type=int,
        default=None,
        help="Sets the distributed port for model communication over multiple gpus.",
    )
    
    parser.add_argument(
        "--truncate-few-shot-samples-globally",
        action="store_true",
        help="Truncate all test samples consistently so we keep the same few-shot samples. "
        "By default we truncate each eval sample separately, so some might end up "
        "with more few-shot samples than others.",
    )

    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Maximum sequence length. Defaults to the training value. When 0 the maximum sequence length is set to --max-tokens.",
    )

    parser.add_argument(
        "--num-training-updates",
        type=int,
        default=-1,
        help="number of training updates at which to evaluate the model",
    )

    parser.add_argument(
        "--no-shuffle-examples",
        action="store_true",
        help="Do not shuffle the few-shot examples for each test instance",
    )

    parser.add_argument("--replace-newline-with-eos", action="store_true")
    parser.add_argument(
        "--train-sep",
        default=" ",
        help="Separator between example prompts in few-shot setting.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument(
        "--recalc-metrics-from-predictions",
        action="store_true",
        help="This will attempt to recalculate metrics from predictions",
    )
    parser.add_argument("--cause-conj", default=" because ")
    parser.add_argument("--effect-conj", default=" so ")
    parser.add_argument(
        "--capitalization",
        default="correct",
        choices=["correct", "bug", "upper", "lower"],
    )

    parser.add_argument(
        f"--any-train-set",
        default=None,
        help="This is used if --[task]-train-set is not passed!",
    )
    parser.add_argument(
        f"--any-valid-set",
        default=None,
        help="This is used if --[task]-valid-set is not passed!",
    )
    parser.add_argument(
        f"--any-eval-set",
        default=None,
        nargs="+",
        help="This is used if --[task]-valid-set is not passed!",
    )
    parser.add_argument(
        f"--any-train-lang",
        default=None,
        help="This is used if --[task]-train-lang is not passed!",
    )
    parser.add_argument(
        f"--any-valid-lang",
        default=None,
        help="This is used if --[task]-train-lang is not passed!",
    )

    for task in get_all_tasks():
        parser.add_argument(
            f"--{task}-template", metavar="TEMPLATE", default=None, nargs="+"
        )
        # data sets
        parser.add_argument(f"--{task}-train-set", metavar="TRAIN_SET", default=None)
        parser.add_argument(f"--{task}-valid-set", metavar="VALID_SET", default=None)
        parser.add_argument(
            f"--{task}-eval-set", metavar="EVAL_SET", default=None, nargs="+"
        )

        parser.add_argument(f"--{task}-calibration-options", default=[], nargs="+")
        languages = get_task_class_by_name(task_name=task).get_supported_languages()
        if len(languages) > 1:
            # langs
            parser.add_argument(
                f"--{task}-languages",
                metavar="LANGUAGE",
                nargs="+",
                choices=languages,
                default=languages,
            )
            parser.add_argument(
                f"--{task}-train-lang",
                metavar="TRAIN_LANG",
                default=None,
                choices=languages,
            )
            parser.add_argument(
                f"--{task}-valid-lang",
                metavar="VALID_LANG",
                default=None,
                choices=languages,
            )

    parser.add_argument(
        "--max-hits-cnt",
        type=int,
        default=0,
        help="Maximum hits count (supporting paragraphs) for EXAMS task.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens will be inferred if this is not passed, which can be slow.",
    )
    parser.add_argument(
        "--compute-vocab-dist",
        action="store_true",
        help="If you need the model to return full vocabulary probability distribtion, set this to true.",
    )
    parser.add_argument(
        "--max-cands",
        type=int,
        default=0,
        help="Limit maximum number of candidates for mLAMA task",
    )

    # args for running eval with FSDP
    parser.add_argument("--fsdp", action="store_true", default=False)

    return parser


def run_evaluations_from_model_name(model_name, **kwargs):
    """Example usage:
    run_evaluations_from_model_name(model_name="124M", tasks=["copa", "cb"], nb_few_shot_samples_values=[0, 1, 32])
    """
    print_r0(f"model_name={model_name}")
    run_params = {k: v for k, v in kwargs.items()}
    run_params["model_name"] = model_name
    kwargs[
        "model_name_display"
    ] = model_name  # HACK to pass the model_name to other functions. Used for logging
    if "distributed_port" not in run_params or run_params["distributed_port"] is None:
        kwargs["distributed_port"] = get_random_port()
    results = load_lm_and_run_func(run_evaluations, model_name, **kwargs)

    return results


def iterate_over_tasks(tasks, **kwargs):
    for task_name in tasks:
        # templates
        template_names = kwargs.get(f"{task_name}_template")
        if template_names is None:
            template_names = [
                (
                    get_task_class_by_name(task_name)
                    .get_default_template_class()
                    .get_template_name()
                )
            ]
        if isinstance(template_names, str):
            template_names = [template_names]
        for template_name in template_names:
            calibration_options = kwargs.get(f"{task_name}_calibration_options", [])
            default_languages = get_task_class_by_name(
                task_name=task_name
            ).get_supported_languages()

            # Handle cases where we passed `{task_name}_language`` (singular) instead of `{task_name}_languages``
            current_task_languages = kwargs.get(
                f"{task_name}_languages",
                kwargs.get(f"{task_name}_language", default_languages),
            )
            if isinstance(current_task_languages, str):
                current_task_languages = [
                    current_task_languages
                ]  # Sometimes we could set only one language as string

            for language in current_task_languages:
                task_default_attributes = get_task_eval_attributes(task_name, {})
                train_set = kwargs.get(
                    f"{task_name}_train_set",
                    kwargs.get(
                        f"any_train_set", task_default_attributes.get("train_set", None)
                    ),
                )
                valid_set = kwargs.get(
                    f"{task_name}_valid_set",
                    kwargs.get(
                        f"any_valid_set", task_default_attributes.get("valid_set", None)
                    ),
                )
                train_lang = kwargs.get(
                    f"{task_name}_train_lang", kwargs.get(f"any_train_lang", language)
                )
                valid_lang = kwargs.get(
                    f"{task_name}_valid_lang", kwargs.get(f"any_valid_lang", language)
                )

                # we can use several eval sets
                eval_sets = kwargs.get(
                    f"{task_name}_eval_set", kwargs.get(f"any_eval_set", None)
                )
                if eval_sets is None:
                    eval_sets = [task_default_attributes.get("eval_set")]

                for eval_set in eval_sets:
                    yield task_name, template_name, calibration_options, eval_set, language, train_set, train_lang, valid_set, valid_lang


def run_evaluations(
    model,
    tasks,
    nb_few_shot_samples_values=None,
    num_trials=10,
    trial_seed=None,
    skip_completed=False,
    dry_run=False,
    return_result_files_list=False,
    **kwargs,
):
    tasks_expanded = []
    for task in tasks:
        all_tasks = get_all_tasks()
        if task in all_tasks:
            tasks_expanded.append(task)
        elif is_task_group(task):
            tasks_expanded.extend(get_tasks_by_group(task))

    assert len(tasks_expanded) > 0, f"Unrecognized task specification {tasks}"
    tasks = tasks_expanded

    if nb_few_shot_samples_values is None:
        nb_few_shot_samples_values = [0, 1, 32]
    results = []
    calibrator_name = kwargs.get("calibrator_name", None)
    model_name = kwargs.get("model_name_display", None)

    # log the information
    results_files_list = []
    results_dir = kwargs.get("results_dir", None) 
    if results_dir is None:
        results_dir = f"few_shot_{model_name}_results"
    os.makedirs(results_dir, exist_ok=True)
    for (
        task_name,
        template_name,
        calibration_options,
        eval_set,
        language,
        train_set,
        train_lang,
        valid_set,
        valid_lang,
    ) in iterate_over_tasks(tasks, **kwargs):
        if not dry_run:
            # The logging below is useful to be printed in the log files
            # but not when we are doing a dry_run which calls this many times for different tasks!
            print_r0(f"task={task_name}")
            print_r0(f"eval_set={eval_set}")
            print_r0(f"eval language={language}")
            print_r0(f"train_set={train_set}")
            print_r0(f"train_lang={train_lang}")
            print_r0(f"template={template_name}")
            print_r0(f"calibration_options={calibration_options}")

        for nb_few_shot_samples in nb_few_shot_samples_values:
            file_prefix = f"task.{task_name}_tmp.{template_name}_train.{train_set}.{train_lang}_val.{valid_set}.{valid_lang}_eval.{eval_set}.{language}_calib.{calibrator_name}_fs{nb_few_shot_samples}"
            results_out_file = os.path.join(results_dir, f"{file_prefix}_results.json")
            results_files_list.append(results_out_file)

            recalc_metrics_from_predictions = kwargs.get(
                "recalc_metrics_from_predictions", False
            )
            if recalc_metrics_from_predictions and not os.path.exists(results_out_file):
                # We only want to recalculate metrics for experiments where we are certain that the prediction files might be available.
                # This is the case when we have successfully obtained results before.
                continue
            elif not recalc_metrics_from_predictions and (
                skip_completed and os.path.exists(results_out_file)
            ):
                print_r0(f"skipping {results_out_file}")
                continue

            if dry_run:
                continue

            print_r0(f"nb_few_shot_samples={nb_few_shot_samples}")
            metric2scores = collections.defaultdict(list)
            # Run multiple trials with different random sets of few shot conditioning examples
            if nb_few_shot_samples == 0:
                # Zero-shot learning does not need multiple runs
                if trial_seed is not None and trial_seed > 0:
                    continue
                trial_seeds = [0]
            else:
                if trial_seed is None:
                    trial_seeds = range(num_trials)
                else:
                    trial_seeds = [trial_seed]
            for seed in trial_seeds:
                for metric, score in run_evaluation(
                    model=model,
                    task_name=task_name,
                    template_name=template_name,
                    nb_few_shot_samples=nb_few_shot_samples,
                    seed=seed,
                    train_set=train_set,
                    valid_set=valid_set,
                    eval_set=eval_set,
                    train_lang=train_lang,
                    valid_lang=valid_lang,
                    language=language,
                    calibration_options=calibration_options,
                    skip_completed=skip_completed,
                    **kwargs,
                ).items():
                    metric2scores[metric].append(score)

            results_loaded = False
            if os.path.exists(results_out_file) and (
                recalc_metrics_from_predictions or skip_completed
            ):
                # we recalculate the metrics but we want to keep some original metrics
                result_row = load_results_json_from_file(results_out_file)
                results_loaded = True
            else:
                result_row = {
                    "model_name": model_name,
                    "task": task_name,
                    "language": language,
                    "template": template_name,
                    "nb_few_shot_samples": nb_few_shot_samples,
                    "calibration_options": calibration_options,
                    "calibrator_name": calibrator_name,
                    "train_set": train_set,
                    "valid_set": valid_set,
                    "eval_set": eval_set,
                    "train_lang": train_lang,
                    "valid_lang": valid_lang,
                }

            exclude_metrics_update_if_results_are_loaded = {
                "execution_time", # execution time during recalc will not reflect the used calculation.
            }  # Please, update accordingly if more metrics would not be accurate after recalc
            for metric, scores in metric2scores.items():
                if (
                    results_loaded
                    and metric in exclude_metrics_update_if_results_are_loaded
                ):
                    continue

                scores_safe = [x if x is not None else np.nan for x in scores]
                multiple_trials = len(scores_safe) > 1
                result_row[metric] = {
                    "scores": scores,
                    "mean": np.mean(scores_safe),
                    "std": np.std(scores_safe) if multiple_trials else 0.0,
                    "mean_confidence_interval": get_mean_confidence_interval(
                        scores_safe
                    )
                    if multiple_trials
                    else np.nan,
                }
            print_r0(f"results={result_row}\n")
            print_readable_results(result_row)
            results.append(result_row)
            result_row["run_params"] = {k: v for k, v in kwargs.items()}
            if trial_seed is None or trial_seed == (num_trials - 1):

                log_results_json_to_file(results_out_file, result_row)

    if return_result_files_list:
        return results, results_files_list
    else:
        return results


def log_results_json_to_file(out_file_name, results):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return
    try:
        with open(out_file_name, mode="w") as f_res:
            json.dump(results, f_res)
    except Exception as we:
        print_r0(f"Error when writing results to file {out_file_name}!:\n" + str(we))


def load_results_json_from_file(out_file_name):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    with open(out_file_name, mode="r") as f_res:
        results = json.load(f_res)

    return results


def print_readable_results(result_row):
    max_len = max([len(x) for x in list((result_row.keys()))])
    for key, val in sorted(list(result_row.items()), key=lambda x: x[0], reverse=True):
        if isinstance(val, dict) and "mean" in val:
            val = round(val["mean"], 4)
            print_r0(f"{key.ljust(max_len)} = {val}")
        elif isinstance(val, float):
            print_r0(f"{key.ljust(max_len)} = {val}")


def run_evaluation_from_model_name(model_name, **kwargs):
    print_r0(f"model_name={model_name}")
    result = load_lm_and_run_func(run_evaluation, model_name, **kwargs)
    return result


def print_confusion_matrix(samples, predictions):
    sample2prediction = {
        prediction.sample: prediction.best_candidate.candidate
        for prediction in predictions
    }
    n_evaluated_samples = 0
    ans2count = {}
    for sample in samples:
        if sample.has_subproblems:
            print_r0("Confusion matrix not supported for tasks with subproblems")
            return
        elif len(sample.correct_candidates) > 1:
            print_r0(
                "Confusion matrix not supported for tasks with multiple correct candidates"
            )
            return
        key = (sample2prediction[sample], sample.correct_candidates[0])
        ans2count[key] = ans2count.get(key, 0) + 1
        n_evaluated_samples += 1

    candidates = sorted({y for x in ans2count.keys() for y in x})
    print_r0("sys/ref\t" + "\t".join(candidates) + "\tTOTAL")
    for c1 in candidates:
        print_r0(c1, end="")
        total = 0
        for c2 in candidates:
            p = 100 * ans2count.get((c1, c2), 0) / n_evaluated_samples
            total += p
            print_r0(f"\t{p:.1f}", end="")
        print_r0(f"\t{total:.1f}")
    print_r0("TOTAL", end="")
    for c in candidates:
        print_r0(
            f"\t{100*sum([n for (c1, c2), n in ans2count.items() if c2 == c])/n_evaluated_samples:.1f}",
            end="",
        )
    print_r0("\t100.0")


def json_encoder(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        return obj.__str__()


def dump_predictions_to_file(samples, predictions, out_file_path):
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        pass

    sample2prediction = {p.sample: p for p in predictions}
    try:
        with open(out_file_path, mode="w") as f_pred:
            for sample in samples:
                if sample.has_subproblems:
                    subsamples = sample.subproblems
                else:
                    subsamples = [sample]

                for curr_sample in subsamples:
                    curr_prediction = sample2prediction[curr_sample]
                    pred_item = {}
                    pred_item["id"] = curr_sample.data.get("id", "unk")
                    pred_item["idx"] = curr_sample.data.get("idx", -1)
                    pred_item["pred_meta"] = curr_prediction.meta
                    pred_item["candidates"] = [
                        [
                            scored_cand.candidate,
                            {
                                "best": scored_cand == curr_prediction.best_candidate,
                                "gold": False
                                if curr_sample.correct_candidates is None
                                else (
                                    scored_cand.candidate
                                    in curr_sample.correct_candidates
                                ),
                                "score": float(scored_cand.score),
                                "meta": scored_cand.meta,
                            },
                        ]
                        for scored_cand in curr_prediction.scored_candidates
                    ]
                    f_pred.write(
                        json.dumps(pred_item, ensure_ascii=False, default=json_encoder)
                    )
                    f_pred.write("\n")
            print_r0(f"Predictions dumped to {out_file_path}")
    except Exception as e:
        raise e


def read_predictions_from_file(samples, in_file_path):
    predictions = []
    with open(in_file_path, mode="r") as f_pred:
        for sample in samples:
            if sample.has_subproblems:
                subsamples = sample.subproblems
            else:
                subsamples = [sample]

            for curr_sample in subsamples:
                pred_item = json.loads(f_pred.readline())
                scored_candidates = []
                for cand, cand_info in pred_item["candidates"]:
                    scored_cand = ScoredCandidate(
                        cand, cand_info["score"], meta=cand_info["meta"]
                    )
                    scored_candidates.append(scored_cand)

                curr_prediction = Prediction(
                    curr_sample,
                    scored_candidates=scored_candidates,
                    meta=pred_item["pred_meta"],
                )
                predictions.append(curr_prediction)

    return predictions


def load_task_template_calibrator_predictor(
    model,
    task_name,
    template_name,
    predictor_name,
    nb_few_shot_samples,
    uniform_sampling,
    seed=0,
    calibrator_name=None,
    use_full_train_set=False,
    **kwargs,
):
    task_custom_kwargs = get_task_class_custom_init_params(task_name)
    if task_custom_kwargs is not None and len(task_custom_kwargs) > 0:
        kwargs = copy.deepcopy(kwargs)
        kwargs.update(task_custom_kwargs)

    task = get_task_class_by_name(task_name=task_name).from_kwargs(**kwargs)

    if nb_few_shot_samples != -1:
        task = task.get_random_subset(
            train_size=nb_few_shot_samples,
            valid_size=nb_few_shot_samples,
            uniform_sampling=uniform_sampling,
            seed=seed,
        )
    elif len(task.valid_samples) == 0 and not use_full_train_set:
        # if we don't have valid data, split train data
        print_r0(f"splitting train data 80/20")
        total_size = len(task.train_samples)
        task = task.get_random_subset(
            train_size=int(total_size * 0.8),
            valid_size=int(total_size * 0.2),
            uniform_sampling=uniform_sampling,
            seed=seed,
        )

    template = get_template_class_by_name(template_name=template_name).from_kwargs(
        **kwargs
    )

    calibrator = (
        None
        if calibrator_name is None
        else get_calibrator_class_by_name(calibrator_name=calibrator_name).from_kwargs(
            **kwargs
        )
    )
    if calibrator is not None:
        print_r0("* using calibrator: {}".format(calibrator))

    predictor = get_predictor_class_by_name(predictor_name=predictor_name).from_kwargs(
        model=model, task=task, template=template, calibrator=calibrator, **kwargs
    )

    return task, template, calibrator, predictor


def run_evaluation(
    model,
    task_name,
    template_name,
    predictor_name,
    nb_few_shot_samples,
    uniform_sampling,
    seed=0,
    calibrator_name=None,
    confusion_matrix=False,
    skip_completed=False,
    **kwargs,
):
    """Run single evaluation"""
    start_time = time.monotonic()

    task, template, calibrator, predictor = load_task_template_calibrator_predictor(
        model=model,
        task_name=task_name,
        template_name=template_name,
        predictor_name=predictor_name,
        nb_few_shot_samples=nb_few_shot_samples,
        uniform_sampling=uniform_sampling,
        seed=seed,
        calibrator_name=calibrator_name,
        **kwargs,
    )

    # dump predictions to a file for further analysis
    predictions_out_file = None
    predictions_dump_dir = kwargs.get("predictions_dump_dir", None)
    if predictions_dump_dir is not None:
        if not os.path.exists(predictions_dump_dir):
            os.makedirs(predictions_dump_dir, exist_ok=True)
        lang_code = task.language
        train_set = task.train_set
        valid_set = task.valid_set
        eval_set = task.eval_set
        train_lang = task.train_lang
        valid_lang = task.valid_lang

        file_prefix = f"task.{task_name}_tmp.{template_name}_train.{train_set}.{train_lang}_val.{valid_set}.{valid_lang}_eval.{eval_set}.{lang_code}_calib.{calibrator_name}_fs{nb_few_shot_samples}"

        predictions_out_file = os.path.join(
            predictions_dump_dir, f"{file_prefix}_seed{seed}_predictions.jsonl"
        )

    # predict
    eval_samples = task.eval_samples

    recalc_metrics_from_predictions = kwargs.get(
        "recalc_metrics_from_predictions", False
    )
    if (
        (skip_completed or recalc_metrics_from_predictions)
        and (isinstance(predictor, PromptingPredictor)
        and not predictor.use_calibration)
        and predictions_out_file is not None
        and os.path.exists(predictions_out_file)
    ):
        # Add support for reading predictions from file. This can be used for resuming failed experiments, recalculating scores, etc.
        # Currently only supports PromptingPredictor without calibration - this is a quick fix and we will likely not use it with
        # calibration often so we leave the implementation or reading ths for future work.
        eval_predictions = read_predictions_from_file(
            eval_samples, predictions_out_file
        )
        print_r0(f"Predictions are loaded from {predictions_out_file}!")
        metrics_scores = (
            {}
        )  # These are "meta" metrics (such as ppl, etc) that are coming from running the predictor. These will not be available after the recalc.
    else:
        # run the full evalulation
        eval_predictions, metrics_scores = predictor.predict(eval_samples)

    # calculate metrics
    metrics_to_collect = (
        task.metrics + (GoldAnswerPPLMetric(),)
        if not any([isinstance(x, GoldAnswerPPLMetric) for x in task.metrics])
        else ()
    )
    task_metrics = get_metric_scores(metrics_to_collect, eval_samples, eval_predictions)
    metrics_scores.update(task_metrics)

    if confusion_matrix:
        print_confusion_matrix(eval_samples, eval_predictions)

    if predictions_out_file is not None:
        dump_predictions_to_file(eval_samples, eval_predictions, predictions_out_file)

    run_time = time.monotonic() - start_time
    metrics_scores["execution_time"] = run_time
    return metrics_scores


def get_metric_scores(metrics, eval_samples, eval_predictions):
    metric_scores = {}
    for metric in metrics:
        name = metric.name
        score = metric.score(eval_samples, eval_predictions)
        if isinstance(score, dict):
            for mn, mv in score.items():
                if mn in metric_scores:
                    raise Exception(f"Metric {mn} is already reported!")
                metric_scores[mn] = mv
        else:
            metric_scores[name] = score

    return metric_scores


def get_mean_confidence_interval(data, confidence=0.95):
    data = np.array(data)
    a = 1.0 * data
    a = a[~np.isnan(a)]
    n = len(a)
    se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return h


if __name__ == "__main__":
    cli_main()
