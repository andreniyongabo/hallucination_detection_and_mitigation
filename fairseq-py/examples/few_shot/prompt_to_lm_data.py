# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path
from typing import List

import fire
import jsonlines

from examples.few_shot import models
from examples.few_shot.gpt3_eval import (
    iterate_over_tasks,
    load_task_template_calibrator_predictor,
)
from examples.few_shot.tasks import get_all_tasks, get_tasks_by_group, is_task_group
from fairseq.data.jsonl_dataset import JsonlDataset

from fb_sweep.sweep.flan_constants import flan_clusters
from fb_sweep.finetune_lm import _flan_task_mappings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def expand_tasks(downstream_tasks) -> List[str]:
    """Retrieve all the task if the downstream_tasks has a task group name"""
    downstream_tasks_expanded = []
    all_tasks = get_all_tasks()
    for task_name in downstream_tasks:
        if task_name in all_tasks:
            downstream_tasks_expanded.append(task_name)
        elif is_task_group(task_name):
            downstream_tasks_expanded.extend(get_tasks_by_group(task_name))
    downstream_tasks = downstream_tasks_expanded
    return downstream_tasks


def _get_examples(task, predictor, template, samples) -> List[str]:
    """Iterate over samples to create a list of verbalized prompts."""
    samples = [
        subproblem
        for sample in samples
        for subproblem in (sample.subproblems if sample.has_subproblems else [sample])
    ]

    # HACK: empty train_samples in task to reuse get_prompts
    # without adding training samples in prompts
    task._train_samples = []
    verbalized_sentences = []
    many_prompts, _, _ = predictor.get_prompts(samples)
    for prompt, sample in zip(many_prompts, samples):
        correct_cands = set(sample.correct_candidates)
        # if task.has_candidates:
        #     candidates = sample.candidates
        # else:
        #     # Generation task
        #     candidates = correct_cands
        for candidate in correct_cands:
            # label = 1 if candidate in correct_cands else 0
            # assert label == 1, f"Label = {label}, candidate = {candidate}, task={task}"
            candidate = template.verbalize(sample, candidate)
            example = prompt.replace("<mask>", candidate)
            verbalized_sentences.append(example)

    return verbalized_sentences


def _load_model_from_hub(mname="125M_gpt3_setting"):
    # read model config from examples.few_shot.models
    eval_lm_cfg, model_config = models.get_lm_config(mname)
    # load model
    hub_model = models.load_and_get_model(
        eval_lm_cfg,
        model_config,
        skip_prepare_for_inference=True,
        fsdp=False,  # wrap using this hook
    )
    hub_model.max_positions = 10000
    return hub_model


def load_examples(
    save_dir: str,
    no_streaming=False,
    task_str="flan",
):
    def write_examples(tasks, split):
        if no_streaming:
            save_path = f"{save_dir}/{split}.txt"
            f = open(save_path, "w")
        else:
            f = None

        for task_name, template_name, _, __, language, *__ in iterate_over_tasks(
            tasks,
        ):
            (
                task,
                template,
                calibrator,
                predictor,
            ) = load_task_template_calibrator_predictor(
                model=dummy_model,
                task_name=task_name,
                template_name=template_name,
                predictor_name="clmprompting",
                language=language,
                nb_few_shot_samples=-1,
                uniform_sampling=False,
                use_full_train_set=False,
                seed=1,
            )

            samples = task.train_samples if split == "train" else task.valid_samples
            logger.info(f"task_name: {task_name}: split: {split}: N={len(samples)}")

            examples: List[str] = _get_examples(task, predictor, template, samples)
            if no_streaming:
                for e in examples:
                    f.write(f"{e}\n\n\n")
            else:
                if split == "train":
                    save_path = f"{save_dir}/{split}/00/{task_name}.jsonl"
                elif split == "valid":
                    save_path = f"{save_dir}/valid/00/{task_name}.jsonl"

                Path(save_path).parent.mkdir(exist_ok=True, parents=True)

                with jsonlines.open(save_path, "w") as f:
                    for e in examples:
                        f.write({"text": e})
                JsonlDataset(save_path, recache=True)

    "Saves to disk in same format as /large_experiments/xlmg/data/gptz/corpus_dedup_10_10_1_0.05_exp29/"
    # Resolve flan_minus_X to the actual training and valid tasks

    downstream_tasks, valid_tasks = _flan_task_mappings(task_str, flan_clusters)
    dummy_model = _load_model_from_hub()
    downstream_tasks = expand_tasks(downstream_tasks.split(","))

    # TODO: This is ugly, but needed for backward compatibility with prompt_tuning_task
    # Should be cleaned up
    valid_tasks = ["_".join(v.split("_")[1:]) for v in valid_tasks.split(",")]
    valid_tasks = expand_tasks(valid_tasks)

    Path(save_dir).mkdir(exist_ok=True)
    write_examples(downstream_tasks, "train")
    write_examples(valid_tasks, "valid")


if __name__ == "__main__":
    fire.Fire(load_examples)
