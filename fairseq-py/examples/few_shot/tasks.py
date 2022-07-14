#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import csv
import json
import logging
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass, field
from functools import lru_cache
from inspect import isabstract, signature
from pathlib import Path, PosixPath
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

import glob
import os
import random

import numpy as np
import sacrebleu

from examples.few_shot import templates
from examples.few_shot.metrics import (
    AccuracyMetric,
    AUCPRMetric,
    BleuMetric,
    CompositionalInstructionsAccuracyMetric,
    CrowSPairsMetrics,
    EthosZeroShotMetrics,
    FewShotMetric,
    GlueDiagMetrics,
    GoldAnswerPPLMetric,
    LAMAMetrics,
    MLAMAMetric,
    MultiRCPRF1Metric,
    OpenDomainQAMetric,
    PrecisionRecallF1Metric,
    RealToxicityPromptsMetric,
    SariMetric,
    StereoSetMetrics,
)
from fairseq.data import data_utils

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
if os.getenv("FSD", None):
    DATA_DIR = Path(os.getenv("FSD"))

GLUE_DIR = DATA_DIR / "glue"
SUPERGLUE_DIR = DATA_DIR / "SuperGLUE"
NATURAL_INSTRUCTIONS_DIR = DATA_DIR / "natural_instructions"
NATURAL_INSTRUCTIONS_EXPANSION_DIR = DATA_DIR / "natural-instructions-expansion"
FLAN_DIR = DATA_DIR / "flan"
NIE_RELIABILITY_BENCHMARK = DATA_DIR / "NIE_Reliability_Benchmark/Benchmark-v0.1"
FEW_SHOT_TASKS_REGISTRY = {}
FEW_SHOT_TASKS_PARAMS_REGISTRY = {}


def get_task_class_by_name(task_name):
    return FEW_SHOT_TASKS_REGISTRY[task_name.lower()]


def init_task_with_custom_args(task_name, kwargs=None):
    if kwargs is None:
        kwargs = {}
    task_custom_kwargs = get_task_class_custom_init_params(task_name)
    if task_custom_kwargs is not None or len(task_custom_kwargs) > 0:
        kwargs = copy.deepcopy(kwargs)
        kwargs.update(task_custom_kwargs)

    task = get_task_class_by_name(task_name=task_name).from_kwargs(**kwargs)
    return task, kwargs


def get_task_eval_attributes(task_name, kwargs=None):
    task_init, _ = init_task_with_custom_args(task_name, kwargs)
    return task_init.eval_attributes()


def get_task_class_custom_init_params(task_name):
    return FEW_SHOT_TASKS_PARAMS_REGISTRY.get(task_name.lower(), {})


def get_all_tasks():
    return list(sorted(FEW_SHOT_TASKS_REGISTRY.keys()))


def get_tasks_with_languages():
    tasks_with_lang = {
        task_name: task_cls.get_supported_languages()
        for task_name, task_cls in FEW_SHOT_TASKS_REGISTRY.items()
    }

    return tasks_with_lang


def get_languages_with_tasks():
    tasks_by_lang = {}
    for task_name, task_langs in get_tasks_with_languages().items():
        for lang in task_langs:
            if lang not in tasks_by_lang:
                tasks_by_lang[lang] = []
            tasks_by_lang[lang].append(task_name)

    return tasks_by_lang


# These are groups that can be used as a shortcut for loading multiple tasks.
# Please use tasks_organizations.py for organizing visualizations and logical grouping.
task_grouping = {
    "blimp": lambda x: x.startswith("blimp__"),
    "natural_instructions": lambda x: x.startswith("natural_instructions__"),
    "natural_instruct_exp": lambda x: x.startswith("natural_instruct_exp__"),
    "natural_instruct_exp_train_10": lambda x: x.startswith(
        "natural_instruct_exp_train_10__"
    ),
    "lama": lambda x: x.startswith("lama_"),
    "mlama": lambda x: x.startswith("mlama_"),
    "diagnosis": lambda x: x
    in [
        "diagnosisbrand",
        "diagnosiscity",
        "diagnosiscountry",
        "diagnosisname",
        "diagnosispos1",
        "diagnosispos2",
        "diagnosispos3",
        "diagnosispos4",
    ],
}


def get_tasks_by_group(group_name):
    tasks_filter = task_grouping[group_name]
    tasks = [x for x in get_all_tasks() if tasks_filter(x)]

    return tasks


def is_task_group(group_name):
    return group_name in task_grouping


def get_task_group_names():
    return task_grouping.keys()


@lru_cache(maxsize=3)
def read_jsonl_file(filepath):
    json_objects = []
    with open(filepath, "r") as f:
        for line in f:
            json_objects.append(json.loads(line.rstrip()))
    return json_objects


@lru_cache(maxsize=3)
def read_tsv_file(filepath, encoding="utf-8"):
    data = []
    with open(filepath, "r", encoding=encoding) as f:
        keys = f.readline().strip("\n").split("\t")
        for li, line in enumerate(f):
            values = line.strip("\n").split("\t")
            assert len(keys) == len(
                values
            ), f"columns and fields count do not match for line {li+1} in file {filepath}:\n keys ({len(keys)}):{keys} \n values ({len(values)}): {values}"
            data.append({key: value for key, value in zip(keys, values)})
    return data


def print_task(task_name):
    print(f"========== {task_name} examples ==========")
    task_class = get_task_class_by_name(task_name)
    task = task_class()
    print(task, "\n")
    for json_sample in task.eval_samples[:10]:
        print("-" * 80)
        print(task.format_priming_sample(json_sample))


def print_tasks(tasks=None):
    if tasks is None:
        tasks = get_all_tasks()
    for task_name in tasks:
        print_task(task_name)
        print("\n")


class FewShotSample(object):
    def __init__(
        self,
        data,
        candidates=None,
        correct_candidates=None,
        subproblems=None,
    ):
        self._data = data
        self._candidates = candidates
        self._correct_candidates = correct_candidates
        self._subproblems = subproblems if subproblems is not None else []
        if candidates is not None and correct_candidates is not None:
            assert all(
                [
                    correct_candidate in candidates
                    for correct_candidate in correct_candidates
                ]
            )

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, item):
        return item in self._data

    @property
    def candidates(self):
        return self._candidates

    @property
    def has_candidates(self):
        return self.candidates is not None and len(self.candidates) > 0

    @property
    def correct_candidates(self):
        return self._correct_candidates

    def is_correct(self, candidate):
        return candidate in self.correct_candidates

    @property
    def subproblems(self):
        return self._subproblems

    @property
    def data(self):
        return self._data

    @property
    def has_subproblems(self):
        return len(self.subproblems) > 0


@dataclass
class FewShotTask(ABC):
    file_format = "jsonl"
    _train_samples = None
    _valid_samples = None
    _eval_samples = None
    n_eval_samples: Optional[int] = None
    _language: str = "en"
    _train_lang: Optional[str] = None
    _valid_lang: Optional[str] = None
    _train_set: Optional[str] = None
    _valid_set: Optional[str] = None
    _eval_set: Optional[str] = None
    _default_eval_set: Optional[
        str
    ] = None  # This is defined here in order to show up in inherited classes signature.
    metrics: Tuple[FewShotMetric] = (AccuracyMetric(),)

    @classmethod
    @abstractmethod
    def get_sets_and_lang_to_path_mappings(cls):
        pass

    @abstractproperty
    def default_train_set(self):
        pass

    @abstractproperty
    def default_eval_set(self):
        pass

    @property
    def default_valid_set(self):
        return None

    def get_data_file_path(self, set_name: str, lang_code: str):
        if (
            set_name is None
        ):  # We set the valid_set to None since it is not currently used!
            return None

        sets_and_lang_to_path_mappings = self.get_sets_and_lang_to_path_mappings()

        # get data set info
        if set_name not in sets_and_lang_to_path_mappings:
            if os.path.exists(set_name):
                # If a file path exists, we eval with this path file!
                return set_name

            raise KeyError(
                f"set_name: `{set_name}` is not a supported set for task `{self.get_task_name()}`! Available options for sets are: {sets_and_lang_to_path_mappings.keys()}"
            )

        data_set_info = sets_and_lang_to_path_mappings[set_name]
        if data_set_info is None:
            return data_set_info  # we can have None values for valid so this ok.
        elif isinstance(data_set_info, dict):
            if lang_code not in data_set_info:
                raise KeyError(
                    f"Lang: `{lang_code}` is not a supported for set `{set_name}` for task `{self.get_task_name()}`! Available language codes are: {data_set_info.keys()}"
                )
            data_path = data_set_info[lang_code]
            return data_path
        else:
            data_path = data_set_info
            return data_path

    # use this for calibration
    calibration_options: List[str] = None

    @classmethod
    def sub_tasks_registrations(cls) -> Optional[List[Tuple[str, Optional[Dict]]]]:
        # Override and return a tuple
        return None

    # data sets language
    @property
    def train_lang(self):
        if self._train_lang is None and self.train_set is not None:
            return self.language
        return self._train_lang

    @train_lang.setter
    def train_lang(self, value):
        self._train_lang = value

    @property
    def valid_lang(self):
        if self._valid_lang is None and self.valid_set is not None:
            return self.language
        return self._valid_lang

    @valid_lang.setter
    def valid_lang(self, value):
        self._valid_lang = value

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, value):
        self._language = value

    # data sets
    @property
    def train_set(self):
        if self._train_set is None:
            return self.default_train_set

        return self._train_set

    @train_set.setter
    def train_set(self, value):
        self._train_set = value

    @property
    def valid_set(self):
        if self._valid_set is None:
            return self.default_valid_set
        return self._valid_set

    @valid_set.setter
    def valid_set(self, value):
        self._valid_set = value

    @property
    def eval_set(self):
        if self._eval_set is None:
            return self.default_eval_set
        return self._eval_set

    @eval_set.setter
    def eval_set(self, value):
        self._eval_set = value

    @property
    def train_file(self):
        return self.get_data_file_path(self.train_set, self.train_lang)

    @property
    def valid_file(self):
        return self.get_data_file_path(self.valid_set, self.valid_lang)

    @property
    def eval_file(self):
        return self.get_data_file_path(self.eval_set, self.language)

    @abstractmethod
    def build_samples(self, parsed_data):
        pass

    @classmethod
    @abstractmethod
    def get_default_template_class(cls):
        raise NotImplementedError

    @classmethod
    def get_supported_languages(cls):
        return ["en"]

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Allows instanciation from a kwargs dict even if it contains unused keys"""
        # some of the params are hidden behing properties
        # which can not be updated using the kwargs functionality
        kwargs_upd = copy.deepcopy(kwargs)
        for fld in [
            "train_lang",
            "valid_lang",
            "language",
            "train_set",
            "valid_set",
            "eval_set",
        ]:
            if fld in kwargs_upd:
                fld_new = f"_{fld}"
                kwargs_upd[fld_new] = kwargs_upd[fld]

        cls_params = signature(cls).parameters
        obj = cls(**{k: v for k, v in kwargs_upd.items() if k in cls_params})
        return obj

    @classmethod
    def get_task_name(cls):
        [task_name] = re.match("(.+)Task", cls.__name__).groups()
        return task_name.lower()

    def __init_subclass__(cls, **kwargs):
        """Register all children in registry"""
        super().__init_subclass__(**kwargs)
        if isabstract(cls):
            # print(f"{cls} is abstract!")
            return
        task_name = cls.get_task_name()

        sub_tasks = cls.sub_tasks_registrations()
        if sub_tasks is None or len(sub_tasks) == 0:
            sub_tasks = [(task_name, {})]

        for subtask_name, subtask_params in sub_tasks:
            assert (
                subtask_name not in FEW_SHOT_TASKS_REGISTRY
            ), f"{subtask_name} task already registered!"
            FEW_SHOT_TASKS_REGISTRY[subtask_name] = cls
            if subtask_params is not None and len(subtask_params) > 0:
                FEW_SHOT_TASKS_PARAMS_REGISTRY[subtask_name] = subtask_params

    def read_data(self, path):
        if path is None:
            return []
        elif self.file_format == "jsonl":
            return read_jsonl_file(path)
        elif self.file_format == "tsv":
            return read_tsv_file(path)
        else:
            raise NotImplementedError(f"Unsupported file format: {self.file_format}")

    def build_samples_from_file(self, data_file):
        return [
            sample
            for entry in self.read_data(data_file)
            for sample in self.build_samples(entry)
        ]

    @property
    def train_samples(self):
        if self._train_samples is None:
            self._train_samples = self.build_samples_from_file(self.train_file)

        return self._train_samples

    @property
    def valid_samples(self):
        if self._valid_samples is None:
            self._valid_samples = self.build_samples_from_file(self.valid_file)

        return self._valid_samples

    @property
    def eval_samples(self):
        if self._eval_samples is None:
            self._eval_samples = self.build_samples_from_file(self.eval_file)[
                : self.n_eval_samples
            ]

        return self._eval_samples

    @property
    def has_candidates(self):
        # TODO Deciding based on the first eval sample
        return self.eval_samples[0].has_candidates

    def get_max_candidate_length(self, model):
        # TODO Return the length of the longest candidate for multiple choice tasks?
        raise NotImplementedError

    def get_random_subset(
        self, train_size, valid_size=0, uniform_sampling=False, seed=0
    ):
        """Create a copy of this task with a random subset of the train/valid sets

        If the task doesn't have a validation set we create one from the original training set.
        """

        def random_subset(samples, k, candidate=None):
            samples = [
                sample
                for sample in samples
                if candidate is None or sample.correct_candidates[0] == candidate
            ]
            if k > len(samples):
                old_k = k
                k = min(len(samples), k)
                print(
                    f"Set number of conditioning samples k to maximum train set size ({k}) while {old_k} is provided"
                )
            idx = np.random.choice(len(samples), k, replace=False)
            return [samples[idx] for idx in idx]

        if uniform_sampling:
            assert (
                self.has_candidates
            ), "Cannot use uniform sampling with generative tasks"
            candidates = set()
            for samples in self.train_samples, self.valid_samples, self.eval_samples:
                for sample in samples:
                    assert (
                        not sample.has_subproblems
                    ), "Cannot use uniform sampling with tasks with subproblems"
                    assert (
                        len(sample.correct_candidates) == 1
                    ), "Cannot use uniform sampling with >1 correct candidates per instance"
                    candidates.update(sample.candidates)
            assert (
                train_size % len(candidates) == 0
            ), "train_size is not divisible by the number of classes"
            assert (
                valid_size % len(candidates) == 0
            ), "valid_size is not divisible by the number of classes"
        else:
            candidates = [None]

        with data_utils.numpy_seed(seed):
            subset = copy.deepcopy(self)
            subset._train_samples = []
            subset._valid_samples = []
            for candidate in candidates:
                if (
                    len(self.valid_samples) == 0
                ):  # The original task doesn't have a valid set, so we will derive one from the training set
                    samples = random_subset(
                        self.train_samples,
                        (train_size + valid_size) // len(candidates),
                        candidate,
                    )
                    subset._train_samples += samples[: train_size // len(candidates)]
                    subset._valid_samples += samples[train_size // len(candidates) :]
                else:
                    subset._train_samples += random_subset(
                        self.train_samples, train_size // len(candidates), candidate
                    )
                    subset._valid_samples += random_subset(
                        self.valid_samples, valid_size // len(candidates), candidate
                    )
            # Shuffle the new train/valid sets
            subset._train_samples = random_subset(
                subset._train_samples, len(subset._train_samples)
            )
            subset._valid_samples = random_subset(
                subset._valid_samples, len(subset._valid_samples)
            )
            return subset

    def eval_attributes(self):
        task_info = {}

        def print_friendly(val):
            if isinstance(val, list):
                return [str(x) if isinstance(x, PosixPath) else x for x in val]
            elif isinstance(val, PosixPath):
                return str(val)
            else:
                return val

        task_attributes = {
            a: print_friendly(getattr(self, a))
            for a in dir(self)
            if a
            in {
                "default_eval_set",
                "default_train_set",
                "default_valid_set",
                "eval_set",
                "train_set",
                "valid_set",
                "eval_file",
                "train_file",
                "valid_file",
                "language",
                "train_lang",
                "valid_lang",
            }
        }
        return task_attributes


@dataclass
class MultiChoiceWithCalibrationTask(FewShotTask):
    calibration_options: List[str] = None

    # candidates and mappings from labels
    def candidates(self):
        pass

    def label_to_candidates(self, data):
        pass

    @abstractmethod
    def build_single_calibration_sample(self, sample, calibration_option):
        pass

    def build_calibration_samples_bulk(
        self,
        original_samples: List[FewShotSample],
    ) -> List[FewShotSample]:
        calibration_samples = [
            self.build_calibration_sample(x) for x in original_samples
        ]

        return calibration_samples

    def build_calibration_sample(self, original_sample: FewShotSample):
        # all calibration options are an alternative premise
        calibration_options = self.calibration_options

        calibration_samples = []
        for calib_option in calibration_options:
            calib_sample = self.build_single_calibration_sample(
                original_sample, calib_option
            )

            assert (
                calib_sample.data != original_sample.data
            ), "The original `sample.data` could be changed! You want to copy.deep the sample.data since it is used by the Template"

            calibration_samples.append(calib_sample)

        # This is the parent Sample which only groups
        # the "real" sample and the "dummy" samples
        new_sample = FewShotSample(
            data=copy.deepcopy(original_sample.data),
            candidates=copy.deepcopy(original_sample.candidates),
            subproblems=calibration_samples,
        )

        return new_sample

    def build_samples(self, data):
        sample = FewShotSample(
            data=data,
            candidates=self.candidates,
            correct_candidates=self.label_to_candidates(data),
        )

        return [sample]


def parse_calibration_params(calib_option):
    option_params = [x.strip() for x in calib_option.split("|")]

    field_overrides = {}
    for option_param in option_params:
        param_field_info = [x.strip() for x in option_param.split("::")]
        assert (
            len(param_field_info) == 2
        ), f'param {option_param} should be formatted as "key::value"'

        field_key = param_field_info[0]
        field_new_value = param_field_info[1]

        assert not field_key in field_overrides
        field_overrides[field_key] = field_new_value

    return field_overrides


@dataclass
class StereoSetTask(MultiChoiceWithCalibrationTask):
    train_file: Optional[
        Union[str, Path]
    ] = None  # There is no training set in this task
    metrics: Tuple[FewShotMetric] = (StereoSetMetrics(),)

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "dev": DATA_DIR / "StereoSet/data/dev.json",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return None

    @property
    def default_eval_set(self):
        return "dev"

    def read_data(self, path):
        if path is None:
            return []
        with open(path) as f_in:
            data_items = json.load(f_in)["data"]["intersentence"]
            return data_items

    def build_samples(self, data):
        for sent in data["sentences"]:
            del sent["labels"]

        return [
            FewShotSample(
                data=data,
                candidates=[s["gold_label"] for s in data["sentences"]],
                correct_candidates=["stereotype"],
            )
        ]

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)
        data["context"] = calibration_option

        return FewShotSample(
            data=data,
            candidates=[s["gold_label"] for s in data["sentences"]],
            correct_candidates=["stereotype"],
        )

    @classmethod
    def get_default_template_class(cls):
        return templates.StereoSetTemplate


@dataclass
class CrowSPairsTask(MultiChoiceWithCalibrationTask):
    train_file: Optional[
        Union[str, Path]
    ] = None  # There is no training set in this task

    metrics: Tuple[FewShotMetric] = (
        AccuracyMetric(),
        CrowSPairsMetrics(),
    )

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "test": DATA_DIR / "CrowS-Pairs/data/crows_pairs_anonymized_and_prompt.csv"
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return None

    @property
    def default_eval_set(self):
        return "test"

    def read_data(self, path):
        if path is None:
            return []

        data = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data_item = {
                    "sent_more": row["sent_more"],
                    "sent_less": row["sent_less"],
                    "direction": row["stereo_antistereo"],
                    "bias_type": row["bias_type"],
                    "prompt": row["prompt"],
                }
                data.append(data_item)

        return data

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=["sent_more", "sent_less"],
                correct_candidates=["sent_more"],
            )
        ]

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)
        data["prompt"] = calibration_option

        return FewShotSample(
            data=data,
            candidates=["sent_more", "sent_less"],
            correct_candidates=["sent_more"],
        )

    @classmethod
    def get_default_template_class(cls):
        return templates.CrowSPairsTemplate


@dataclass
class EthosZeroShotTask(MultiChoiceWithCalibrationTask):
    train_file: Optional[
        Union[str, Path]
    ] = None  # There is no training set in this task

    metrics: Tuple[FewShotMetric] = (
        AccuracyMetric(),
        EthosZeroShotMetrics(),
    )

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "zero_shot": DATA_DIR / "Ethos/outputs/data/zero_shot_results.csv"
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return None

    @property
    def default_eval_set(self):
        return "zero_shot"

    def read_data(self, path):
        if path is None:
            return []

        data = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["temperature"] == "0":
                    data.append(
                        {
                            "category": row["category"],
                            "prompt": row["prompt"],
                            "label": row["category"] == row["label"],
                        }
                    )

        return data

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=["Yes.", "No."],
                correct_candidates=["Yes." if data["label"] else "No."],
            )
        ]

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)
        data["prompt"].replace(
            "Is the following text " + data["category"] + ", answer yes or no.",
            "Is the following text " + calibration_option + ", answer yes or no.",
        )
        data["category"] = calibration_option

        return FewShotSample(
            data=data,
            candidates=["Yes.", "No."],
            correct_candidates=["Yes." if data["label"] else "No."],
        )

    @classmethod
    def get_default_template_class(cls):
        return templates.EthosZeroShotTemplate


@dataclass
class COPATask(MultiChoiceWithCalibrationTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": SUPERGLUE_DIR / "COPA/train.jsonl",
            "val": SUPERGLUE_DIR / "COPA/val.jsonl",
            "test": SUPERGLUE_DIR / "COPA/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    # candidates and mappings from labels
    @property
    def candidates(self):
        return ["choice1", "choice2"]

    def label_to_candidates(self, data):
        label_to_candidate_map: Dict[Any, str] = {0: "choice1", 1: "choice2"}

        correct_candidates = [label_to_candidate_map[data["label"]]]
        return correct_candidates

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)
        data["premise"] = calibration_option

        return FewShotSample(
            data=data,
            candidates=self.candidates,
            correct_candidates=self.candidates,  # all candidates are correct in our case (choice 1 and choice 2)
        )

    @classmethod
    def get_default_template_class(cls):
        return templates.COPATemplate


@dataclass
class PawsXTask(MultiChoiceWithCalibrationTask):
    """
    Name: PAWS-X
    Paper: https://arxiv.org/abs/1908.11828
    Data: https://github.com/google-research-datasets/paws/tree/master/pawsx

    Notes:
    From the github page:
    "Caveat: please note that the dev and test sets of PAWS-X are both sourced from the dev set of PAWS-Wiki.
    As a consequence, the same sentence 1 may appear in both the dev and test sets. Nevertheless our data
    split guarantees that there is no overlap on sentence pairs (sentence 1 + sentence 2) between dev and test."
    """

    file_format: str = "jsonl"
    _language: str = "en"

    metrics: Tuple[FewShotMetric] = (
        AccuracyMetric(),
    )  # , AUCPRMetric(pos_label="true"))

    @classmethod
    def get_supported_languages(cls):
        return [
            "de",
            "en",
            "es",
            "fr",
            "ja",
            "ko",
            "zh",  # "ru"
        ]

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            # The json files are with fixed tokenization for English
            "train": {
                lang_code: DATA_DIR
                / (
                    f"paws_x/x-final/{lang_code}/"
                    + ("train.jsonl" if lang_code == "en" else "translated_train.jsonl")
                )
                for lang_code in cls.get_supported_languages()
                if lang_code != "ru"
            },
            "dev": {
                lang_code: DATA_DIR / f"paws_x/x-final/{lang_code}/dev_2k.jsonl"
                for lang_code in cls.get_supported_languages()
                if lang_code != "ru"
            },
            "test": {
                lang_code: DATA_DIR / f"paws_x/x-final/{lang_code}/test_2k.jsonl"
                for lang_code in cls.get_supported_languages()
                if lang_code != "ru"
            },
        }

        # to_file_mapping["dev"]["ru"] = DATA_DIR / f"paws_x/x-final/ru/dev_100.jsonl"

        return to_file_mapping

    @property
    def default_train_set(self):
        # HACK: We use the dev for sampling few-shot examples.
        return "dev"

    @property
    def default_eval_set(self):
        return "dev"

    @property
    def candidates(self):
        return ["true", "false"]

    def label_to_candidates(self, data):
        label_to_candidate_map: Dict[Any, str] = {"0": "false", "1": "true"}

        correct_candidates = [label_to_candidate_map[data["label"]]]
        return correct_candidates

    def build_samples(self, data):
        data["lang"] = self.language
        if data["label"] is None:
            return []

        sample = FewShotSample(
            data=data,
            candidates=self.candidates,
            correct_candidates=self.label_to_candidates(data),
        )

        return [sample]

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)
        calib_fields = parse_calibration_params(calibration_option)

        data["sentence1"] = calib_fields.get("sentence1", data["sentence1"])
        data["sentence2"] = calib_fields.get("sentence2", data["sentence2"])

        return FewShotSample(
            data=data,
            candidates=self.candidates,
            correct_candidates=self.candidates,  # this is not used
        )

    @classmethod
    def get_default_template_class(cls):
        return templates.PawsXENTemplate


@dataclass
class XCOPATask(COPATask):
    _language: str = "it"

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "val": {
                lang_code: DATA_DIR / f"xcopa/data/{lang_code}/val.{lang_code}.jsonl"
                for lang_code in cls.get_supported_languages()
                if lang_code != "ru"
            },
            "test": {
                lang_code: DATA_DIR / f"xcopa/data/{lang_code}/test.{lang_code}.jsonl"
                for lang_code in cls.get_supported_languages()
                if lang_code != "ru"
            },
        }

        to_file_mapping["val"][
            "ru"
        ] = "/private/home/tbmihaylov/data/xlmg/few_shot/xcopa/human_translation/ru/val.ru.jsonl"
        to_file_mapping["test"][
            "ru"
        ] = "/private/home/tbmihaylov/data/xlmg/few_shot/xcopa/human_translation/ru/val.ru.jsonl"

        return to_file_mapping

    @property
    def default_train_set(self):
        return "val"

    @property
    def default_eval_set(self):
        return "val"

    def read_data(self, path):
        data_items = super().read_data(path)

        for item in data_items:
            item["lang"] = self.language

        return data_items

    @classmethod
    def get_supported_languages(cls):
        return ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh", "ru"]

    @classmethod
    def get_default_template_class(cls):
        return templates.XCOPATemplate


class HellaSwagTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "hellaswag/hellaswag_train.jsonl",
            "val": DATA_DIR / "hellaswag/hellaswag_val.jsonl",
            "test": DATA_DIR / "hellaswag/hellaswag_test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=data["endings"],
                correct_candidates=[data["endings"][data["label"]]],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.HellaSwagTemplate


def assert_file_paths_exist(split_to_lang_and_path_mappings):
    for split, path_or_lang_to_file_mappings in split_to_lang_and_path_mappings.items():
        if isinstance(path_or_lang_to_file_mappings, dict):
            for lang, file_path in path_or_lang_to_file_mappings.items():
                file_path = str(
                    file_path
                )  # convert to str because it could be PossixPath
                assert os.path.exists(
                    file_path
                ), f"File for split:{split}, lang:{lang} - {file_path} does not exist!"
        else:
            # just a file
            file_path = str(
                path_or_lang_to_file_mappings
            )  # convert to str because it could be PossixPath
            assert os.path.exists(
                file_path
            ), f"File for split:{split} - {file_path} does not exist!"


@dataclass
class StoryClozeTask(FewShotTask):
    file_format: str = "tsv"

    @classmethod
    def get_supported_languages(cls):
        return ["en", "ru", "zh", "ar", "sw", "hi", "es", "my", "id", "eu", "te"]

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "val2016": {
                "en": DATA_DIR / "storycloze/spring2016.val.tsv",
                "zh": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.zh.tsv",
                "ru": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.ru.tsv",
                "ar": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.ar.tsv",
                "sw": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.sw.tsv",
                "hi": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.hi.tsv",
                "es": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.es.tsv",
                "my": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.my.tsv",
                "id": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.id.tsv",
                "eu": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.eu.tsv",
                "te": "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.te.tsv",
            },
            "test2016": DATA_DIR / "storycloze/spring2016.test.tsv",
            "val2018": DATA_DIR / "storycloze/converted/winter2018.val.tsv",
            "test2018": DATA_DIR / "storycloze/converted/winter2018.test.tsv",
        }

        # Add the val2016 split 20:80 for train:eval

        # val2016_split_20_80_train
        train_split_suffix = "split_20_80_train"
        to_file_mapping["val2016_" + train_split_suffix] = {
            lang: f"{input_path}.{train_split_suffix}.tsv"
            for lang, input_path in to_file_mapping["val2016"].items()
        }
        to_file_mapping["val2016_" + train_split_suffix][
            "en"
        ] = "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.en_original.tsv.split_20_80_train.tsv"

        # val2016_split_20_80_eval
        eval_split_suffix = "split_20_80_eval"
        to_file_mapping["val2016_" + eval_split_suffix] = {
            lang: f"{input_path}.{eval_split_suffix}.tsv"
            for lang, input_path in to_file_mapping["val2016"].items()
        }
        to_file_mapping["val2016_" + eval_split_suffix][
            "en"
        ] = "/private/home/tbmihaylov/data/xlmg/few_shot/story_cloze/translation/spring2016.val.en_original.tsv.split_20_80_eval.tsv"

        assert_file_paths_exist(to_file_mapping)
        return to_file_mapping

    @property
    def default_train_set(self):
        return "val2016"  # GPT-3 also uses the validation set for training

    @property
    def default_eval_set(self):
        return "test2016"

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=[
                    data["RandomFifthSentenceQuiz1"],
                    data["RandomFifthSentenceQuiz2"],
                ],
                correct_candidates=[
                    data["RandomFifthSentenceQuiz1"]
                    if data["AnswerRightEnding"] == "1"
                    else data["RandomFifthSentenceQuiz2"]
                ],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.StoryClozeTemplate


@dataclass
class WinogradTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "test": DATA_DIR / "winograd/WSCollection.xml",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "test"

    @property
    def default_eval_set(self):
        return "test"

    def read_data(self, path):
        if path is None:
            return []
        data = []
        tree = ET.parse(path)
        for schema in tree.getroot().findall("schema"):
            text = schema.find("text")
            candidates = [
                candidate.text.strip()
                for candidate in schema.find("answers").findall("answer")
            ]
            correct = schema.find("correctAnswer").text.strip().strip(".")
            assert correct in ["A", "B"]
            correct_candidate = candidates[0] if correct == "A" else candidates[1]
            data.append(
                {
                    "txt1": text.find("txt1").text.strip(),
                    "txt2": text.find("txt2").text.strip(),
                    "pron": text.find("pron").text.strip(),
                    "candidates": candidates,
                    "correct_candidates": [correct_candidate],
                }
            )
        assert len(data) == 285
        data = data[
            :273
        ]  # The last 12 examples were added recently and are commonly excluded from evaluation
        return data

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=data["candidates"],
                correct_candidates=data["correct_candidates"],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.WinogradTemplate


@dataclass
class XWinogradTask(WinogradTask):
    _language: str = "en"

    @classmethod
    def get_supported_languages(cls):
        return ["en", "fr", "jp", "pt", "ru", "zh"]

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "test": {
                lang_code: DATA_DIR
                / f"crosslingual_winograd/data_per_lang/{lang_code}.tsv"
                for lang_code in cls.get_supported_languages()
            },
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "test"

    @property
    def default_eval_set(self):
        return "test"

    def read_data(self, path):
        data = []
        if path is None:
            return data
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                lang, text1, pron, text2, ref, answers = line.strip("\n").split("\t")
                candidates = []
                correct_candidate = None
                for answer in json.loads(answers):
                    candidates.append(answer[0])
                    if answer[3]:
                        correct_candidate = answer[0]
                assert correct_candidate is not None
                data.append(
                    {
                        "lang": lang,
                        "txt1": text1,
                        "txt2": text2,
                        "pron": pron,
                        "candidates": candidates,
                        "correct_candidates": [correct_candidate],
                    }
                )
            return data

    @classmethod
    def get_supported_languages(cls):
        return ["en", "fr", "jp", "pt", "ru", "zh"]


@dataclass
class WinograndeTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train_xs": DATA_DIR / "winogrande/train_xs.jsonl",
            "train_s": DATA_DIR / "winogrande/train_s.jsonl",
            "train_m": DATA_DIR / "winogrande/train_m.jsonl",
            "train_l": DATA_DIR / "winogrande/train_l.jsonl",
            "train_xl": DATA_DIR / "winogrande/train_xl.jsonl",
            "dev": DATA_DIR / "winogrande/dev.jsonl",
            "test": DATA_DIR / "winogrande/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train_xl"

    @property
    def default_eval_set(self):
        return "dev"

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=[data["option1"], data["option2"]],
                correct_candidates=[data["option" + data["answer"]]],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.WinograndeTemplate


@dataclass
class PIQATask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "piqa/train",
            "valid": DATA_DIR / "piqa/valid",
            "test": DATA_DIR / "piqa/tests",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "valid"

    def read_data(self, path):
        if path is None:
            return []
        data = super().read_data(str(path) + ".jsonl")
        try:
            with open(str(path) + "-labels.lst", encoding="utf-8") as f:
                labels = [int(line) for line in f]
        except:
            labels = [0] * len(data)  # label is unknown for test!
        assert len(data) == len(labels)
        for sample, label in zip(data, labels):
            sample["label"] = label
        return data

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=["sol1", "sol2"],
                correct_candidates=["sol" + str(data["label"] + 1)],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.PIQATemplate


@dataclass
class ARCChallengeTask(MultiChoiceWithCalibrationTask):
    max_hits_cnt: int = 0

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "arc/ARC-Challenge/ARC-Challenge-Train.jsonl",
            "dev": DATA_DIR / "arc/ARC-Challenge/ARC-Challenge-Dev.jsonl",
            "test": DATA_DIR / "arc/ARC-Challenge/ARC-Challenge-Test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)

        # parse calibration options
        calib_fields = {}
        if "::" in calibration_option:
            calib_fields = parse_calibration_params(calibration_option)

        # context
        calibration_para = calib_fields.get(
            "context", calib_fields.get("paragraph", None)
        )

        # question
        question = data["question"]["stem"]
        question = calib_fields.get("question", question)
        data["question"]["stem"] = question

        for ch in data["question"]["choices"]:
            if calibration_para is not None:
                # set all choice paragraphs to the calibration paragraph
                ch["para"] = calibration_para

        return FewShotSample(
            data=data,
            candidates=[choice["label"] for choice in data["question"]["choices"]],
            correct_candidates=[data["answerKey"]],
        )

    def build_samples(self, data):
        def clean_text(txt):
            txt = re.sub(r"\n+", " ", txt).strip()
            return txt

        for ch in data["question"]["choices"]:
            if "hits" in ch:
                if self.max_hits_cnt > 0:
                    hits = [clean_text(x) for x in ch["hits"][: self.max_hits_cnt]]
                    ch["para"] = " ".join(hits)

                del ch["hits"]  # save memory

        return [
            FewShotSample(
                data=data,
                candidates=[choice["label"] for choice in data["question"]["choices"]],
                correct_candidates=[data["answerKey"]],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.ARCOldTemplate


@dataclass
class ARCEasyTask(ARCChallengeTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "arc/ARC-Easy/ARC-Easy-Train.jsonl",
            "dev": DATA_DIR / "arc/ARC-Easy/ARC-Easy-Dev.jsonl",
            "test": DATA_DIR / "arc/ARC-Easy/ARC-Easy-Test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"


@dataclass
class OpenBookQATask(ARCChallengeTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "openbookqa/Main/train.jsonl",
            "dev": DATA_DIR / "openbookqa/Main/dev.jsonl",
            "test": DATA_DIR / "openbookqa/Main/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"

    @classmethod
    def get_default_template_class(cls):
        return templates.OpenBookQATemplate


@dataclass
class CommonsenseQATask(ARCChallengeTask):
    metrics: Tuple[FewShotMetric] = (AccuracyMetric(),)

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "commonsenseqa/train.jsonl",
            "dev": DATA_DIR / "commonsenseqa/dev.jsonl",
            "test": DATA_DIR / "commonsenseqa/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"


EXAMS_LANG2CODE = {
    "Albanian": "sq",
    "Arabic": "ar",
    "Bulgarian": "bg",
    "Croatian": "hr",
    "French": "fr",
    "German": "de",
    "Hungarian": "hu",
    "Italian": "it",
    "Lithuanian": "lt",
    "North Macedonian": "mk",
    "Polish": "pl",
    "Portuguese": "pt",
    "Serbian": "sr",
    "Spanish": "es",
    "Turkish": "tr",
    "Vietnamese": "vi",
}


@dataclass
class ExamsTask(ARCChallengeTask):
    max_hits_cnt: int = 0
    _language: str = "bg"

    @classmethod
    def get_supported_languages(cls):
        # We are evaluating on train/dev for now so return languages that have them
        return ["sq", "bg", "hr", "hu", "it", "mk", "pl", "pt", "sr", "tr", "vi"]
        # return list(sorted(EXAMS_LANG2CODE.values()))

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            # we set these explicitly to be able to retrieve the supported languages automatically!
            "train": {
                lang_code: DATA_DIR / "exams/multilingual/with_hits/train.jsonl"
                for lang_code in cls.get_supported_languages()
            },
            "dev": {
                lang_code: DATA_DIR / "exams/multilingual/with_hits/dev.jsonl"
                for lang_code in cls.get_supported_languages()
            },
            "test": {
                lang_code: DATA_DIR / "exams/multilingual/with_hits/test.jsonl"
                for lang_code in EXAMS_LANG2CODE.values()
            },
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"

    def build_samples(self, data):
        if data["answerKey"] == "@":
            # Manually skipping invalid samples where the gold answer is not a valid choice
            return []
        else:
            return super().build_samples(data)

    def read_data(self, path):
        samples = super().read_data(path)
        return [
            sample
            for sample in samples
            if EXAMS_LANG2CODE[sample["info"]["language"]] == self.language
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.ExamsTemplate


@dataclass
class AbstractODQATask(FewShotTask):
    metrics: Tuple[FewShotMetric] = (OpenDomainQAMetric(),)

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                correct_candidates=data["answer"],
            )
        ]

    def get_max_candidate_length(self, *args, **kwargs):
        return 20  # Should be large enough for all answers

    @classmethod
    def get_default_template_class(cls):
        return templates.ODQATemplate


@dataclass
class RAIPIILeaksTask(FewShotTask):
    metrics: Tuple[FewShotMetric] = ((),)

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "eval_demo": DATA_DIR / "rai_pii_leaks/eval_demo.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return None

    @property
    def default_eval_set(self):
        return "eval_demo"

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                correct_candidates=data["expected_output"],
            )
        ]

    def get_max_candidate_length(self, *args, **kwargs):
        return 50  # Should be large enough for all answers

    @classmethod
    def get_default_template_class(cls):
        return templates.GenerateTextTemplate

    @classmethod
    def get_task_name(cls):
        return "rai_pii_leaks"


@dataclass
class NaturalQuestionsTask(AbstractODQATask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "odqa/nq.train.jsonl",
            "dev": DATA_DIR / "odqa/nq.dev.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"


@dataclass
class TriviaQATask(AbstractODQATask):
    # TODO GPT-3 was evaluated on the wiki test server
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "odqa/triviaqa.train.jsonl",
            "dev": DATA_DIR / "odqa/triviaqa.dev.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"


@dataclass
class WebQuestionsTask(AbstractODQATask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "odqa/webquestions.train.jsonl",
            "test": DATA_DIR / "odqa/webquestions.test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "test"


@dataclass
class WiCTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": SUPERGLUE_DIR / "WiC/train.jsonl",
            "val": SUPERGLUE_DIR / "WiC/val.jsonl",
            "test": SUPERGLUE_DIR / "WiC/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=["false", "true"],
                correct_candidates=["true" if data["label"] else "false"],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.WiCTemplate


@dataclass
class BoolQTask(MultiChoiceWithCalibrationTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": SUPERGLUE_DIR / "BoolQ/train.jsonl",
            "val": SUPERGLUE_DIR / "BoolQ/val.jsonl",
            "test": SUPERGLUE_DIR / "BoolQ/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    # candidates and mappings from labels
    @property
    def candidates(self):
        return ["false", "true"]

    def label_to_candidates(self, data):
        label_to_candidate_map: Dict[Any, str] = {
            False: "false",
            True: "true",
            "False": "false",
            "True": "true",
        }

        correct_candidates = [label_to_candidate_map[data["label"]]]
        return correct_candidates

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)
        calib_fields = parse_calibration_params(calibration_option)

        data["passage"] = calib_fields.get(
            "passage", calib_fields.get("paragraph", data["passage"])
        )
        data["question"] = calib_fields.get("question", data["question"])
        data["label"] = calib_fields.get("label", data["label"])

        return FewShotSample(
            data=data,
            candidates=self.candidates,
            correct_candidates=self.label_to_candidates(data),
        )

    @classmethod
    def get_default_template_class(cls):
        return templates.BoolQTemplate


@dataclass
class CBTask(MultiChoiceWithCalibrationTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": SUPERGLUE_DIR / "CB/train.jsonl",
            "val": SUPERGLUE_DIR / "CB/val.jsonl",
            "test": SUPERGLUE_DIR / "CB/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    metrics: Tuple[FewShotMetric] = (
        AccuracyMetric(),
        PrecisionRecallF1Metric(
            average="macro", labels=["entailment", "contradiction", "neutral"]
        ),
        PrecisionRecallF1Metric(
            average="micro", labels=["entailment", "contradiction", "neutral"]
        ),
    )

    # candidates and mappings from labels
    @property
    def candidates(self):
        return ["entailment", "contradiction", "neutral"]

    def label_to_candidates(self, data):
        correct_candidates = [data["label"]]
        return correct_candidates

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)
        calib_fields = parse_calibration_params(calibration_option)

        data["premise"] = calib_fields.get("premise", data["premise"])
        data["hypothesis"] = calib_fields.get("hypothesis", data["hypothesis"])
        data["label"] = calib_fields.get("label", data["label"])

        return FewShotSample(
            data=data,
            candidates=self.candidates,
            correct_candidates=self.label_to_candidates(data)
            if self.label_to_candidates(data) in self.candidates
            else [],
        )

    @classmethod
    def get_default_template_class(cls):
        return templates.CBTemplate


@dataclass
class RTETask(MultiChoiceWithCalibrationTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": SUPERGLUE_DIR / "RTE/train.jsonl",
            "val": SUPERGLUE_DIR / "RTE/val.jsonl",
            "test": SUPERGLUE_DIR / "RTE/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    # candidates and mappings from labels
    @property
    def candidates(self):
        return ["entailment", "not_entailment"]

    def label_to_candidates(self, data):
        correct_candidates = [data["label"]]
        return correct_candidates

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)
        calib_fields = parse_calibration_params(calibration_option)

        data["premise"] = calib_fields.get("premise", data["premise"])
        data["hypothesis"] = calib_fields.get("hypothesis", data["hypothesis"])
        data["label"] = calib_fields.get("label", data["label"])

        return FewShotSample(
            data=data,
            candidates=self.candidates,
            correct_candidates=self.label_to_candidates(data)
            if data["label"] in self.candidates
            else [],
        )

    @classmethod
    def get_default_template_class(cls):
        return templates.RTETemplate


@dataclass
class WSCTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": SUPERGLUE_DIR / "WSC/train.jsonl",
            "val": SUPERGLUE_DIR / "WSC/val.jsonl",
            "test": SUPERGLUE_DIR / "WSC/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=["false", "true"],
                correct_candidates=["true" if data["label"] else "false"],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.WSCTemplate


@dataclass
class ReCoRDTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": SUPERGLUE_DIR / "ReCoRD/train.jsonl",
            "val": SUPERGLUE_DIR / "ReCoRD/val.jsonl",
            "test": SUPERGLUE_DIR / "ReCoRD/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    def build_samples(self, data):
        candidates = [
            (entity["start"], entity["end"] + 1)
            for entity in data["passage"]["entities"]
        ]
        samples = []
        for qas in data["qas"]:
            correct_candidates = []
            for answer in qas["answers"]:
                candidate = (answer["start"], answer["end"] + 1)
                assert (
                    data["passage"]["text"][candidate[0] : candidate[1]]
                    == answer["text"]
                )
                correct_candidates.append(candidate)
            qas_data = dict(data)
            qas_data["qas"] = qas
            samples.append(
                FewShotSample(
                    data=qas_data,
                    candidates=candidates,
                    correct_candidates=correct_candidates,
                )
            )
        return samples

    @classmethod
    def get_default_template_class(cls):
        return templates.ReCoRDTemplate


@dataclass
class MultiRCTask(MultiChoiceWithCalibrationTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": SUPERGLUE_DIR / "MultiRC/train.jsonl",
            "val": SUPERGLUE_DIR / "MultiRC/val.jsonl",
            "test": SUPERGLUE_DIR / "MultiRC/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    metrics: Tuple[FewShotMetric] = (
        MultiRCPRF1Metric(positive_candidate="true"),
        AccuracyMetric(),
    )

    @property
    def candidates(self):
        return ["false", "true"]

    def label_to_candidates(self, data):
        correct_candidates = [self.candidates[data["question"]["answer"]["label"]]]
        return correct_candidates

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)

        # parse calibration options
        if "::" in calibration_option:
            calib_fields = parse_calibration_params(calibration_option)
        else:
            # If the calib option does not contain fields,
            # it is considered a "dummy" answer
            calib_fields = {"answer": calibration_option}

        # answer
        answer = data["question"]["answer"]
        answer["label"] = 0
        answer["text"] = calib_fields.get("answer", answer["text"])

        # context
        paragraph_text = data["text"]
        paragraph_text = calib_fields.get(
            "text",
            calib_fields.get("context", calib_fields.get("paragraph", paragraph_text)),
        )
        data["text"] = paragraph_text

        # question
        question = data["question"]["question"]
        question = calib_fields.get("question", question)
        data["question"]["question"] = question

        return FewShotSample(
            data=data,
            candidates=copy.deepcopy(original_sample.candidates),
            correct_candidates=self.label_to_candidates(data),
        )

    def build_samples(self, data):
        data = data["passage"]
        samples = []

        for question in data["questions"]:
            sample_data = copy.deepcopy(data)
            del sample_data["questions"]
            sample_data["question"] = question
            subproblems = []
            for answ_id, answer in enumerate(question["answers"]):
                subproblem_data = copy.deepcopy(sample_data)
                del subproblem_data["question"]["answers"]
                subproblem_data["question"]["answer"] = answer
                subproblems.append(
                    FewShotSample(
                        data=subproblem_data,
                        candidates=self.candidates,
                        correct_candidates=self.label_to_candidates(subproblem_data),
                    )
                )

            samples.append(
                FewShotSample(
                    data=copy.deepcopy(subproblems[0].data),
                    candidates=self.candidates,
                    subproblems=subproblems,
                )
            )
        return samples

    @classmethod
    def get_default_template_class(cls):
        return templates.MultiRCTemplate


@dataclass
class SST2Task(FewShotTask):
    file_format: str = "tsv"
    eval_file: Optional[Union[str, Path]] = GLUE_DIR / "SST-2/dev.tsv"
    train_file: Optional[Union[str, Path]] = GLUE_DIR / "SST-2/train.tsv"

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": GLUE_DIR / "SST-2/train.tsv",
            "dev": GLUE_DIR / "SST-2/dev.tsv",
            "test": GLUE_DIR / "SST-2/test.tsv",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"

    def build_samples(self, data):
        # def clean_sentence(sent):
        #     return sent.replace(' , ', ', ').replace(" 's", "'s").replace(' .', '.')
        # data['sentence'] = clean_sentence(data['sentence'])
        return [
            FewShotSample(
                data=data,
                candidates=["0", "1"],
                correct_candidates=[data["label"]],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.SentimentAnalysisTemplate


@dataclass
class CompositionalInstructionsClassificationTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        INSTRUCTIONS_DIR = (
            "/private/home/tbmihaylov/data/synthetic-instructions/v1-h2-21/"
        )
        to_file_mapping = {
            "wiki40b-val-fact-composition-1": INSTRUCTIONS_DIR
            + "instruction-generation-v1-per-token-ds-hf-wiki40b-validation-docs-500-inputlens1-compfacts1.jsonl",
            "wiki40b-val-fact-composition-2": INSTRUCTIONS_DIR
            + "instruction-generation-v1-per-token-ds-hf-wiki40b-validation-docs-500-inputlens1-compfacts2.jsonl",
            "wiki40b-val-fact-composition-3": INSTRUCTIONS_DIR
            + "instruction-generation-v1-per-token-ds-hf-wiki40b-validation-docs-500-inputlens1-compfacts3.jsonl",
            "wiki40b-val-fact-composition-4": INSTRUCTIONS_DIR
            + "instruction-generation-v1-per-token-ds-hf-wiki40b-validation-docs-500-inputlens1-compfacts4.jsonl",
            "wiki40b-val-fact-composition-5": INSTRUCTIONS_DIR
            + "instruction-generation-v1-per-token-ds-hf-wiki40b-validation-docs-500-inputlens1-compfacts5.jsonl",
        }

        for k, v in to_file_mapping.items():
            assert os.path.exists(v)

        return to_file_mapping

    @property
    def default_train_set(self):
        return "wiki40b-val-fact-composition-1"

    @property
    def default_eval_set(self):
        return "wiki40b-val-fact-composition-2"

    metrics: Tuple[FewShotMetric] = (CompositionalInstructionsAccuracyMetric(),)

    def build_samples_from_file(self, data_file):
        with data_utils.numpy_seed(0):
            samples = []
            sample_id = 0
            for entry in self.read_data(data_file):
                for sample in self.build_samples(entry):
                    sample.data["sample_id"] = sample_id
                    samples.append(sample)
                    sample_id += 1

        return samples

    def build_samples(self, data):
        samples = []

        full_task_labels = data["labels"][:]
        full_task_correct_candidate = data["gold_label"]

        # this is full problem
        full_problem = FewShotSample(
            data={
                "instruction": data["full_task"],
                "input": data["input_text"],
                "full_task": True,
                "source": data,
                "gold_label": full_task_correct_candidate,
                "labels": full_task_labels,
            },
            candidates=random.sample(full_task_labels, len(full_task_labels)),
            correct_candidates=[full_task_correct_candidate],
        )

        sub_problems = []
        for step_id, instruction_step in enumerate(data["instruction_steps"]):
            curr_candidates = ["True", "False"]
            curr_correct_candidate = "True"

            sub_problems.append(
                FewShotSample(
                    data={
                        "instruction": instruction_step["statement"],
                        "input": data["input_text"],
                        "full_task": False,
                        "source": data,
                        "sub_task_id": step_id,
                        "gold_label": curr_correct_candidate,
                        "labels": curr_candidates,
                    },
                    candidates=random.sample(curr_candidates, len(curr_candidates)),
                    correct_candidates=[curr_correct_candidate],
                )
            )

        samples.append(
            FewShotSample(
                data=data,
                candidates=full_problem.candidates,
                subproblems=[full_problem] + sub_problems,
            )
        )

        return samples

    @classmethod
    def get_default_template_class(cls):
        return templates.CompositionalInstructionsClassificationTemplate

    @classmethod
    def get_task_name(cls):
        return "compositional_instructions_classification"


def numpy_random_sample(items: List[Any], k: int):
    """Samples k items using the numpy random choice.
    Call this inside

    Args:
        items (List[Any]): Items to sample from
        k (int): Number of items to sample
    """
    sampled_idxs = np.random.choice(len(items), k, replace=False)
    return [items[i] for i in sampled_idxs]


@dataclass
class CompositionalInstructionsClassificationWithCompositionalSubTasksv1_1Task(
    CompositionalInstructionsClassificationTask
):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        INSTRUCTIONS_DIR = "/large_experiments/xlmg/data/instruction_understanding/compositional_instructions/2022-01-18-v1.1/"
        to_file_mapping = {
            "wiki40b-val-fact-500-docs-comp-2": INSTRUCTIONS_DIR
            + "instruction-generation-v1-per-token-ds-hf-wiki40b-validation-docs-500-inputlens1-compfacts2-maxexcheck-20.jsonl",
            "wiki40b-val-fact-500-docs-comp-3": INSTRUCTIONS_DIR
            + "instruction-generation-v1-per-token-ds-hf-wiki40b-validation-docs-500-inputlens1-compfacts3-maxexcheck-20.jsonl",
            "wiki40b-val-fact-500-docs-comp-4": INSTRUCTIONS_DIR
            + "instruction-generation-v1-per-token-ds-hf-wiki40b-validation-docs-500-inputlens1-compfacts4-maxexcheck-20.jsonl",
            "wiki40b-val-fact-500-docs-comp-5": INSTRUCTIONS_DIR
            + "instruction-generation-v1-per-token-ds-hf-wiki40b-validation-docs-500-inputlens1-compfacts5-maxexcheck-20.jsonl",
        }

        for k, v in to_file_mapping.items():
            assert os.path.exists(v)

        return to_file_mapping

    @property
    def default_train_set(self):
        return "wiki40b-val-fact-500-docs-comp-2"

    @property
    def default_eval_set(self):
        return "wiki40b-val-fact-500-docs-comp-3"

    metrics: Tuple[FewShotMetric] = (CompositionalInstructionsAccuracyMetric(),)

    def build_samples(self, data):
        samples = []

        label_options = [f"Label{x}" for x in range(21, 95)]
        full_task_labels = numpy_random_sample(label_options, 2)

        full_task_correct_candidate = full_task_labels[0]
        full_task_incorrect_candidate = full_task_labels[1]

        if (
            len(data["examples"]["positive"]) > 0
            and len(data["examples"]["negative"]) > 0
        ):
            balance_min_count = min(
                len(data["examples"]["positive"]), len(data["examples"]["negative"])
            )
            example_options = numpy_random_sample(
                data["examples"]["positive"], balance_min_count
            ) + numpy_random_sample(data["examples"]["negative"], balance_min_count)
        elif len(data["examples"]["positive"]) > 0:
            example_options = data["examples"]["positive"]
        else:
            example_options = data["examples"]["negative"]

        input_example = numpy_random_sample(example_options, 1)[0]

        def has_matches(query_results: Dict[str, List[Any]]):
            return all([len(res) > 0 for _, res in query_results.items()])

        def get_atomic_steps_predicates_with_polarity(instruction_task):
            atomic_steps_types = []
            for step_args in instruction_task["instruction_facts"]:
                step_type = step_args[:2]  # ["+", "is_uppercase"]
                atomic_steps_types.append(step_type)

            return atomic_steps_types

        # this is full problem
        full_problem = FewShotSample(
            data={
                "instruction": data["full_task"],
                "input": input_example["text"],
                "full_task": True,
                "source": data,
                "instruction_steps_types": get_atomic_steps_predicates_with_polarity(
                    data
                ),
                "input_example": input_example,
                "gold_label": full_task_correct_candidate,
                "labels": full_task_labels,
            },
            candidates=numpy_random_sample(full_task_labels, len(full_task_labels)),
            correct_candidates=[
                full_task_correct_candidate
                if input_example["match"]
                else full_task_incorrect_candidate
            ],  # We flip when the input example is negative
        )

        sub_problems = []
        for subtask_id, sub_task in enumerate(data["comp_sub_tasks"]):
            curr_candidates = numpy_random_sample(label_options, 2)
            curr_correct_candidate = curr_candidates[0]
            curr_incorrect_candidate = curr_candidates[1]

            if len(input_example["states"]) > subtask_id:
                sub_problem_example = input_example["states"][subtask_id]
            else:
                sub_problem_example = None

            if sub_problem_example is None:
                sub_problem_example_match = False
            else:
                if "match" not in sub_problem_example:
                    sub_problem_example_match = has_matches(
                        sub_problem_example["results"]
                    )
                    sub_problem_example["match"] = sub_problem_example_match
                else:
                    sub_problem_example_match = sub_problem_example["match"]

            sub_problems.append(
                FewShotSample(
                    data={
                        "instruction": sub_task["full_task"],
                        "input": input_example["text"],
                        "full_task": True,
                        "source": sub_task,
                        "instruction_steps_types": get_atomic_steps_predicates_with_polarity(
                            sub_task
                        ),
                        "input_example": sub_problem_example,
                        "sub_task_id": subtask_id,
                        "gold_label": curr_correct_candidate,
                        "labels": curr_candidates,
                    },
                    candidates=numpy_random_sample(
                        curr_candidates, len(curr_candidates)
                    ),
                    correct_candidates=[
                        curr_correct_candidate
                        if sub_problem_example_match
                        else curr_incorrect_candidate
                    ],
                )
            )

        samples.append(
            FewShotSample(
                data=data,
                candidates=full_problem.candidates,
                subproblems=[full_problem] + sub_problems,
            )
        )

        return samples

    @classmethod
    def get_default_template_class(cls):
        return templates.CompositionalInstructionsClassificationCustomv3SimpleTemplate

    @classmethod
    def get_task_name(cls):
        return "cic_v1_1_comp_subtasks"


@dataclass
class CompositionalInstructionsClassificationRandomLabelsTask(
    CompositionalInstructionsClassificationTask
):
    def build_samples(self, data):
        samples = []

        label_options = [f"Label{x}" for x in range(21, 50)]
        full_task_labels = [
            label_options[x]
            for x in np.random.choice(len(label_options), 2, replace=False)
        ]

        full_task_correct_candidate = full_task_labels[0]

        # this is full problem
        full_problem = FewShotSample(
            data={
                "instruction": data["full_task"],
                "input": data["input_text"],
                "full_task": True,
                "source": data,
                "gold_label": full_task_correct_candidate,
                "labels": full_task_labels,
            },
            candidates=random.sample(full_task_labels, len(full_task_labels)),
            correct_candidates=[full_task_correct_candidate],
        )

        sub_problems = []
        for step_id, instruction_step in enumerate(data["instruction_steps"]):
            curr_candidates = [
                label_options[x]
                for x in np.random.choice(len(label_options), 2, replace=False)
            ]
            curr_correct_candidate = curr_candidates[0]

            sub_problems.append(
                FewShotSample(
                    data={
                        "instruction": instruction_step["statement"],
                        "input": data["input_text"],
                        "full_task": False,
                        "source": data,
                        "sub_task_id": step_id,
                        "gold_label": curr_correct_candidate,
                        "labels": curr_candidates,
                    },
                    candidates=random.sample(curr_candidates, len(curr_candidates)),
                    correct_candidates=[curr_correct_candidate],
                )
            )

        samples.append(
            FewShotSample(
                data=data,
                candidates=full_problem.candidates,
                subproblems=[full_problem] + sub_problems,
            )
        )

        return samples

    @classmethod
    def get_default_template_class(cls):
        return templates.CompositionalInstructionsClassificationCustomv1Template

    @classmethod
    def get_task_name(cls):
        return "cic_random_labels"


@dataclass
class SNLITask(MultiChoiceWithCalibrationTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": GLUE_DIR / "SNLI/train.jsonl",
            "dev": GLUE_DIR / "SNLI/dev.jsonl",
            "test": GLUE_DIR / "SNLI/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"

    @property
    def candidates(self):
        return ["entailment", "contradiction", "neutral"]

    def label_to_candidates(self, data):
        correct_candidates = [data["gold_label"]]
        return correct_candidates

    def build_samples(self, data):
        if data["gold_label"] == "-":
            return []
        if "premise" not in data and "sentence1" in data:
            data["premise"] = data["sentence1"]
        if "hypothesis" not in data and "sentence2" in data:
            data["hypothesis"] = data["sentence2"]
        return [
            FewShotSample(
                data=data,
                candidates=self.candidates,
                correct_candidates=self.label_to_candidates(data),
            )
        ]

    def build_single_calibration_sample(self, original_sample, calibration_option):
        data = copy.deepcopy(original_sample.data)
        calib_fields = parse_calibration_params(calibration_option)

        data["premise"] = calib_fields.get(
            "sentence1",
            calib_fields.get("premise", data.get("premise", data.get("sentence1"))),
        )
        data["sentence1"] = data[
            "premise"
        ]  # some templates use sentence1 and some premise

        data["hypothesis"] = calib_fields.get(
            "sentence2",
            calib_fields.get(
                "hypothesis", data.get("hypothesis", data.get("sentence2"))
            ),
        )
        data["sentence2"] = data[
            "hypothesis"
        ]  # some templates use sentence2 and some hypothesis

        return FewShotSample(
            data=data,
            candidates=self.candidates,
            correct_candidates=self.candidates,  # this is not used
        )

    @classmethod
    def get_default_template_class(cls):
        return templates.GPT3StyleNLITemplate


@dataclass
class MNLIMatchedTask(SNLITask):
    file_format: str = "tsv"

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": GLUE_DIR / "MNLI/train.tsv",
            "dev_matched": GLUE_DIR / "MNLI/dev_matched.tsv",
            "dev_mismatched": GLUE_DIR / "MNLI/dev_mismatched.tsv",
            "test_matched": GLUE_DIR / "MNLI/test_matched.tsv",
            "test_mismatched": GLUE_DIR / "MNLI/test_mismatched.tsv",
            "diagnostic": GLUE_DIR / "MNLI/diagnostic.tsv",
            "diagnostic_full": GLUE_DIR / "MNLI/diagnostic-full.tsv",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev_matched"


@dataclass
class MNLIMismatchedTask(MNLIMatchedTask):
    @property
    def default_eval_set(self):
        return "dev_mismatched"


@dataclass
class ANLIR1Task(SNLITask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "anli/R1/train.jsonl",
            "dev": DATA_DIR / "anli/R1/dev.jsonl",
            "test": GLUE_DIR / "anli/R1/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"

    name2label = {"e": "entailment", "c": "contradiction", "n": "neutral"}

    def build_samples(self, data):
        data["premise"] = data["context"]
        data["sentence1"] = data["premise"]
        data["sentence2"] = data["hypothesis"]
        return [
            FewShotSample(
                data=data,
                candidates=["entailment", "contradiction", "neutral"],
                correct_candidates=[self.name2label[data["label"]]],
            )
        ]


@dataclass
class ANLIR2Task(ANLIR1Task):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "anli/R2/train.jsonl",
            "dev": DATA_DIR / "anli/R2/dev.jsonl",
            "test": GLUE_DIR / "anli/R2/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"


@dataclass
class ANLIR3Task(ANLIR1Task):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "anli/R3/train.jsonl",
            "dev": DATA_DIR / "anli/R3/dev.jsonl",
            "test": GLUE_DIR / "anli/R3/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "dev"


@dataclass
class GlueDiagTask(SNLITask):
    """GLUE benchmark diagnostics set"""

    file_format: str = "tsv"
    eval_file: Optional[Union[str, Path]] = DATA_DIR / "glue_diagnostics/test.tsv"
    train_file: Optional[Union[str, Path]] = None

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {"test": DATA_DIR / "glue_diagnostics/test.tsv"}

        return to_file_mapping

    @property
    def default_train_set(self):
        return None

    @property
    def default_eval_set(self):
        return "test"

    metrics: Tuple[FewShotMetric] = (GlueDiagMetrics(),)

    def build_samples(self, data):
        if data["Label"] == "-":
            return []
        data["sentence1"] = data["Premise"]
        data["sentence2"] = data["Hypothesis"]
        return [
            FewShotSample(
                data=data,
                candidates=["entailment", "contradiction", "neutral"],
                correct_candidates=[data["Label"]],
            )
        ]


@dataclass
class XNLITask(SNLITask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "dev": {
                lang_code: DATA_DIR / "XNLI-1.0/xnli.dev.jsonl"
                for lang_code in cls.get_supported_languages()
            },
            "test": {
                lang_code: DATA_DIR / "XNLI-1.0/xnli.test.jsonl"
                for lang_code in cls.get_supported_languages()
            },
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "dev"  # HACK: we use the dev set for few-shot for now

    @property
    def default_eval_set(self):
        return "dev"

    @classmethod
    def get_supported_languages(cls):
        return [
            "en",
            "fr",
            "es",
            "de",
            "el",
            "bg",
            "ru",
            "tr",
            "ar",
            "vi",
            "th",
            "zh",
            "hi",
            "sw",
            "ur",
        ]

    def read_data(self, path):
        samples = super().read_data(path)
        return [sample for sample in samples if sample["language"] == self.language]


@dataclass
class DiagnosisTask(FewShotTask):
    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=list(range(len(data["choices"]))),
                correct_candidates=[data["label"]],
            )
        ]

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "test"

    # Setting this argument for all diagnostic tasks
    # since the train set has only 32 examples and
    # 32-shot eval will throw an error with out setting this.
    @property
    def default_valid_set(self):
        return "train"

    @classmethod
    def get_default_template_class(cls):
        return templates.DiagnosisTemplate


@dataclass
class DiagnosisCountryTask(DiagnosisTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "diagnosis/country.train.jsonl",
            "test": DATA_DIR / "diagnosis/country.test.jsonl",
        }

        return to_file_mapping


@dataclass
class DiagnosisCityTask(DiagnosisTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "diagnosis/city.train.jsonl",
            "test": DATA_DIR / "diagnosis/city.test.jsonl",
        }

        return to_file_mapping


@dataclass
class DiagnosisNameTask(DiagnosisTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "diagnosis/name.train.jsonl",
            "test": DATA_DIR / "diagnosis/name.test.jsonl",
        }

        return to_file_mapping


@dataclass
class DiagnosisBrandTask(DiagnosisTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "diagnosis/brand.train.jsonl",
            "test": DATA_DIR / "diagnosis/brand.test.jsonl",
        }

        return to_file_mapping


@dataclass
class DiagnosisPos1Task(DiagnosisTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "diagnosis/pos1.train.jsonl",
            "test": DATA_DIR / "diagnosis/pos1.test.jsonl",
        }

        return to_file_mapping


@dataclass
class DiagnosisPos2Task(DiagnosisTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "diagnosis/pos2.train.jsonl",
            "test": DATA_DIR / "diagnosis/pos2.test.jsonl",
        }

        return to_file_mapping


@dataclass
class DiagnosisPos3Task(DiagnosisTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "diagnosis/pos3.train.jsonl",
            "test": DATA_DIR / "diagnosis/pos3.test.jsonl",
        }

        return to_file_mapping


@dataclass
class DiagnosisPos4Task(DiagnosisTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "diagnosis/pos4.train.jsonl",
            "test": DATA_DIR / "diagnosis/pos4.test.jsonl",
        }

        return to_file_mapping


@dataclass
class SyntheticTask(FewShotTask):
    @property
    def default_train_set(self):
        return self.default_eval_set

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                correct_candidates=[data["completion"]],
            )
        ]

    def get_max_candidate_length(self, *args, **kwargs):
        return 10  # Should be large enough for all words


@dataclass
class ArithmeticTask(SyntheticTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            task_set: DATA_DIR / f"gpt-3/{task_set}.jsonl"
            for task_set in [
                "two_digit_addition",
                "three_digit_addition",
                "four_digit_addition",
                "five_digit_addition",
                "six_digit_addition",
                "two_digit_subtraction",
                "three_digit_subtraction",
                "four_digit_subtraction",
                "five_digit_subtraction",
                "six_digit_subtraction",
                "single_digit_three_ops",
                "sum_of_digits",
            ]
        }

        return to_file_mapping

    _default_eval_set = None

    @property
    def default_eval_set(self):
        return self._default_eval_set

    @classmethod
    def sub_tasks_registrations(cls):
        assert (
            "_default_eval_set" in signature(cls).parameters
        ), "Make sure that `_default_eval_set` is a class attribute (not a property) so it can be set from kwargs!"

        task_names_to_eval_set = {
            # These are used for backward name compatability
            "Addition2Digit": "two_digit_addition",
            "Addition3Digit": "three_digit_addition",
            "Addition4Digit": "four_digit_addition",
            "Addition5Digit": "five_digit_addition",
            "Addition6Digit": "six_digit_addition",
            "Subtraction2Digit": "two_digit_subtraction",
            "Subtraction3Digit": "three_digit_subtraction",
            "Subtraction4Digit": "four_digit_subtraction",
            "Subtraction5Digit": "five_digit_subtraction",
            "Subtraction6Digit": "six_digit_subtraction",
            "Multiplication2Digit": "single_digit_three_ops",
            "Singledigit3Ops": "single_digit_three_ops",
            "SumOfDigits": "sum_of_digits",
        }

        sub_tasks = []
        for subtask_name, eval_set_name in task_names_to_eval_set.items():
            task_reg_name = subtask_name.lower()

            sub_tasks.append((task_reg_name, {"_default_eval_set": eval_set_name}))

        return sub_tasks

    @classmethod
    def get_default_template_class(cls):
        return templates.SyntheticTemplate


@dataclass
class UnscramblingTask(SyntheticTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            task_set: DATA_DIR / f"gpt-3/{task_set}.jsonl"
            for task_set in [
                "cycle_letters_in_word",
                "mid_word_1_anagrams",
                "mid_word_2_anagrams",
                "random_insertion_in_word",
                "reversed_words",
            ]
        }

        return to_file_mapping

    @property
    def default_eval_set(self):
        return self._default_eval_set

    @classmethod
    def sub_tasks_registrations(cls):
        assert (
            "_default_eval_set" in signature(cls).parameters
        ), "Make sure that `_default_eval_set` is a class attribute (not a property) so it can be set from kwargs!"
        task_names_to_eval_set = {
            # These are used for backward name compatability
            "CycledLetters": "cycle_letters_in_word",
            "Anagrams1": "mid_word_1_anagrams",
            "Anagrams2": "mid_word_2_anagrams",
            "SymbolInsertion": "random_insertion_in_word",
            "ReversedWords": "reversed_words",
        }

        sub_tasks = []
        for subtask_name, eval_set_name in task_names_to_eval_set.items():
            task_reg_name = subtask_name.lower()
            sub_tasks.append((task_reg_name, {"_default_eval_set": eval_set_name}))

        return sub_tasks

    @classmethod
    def get_default_template_class(cls):
        return templates.UnscramblingTemplate


@dataclass
class MTTask(FewShotTask):
    metrics: Tuple[FewShotMetric] = (BleuMetric(),)

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        # this is not used
        pass

    @property
    def default_train_set(self):
        return self.trainset

    @property
    def default_eval_set(self):
        return self.testset

    @property
    def language(self):
        return self.langpair

    @abstractproperty
    def srclang(self):
        pass

    @abstractproperty
    def tgtlang(self):
        pass

    @abstractproperty
    def trainset(self):
        pass

    @abstractproperty
    def testset(self):
        pass

    @property
    def langpair(self):
        return f"{self.srclang}-{self.tgtlang}"

    def get_src_and_ref_files(self, testset, langpair):
        src = sacrebleu.get_source_file(test_set=testset, langpair=langpair)
        ref = sacrebleu.get_reference_files(test_set=testset, langpair=langpair)
        assert len(ref) == 1, "Multiple references not supported"
        return src, ref[0]

    def get_data_file_path(self, set_name: str, lang_code: str):
        if set_name is None:
            return None
        return self.get_src_and_ref_files(set_name, lang_code)

    def read_data(self, path):
        if path is None:
            return []
        lines = []
        for p in path:
            with open(p, encoding="utf-8") as f:
                lines.append(f.read().splitlines())
        assert len(lines) == 2, "Multiple references not supported"
        src, ref = lines
        assert len(src) == len(
            ref
        ), "Source and reference must have the same number of lines"
        return [{"src": s, "ref": t} for s, t in zip(src, ref)]

    def build_samples(self, data):
        return [FewShotSample(data=data, correct_candidates=[data["ref"]])]

    def get_max_candidate_length(self, *args, **kwargs):
        return 256  # TODO Is this enough for all testsets?

    @classmethod
    def get_default_template_class(cls):
        return templates.MTTemplate


@dataclass
class WMT14FrEnTask(MTTask):
    srclang: str = "fr"
    tgtlang: str = "en"
    testset: str = "wmt14/full"
    trainset: str = "wmt13"


@dataclass
class WMT14EnFrTask(MTTask):
    srclang: str = "en"
    tgtlang: str = "fr"
    testset: str = "wmt14/full"
    trainset: str = "wmt13"


@dataclass
class WMT16DeEnTask(MTTask):
    srclang: str = "de"
    tgtlang: str = "en"
    testset: str = "wmt16"
    trainset: str = "wmt15"


@dataclass
class WMT16EnDeTask(MTTask):
    srclang: str = "en"
    tgtlang: str = "de"
    testset: str = "wmt16"
    trainset: str = "wmt15"


@dataclass
class WMT16RoEnTask(MTTask):
    srclang: str = "ro"
    tgtlang: str = "en"
    testset: str = "wmt16"
    trainset: str = (
        "wmt16"  # HACK We are taking the few-shot examples from the test set
    )


@dataclass
class WMT16EnRoTask(MTTask):
    srclang: str = "en"
    tgtlang: str = "ro"
    testset: str = "wmt16"
    trainset: str = (
        "wmt16"  # HACK We are taking the few-shot examples from the test set
    )


@dataclass
class SATAnalogiesTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "test": DATA_DIR / "sat/SAT-package-V3.txt",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "test"  # HACK Using the eval set for training

    @property
    def default_eval_set(self):
        return "test"

    def read_data(self, path):
        if path is None:
            return []
        data = []
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if not line.startswith("#")]
            assert len(lines) % 9 == 0  # 9 lines per example
            for i in range(0, len(lines), 9):
                assert lines[i] == ""  # First line is empty
                w1, w2, _ = lines[i + 2].split()
                sample = {
                    "stem": {"word1": w1, "word2": w2},
                    "candidates": {},
                    "label": lines[i + 8].strip(),
                }
                for j, candidate in enumerate(["a", "b", "c", "d", "e"]):
                    w1, w2, _ = lines[i + 3 + j].split(" ")
                    sample["candidates"][candidate] = {"word1": w1, "word2": w2}
                data.append(sample)
        return data

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=["a", "b", "c", "d", "e"],
                correct_candidates=[data["label"]],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.SATAnalogiesTemplate


@dataclass
class SimplificationTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "valid": DATA_DIR / "asset/valid.jsonl",
            "test": DATA_DIR / "asset/test.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "test"  # HACK: we use the test set for few-shot for now

    @property
    def default_eval_set(self):
        return "valid"

    metrics: Tuple[FewShotMetric] = (SariMetric(),)

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                correct_candidates=[
                    data["references"][0]
                ],  # We only take the first reference
            )
        ]

    def get_max_candidate_length(self, *args, **kwargs):
        return 100  # Should be large enough for all samples

    @classmethod
    def get_default_template_class(cls):
        return templates.SimplificationTemplate


@dataclass
class RealToxicityPromptsTask(FewShotTask):
    """
    Evaluates on 600 challenging examples from the RealToxicityPrompts task.

    NOTE: some differences from RealToxicityPrompts
    - RTP uses the perspective API. We use a toxicity classifier from ParlAI
        which does was trained on full responses rather than substrings
    - RTP samples generations with nucleus sampling. We return top generation from
        beam search.

    See <https://arxiv.org/pdf/2009.11462.pdf> for more info.
    """

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "train": DATA_DIR / "realtoxicityprompts/train.jsonl",
            "val": DATA_DIR / "realtoxicityprompts/val.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return "train"

    @property
    def default_eval_set(self):
        return "val"

    metrics: Tuple[FewShotMetric] = (RealToxicityPromptsMetric(),)

    @classmethod
    def get_default_template_class(cls):
        return templates.PromptCompletionTemplate

    def build_samples(self, data) -> List[FewShotSample]:
        return [FewShotSample(data=data)]

    def get_max_candidate_length(self, *args, **kwargs):
        return 100  # Should be large enough for all samples


@dataclass
class NaturalInstructionsTask(FewShotTask):
    metrics: Tuple[FewShotMetric] = (BleuMetric(),)

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}
        task_data_dir = str(NATURAL_INSTRUCTIONS_DIR)
        data_files = sorted(list(glob.glob(task_data_dir + "/subtask*.json")))

        sub_tasks = []
        for data_file in data_files:
            task_set_name = os.path.basename(data_file).split(".")[0]
            task_set_name = task_set_name.lower()

            to_file_mapping[task_set_name] = data_file

        return to_file_mapping

    @property
    def default_train_set(self):
        return (
            self.default_eval_set
        )  # HACK We are taking the few-shot examples from the test set

    _default_eval_set: Optional[str] = None

    @property
    def default_eval_set(self):
        return self._default_eval_set

    @classmethod
    def sub_tasks_registrations(cls):
        assert (
            "_default_eval_set" in signature(cls).parameters
        ), "Make sure that `_default_eval_set` is a class attribute (not a property) so it can be set from kwargs!"

        task_sets_to_path = cls.get_sets_and_lang_to_path_mappings()
        sub_tasks = []
        for task_set, data_file in task_sets_to_path.items():
            task_name = cls.get_task_name() + "__" + task_set
            sub_tasks.append((task_name, {"_default_eval_set": task_set}))

        return sub_tasks

    def read_data(self, path):
        if path is None:
            return []

        data_items = []
        with open(path) as f_in:
            data = json.load(f_in)
            instances = data["Instances"]
            del data["Instances"]

            for idx, instance in enumerate(instances):
                item = {
                    "ids": idx,
                    "instructions": data,
                    "instance": instance,
                }

                data_items.append(item)

        return data_items

    def build_samples(self, data):
        outputs = data["instance"]["output"]
        if not isinstance(outputs, list):
            outputs = [outputs]

        samples = []
        for output in outputs:
            samples.append(
                FewShotSample(
                    data=data,
                    correct_candidates=[output],
                )
            )

        return samples

    def get_max_candidate_length(self, *args, **kwargs):
        return 10  # Should be large enough for all words

    @classmethod
    def get_default_template_class(cls):
        return templates.NaturalInstructionsTemplate


@dataclass
class FLANTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}
        task_data_dir = str(FLAN_DIR)
        data_files = sorted(list(glob.glob(task_data_dir + "/*")))

        sub_tasks = []
        for data_file in data_files:
            task_set_name = os.path.basename(data_file)
            task_set_name = task_set_name.lower()

            to_file_mapping[task_set_name] = data_file

        return to_file_mapping

    @property
    def has_candidates(self):
        return True  # For generation tasks, we make the gold generation as the only candidate

    _default_train_set: Optional[str] = None
    _default_valid_set: Optional[str] = None
    _default_eval_set: Optional[str] = None

    @property
    def default_train_set(self):
        return self._default_train_set

    @property
    def default_valid_set(self):
        return self._default_valid_set

    @property
    def default_eval_set(self):
        return self._default_eval_set

    @classmethod
    def sub_tasks_registrations(cls):
        assert (
            "_default_eval_set" in signature(cls).parameters
        ), "Make sure that `_default_eval_set` is a class attribute (not a property) so it can be set from kwargs!"

        task_sets_to_path = cls.get_sets_and_lang_to_path_mappings()
        sub_tasks = set(
            ["_".join(subtask.split("_")[1:]) for subtask in task_sets_to_path.keys()]
        )
        subtasks_reg = []
        for sub_task in sub_tasks:
            task_name = cls.get_task_name() + "__" + sub_task
            if "10templates" in task_name:  # These tasks are used for training
                subtasks_reg.append(
                    (
                        task_name,
                        {
                            "_default_train_set": "train_" + sub_task,
                            "_default_valid_set": "valid_" + sub_task,
                            "_default_eval_set": "test_" + sub_task,
                        },
                    )
                )
            elif "alltemplates" in task_name:  # These tasks are only used for eval
                subtasks_reg.append(
                    (task_name, {"_default_eval_set": "test_" + sub_task})
                )

        return subtasks_reg

    def read_data(self, path):
        if path is None:
            return []

        data_items = []
        with open(path) as f_in:
            for idx, line in enumerate(f_in):
                example = json.loads(line)
                item = {
                    "ids": idx,
                    "input": example["input"],
                    "target": example["target"][: self.get_max_candidate_length()],
                    "candidates": [
                        c[: self.get_max_candidate_length()]
                        for c in example["candidates"]
                    ],
                }

                data_items.append(item)

        return data_items

    def build_samples(self, data):
        return [
            FewShotSample(
                data=data,
                candidates=data["candidates"]
                if len(data["candidates"]) > 0
                else [data["target"]],
                correct_candidates=[data["target"]],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.FLANTemplate

    def get_max_candidate_length(self, *args, **kwargs):
        return 256  # From FLAN: The input and target sequence lengths used in finetuning are 1024 and 256, respectively

    @classmethod
    def get_task_name(cls):
        return "flan"


@dataclass
class NaturalInstructionsExpansionTask(FewShotTask):
    metrics: Tuple[FewShotMetric] = (BleuMetric(),)

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}
        task_data_dir = str(NATURAL_INSTRUCTIONS_EXPANSION_DIR)
        data_files = sorted(list(glob.glob(task_data_dir + "/tasks/task*.json")))

        sub_tasks = []
        for data_file in data_files:
            task_set_name = os.path.basename(data_file).split(".json")[0]
            task_set_name = task_set_name.lower()

            to_file_mapping[task_set_name] = data_file

        return to_file_mapping

    @property
    def default_train_set(self):
        return (
            self.default_eval_set
        )  # HACK We are taking the few-shot examples from the test set

    _default_eval_set: Optional[str] = None

    @property
    def default_eval_set(self):
        return self._default_eval_set

    @classmethod
    def sub_tasks_registrations(cls):
        assert (
            "_default_eval_set" in signature(cls).parameters
        ), "Make sure that `_default_eval_set` is a class attribute (not a property) so it can be set from kwargs!"

        task_sets_to_path = cls.get_sets_and_lang_to_path_mappings()
        sub_tasks = []
        for task_set, data_file in task_sets_to_path.items():
            task_name = cls.get_task_name() + "__" + task_set
            sub_tasks.append((task_name, {"_default_eval_set": task_set}))

        return sub_tasks

    def read_data(self, path):
        if path is None:
            return []

        data_items = []
        with open(path) as f_in:
            data = json.load(f_in)
            instances = data["Instances"]
            del data["Instances"]

            for idx, instance in enumerate(instances):
                item = {
                    "ids": idx,
                    "instructions": data,
                    "instance": instance,
                }

                data_items.append(item)

        return data_items

    def build_samples(self, data):
        outputs = data["instance"]["output"]
        if not isinstance(outputs, list):
            outputs = [outputs]
        else:
            outputs = [outputs[0]]  # take only one output

        return [FewShotSample(data=data, correct_candidates=outputs)]

    def get_max_candidate_length(self, *args, **kwargs):
        return 50  # TODO: need to verify this method

    @classmethod
    def get_default_template_class(cls):
        return templates.NaturalInstructionsExpansionTemplate

    @classmethod
    def get_task_name(cls):
        return "natural_instruct_exp"


@dataclass
class NaturalInstructionsExpansionTrain10Task(NaturalInstructionsExpansionTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}
        task_data_dir = str(NATURAL_INSTRUCTIONS_EXPANSION_DIR)
        with open(os.path.join(task_data_dir, "train_tasks.txt"), "r") as f:
            train_tasks = [line for line in f.read().splitlines()]
        data_files = [
            os.path.join(task_data_dir, "tasks", f"{task}.json") for task in train_tasks
        ]

        for data_file in data_files[:10]:
            task_set_name = os.path.basename(data_file).split(".json")[0]
            task_set_name = task_set_name.lower()

            to_file_mapping[task_set_name] = data_file

        return to_file_mapping

    @classmethod
    def get_task_name(cls):
        return "natural_instruct_exp_train_10"


@dataclass
class NaturalInstructionsExpansionTestTask(NaturalInstructionsExpansionTask):
    # White testing, convert NIE tasks into classification
    metrics: Tuple[FewShotMetric] = (AccuracyMetric(),)
    candidate_choices: List[str] = None  # set this while reading data for each subtask

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}
        task_data_dir = str(NATURAL_INSTRUCTIONS_EXPANSION_DIR)
        with open(os.path.join(task_data_dir, "test_tasks.txt"), "r") as f:
            test_tasks = [line for line in f.read().splitlines()]
        data_files = [
            os.path.join(task_data_dir, "tasks", f"{task}.json") for task in test_tasks
        ]

        for data_file in data_files:
            task_set_name = os.path.basename(data_file).split(".json")[0]
            task_set_name = task_set_name.lower()
            task_set_name = task_set_name.split("_")[
                0
            ]  # TODO: avoiding long task name issues.

            to_file_mapping[task_set_name] = data_file

        return to_file_mapping

    def read_data(self, path):
        if path is None:
            return []
        candidate_choices = []
        data_items = []
        with open(path) as f_in:
            data = json.load(f_in)
            instances = data["Instances"]
            del data["Instances"]

            for idx, instance in enumerate(instances):
                item = {
                    "ids": idx,
                    "instructions": data,
                    "instance": instance,
                }
                if isinstance(instance["output"], list):
                    choice = instance["output"][0]
                else:
                    choice = instance["output"]
                candidate_choices.append(choice)
                data_items.append(item)

        self.candidate_choices = list(set(candidate_choices))
        return data_items

    def build_samples(self, data):
        outputs = data["instance"]["output"]
        if not isinstance(outputs, list):
            outputs = [outputs]

        return [
            FewShotSample(
                data=data, candidates=self.candidate_choices, correct_candidates=outputs
            )
        ]

    @classmethod
    def get_task_name(cls):
        return "natural_instruct_exp_test"


@dataclass
class NaturalInstructionsExpansionTrainTestTask(NaturalInstructionsExpansionTestTask):
    metrics: Tuple[FewShotMetric] = (AccuracyMetric(),)
    candidate_choices: List[str] = None  # set this while reading data for each subtask

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}
        task_data_dir = str(NATURAL_INSTRUCTIONS_EXPANSION_DIR)
        with open(os.path.join(task_data_dir, "train_tasks.txt"), "r") as f:
            test_tasks = [line for line in f.read().splitlines()]
        data_files = [
            os.path.join(task_data_dir, "tasks", f"{task}.json") for task in test_tasks
        ]

        for data_file in data_files:
            task_set_name = os.path.basename(data_file).split(".json")[0]
            task_set_name = task_set_name.lower()
            task_set_name = task_set_name.split("_")[
                0
            ]  # TODO: avoiding long task name issues.

            to_file_mapping[task_set_name] = data_file

        return to_file_mapping

    @classmethod
    def get_task_name(cls):
        return "natural_instruct_exp_train_test"


@dataclass
class NIEReliabilityBenchmark(NaturalInstructionsExpansionTestTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}
        task_data_dir = str(NIE_RELIABILITY_BENCHMARK)
        data_files = sorted(list(glob.glob(task_data_dir + "/*/*/*.json")))

        for data_file in data_files:
            cluster_name, eval_name, task_name = data_file.lower().split("/")[-3:]
            task_name = task_name.split(".json")[0].split("_")[
                0
            ]  # only taking task number
            task_set_name = f"{cluster_name}--{eval_name}--{task_name}"
            to_file_mapping[task_set_name] = data_file

        return to_file_mapping

    @classmethod
    def get_task_name(cls):
        return "nie_reliability_benchmark"

    @classmethod
    def get_default_template_class(cls):
        # return templates.NIENoExampleToFLANTemplate # Use FLAN template for FLAN finetuned models
        return templates.NIENoExampleTemplate


@dataclass
class RegexTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}
        task_data_dir = str(DATA_DIR / "regexp")
        data_files = sorted(list(glob.glob(task_data_dir + "/*")))

        for data_file in data_files:
            task_set_name = os.path.basename(data_file)
            task_set_name = task_set_name.lower()

            to_file_mapping[task_set_name] = data_file

        return to_file_mapping

    _default_eval_set: Optional[str] = None
    _default_valid_set: Optional[str] = None
    _default_train_set: Optional[str] = None

    @property
    def default_train_set(self):
        return self._default_train_set

    @property
    def default_valid_set(self):
        return self._default_valid_set

    @property
    def default_eval_set(self):
        return self._default_eval_set

    @classmethod
    def sub_tasks_registrations(cls):
        assert (
            "_default_eval_set" in signature(cls).parameters
        ), "Make sure that `_default_eval_set` is a class attribute (not a property) so it can be set from kwargs!"

        task_sets_to_path = cls.get_sets_and_lang_to_path_mappings()
        sub_tasks = set(
            [subtask.split("_")[-1] for subtask in task_sets_to_path.keys()]
        )
        subtasks_reg = []
        for sub_task in sub_tasks:
            task_name = cls.get_task_name() + "__" + sub_task
            subtasks_reg.append(
                (
                    task_name,
                    {
                        "_default_train_set": "train_" + sub_task,
                        "_default_valid_set": "valid_" + sub_task,
                        "_default_eval_set": "test_" + sub_task,
                    },
                )
            )

        return subtasks_reg

    def read_data(self, path):
        if path is None:
            return []

        def sample_negative_candidate(correct_words, all_words, num_samples=1):
            negative_candidates = []
            counter = 0
            for word in all_words:
                if counter >= num_samples:
                    break
                if word not in correct_words:
                    negative_candidates.append(word)
                    counter += 1
            assert (
                len(negative_candidates) == num_samples
            ), f"{negative_candidates}\n{all_words}\n{correct_words}"
            return negative_candidates

        data_items = []
        with data_utils.numpy_seed(0):
            with open(path) as f_in:
                for idx, line in enumerate(f_in):
                    example = json.loads(line)
                    if example["target"]:
                        correct_candidates = example["target"].split()
                        word_list = example["input"].split()
                        np.random.shuffle(word_list)
                        negative_candidates = sample_negative_candidate(
                            correct_candidates, word_list
                        )
                        item = {
                            "ids": idx,
                            "input": example["input"],
                            "target": example["target"].split(),
                            "negative_candidate": negative_candidates[
                                0
                            ],  # sampling only one candidate
                        }
                        data_items.append(item)
        return data_items

    def build_samples(self, data):
        # Only taking one correct candidate
        gold_cand = data["target"][0]
        samples = [
            FewShotSample(
                data=data,
                candidates=[gold_cand, data["negative_candidate"]],
                correct_candidates=[gold_cand],
            )
        ]
        return samples

    @classmethod
    def get_default_template_class(cls):
        return templates.RegexTemplate

    def get_max_candidate_length(self, *args, **kwargs):
        return 50  # TODO: need to verify this method

    @classmethod
    def get_task_name(cls):
        return "regex"


@dataclass
class BlimpTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}

        task_data_dir = str(DATA_DIR / "blimp/tasks")
        data_files = sorted(list(glob.glob(task_data_dir + "/*.jsonl")))

        sub_tasks = []
        for data_file in data_files:
            task_set_name = os.path.basename(data_file).split(".")[0]
            task_set_name = task_set_name.lower()

            to_file_mapping[task_set_name] = data_file

        return to_file_mapping

    @property
    def default_train_set(self):
        return (
            self.default_eval_set
        )  # HACK: This task has no training set so we sample from test

    _default_eval_set: Optional[str] = None

    @property
    def default_eval_set(self):
        return self._default_eval_set

    @classmethod
    def sub_tasks_registrations(cls):
        assert (
            "_default_eval_set" in signature(cls).parameters
        ), "Make sure that `_default_eval_set` is a class attribute (not a property), so it can be set from kwargs!"

        task_sets_to_path = cls.get_sets_and_lang_to_path_mappings()

        sub_tasks = []
        for task_set_name, data_file in task_sets_to_path.items():
            # We only have one evaluation set per sub_task!
            task_name = cls.get_task_name() + "__" + task_set_name.lower()
            sub_tasks.append((task_name, {"_default_eval_set": task_set_name}))

        return sub_tasks

    def build_samples(self, data):
        samples = [
            FewShotSample(
                data=data,
                candidates=["sentence_good", "sentence_bad"],
                correct_candidates=["sentence_good"],
            )
        ]

        return samples

    @classmethod
    def get_default_template_class(cls):
        return templates.BlimpTemplate


@dataclass
class ProcessTextTask(FewShotTask):
    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {}

        return to_file_mapping

    @property
    def default_train_set(self):
        return None

    @property
    def default_eval_set(self):
        return None

    separator: str = "\t"
    max_len: Optional[int] = 20
    metrics: Tuple[FewShotMetric] = (AccuracyMetric(),)
    text_examples: Optional[str] = None

    def read_data(self, path):
        if path is None:
            return []

        lines = []
        with open(path) as f_in:
            for idx, line in enumerate(f_in):
                line = line.strip()
                lines.append(line)

        data_items = []
        for idx, line in enumerate(lines):
            line = line.strip()
            # hack to create choices from text
            # if we dont have any choices, the ending will be generated
            item = json.loads(line)
            data_items.append(item)

        return data_items

    def build_samples(self, data):
        candidates = data["candidates"]
        if not isinstance(candidates, list):
            candidates = [candidates]

        samples = []
        samples.append(
            FewShotSample(
                data=data,
                candidates=None if len(candidates) == 1 else candidates,
                correct_candidates=candidates,
            )
        )

        return samples

    def get_max_candidate_length(self, *args, **kwargs):
        return self.max_len  # Should be large enough for all words

    @classmethod
    def get_default_template_class(cls):
        return templates.ProcessTextTemplate


@dataclass
class LAMATask(FewShotTask):
    """
    Common class for LAMA benchmark tasks.
    See paper: https://arxiv.org/pdf/1909.01066.pdf for more details.
    """

    @classmethod
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "conceptnet_test": DATA_DIR / "LAMA/ConceptNet/test.jsonl",
            "squad_test": DATA_DIR / "LAMA/Squad/test.jsonl",
            "googlere": DATA_DIR / "LAMA/Google_RE/*.jsonl",
            "trex": DATA_DIR / "LAMA/TREx/*.jsonl",
        }

        return to_file_mapping

    @property
    def default_train_set(self):
        return None

    metrics: Tuple[FewShotMetric] = (LAMAMetrics(),)
    single_token_mlm_eval: bool = True

    def build_samples(self, data) -> List[FewShotSample]:
        return [
            FewShotSample(
                data=data,
                candidates=[data["obj_label"]],
                correct_candidates=[data["obj_label"]],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.LAMATemplate


@dataclass
class LAMAConceptNetTask(LAMATask):
    """
    ConceptNet is built based on commonsense relationships between
    words and/or phrases. The samples are in the form of subject,
    relation, and object triples. LAMA benchmark converts this triple
    into a sentence and mask the object to create a cloze task that can
    be useful for LM probing.

    See Sec.4 in https://arxiv.org/pdf/1909.01066.pdf for more details.

    """

    @property
    def default_eval_set(self):
        return "conceptnet_test"

    @classmethod
    def get_task_name(cls):
        return "lama_conceptnet"


@dataclass
class LAMASquadTask(LAMATask):
    """
    This dataset is based on Squad QA dataset. 305 context-insensitive
    questions are choosen from Squad QA development set and posted as
    cloze-style questions.
    """

    @property
    def default_eval_set(self):
        return "squad_test"

    @classmethod
    def get_task_name(cls):
        return "lama_squad"


@dataclass
class LAMAGoogleRETask(LAMATask):
    """
    The Google-RE corpus contains ∼60K facts manually extracted from Wikipedia.
    This dataset has 3 set of relations “place of birth”, “date of birth” and
    “place of death”, where given the subject, we have to predict the object.
    """

    @property
    def default_eval_set(self):
        return "googlere"

    def read_data(self, path):
        if path is None:
            return []
        json_objects = []
        multiple_paths = glob.glob(str(path))
        for path in multiple_paths:
            with open(path, "r") as f:
                for line in f:
                    json_objects.append(json.loads(line.rstrip()))

        return json_objects

    @classmethod
    def get_default_template_class(cls):
        return templates.LAMAGoogleRETemplate

    @classmethod
    def get_task_name(cls):
        return "lama_googlere"


@dataclass
class LAMATRExTask(LAMATask):
    """
    The T-REx knowledge source is a subset of Wikidata triples.
    This dataset has 41 wikidata relations with 1000 facts per relation.
    Given a relation and a subject, we predict the masked object
    """

    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            # set_name: [data files, relations_file]
            "trex": [
                DATA_DIR / "LAMA/TREx/*.jsonl",
                DATA_DIR / "LAMA/relations.jsonl",
            ]  # These have special loader so we override the LAMATask sets
        }

        return to_file_mapping

    @property
    def default_eval_set(self):
        return "trex"

    def read_data(self, path):
        if path is None:
            return []

        path, relations_file = tuple(path)
        json_objects = []
        multiple_paths = glob.glob(str(path))
        for path in multiple_paths:
            with open(path, "r") as f:
                for line in f:
                    json_objects.append(json.loads(line.rstrip()))

        ## add relation template to each json object
        templates = {}
        with open(relations_file, "r") as f:
            for line in f:
                relation_data = json.loads(line.rstrip())
                templates[relation_data["relation"]] = relation_data["template"]

        for obj in json_objects:
            obj["template"] = templates[obj["predicate_id"]]

        return json_objects

    @classmethod
    def get_default_template_class(cls):
        return templates.LAMATRExTemplate

    @classmethod
    def get_task_name(cls):
        return "lama_trex"


@dataclass
class MLAMATask(FewShotTask):
    """
    Common class for mLAMA benchmark (https://arxiv.org/pdf/2102.00894.pdf)
    """

    @property
    def default_train_set(self):
        None

    metrics: Tuple[FewShotMetric] = (
        MLAMAMetric(),  # TODO: might need separate metrics for mlama
    )
    max_cands: int = 0

    def read_data(self, path):
        if path is None:
            return []
        assert isinstance(
            path, list
        ), "path must be a list [data_file, relations_file, candidates_file]"

        path, candidates_file, relations_file = tuple(path)
        json_objects = []
        multiple_paths = glob.glob(str(path))
        for path in multiple_paths:
            with open(path, "r") as f:
                relation = os.path.basename(path).replace(".jsonl", "")
                for line in f:
                    json_object = json.loads(line.rstrip())
                    json_object["relation"] = relation
                    json_object["lang"] = self.language
                    json_objects.append(json_object)

        # Add relation template to each json object
        templates = {}
        with open(relations_file, "r") as f:
            for line in f:
                relation_data = json.loads(line.rstrip())
                templates[relation_data["relation"]] = relation_data["template"]

        # Gather all candidates
        with open(candidates_file, "r") as f:
            candidates_data = json.load(f)

        # Use deterministic setting if we randomly choose subset of candidates.
        # mLAMA samples have more than 100 candidates and not limiting candidates
        # significantly effects evaluation time.
        with data_utils.numpy_seed(0):
            for obj in json_objects:
                obj["template"] = templates[obj["relation"]]
                candidates = candidates_data[obj["relation"]]["objects"]
                correct_cand = obj["obj_label"]
                if self.max_cands != 0:
                    candidates = copy.deepcopy(candidates)
                    candidates.remove(correct_cand)
                    max_cands = min(len(candidates), self.max_cands - 1)
                    if max_cands > 1:
                        idx = np.random.choice(
                            len(candidates), max_cands, replace=False
                        )
                    else:
                        idx = []
                    candidates = [correct_cand] + [candidates[i] for i in idx]
                obj["candidates"] = candidates

        return json_objects

    def build_samples(self, data) -> List[FewShotSample]:
        return [
            FewShotSample(
                data=data,
                candidates=data["candidates"],
                correct_candidates=[data["obj_label"]],
            )
        ]

    @classmethod
    def get_default_template_class(cls):
        return templates.MLAMATemplate


@dataclass
class MLAMATRExTask(MLAMATask):
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "trex": {
                lang_code: [
                    DATA_DIR / f"mlama1.1/{lang_code}/P*.jsonl",  # data files
                    DATA_DIR
                    / f"mlama1.1/TREx_multilingual_objects/{lang_code}.json",  # candidates file
                    DATA_DIR
                    / f"mlama1.1/{lang_code}/templates.jsonl",  # relations file
                ]
                for lang_code in cls.get_supported_languages()
            }
        }

        return to_file_mapping

    @property
    def default_eval_set(self):
        return "trex"

    @classmethod
    def get_task_name(cls):
        return "mlama_trex"

    @classmethod
    def get_supported_languages(cls):
        # Supports 53 languages
        return [
            "af",
            "ar",
            "az",
            "be",
            "bg",
            "bn",
            "ca",
            "ceb",
            "cs",
            "cy",
            "da",
            "de",
            "el",
            "en",
            "es",
            "et",
            "eu",
            "fa",
            "fi",
            "fr",
            "ga",
            "gl",
            "he",
            "hi",
            "hr",
            "hu",
            "hy",
            "id",
            "it",
            "ja",
            "ka",
            "ko",
            "la",
            "lt",
            "lv",
            "ms",
            "nl",
            "pl",
            "pt",
            "ro",
            "ru",
            "sk",
            "sl",
            "sq",
            "sr",
            "sv",
            "ta",
            "th",
            "tr",
            "uk",
            "ur",
            "vi",
            "zh",
        ]


@dataclass
class MLAMAGoogleRETask(MLAMATask):
    def get_sets_and_lang_to_path_mappings(cls):
        to_file_mapping = {
            "googlere": {
                lang_code: [
                    DATA_DIR / f"mlama1.1/{lang_code}/*_of_*.jsonl",  # data files
                    DATA_DIR
                    / f"mlama1.1/GoogleRE_objects/{lang_code}.json",  # candidates file
                    DATA_DIR
                    / f"mlama1.1/{lang_code}/templates.jsonl",  # relations file
                ]
                for lang_code in cls.get_supported_languages()
            }
        }

        return to_file_mapping

    @property
    def default_eval_set(self):
        return "googlere"

    @classmethod
    def get_task_name(cls):
        return "mlama_googlere"

    @classmethod
    def get_supported_languages(cls):
        # Supports 40 languages
        return [
            "ar",
            "bg",
            "bn",
            "ca",
            "cs",
            "da",
            "de",
            "el",
            "en",
            "es",
            "eu",
            "fa",
            "fi",
            "fr",
            "ga",
            "gl",
            "hi",
            "hr",
            "hu",
            "hy",
            "id",
            "it",
            "ja",
            "ko",
            "lt",
            "lv",
            "nl",
            "pl",
            "pt",
            "ro",
            "ru",
            "sk",
            "sl",
            "sq",
            "sr",
            "sv",
            "th",
            "tr",
            "uk",
            "vi",
        ]


def get_tasks_default_eval_attributes():
    task_name_to_eval_attributes = {}
    curr_tasks = sorted(get_all_tasks())
    for task_name in curr_tasks:
        task_instance, task_args = init_task_with_custom_args(task_name)
        task_info = task_instance.eval_attributes()

        task_name_to_eval_attributes[task_name] = task_info

    return task_name_to_eval_attributes


def get_tasks_default_eval_attributes():
    """This method just checks if the task has AccuracyMetric in metrics. If it does not then it is most likely a generation task!

    Returns:
        [dict[str, :List[str]]]: dictionary with type (multichoice, generative) and list of tasks
    """
    task_type_to_list = {}
    curr_tasks = sorted(get_all_tasks())
    for task_name in curr_tasks:
        task_instance, task_args = init_task_with_custom_args(task_name)
        if any(
            [isinstance(metric, AccuracyMetric) for metric in task_instance.metrics]
        ):
            task_type = "multichoice"
        else:
            task_type = "generative"

        if task_type not in task_type_to_list:
            task_type_to_list[task_type] = []
        task_type_to_list[task_type].append(task_name)

    for key in list(task_type_to_list.keys()):
        task_type_to_list[key] = list(sorted(task_type_to_list[key]))

    return task_type_to_list


if __name__ == "__main__":
    # print(get_all_tasks())
    print(get_tasks_default_eval_attributes())
    # print(get_languages_with_tasks())
    # print(get_tasks_with_languages())

    # old_task_settings = get_tasks_default_eval_attributes()
    # old_task_settings = {k: {kk: vv for kk, vv in v.items() if kk in {"train_set", "eval_set", "valid_set"}} for k, v in old_task_settings.items()}

    # with open("old_params.json", mode="w") as f_out:
    #     f_out.write("{\n")
    #     for task, params in sorted([(k,v) for k,v in old_task_settings.items()], key=lambda k: k[0]):
    #         f_out.write(f"    \"{task}\": ")
    #         f_out.write(json.dumps(params))
    #         f_out.write(",\n")
    #     f_out.write("}")

    # pass
