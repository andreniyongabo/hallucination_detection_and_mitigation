#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib
import os
import typing as tp
from collections import Counter, namedtuple
from dataclasses import dataclass
from pathlib import Path, PurePath

import wandb
from examples.nllb.mining.monolingual.utils import predict_lid, predict_script

LID_LABEL_PREFIX = "__label__"
LANG_SCRIPT_MAPPING = "language_scripts_200.tsv"
DEFAULT_CHECK_FOLDER = "/large_experiments/nllb/mmt/flores101_beta/devtest"
WANDB_ENTITY = "nllb"
WANDB_PROJECT = "data-sanity"
WANDB_TABLE_NAME = "sanity_check_results"


class WandbTableRow(tp.NamedTuple):
    file_path: str
    nb_examples: int
    expected_lang: str
    expected_script: str
    fishy_lid: bool
    fishy_script: bool
    lid_stats: tp.List[tp.Union[str, float, None]]
    sid_stats: tp.List[tp.Union[str, float, None]]


def standardize_lang_name(raw_lang_name: str) -> str:
    """Making sure a lang name only contains underscores"""
    return raw_lang_name.replace("-", "_")


@dataclass
class DataSanityChecker:
    """
    Class holding an LID model and a SID routine to perform sanity
    checks within a folder and save results into weight & biases
    """

    lid_clf: tp.Callable[[str], tp.Tuple[tp.Optional[str], float]]
    sid_clf: tp.Callable[[str], tp.Tuple[tp.Optional[str], float]]
    script_map: tp.Dict[str, str]

    def _compute_stats(
        self, preds: Counter, topk: int = 3
    ) -> tp.Tuple[tp.List[tp.Union[str, float, None]], int]:
        """Returns the top k langs occuring in a list and their prevalence"""

        nb_lines = sum(preds.values())

        top3 = preds.most_common(topk)
        top3_langs, top3_stats = [list(t) for t in zip(*top3)]
        top3_stats = [cnt / nb_lines for cnt in top3_stats]

        for i, lang in enumerate(top3_langs):
            if not lang:
                continue
            # TODO: replace with .removeprefix when moving to Python 3.9
            if lang.startswith(LID_LABEL_PREFIX):
                top3_langs[i] = standardize_lang_name(lang[len(LID_LABEL_PREFIX) :])
            else:
                top3_langs[i] = standardize_lang_name(lang)

        # lid/sid might return less than 3 langs, but wandb tables expect constant len
        while len(top3_langs) < 3:
            top3_langs.append(None)
            top3_stats.append(None)

        return top3_langs + top3_stats, nb_lines

    def _process_file(
        self,
        file_path: str,
    ) -> WandbTableRow:
        """Performs LID & SID on a file"""

        with open(file_path, "r") as file:
            all_lid_preds: Counter = Counter()
            all_sid_preds: Counter = Counter()
            for line in file:
                line = line.rstrip()
                # TODO: deeper analysis of lid & sid scores
                lid_pred, _ = self.lid_clf(line)
                sid_pred, _ = self.sid_clf(line)
                all_lid_preds[lid_pred] += 1
                all_sid_preds[sid_pred] += 1
        lid_stats, nb_lines = self._compute_stats(all_lid_preds)
        sid_stats, _ = self._compute_stats(all_sid_preds)

        expected_lang = standardize_lang_name(PurePath(file_path).stem)
        expected_script = self.script_map[expected_lang]
        fishy_lid = not (expected_lang == lid_stats[0])
        fishy_script = not (expected_script == sid_stats[0])

        return WandbTableRow(
            os.path.abspath(file_path),
            nb_lines,
            expected_lang,
            expected_script,
            fishy_lid,
            fishy_script,
            lid_stats,
            sid_stats,
        )

    def _process_folder(
        self,
        folder_path: str,
        wandb_table: wandb.Table,
    ):
        """Performs LID & SID on a folder and saves stats to a wandb table"""

        for filename in os.listdir(folder_path):
            stats = self._process_file(os.path.join(folder_path, filename))
            wandb_table.add_data(
                stats.file_path,
                stats.nb_examples,
                stats.expected_lang,
                stats.expected_script,
                stats.fishy_lid,
                stats.fishy_script,
                *stats.lid_stats,
                *stats.sid_stats,
            )

    def _process_multiple_folders(
        self,
        folder_paths: list[str],
    ) -> wandb.Table:
        """Performs LID & SID on several folders and saves stats to a table"""

        wandb_table = wandb.Table(
            columns=[
                "filename",
                "nb lines",
                "expected_lang",
                "expected_script",
                "fishy_lid",
                "fishy_sid",
                "top1 lid",
                "top2 lid",
                "top3 lid",
                "top1 lid %",
                "top2 lid %",
                "top3 lid%",
                "top1 sid",
                "top2 sid",
                "top3 sid",
                "top1 sid %",
                "top2 sid %",
                "top3 sid %",
            ]
        )

        for folder in folder_paths:
            self._process_folder(folder, wandb_table)

        return wandb_table

    def check(self, args: argparse.Namespace):
        res_table = self._process_multiple_folders(args.paths)
        wandb_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY)
        wandb.config.update(args)
        if wandb_run:  # wandb.init can return None
            wandb_run.log({WANDB_TABLE_NAME: res_table})


def main():
    parser = argparse.ArgumentParser(
        description="Run LID & SID sanity check on given folder"
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=[DEFAULT_CHECK_FOLDER],
        help="path to folder to run the check on",
    )
    parser.add_argument("--model", type=str, default=None, help="fasttext model")
    parser.add_argument("--thresholds", type=str, default=None, help="thresholds file")
    parser.add_argument(
        "--model-date",
        type=str,
        help=(
            "specify the model and threshold by train date. Set 'last' for the latest"
            " one."
        ),
    )
    args = parser.parse_args()

    assert (
        args.model_date or args.model
    ), "Select a model: for example `--model-date last`"

    if args.model_date:
        assert args.model is None and args.thresholds is None, (
            "You can't specify the model or the thresholds with the `--model-date`"
            " argument"
        )

        lid_clf = predict_lid.get_lid_predictor_date(args.model_date)
    else:
        lid_clf = predict_lid.get_lid_predictor(
            Path(args.model), Path(args.thresholds) if args.thresholds else None
        )

    sid_clf = predict_script.get_script_predictor()
    with importlib.resources.path(
        "examples.nllb.mining.monolingual", LANG_SCRIPT_MAPPING
    ) as language_script_file:
        script_map = predict_script.get_script_map(language_script_file)
    data_sanity_checker = DataSanityChecker(lid_clf, sid_clf, script_map)
    data_sanity_checker.check(args)


if __name__ == "__main__":
    main()
