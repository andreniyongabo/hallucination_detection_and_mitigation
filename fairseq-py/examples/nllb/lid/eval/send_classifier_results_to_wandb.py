#!python3 -u

import argparse
import fnmatch
import os
import re
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    print("Please install `wandb` module; `pip install wandb`", file=sys.stderr)
    exit(1)


class TestDataframes(unittest.TestCase):
    def test_fasttext_file_read(self):
        mock_fasttext = os.path.join(tempfile.gettempdir(), "results.txt")
        mock_content = """
F1-Score : 0.998498  Precision : 1.000000  Recall : 0.997000  FPR : 0.000000   __label__abk
F1-Score : 1.000000  Precision : 1.000000  Recall : 1.000000  FPR : 0.000000   __label__ady
F1-Score : 0.996543  Precision : 0.996051  Recall : 0.997036  FPR : 0.000022   __label__afr
F1-Score : 0.994606  Precision : --------  Recall : 0.989270  FPR : 0.000000   __label__alt
F1-Score : 0.999506  Precision : 0.999013  Recall : 1.000000  FPR : 0.000005   __label__amh
F1-Score : 0.872414  Precision : 0.773700  Recall : --------  FPR : 0.001608   __label__ara
F1-Score : 0.995988  Precision : 0.998994  Recall : 0.993000  FPR : 0.000005   __label__arn
F1-Score : 0.825602  Precision : 1.000000  Recall : 0.703000  FPR : 0.000000   __label__arz
Precision : 1.000000 # should be ignored
Recall : 1.000000 # should be ignored
        """
        with open(mock_fasttext, "w") as f:
            f.write(mock_content)
        # fmt: off
        reference_df_data = [
            ["abk", 0.998498, 1.000000, 0.997000, 0.000000],
            ["ady", 1.000000, 1.000000, 1.000000, 0.000000],
            ["afr", 0.996543, 0.996051, 0.997036, 0.000022],
            ["alt", 0.994606,   np.nan, 0.989270, 0.000000],
            ["amh", 0.999506, 0.999013, 1.000000, 0.000005],
            ["ara", 0.872414, 0.773700,   np.nan, 0.001608],
            ["arn", 0.995988, 0.998994, 0.993000, 0.000005],
            ["arz", 0.825602, 1.000000, 0.703000, 0.000000],
        ]
        # fmt: on
        reference_df = pd.DataFrame(
            reference_df_data,
            columns=["label", "f1-score", "precision", "recall", "fpr"],
        )
        reference_df = reference_df.set_index("label")

        with open(mock_fasttext, "r") as f:
            reached_df = get_dataframe_from_fasttext_result(f.name)
            self.assertTrue(reached_df.equals(reference_df))

    def test_experiment_name(self):
        self.assertEqual(
            get_experiment_name_from_path(
                "/path/to/lidruns/2021-10-26-13-04-optim/foo/bar/result.ft.txt"
            ),
            "2021-10-26-13-04-optim_result.ft.txt",
        )
        self.assertEqual(
            get_experiment_name_from_path(
                "/path/to/2022-01-01-13-04-this-is-the-future/foo/bar/foo/foo/result.thresholds3.txt"
            ),
            "2022-01-01-13-04-this-is-the-future_result.thresholds3.txt",
        )


def get_experiment_name_from_path(path):
    path_parts = path.split("/")
    return fnmatch.filter(path_parts, "202?-*")[0] + "_" + path_parts[-1]


def get_dataframe_from_fasttext_result(path):
    assert os.path.exists(path)

    # lines should be like this:
    # something : 0.32423 something2 : 0.174714 something3 : 0.1747124 ... __label__foo
    # something : 0.32785 something2 : -------- something3 : 0.8324712 ... __label__bar
    ft_label_score_line = re.compile(
        r"([\w-]+\s*:\s*(?:--------|[+-]?[0-9]+\.[0-9]+)\s+)*\s+__label__\w+"
    )

    def get_nvalue(st):
        if st == "--------":
            return np.nan
        else:
            return float(st)

    values = []
    with open(path, "r") as fl:
        for line in fl:
            if ft_label_score_line.match(line):
                row = {}
                i = iter(line.split())
                while True:
                    column_or_label = next(i)
                    if column_or_label.startswith("__label__"):
                        row["label"] = column_or_label[9:]
                        break
                    else:
                        separator = next(i)
                        cell_value = next(i)
                        row[column_or_label] = get_nvalue(cell_value)

                values.append(row)

    df = pd.DataFrame(values)
    df = df.set_index("label")
    df.columns = [str.lower(c) for c in df.columns]

    return df


def send_data_to_wandb(result_path):
    df = get_dataframe_from_fasttext_result(result_path)
    df = df.sort_values(by=["label"], ascending=True)
    df["langs"] = df.index  # wandb can't read the index

    run_id = get_experiment_name_from_path(result_path)
    wandb_run = wandb.init(project="lid", entity="nllb", id=run_id)
    frame_data = wandb.Table(dataframe=df)
    wandb_run.log({"classifier_results": frame_data})


def main():
    parser = argparse.ArgumentParser(
        description="This script takes a fasttext-like result file as input, creates a pandas dataframe and uploads the results to Weight&Biases."
    )
    parser.add_argument("resultfile", type=str)
    args = parser.parse_args()

    send_data_to_wandb(args.resultfile)


if __name__ == "__main__":
    main()
