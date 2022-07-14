#!python3 -u


import os
import re
import difflib

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_dataframe_from_fasttext_result(path):
    assert os.path.exists(path)

    ft_label_score_line = re.compile("F1-Score")

    values = []
    with open(path, "r") as fl:
        for line in fl:
            if ft_label_score_line.match(line):
                spl = line.split()
                if spl[5] != '--------' and spl[8] != '--------':
                    values.append([float(spl[2]), float(spl[5]), float(spl[8]), float(spl[11]), spl[12][9:]])
    df = pd.DataFrame(values, columns=['f1-score', 'precision', 'recall', 'fpr', 'label'])
    df = df.set_index('label')

    return df


def get_experiment_name_from_path(path):
    spl = path.split("/")
    ldr = spl.index("lidruns")
    if ldr != -1:
        experiment_name = spl[ldr+1]
        experiment_name = re.sub(r'\d+-\d+-\d+-\d+-\d+-', '', experiment_name)

        return experiment_name

    return "Name?"

def dataframe_from_experiment_paths(experiment_paths, values_column='precision'):
    df_merged = None
    fig, ax = plt.subplots()

    ascending = (values_column != 'fpr')

    global_xp_name = get_experiment_name_from_path(experiment_paths[0][1])

    for experiment_name, res_path in experiment_paths:
        result_file = res_path
        print(f"result_file = {result_file}")

        df_result = get_dataframe_from_fasttext_result(result_file)
        df_result = df_result[[values_column]]

        column_name = f"{values_column}-{experiment_name}"
        df_result.rename(columns={values_column: column_name}, inplace=True)

        if df_merged is None:
            df_merged = df_result
        else:
            df_merged = df_merged.merge(df_result, on='label', how='outer')

    for experiment_name, _ in experiment_paths:
        column_name = f"{values_column}-{experiment_name}"
        vals = df_merged[column_name].values

        if not ascending:
            nan_value = 100.0
        else:
            nan_value = -100.0
        vals[np.isnan(vals)] = nan_value
        vals = np.sort(vals)
        if not ascending:
            vals = vals[::-1]
        vals[vals == 100.0] = np.nan
        vals[vals == -100.0] = np.nan

        ax.plot(vals, label=experiment_name)

    plt.legend()
    plt.xlabel("Ordered langs")
    plt.ylabel(values_column)
    plt.title(global_xp_name)
    plt.savefig(f"lang_compare_result_{values_column}.png", dpi=300)

    return df_merged



def main():
    # experiment_paths = [
    #     "/large_experiments/mmt/lidruns/2021-05-24-16-56-flores-99-match/result/result.test-label-flores-dev.txt",
    #     "/large_experiments/mmt/lidruns/2021-05-24-22-35-flores-99-match-jw300-only/result/result.test-label-flores-dev.txt",
    #     "/large_experiments/mmt/lidruns/2021-05-24-22-35-flores-99-match-lid187-only/result/result.test-label-flores-dev.txt",
    #     "/large_experiments/mmt/lidruns/2021-06-10-11-38-flores-99-match-cld3/result/result.classifiermetrics-flores-dev.txt",
    # ]

    # experiment_paths = [
    #     ("cld3",     "/large_experiments/mmt/lidruns/2021-06-15-17-38-compare-on-common3/result/result.classifiermetrics-flores-dev.cld3.txt"),
    #     ("fasttext", "/large_experiments/mmt/lidruns/2021-06-15-17-38-compare-on-common3/result/result.classifiermetrics-flores-dev.fasttext.txt"),
    # ]


    # experiment_paths = [
    #     ("cld3",     "/large_experiments/mmt/lidruns/2021-06-16-09-32-compare-on-flores99/result/result.classifiermetrics-flores-dev.cld3.txt"),
    #     ("fasttext", "/large_experiments/mmt/lidruns/2021-06-16-09-32-compare-on-flores99/result/result.classifiermetrics-flores-dev.fasttext.txt"),
    # ]

    # experiment_paths = [
    #     ("cld3",     "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/result.classifiermetrics-earl.cld3.txt"),
    #     ("fasttext", "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/result.classifiermetrics-earl.fasttext.6.txt"),
    # ]

    # experiment_paths = [
    #     ("fasttext.1", "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/result.classifiermetrics-earl.fasttext.1.txt"),
    #     ("fasttext.2", "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/result.classifiermetrics-earl.fasttext.2.txt"),
    #     ("fasttext.3", "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/result.classifiermetrics-earl.fasttext.3.txt"),
    #     ("fasttext.6", "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/result.classifiermetrics-earl.fasttext.6.txt"),
    # ]

    # experiment_paths = [
    #     ("fasttext.6", "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/result.classifiermetrics-earl.fasttext.6.txt"),
    #     ("fasttext.7", "/large_experiments/mmt/lidruns/2021-06-16-09-44-compare-on-earl/result/result.classifiermetrics-earl.fasttext.7.txt"),
    # ]

    experiment_paths = [
        ("fasttext", "/large_experiments/mmt/lidruns/2021-07-01-00-17-compare-on-common4/result/result.classifiermetrics-flores-dev.fasttext.txt"),
        ("langdetect", "/large_experiments/mmt/lidruns/2021-07-01-00-17-compare-on-common4/result/result.classifiermetrics-flores-dev.langdetect.txt"),
    ]


    df_fpr = dataframe_from_experiment_paths(experiment_paths,
                                             values_column='fpr')

    df_precision = dataframe_from_experiment_paths(experiment_paths,
                                             values_column='precision')

    df_recall = dataframe_from_experiment_paths(experiment_paths,
                                             values_column='recall')

    df_f1_score = dataframe_from_experiment_paths(experiment_paths,
                                             values_column='f1-score')



    df_merged = df_fpr.merge(df_precision, on='label').merge(df_recall, on='label').merge(df_f1_score, on='label')
    print(df_merged.columns)
    sort_column_name = difflib.get_close_matches("precision-fasttext", df_merged.columns)
    print(f"sort_column_name = {sort_column_name}")
    df_merged = df_merged.sort_values(by=[sort_column_name[0]], ascending=True)

    print(df_merged.T)
    df_merged.T.to_excel("lidresults_horizontal.xlsx", sheet_name="LID results")
    df_merged.to_excel("lidresults_vertical.xlsx", sheet_name="LID results")



if __name__ == "__main__":
    main()
