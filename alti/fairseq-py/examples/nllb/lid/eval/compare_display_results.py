#!python3 -u


import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


result_filename = "result.test-label-flores-dev.txt"


def get_dataframe_from_fasttext_result(path):
    assert os.path.exists(path)

    ft_label_score_line = re.compile("F1-Score")

    values = []
    with open(path, "r") as fl:
        for line in fl:
            if ft_label_score_line.match(line):
                spl = line.split()
                values.append([float(spl[2]), float(spl[5]), float(spl[8]), float(spl[11]), spl[12][9:]])
    df = pd.DataFrame(values, columns=['f1-score', 'precision', 'recall', 'fpr', 'label'])
    # print(df)

    return df


def get_experiment_name_from_path(path):
    spl = path.split("/")
    ldr = spl.index("lidruns")
    if ldr != -1:
        experiment_name = spl[ldr+1]
        experiment_name = re.sub(r'\d+-\d+-\d+-\d+-\d+-', '', experiment_name)

        return experiment_name

    return "Name?"


def compare_display(experiment_paths, sort_column='precision', values_column='precision', ref_value=0.99):
    ref_df = None
    dfs = []
    for res_path in experiment_paths:
        result_file = os.path.join(res_path, result_filename)
        print(f"result_file = {result_file}")

        df = get_dataframe_from_fasttext_result(result_file)

        if ref_df is None:
            ref_df = df
        else:
            df_exp_name = get_experiment_name_from_path(result_file)
            dfs.append((df, df_exp_name))

    # ref_df = ref_df.sort_values(by=[sort_column], ascending=True)
    ref_df = ref_df.sort_values(by=[sort_column], ascending=False)
    ref_df = ref_df.head(30)

    print(ref_df)
    print(len(ref_df))
    langs = ref_df['label'].values

    lang_indices = np.arange(len(langs))
    ax = plt.subplot()
    width=0.5

    disply_y_coef = 2

    ref_exp_name = get_experiment_name_from_path(experiment_paths[0])
    ax.barh(lang_indices * disply_y_coef + width, ref_df[values_column], width, align='center', alpha=0.8, color='#5fb877', label=ref_exp_name)


    for df, df_exp_name in dfs:
        idx = pd.Index(df['label']).get_indexer(langs)
        idx = idx[idx != -1]
        df_selected = df.iloc[idx]
        df_selected_langs = df_selected['label'].values
        df_selected_langs_indices = np.array([np.argwhere(langs == lng).item() for lng in df_selected_langs])

        print("df_selected")
        print(df_selected)
        print(len(df_selected))
        vals = df_selected[values_column]
        ax.barh(df_selected_langs_indices*disply_y_coef, vals, width, align='center', alpha=0.8, color='#b87b5f', label=df_exp_name)


    print(len(langs))
    # print(df_selected['label'].values)

    # 98 => 3
    # 20 => 12

    font_size = int(12 - (len(langs)-20) * 9.0/78.0)

    print(f"font_size = {font_size}")
    ax.axvline(x=ref_value, color='k', linestyle='--')

    ax.set(yticks=lang_indices*disply_y_coef + width, yticklabels=langs, ylim=[0, len(langs)*disply_y_coef])
    # ax.set(xticks=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99], xlim=[0.5, 1.0])
    ax.set(xlim=[0.0, 0.01])
    plt.setp(ax.get_yticklabels(), fontsize=font_size)
    plt.xlabel(values_column)
    plt.title(f"{values_column} - Languages")
    plt.legend(loc='upper left')
    plt.savefig("lang_compare_result.png", dpi=300)
    # plt.show()


def main():
    # experiment1_path = "/large_experiments/mmt/lidruns/2021-05-19-12-53-flores-99/result"
    # experiment2_path = "/large_experiments/mmt/lidruns/2021-05-24-16-56-flores-99-match/result"
    # experiment2_path = "/large_experiments/mmt/lidruns/2021-05-24-16-56-flores-99-match/result"

    experiment1_path = "/large_experiments/mmt/lidruns/2021-05-24-16-56-flores-99-match/result"
    # experiment2_path = "/large_experiments/mmt/lidruns/2021-05-24-22-35-flores-99-match-jw300-only/result"
    experiment2_path = "/large_experiments/mmt/lidruns/2021-05-24-22-35-flores-99-match-lid187-only/result"

    # experiment1_path = "/large_experiments/mmt/lidruns/2021-05-19-12-53-flores-99/result"
    # experiment2_path = "/large_experiments/mmt/lidruns/2021-05-24-16-56-flores-99-match/result"

    # lst = [experiment1_path]    # , experiment2_path]
    lst = [experiment1_path, experiment2_path]
    compare_display(lst,
                    sort_column='fpr',
                    values_column='fpr',
                    ref_value=0.0001)

if __name__ == "__main__":
    main()
