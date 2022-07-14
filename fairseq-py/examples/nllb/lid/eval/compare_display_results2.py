#!python3 -u

import argparse
import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from colour import Color
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

def get_dataframe_from_fasttext_result(path):
    assert os.path.exists(path)

    ft_label_score_line = re.compile("F1-Score")

    def get_nvalue(st):
        if st == "--------":
            return np.nan
        else:
            return float(st)

    values = []
    with open(path, "r") as fl:
        for line in fl:
            if ft_label_score_line.match(line):
                spl = line.split()
                values.append(
                    [
                        get_nvalue(spl[2]),
                        get_nvalue(spl[5]),
                        get_nvalue(spl[8]),
                        get_nvalue(spl[11]),
                        spl[12][9:],
                    ]
                )
    df = pd.DataFrame(
        values, columns=["f1-score", "precision", "recall", "fpr", "label"]
    )
    df = df.set_index("label")

    return df


def get_color_range(vals):
    epsilon = 0.00001
    negligeables = np.abs(vals) <= epsilon
    n_real_vals = np.where(np.logical_and(~negligeables, ~np.isnan(vals)))
    insta = ["#F58529", "#FEDA77", "#DD2A7B", "#8134AF", "#515BD4"]
    n = n_real_vals[0].shape[0]

    colors = list(Color(insta[0]).range_to(Color(insta[-1]), n))
    res = np.array(["#000000"] * vals.shape[0])
    res[n_real_vals] = colors

    return res


def dataframe_from_experiment_paths(experiment_paths, values_column="precision"):
    df_merged = None
    ascending = values_column != "fpr"

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
            df_merged = df_merged.merge(df_result, on="label", how="outer")

    return df_merged


def draw_improvement_plot(df, metric, ax):
    print(f"{metric}")
    ascending = metric == "fpr"
    diff = df[f"{metric}-new"] - df[f"{metric}-old"]
    for percentile in [0.05, 0.1, 0.15]:
        percentile_old = df[f"{metric}-old"].quantile(percentile)
        percentile_new = df[f"{metric}-new"].quantile(percentile)

        print(f"\t{percentile*100} percentile_old = {percentile_old:.5f}")
        print(f"\t{percentile*100} percentile_new = {percentile_new:.5f}")

    langs = df.index.values
    diffs = diff.values
    order = np.argsort(diffs if ascending else -diffs)
    langs = langs[order]
    diffs = diffs[order]

    colors = get_color_range(diffs)
    ax.bar(langs, diffs, color=colors)

    ax.tick_params(axis="x", which="major", labelsize=1.3)
    ax.tick_params(axis="y", which="major", labelsize=5)

    ax.set_xlabel("Languages")
    ax.set_ylabel(f"Diff {metric.capitalize()}")

    ax.xaxis.label.set_size(4)
    ax.yaxis.label.set_size(5)
    title = "Improvement"
    if ascending:
        title += " (less is better)"

    ax.set_title(title)
    ax.get_xticklabels()
    ax.set_xticklabels(labels=langs, rotation="vertical")


def draw_sorted_plot(df, metric, ax):
    ascending = metric == "fpr"

    low_resource_langs = ['bam', 'epo', 'kin', 'ady', 'aka', 'awa', 'bjn', 'bis', 'che',
       'chr', 'nya', 'din', 'dzo', 'ewe', 'fij', 'fon', 'gom', 'kal',
       'grn', 'haw', 'kbp', 'kau', 'krc', 'kas', 'kik', 'kon', 'ltg',
       'mni', 'nia', 'pag', 'pap', 'roh', 'run', 'bxr', 'smo', 'sag',
       'skr', 'alt', 'sot', 'tah', 'bod', 'tpi', 'tog', 'tso', 'tum',
       'twi', 'uig', 'cre', 'iku', 'aym', 'bos', 'est', 'ful', 'lug',
       'ibo', 'gle', 'kmb', 'lao', 'lav', 'lin', 'lit', 'mkd', 'mlt',
       'ary', 'orm', 'slv', 'ssw', 'tir', 'tsn', 'wol', 'xho']
    afrikans= ['afr', 'amh', 'ara', 'dyu', 'ful', 'hau', 'heb', 'ibo', 'kam',
       'kmb', 'kon', 'lin', 'lug', 'luo', 'nso', 'nya', 'orm', 'sna',
       'som', 'ssw', 'swh', 'tir', 'tsn', 'umb', 'zul', 'bam', 'kin',
       'bem', 'din', 'ewe', 'fon', 'kbp', 'kab', 'kik', 'lua', 'mos',
       'nus', 'run', 'sag', 'sot', 'tmh', 'tog', 'tso', 'tum', 'twi',
       'cjk']

    ax.patches = []
    for experiment_name in ["new", "old"]:
        vals = df[f"{metric}-{experiment_name}"].values
        langs = df.index.values
        order = np.argsort(vals)
        vals = vals[order]
        langs = langs[order]
        if ascending:
            vals = vals[::-1]
            langs = langs[::-1]

        ax.plot(vals, linewidth=0.7, label=experiment_name)

        rectboxes = []
        if experiment_name == 'new':
            for i, lang in enumerate(langs):
                if metric != "fpr":
                    if lang in low_resource_langs:
                        rectboxes.append(Rectangle((i, 0.0), 1.0, 1))

            facecolor = 'r'
            alpha=0.2
            edgecolor="None"
            pc = PatchCollection(rectboxes, facecolor=facecolor, alpha=alpha,
                                 edgecolor=edgecolor)
            ax.add_collection(pc)

    ax.legend()
    ax.set_xlabel("Languages")
    ax.set_ylabel(f"(Ordered) {metric.capitalize()}")
    if metric != "fpr":
        ax.axhline(0.98, c='black', ls='--', lw=1)


def compare(old, new, evaluated_langs_only):
    experiment_paths = [("old", old), ("new", new)]

    metrics = ["fpr", "precision", "recall", "f1-score"]
    df = None
    for metric in metrics:
        df_metric = dataframe_from_experiment_paths(
            experiment_paths, values_column=metric
        )
        df = df.merge(df_metric, on="label") if df is not None else df_metric

    df = df.sort_values(by=["label"], ascending=True)

    if evaluated_langs_only:
        df.dropna(subset=["recall-old"], inplace=True)
        df.dropna(subset=["recall-new"], inplace=True)
    print(df.columns)


    fig, axs = plt.subplots(2, 2)
    plt.rcParams["axes.labelsize"] = 2
    plt.rcParams["axes.titlesize"] = 4

    for metric, ax in zip(metrics, axs.reshape(-1)):
        print(metric, ax)
        draw_improvement_plot(df, metric, ax)

    fig.suptitle(f'{old}\nvs\n{new}', fontsize=4)

    plt.tight_layout()
    plt.savefig(f"improv.png", dpi=600)

    fig, axs = plt.subplots(2, 2)
    plt.rcParams["axes.labelsize"] = 2
    plt.rcParams["axes.titlesize"] = 4

    for metric, ax in zip(metrics, axs.reshape(-1)):
        print(metric, ax)
        draw_sorted_plot(df, metric, ax)

    fig.suptitle(f'{old}\nvs\n{new}', fontsize=4)

    plt.tight_layout()
    plt.savefig(f"sorted.png", dpi=600)


def main():
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument("--old", required=True, type=str)
    parser.add_argument("--new", required=True, type=str)
    parser.add_argument("--eval-langs-only", action="store_true")
    args = parser.parse_args()

    compare(args.old, args.new, args.eval_langs_only)


if __name__ == "__main__":
    main()
