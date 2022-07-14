import os
import sys

import glob
import json
import re
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
import time
import copy
import pandas as pd
from pandas import DataFrame
from examples.few_shot.scripts.collect_results import (
    get_checkpoint_step_form_modelname,
    load_results,
)
from examples.few_shot.tasks_organization import (
    get_task_display_groups,
    get_tasks_to_groups_mapping,
    get_groups_to_groups_mapping,
    invert_dict,
)

task_display_groups = get_task_display_groups()
task_display_groups["all_no_blimp"] = [
    x for x in task_display_groups["all"] if not x.startswith("blimp")
]
task_to_groups = get_tasks_to_groups_mapping(task_display_groups)
groups_to_groups = get_groups_to_groups_mapping(task_display_groups)


# Generate grouped results


def grouped_results_to_index_based_json(metric_pt):
    idx_to_metrics_by_model = {}

    metric_cols_all = []
    ppl_cols_all = []

    for idx, row in metric_pt.iterrows():
        # print(idx, row.items())
        curr_task_name = idx[0]
        curr_shots = idx[3]
        # print(curr_task_name)
        # break

        cols_with_vals_by_model = {}

        for [col, model], val in row.items():
            if not col.endswith("::mean") and col != "_metric_val":
                # Use only result columns
                continue
            if col not in cols_with_vals_by_model:
                cols_with_vals_by_model[col] = []
            cols_with_vals_by_model[col].append((model, val))

        # sort model names and values
        for col in cols_with_vals_by_model.keys():
            cols_with_vals_by_model[col] = sorted(
                cols_with_vals_by_model[col], key=lambda x: x[0]
            )

        idx_to_metrics_by_model[idx] = cols_with_vals_by_model
    return idx_to_metrics_by_model


def read_results_and_prepare_for_checkpoint_visualization(
    results_json,
    step_min=None,
    step_max=None,
    language="any",
    debug=False,
    export_intermediate_results_to_tsv=False,
    filter_item_func=None,
):
    def filter_item(item):
        """Use this to implement special filters

        Args:
            item ([type]): [description]

        Returns:
            [type]: [description]
        """
        skip = False

        if language != "any" and item["language"] != language:
            return True

        if "checkpoint_steps" in item:
            if step_min is not None and item["checkpoint_steps"] < step_min:
                return True
            if step_max is not None and item["checkpoint_steps"] > step_max:
                return True

        if filter_item_func is not None:
            return filter_item_func(item)

        return skip

    def postprocess(item):
        """Postprocess loaded item with some fixes, model renames, etc.

        Args:
            item (List[Dict[str, any]]): Results item
        """
        # Clean unused fields that slow down pandas
        for field_name, _ in item.items():
            if field_name.startswith("ppl_candidates_full_prompt__"):
                del item[field_name]

        # Map soem model names to step-like names
        if item["model_name"] == "moe_15B":
            item["model_name"] = "moe_15B__step00572000"
        pass

    if not results_json.endswith(".tsv.raw.jsonl"):
        raise ValueError("Your file must end with tsv.raw.jsonl")

    # Load results
    results_loaded, expanded_results, current_tasks_to_groups = load_results(
        results_json, task_to_groups=task_to_groups, filter_item=None, postprocess=None
    )

    # See what are the checkpoints for current results
    if debug:
        model_name_to_checkpoints = {
            item["model_name"]: item["checkpoint_steps"] for item in results_loaded
        }
        print(f"{len(results_loaded)} items + {len(expanded_results)} expanded items")

    results = results_loaded + expanded_results

    # pring result grouping
    current_groups = invert_dict(current_tasks_to_groups)
    for k, v in current_groups.items():
        print_v = v[:10] + (["..."] if len(v) > 10 else [])
        print(f"-{k} ({len(v)}): {', '.join(print_v)}")
    print()

    # Load data in table
    df = DataFrame.from_records(results)
    if debug:
        print({x for x in set(df["model_name"])})

    if debug:
        print(list(df.columns))

    #####
    # Group data by desired unique attributes
    #####
    selected_df = df

    # main_target_metric = 'accuracy::mean'
    main_target_metric = "_metric_val"
    metrics_cols = [
        "_metric_val"
    ]  # sorted([x + "::mean" for x in list(set(selected_df["_metric"]))])

    if debug:
        print(f"metrics_cols:{metrics_cols}")

    # Pivot table by metric values
    value_columns = metrics_cols
    metric_pt = pd.pivot_table(
        selected_df,
        values=value_columns,
        index=["task", "language", "template", "nb_few_shot_samples", "_metric"],
        columns=[
            "model_name",
            #'calib'
        ],
        aggfunc=np.mean,
    )

    if export_intermediate_results_to_tsv:
        out_file = results_json + "_ppl_selected_metrics.tsv"
        metric_pt.to_csv(out_file, sep="\t")
        if debug:
            print(f"exported to {out_file}")

    # Prepare the data for visualization.
    # Each unique results group, there is a dictionary with metrics as keys and List[(model name, val)] as value
    idx_to_metrics_by_model = grouped_results_to_index_based_json(metric_pt)

    return idx_to_metrics_by_model


def normalize(vals, lower=0.0, upper=1.0):
    min_val = min(vals)
    max_val = max(vals)
    return [
        lower + (x - min_val) * (upper - lower) / (0.0001 + max_val - min_val)
        for x in vals
    ]


def get_plot_fields(v, col_info, related_vals=False):
    metric_key = col_info["key"]
    vals_func = col_info["vals_func"]
    if related_vals:
        rel_vals_func = col_info.get("related_vals_func", None)
        if rel_vals_func is not None:
            vals_func = rel_vals_func

        rel_lbl = False

    vals = vals_func(v[metric_key])
    lbl = (
        col_info["lbl"]
        if not related_vals
        else col_info.get("rel_lbl", col_info["lbl"])
    )

    return lbl, vals


def plot_checkpoint_results_for_single_model(
    idx_to_metrics_by_model,
    main_target_metric="accuracy",
    nshots_eq=None,
    max_plotted=0,
    debug=False,
):
    """Plot results from idx_to_metrics_by_model to matplotlib.
       This function must be called from jupyter notebook.

    Args:
        idx_to_metrics_by_model ([type]): Expected dict with keys, unique setting (see read_results_and_prepare_for_checkpoint_visualization)
        nshots_eq ([type], optional): Display only nshot results. Defaults to None.
        max_plotted (int, optional): This is used for debugging - plots only some results. Defaults to 0.
        debug (bool, optional): If true some helper/debug information is printed. Defaults to False.
    """

    # task_to_groups
    # task_display_groups

    # Line styles
    main_style = {"color": "tab:blue", "linewidth": 4, "marker": "o"}
    special_styles = {"all": {"linestyle": "-.", "color": "tab:gray"}}
    related_styles = [
        {"color": cl, "linestyle": lst}
        for cl in [
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:olive",
            "tab:cyan",
        ]
        for lst in ["dashed", "dotted"]
    ]

    # ('arceasy', 'en', 'arc_old', 0, 'accuracy'), {'accuracy::mean': [('1.3B_gpt3_setting__step00020000', 46.11283873833852), ('1.3B_gpt3_setting__step00050000', 50.06663705019991), ('1.3B_gpt3_setting__step00100000', 51.66592625499778), ('1.3B_gpt3_setting__step00150000', 52.59884495779654)
    x_columns = [
        {
            "lbl": "checkpoint_steps",
            "key": main_target_metric,
            "vals_func": lambda vals: [
                str(int(get_checkpoint_step_form_modelname(x) / 1000)) + "k"
                for x, y in vals
            ],
        }
    ]
    y_columns = [
        {
            "lbl": "performance",
            "key": main_target_metric,
            "vals_func": lambda vals: [y for x, y in vals],
            "rel_lbl": "performance (normalized)",
            "related_vals_func": lambda vals: normalize([y for x, y in vals]),
            "plot_related": True,
        },
    ]

    plotted_cnt = 0

    if nshots_eq is not None:
        if debug:
            print(
                f"Filtering only nshots=={nshots_eq}! Set nshots_eq to None to remove the filter!"
            )

    for exp_setting, exp_results in idx_to_metrics_by_model.items():
        task = exp_setting[0]
        if nshots_eq is not None and exp_setting[3] != nshots_eq:
            continue

        if debug:
            print(f"fig:{exp_setting}")

        group_prefix = ""
        if task in groups_to_groups:
            group_prefix = "_"
        title = "{group_prefix}{task}-{lang} {shots}-shot {template}".format(
            task=task,
            lang=exp_setting[1],
            template=exp_setting[2],
            shots=exp_setting[3],
            group_prefix=group_prefix,
        )
        main_results = exp_results
        # print(main_results)

        # we also plate the grouped results where the current setting participates such as LM, ALL, etc.
        related_results = []
        curr_task_groups = list(
            set(task_to_groups.get(task, []) + groups_to_groups.get(task, []))
        )
        for ctg in curr_task_groups:
            rel_result_key = (ctg, exp_setting[1], "*", exp_setting[3], exp_setting[4])
            if rel_result_key in idx_to_metrics_by_model:
                related_results.append(
                    (rel_result_key, idx_to_metrics_by_model[rel_result_key])
                )
        related_results.sort(key=lambda x: x[0][0])

        for x_col in x_columns:
            for y_col in y_columns:
                plot_rel_on_sep_ax = "related_vals_func" in y_col
                plt.rcParams["axes.facecolor"] = "white"
                fig, ax_main = plt.subplots()
                rect = fig.patch
                rect.set_facecolor("white")

                ax_rel = ax_main.twinx() if plot_rel_on_sep_ax else ax_main

                # main results
                x_lbl, x_vals, y_lbl, y_vals = get_plot_fields(
                    main_results, x_col
                ) + get_plot_fields(main_results, y_col)
                ax_main.plot(x_vals, y_vals, label=title, **main_style)
                ax_main.set_xticklabels(ax_main.get_xticks(), rotation=45)
                ax_main.set_ylabel(
                    f"{y_lbl} ({main_style['color']})", color=main_style["color"]
                )
                ax_main.set_xlabel(x_lbl)

                if y_col.get("plot_related", False):
                    rel_style_id = 0
                    # related results (groups)
                    for rel_setting, rel_result in related_results:
                        rel_task_name = rel_setting[0]
                        rel_title = "{task}-{lang} {shots}-shot".format(
                            task=rel_task_name,
                            lang=rel_setting[1],
                            shots=rel_setting[3],
                        )
                        _, rel_x_vals, rel_y_lbl, rel_y_vals = get_plot_fields(
                            rel_result, x_col, related_vals=True
                        ) + get_plot_fields(rel_result, y_col, related_vals=True)

                        # style
                        rel_style = special_styles.get(rel_task_name, None)
                        if rel_style is None:
                            rel_style = related_styles[rel_style_id]
                            rel_style_id += 1

                        ax_rel.plot(
                            rel_x_vals, rel_y_vals, label=rel_title, **rel_style
                        )

                if plot_rel_on_sep_ax:
                    ax_rel.legend()
                    ax_main.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
                else:
                    ax_main.legend()

                plt.title(title)
                plt.show()

        plotted_cnt += 1
        if max_plotted > 0 and plotted_cnt > max_plotted:
            break


if __name__ == "__main__":
    results_json = "/large_experiments/xlmg/results/intermediate_eval/1.3B_gpt3_setting_kitchen_sink_14/results.tsv.raw.jsonl"
    #                 "/large_experiments/xlmg/results/intermediate_eval/1.3B_gpt3_setting_kitchen_sink_15/results.tsv.raw.jsonl",
    #                 "/large_experiments/xlmg/results/intermediate_eval/1.3B_gpt3_setting_kitchen_sink_16/results.tsv.raw.jsonl",
    #                 "/large_experiments/xlmg/results/intermediate_eval/1.3B_gpt3_setting_kitchen_sink_17/results.tsv.raw.jsonl",

    res = read_results_and_prepare_for_checkpoint_visualization(results_json)

    print(res)
