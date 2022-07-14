from fire import Fire
import torch
import re
from fairseq.file_io import torch_load_cpu
from functools import lru_cache
from tqdm import tqdm
from fb_sweep.agg_results import find_all_matching_lines

from fairseq.criterions.moe_cross_entropy import EXPERT_STATS_PREFIX
from pathlib import Path

def subset(df, pat):
    return df[[x for x in df.columns if pat in x]]

def make_expert_stats_ser(train_log_path, pattern='train_inner'):
    expert_stats_df = find_all_matching_lines(Path(train_log_path).open().readlines(), pattern)
    expert_stats_ser = expert_stats_df.pipe(subset, EXPERT_STATS_PREFIX).T.median().sort_values()
    return expert_stats_ser


def make_mask(expert_stats_ser, n_to_mask, return_worst=False):
    """Returns worst_mask, best_mask"""
    colname = 'avg'
    expert_stats_df = expert_stats_ser.to_frame(colname).T
    meds = expert_stats_df.pipe(subset, EXPERT_STATS_PREFIX).T
    assert meds.shape[1] == 1
    meds['id'] = [int(x.split('_')[-1]) for x in meds.index]
    meds['lnum'] = [int(x.split('_')[-4].lstrip('l')) for x in meds.index]
    meds['enum'] = [2 if 'exp2' in x else 1 for x in meds.index]
    moe_layers = meds.lnum.unique()
    lw_worst_mask = {}
    lw_best_mask = {}
    for i in moe_layers:
        escores = meds[meds.lnum == i].groupby('id').sum().sort_values(colname)[colname]
        lw_worst_mask[i] = escores.index[:n_to_mask].tolist()
        lw_best_mask[i] = escores.index[-n_to_mask:].tolist()
    if return_worst:
        return lw_best_mask, lw_worst_mask
    else:
        return lw_best_mask

