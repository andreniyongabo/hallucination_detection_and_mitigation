import pathlib
import pkg_resources
import sys
from typing import Optional, Union, TextIO

import numpy as np
import pandas as pd


def eprint(*kargs, **kwargs):
    print(*kargs, **kwargs, file=sys.stderr)


def print_help_columns(title: str, df: pd.DataFrame):
    print(title)
    [print(f"\t{a}") for a in df.columns]


def get_df(f: Union[str, TextIO], header: Optional[int]):
    filename = getattr(f, "name", f)
    extension = pathlib.Path(filename).suffix

    if extension == ".xlsx":
        df = pd.read_excel(f, header=header)
    elif extension == ".tsv":
        df = pd.read_csv(f, sep="\t", header=header)

    # We want columns to be referenced from command line
    if df.columns.dtype == np.int64:
        df.columns = [str(col) for col in df.columns]

    return df


def _get_df_from_resources(name: str, header: Optional[int]):
    with pkg_resources.resource_stream(__name__, f"data/{name}") as f:
        df = get_df(f, header)

    return df


def get_lang_df():
    df_general = _get_df_from_resources("general.tsv", header=0)

    return df_general
