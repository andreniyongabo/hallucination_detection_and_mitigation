from collections import OrderedDict
import json
import os
import sys
from typing import Optional

from .core import *

import pandas as pd

column_name_code = "Code"
column_name_language_name = "Language Name"

default_columns = [column_name_code, column_name_language_name]


class Exporter:
    """
    Base class. Calls itself if the export_argument is `_export_is`
    or endswith `_export_endswith`. Examples of export_argument:
    simple, python, filename.tsv, filename.xlsx
    """

    _export_is = None
    _export_endswith = None

    def matched(self, df: pd.DataFrame, export_argument: str):
        if export_argument == self._export_is or (
            self._export_endswith and export_argument.endswith(self._export_endswith)
        ):
            self(df, export_argument)
            return True

        return False

    def __call__(self):
        raise NotImplementedError


class SimpleExporter(Exporter):
    """
    Prints a simple tsv to stdout
    """

    _export_is = "simple"

    def __call__(self, df: pd.DataFrame, _):
        df.to_csv(sys.stdout, index=False, sep="\t", header=False)


class PythonExporter(Exporter):
    _export_is = "python"

    def __call__(self, df: pd.DataFrame, _):
        "Prints a python dictionary to stdout"
        key_column = column_name_code
        assert len(df[key_column].unique()) == len(df[key_column])

        obj = {}
        for ind, row in df.iterrows():
            obj[row[key_column]] = {col: val for col, val in row.iteritems()}

        print(obj.__repr__())


class TsvExporter(Exporter):
    _export_endswith = ".tsv"

    def __call__(self, df: pd.DataFrame, filename: str):
        df.to_csv(filename, index=False, sep="\t")
        print(f"Exported to {filename}")


class ExcelExporter(Exporter):
    _export_endswith = ".xlsx"

    def __call__(self, df: pd.DataFrame, filename: str):
        df.to_excel(filename, index=False)
        print(f"Exported to {filename}")


def default_filter(df: pd.DataFrame):
    df = df[~pd.isna(df[column_name_code])]
    return df


def get_columns(columns: Optional[list], df: pd.DataFrame):
    if not columns:
        return default_columns

    not_found = list(filter(lambda x: x not in df.columns, columns))
    if len(not_found):
        eprint(f"Column(s) not found: {not_found}\n")
        print_help_columns("Use one of these columns:", df)
        exit(1)

    return list(OrderedDict.fromkeys(default_columns + columns))


def export_df(df: pd.DataFrame, export_argument: str):
    exporters = [SimpleExporter(), PythonExporter(), TsvExporter(), ExcelExporter()]
    for e in exporters:
        if e.matched(df, export_argument):
            break


def list_langs(export: str, columns: Optional[list]=None):
    df = get_lang_df()
    df = default_filter(df)

    if columns and columns[0] == "list":
        print_help_columns("Available columns:", df)
    else:
        selected_columns = get_columns(columns, df)
        df = df[selected_columns].sort_values(by=column_name_language_name)

        export_df(df, export)
