from typing import Optional

from .core import *
from .export import export_df


def merge_files(
    left: str,
    right: str,
    how: str,
    export: str,
    left_column: Optional[str] = None,
    right_column: Optional[str] = None,
    left_header: Optional[int] = None,
    right_header: Optional[int] = None,
):
    df_left = get_df(left, left_header)
    df_right = get_df(right, right_header)

    if left_column is None or left_column not in df_left.columns:
        print_help_columns("Available columns in left:", df_left)
    if right_column is None or right_column not in df_right.columns:
        print_help_columns("Available columns in right:", df_right)
    if left_column is None or right_column is None:
        exit(0)

    left_values = set(df_left[left_column].values.tolist())
    right_values = set(df_right[right_column].values.tolist())

    lr_difference = left_values.difference(right_values)
    rl_difference = right_values.difference(left_values)

    if lr_difference:
        eprint("Warning: these values appear only on the left:", lr_difference)
    if rl_difference:
        eprint("Warning: these values appear only on the right:", rl_difference)

    merged = df_left.merge(
        df_right, left_on=left_column, right_on=right_column, how=how
    )

    eprint(f"Left size: {len(df_left)}")
    eprint(f"Right size: {len(df_right)}")

    export_df(merged, export)
