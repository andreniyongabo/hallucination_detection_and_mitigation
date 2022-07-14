import argparse

from ..export import *
from ..merge import *

from .utils import set_default_subparser


def command_list_langs(args: argparse.Namespace):
    list_langs(
        *[
            getattr(args, a)
            for a in [
                "export",
                "columns",
            ]
        ]
    )


def command_merge_files(args: argparse.Namespace):
    merge_files(
        *[
            getattr(args, a)
            for a in [
                "left",
                "right",
                "how",
                "export",
                "left_column",
                "right_column",
                "left_header",
                "right_header",
            ]
        ]
    )


def main():
    parser = argparse.ArgumentParser("nllblangs")
    subparsers = parser.add_subparsers(dest="command")

    parser_list = subparsers.add_parser("list", help="list help")
    parser_list.add_argument("--columns", nargs="+", help="Columns to export")
    parser_list.set_defaults(func=command_list_langs)

    parser_merge = subparsers.add_parser("merge", help="merge help")
    parser_merge.add_argument("left", help="Left file")
    parser_merge.add_argument("right", help="Right file")
    parser_merge.add_argument(
        "--left-column", help="Left file column to use for merging"
    )
    parser_merge.add_argument(
        "--right-column", help="Right file column to use for merging"
    )
    parser_merge.add_argument(
        "--left-header", type=int, help="Number of header rows in left file"
    )
    parser_merge.add_argument(
        "--right-header",
        type=int,
        help="Number of header rows in right file",
    )
    parser_merge.add_argument(
        "--how",
        choices=["left", "right", "outer", "inner", "cross"],
        default="inner",
        help="How to perform merge algorithm",
    )

    for _, subparser in subparsers.choices.items():
        subparser.add_argument(
            "--export",
            default="simple",
            help="export to supported file format, determined by extension: excel (.xlsx), tsv (.tsv), or 'simple', 'python'. Default: simple",
        )

    parser_merge.set_defaults(func=command_merge_files)

    set_default_subparser(parser, "list")
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
