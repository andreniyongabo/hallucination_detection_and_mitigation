import argparse
import os
import sys
from pathlib import Path

import hanzidentifier
from tqdm import tqdm


def _split(
    zho_infile,
    parallel_infile,
    zho_simplified_outfile,
    zho_traditional_outfile,
    parallel_simplified_outfile,
    parallel_traditional_outfile,
    size,
):
    simplified_cnt = 0
    traditional_cnt = 0
    read_lines_cnt = 0
    parallel_line = parallel_infile.readline() if parallel_infile else None
    with tqdm(total=size) as pbar:
        zho_line = zho_infile.readline()
        while zho_line:
            pbar.update(zho_infile.tell() - pbar.n)
            read_lines_cnt += 1
            if hanzidentifier.is_simplified(zho_line):
                simplified_cnt += 1
                zho_simplified_outfile.write(zho_line)
                if parallel_line:
                    parallel_simplified_outfile.write(parallel_line)
            if hanzidentifier.is_traditional(zho_line):
                traditional_cnt += 1
                zho_traditional_outfile.write(zho_line)
                if parallel_line:
                    parallel_traditional_outfile.write(parallel_line)
            zho_line = zho_infile.readline()
            parallel_line = parallel_infile.readline() if parallel_infile else None
    return read_lines_cnt, simplified_cnt, traditional_cnt


def split(
    zho_infile: str,
    parallel_infile: str,
    zho_simplified_outfile: str,
    zho_traditional_outfile: str,
    parallel_simplified_outfile: str,
    parallel_traditional_outfile: str,
):
    size = Path(zho_infile).stat().st_size
    with open(zho_infile) as zho, open(
        zho_simplified_outfile, mode="w"
    ) as zho_simplified, open(zho_traditional_outfile, mode="w") as zho_traditional:
        if parallel_infile:
            assert parallel_simplified_outfile
            assert parallel_traditional_outfile
            with open(parallel_infile) as parallel, open(
                parallel_simplified_outfile, mode="w"
            ) as parallel_simplified, open(
                parallel_traditional_outfile, mode="w"
            ) as parallel_traditional:
                return _split(
                    zho,
                    parallel,
                    zho_simplified,
                    zho_traditional,
                    parallel_simplified,
                    parallel_traditional,
                    size,
                )
        else:
            return _split(zho, None, zho_simplified, zho_traditional, None, None, size)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Split Chinese files into Simplified and Traditional scripts, "
        "supports both monolingual and parallel texts"
    )
    parser.add_argument("--zho", required=True, help="path to Chinese text file")
    parser.add_argument(
        "--parallel",
        required=False,
        help="Optional path to a parallel text file in another language",
    )
    parser.add_argument("--output_suffix_simplified", default="zho_Hans")
    parser.add_argument("--output_suffix_traditional", default="zho_Hant")
    parser.add_argument(
        "--destination_simplified",
        required=True,
        help="Where to store transformed simplified files",
    )
    parser.add_argument(
        "--destination_traditional",
        required=True,
        help="Where to store transformed traditional files",
    )
    parser.add_argument(
        "--force",
        default=False,
        help="Override target files if exist",
        action="store_true",
    )

    return parser


def cli_main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.destination_simplified, exist_ok=True)
    os.makedirs(args.destination_traditional, exist_ok=True)

    zho = args.zho
    zho_simplified = os.path.splitext(zho)[0] + "." + args.output_suffix_simplified
    zho_traditional = os.path.splitext(zho)[0] + "." + args.output_suffix_traditional
    zho_simplified = (
        args.destination_simplified + os.path.sep + os.path.basename(zho_simplified)
    )
    zho_traditional = (
        args.destination_traditional + os.path.sep + os.path.basename(zho_traditional)
    )

    if not args.force:
        if os.path.exists(zho_simplified):
            print("ERROR: {} exists, use --force to override".format(zho_simplified))
            sys.exit(1)
        if os.path.exists(zho_traditional):
            print("ERROR: {} exists, use --force to override".format(zho_traditional))
            sys.exit(1)

    parallel = args.parallel
    parallel_simplified = None
    parallel_traditional = None
    if parallel:
        parallel_simplified = (
            args.destination_simplified + os.path.sep + os.path.basename(parallel)
        )
        parallel_traditional = (
            args.destination_traditional + os.path.sep + os.path.basename(parallel)
        )
        if not args.force:
            if os.path.exists(parallel_simplified):
                print(
                    "ERROR: {} exists, use --force to override".format(
                        parallel_simplified
                    )
                )
                sys.exit(1)
            if os.path.exists(parallel_traditional):
                print(
                    "ERROR: {} exists, use --force to override".format(
                        parallel_traditional
                    )
                )
                sys.exit(1)

    read_lines_cnt, simplified_cnt, traditional_cnt = split(
        zho,
        parallel,
        zho_simplified,
        zho_traditional,
        parallel_simplified,
        parallel_traditional,
    )
    print(
        "Processed {} read {} lines. Simplified {} lines. Traditional {} lines".format(
            zho, read_lines_cnt, simplified_cnt, traditional_cnt
        )
    )


cli_main()
