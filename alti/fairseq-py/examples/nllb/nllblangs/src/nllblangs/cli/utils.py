import argparse
import sys


# From
# https://sourceforge.net/projects/ruamel-std-argparse/


def set_default_subparser(parser, name):
    """default subparser selection. Call after setup, just before parse_args()"""
    subparser_found = False
    for arg in sys.argv[1:]:
        if arg in ["-h", "--help"]:  # global help if no subparser
            break
    else:
        for x in parser._subparsers._actions:
            if not isinstance(x, argparse._SubParsersAction):
                continue
            for sp_name in x._name_parser_map.keys():
                if sp_name in sys.argv[1:]:
                    subparser_found = True
        if not subparser_found:
            sys.argv.insert(1, name)
