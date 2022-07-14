#!/usr/bin/env python3

"""
Generic utils.
"""

import argparse
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)

try:
    import coloredlogs

    COLORED_LOGS = True
except ImportError:
    COLORED_LOGS = False


def get_argparse():
    """
    Return an ArgParse that can set logging to verbose.
    """
    # black magic, sorry y'all
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Turn on verbose logging."
    )
    f = parser.parse_args

    def _parse_args_with_logging(*args, **kwargs):
        parsed = f(*args, **kwargs)
        fmt = "%(asctime)s %(levelname)-8s %(name)-10s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"
        level = logging.DEBUG if parsed.verbose else logging.INFO
        # set the global logs to verbose
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        if COLORED_LOGS:
            coloredlogs.install(fmt=fmt, datefmt=datefmt, level=level)
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
            handler.setLevel(level)
            logging.basicConfig(handlers=[handler])
        return parsed

    parser.parse_args = _parse_args_with_logging
    return parser


def get_script_dir():
    """
    Finds the directory of the script being run.
    """
    return os.path.dirname(os.path.abspath(__file__))


def bash(cmd: str, silent_stderr: bool = False) -> str:
    """
    Run a command in shell in return the output as a string.

    if silent_stderr is True, then stderr will be nulled.
    Otherwise, it will be passed through to root stderr.
    """
    logger.debug(f"Running command '{cmd}'")
    if silent_stderr:
        stderr = subprocess.DEVNULL
    else:
        stderr = None
    return subprocess.check_output(cmd, shell=True, stderr=stderr).decode()


def attach_log_file(logfile):
    """
    Begins streaming logs to a file on disk.
    """
    handler = logging.FileHandler(logfile)
    # always log in verbose mode on disk
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        fmt=f"%(asctime)s [%(process)5d] %(levelname)-8s %(name)-10s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    # attach to the root logger
    logging.getLogger().addHandler(handler)
    # announce this happened
    logger.info(f"Recording logs to {logfile}")
