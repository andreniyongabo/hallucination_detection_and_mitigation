#!/usr/bin/env python3

import logging
import os
import sys
from typing import List

from slurm import find_idle_nodes
from ssh import pdsh
from utils import bash, get_argparse

logger = logging.getLogger(__name__)

KNOWN_BAD = {}
KNOWN_WARN = {
    # this machine was repaired as of 2022-01-19
    # "BL24A1060509037": "Slow IB",
    # This machine was repaired as of 2021-12-23 11:50 ET
    # "BL24A1060109037": "Previously had high DRAM correctable errors",
}


def get_physical_name():
    data = bash("/data/users/common/bin/kvp_client")
    lines = data.split("\n")
    for line in lines:
        if "PhysicalHostNameFullyQualified" in line:
            return line.split(": ")[-1]


def _get_this_script():
    """Return the path to this script"""
    return os.path.abspath(__file__)


def addr(host):
    """
    Finds the physical address of a given (remote) host.
    """
    return bash(f"ssh {host} python3 {_get_this_script()} --print").strip()


def find_bad_hosts(hosts: str) -> List[str]:
    logger.info(f"Checking {hosts} for hardware on our blocklist.")
    raw_output = pdsh(f"python3 {_get_this_script()} --print", hosts).strip()
    bad_hosts = []
    for line in raw_output.split("\n"):
        hostname, physical = line.split(": ")
        if physical in KNOWN_BAD:
            reason = KNOWN_BAD[physical]
            logger.critical(f"{hostname}: known bad host {physical} ({reason})")
            bad_hosts.append(hostname)
        elif physical in KNOWN_WARN:
            reason = KNOWN_WARN[physical]
            logger.warning(f"{hostname}: known iffy host {physical} ({reason})")
        else:
            logger.debug(f"{hostname}: {physical} passes known bad node list")
    return bad_hosts


def main():
    parser = get_argparse()
    parser.add_argument(
        "--print", action="store_true", help="Print this machine's hardware ID and exit"
    )
    parser.add_argument(
        "hosts",
        nargs="?",
        default=None,
        help="Hosts to run this on. Defaults to all idle nodes.",
    )
    args = parser.parse_args()
    if args.print:
        print(get_physical_name())
    else:
        if args.hosts is None:
            args.hosts = find_idle_nodes()
        for host in find_bad_hosts(args.hosts):
            print(f"{host}: known bad host")


if __name__ == "__main__":
    main()
