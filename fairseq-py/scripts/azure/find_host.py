#!/usr/bin/env python3

"""
Find the hpc-pg0-X name of a host, from any form.

Examples:
$ python scripts/azure/find_host.py buo1u00000E
hpc-pg0-4

$ python scripts/azure/find_host.py 10.30.4.12
hpc-pg0-4

$ python scripts/azure/find_host.py ip-0A1E040C
hpc-pg0-4
"""

import logging
import re
import socket

from slurm import parse_slurm_properties
from utils import bash, get_argparse

logger = logging.getLogger("find_host")


def find_my_hostname():
    hostname = bash("hostname").strip()
    return find_hostname(hostname)


def find_hostname(addr):
    ipaddr = socket.gethostbyname(addr)
    logging.debug(f"Resolved {addr} to {ipaddr}")
    nodes = parse_slurm_properties(bash("scontrol show node"))

    for node in nodes:
        if (
            "NodeName" in node
            and "NodeAddr" in node
            and node["NodeName"] != node["NodeAddr"]
            and node["NodeAddr"] == ipaddr
        ):
            return node["NodeName"]


def main():
    argparse = get_argparse()
    argparse.add_argument("addr", help="Host to find. can be an IP or buo1uXX or ip-XX")
    args = argparse.parse_args()
    print(find_hostname(args.addr))


if __name__ == "__main__":
    main()
