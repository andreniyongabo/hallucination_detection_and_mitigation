#!/usr/bin/env python3

"""
Script for updating hostfiles on all nodes.
"""

import logging
import os
import re
import shutil
import socket
import sys
from typing import List

from slurm import find_all_nodes, parse_slurm_properties
from ssh import clean_known_hosts, pdsh
from utils import bash, get_argparse

logger = logging.getLogger("updatehost")


def update_this_node():
    """
    Update only the node this code is being run on.
    """
    logger.debug(f"Updating hosts file on {socket.gethostname()}")

    nodes = parse_slurm_properties(bash("scontrol show node"))
    logger.debug(f"Nodes: {len(nodes)}")

    # Check to see if /etc/hosts.orig exists
    if not os.path.isfile("/etc/hosts.orig"):
        shutil.copyfile("/etc/hosts", "/etc/hosts.orig")
    else:
        shutil.copyfile("/etc/hosts.orig", "/etc/hosts")

    outfilename = "/etc/hosts"
    # Append to /etc/hosts
    outfile = open(outfilename, "a")
    outfile.write("\n# --------------------Slurm Aliases----------------------")

    for node in nodes:
        if (
            "NodeName" in node
            and "NodeAddr" in node
            and node["NodeName"] != node["NodeAddr"]
        ):
            outfile.write("\n{} {}".format(node["NodeAddr"], node["NodeName"]))
    outfile.close()


def _get_this_script():
    """Return the path to this script"""
    return os.path.abspath(__file__)


def _sudo_run_one(verbose: bool = False):
    verbose_cli = "--verbose" if verbose else ""
    bash(f"sudo python3 {_get_this_script()} --one {verbose_cli}")


def check_hosts_file() -> List[str]:
    """
    Checks for irregularities in the hosts file.

    Returns a list of possible bad nodes from this.
    """
    mappings = {}
    bad_nodes = set()
    with open("/etc/hosts") as f:
        for line in f:
            if "hpc-pg0" not in line:
                continue
            ip, name = line.strip().split()
            if ip in mappings:
                first = mappings[ip]
                bad_nodes.add(first)
                bad_nodes.add(name)
                logger.error(f"Nodes {first} and {name} have same IP ({ip})")
            mappings[ip] = name
    return list(bad_nodes)


def update_all_nodes(hosts=None, verbose: bool = False):
    """
    Run the update script on the current host, and all provided hosts.

    If hosts is left None, will use all hosts on the cluster.
    """
    if hosts is None:
        hosts = find_all_nodes()
    _sudo_run_one(verbose)

    clean_known_hosts()

    logger.info(f"Updating hosts file on hosts {hosts}")
    verbose_cli = "--verbose" if verbose else ""
    pdsh(f"sudo python3 {_get_this_script()} --one {verbose_cli}", hosts=hosts)


def main():
    parser = get_argparse()
    parser.add_argument(
        "--one", action="store_true", help="Only run this on the current node."
    )
    parser.add_argument("hosts", nargs="?", default=None, help="Nodes to run this on")
    args = parser.parse_args()

    if args.one:
        update_this_node()
    else:
        if args.hosts is None:
            args.hosts = find_all_nodes()
        update_all_nodes(args.hosts, verbose=args.verbose)
    check_hosts_file()


if __name__ == "__main__":
    main()
