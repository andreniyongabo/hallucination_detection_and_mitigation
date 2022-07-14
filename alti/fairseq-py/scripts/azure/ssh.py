#!/usr/bin/env python3

"""
SSH Utilities.
"""

import logging
import os

from slurm import expand_nodes, find_all_nodes
from utils import bash, get_argparse

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(itr):
        return itr


logger = logging.getLogger("ssh")

# github's ssh key's. added automatically for convenience.
GITHUB_HOSTS = """
|1|diLJva1ldlvIkh/vsQ4QrfH9D64=|gfYHF6vxRdUd40/dCLlp+z1Ttz8= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=
|1|Ha9/dblK4LrKlaqvVlpC9DIVCT4=|dijiEzGXy93Arft/t274K8DswOI= ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg=
"""


def check_for_strict_hosts():
    """
    Ensures the user has set up SSH correctly.
    """
    logger.debug("Checking your ssh config for issues")
    fname = os.path.expanduser("~/.ssh/config")
    with open(fname) as f:
        data = f.read()
        lines = data.split("\n")
    for l1, l2 in zip(lines, lines[1:]):
        if "Host hpc-pg0-*" in l1 and "StrictHostKeyChecking no" in l2:
            logger.debug("Confirmed StrictHostKeyChecking no")
            break
    else:
        logger.warning(
            "Detected you are missing StrictHostKeyChecking. Fixing for you."
        )
        with open(fname, "w") as f:
            f.write(data)
            f.write("\n\n")
            f.write("# ---- set up by azure tools ---\n")
            f.write("Host hpc-pg0-*\n")
            f.write("    StrictHostKeyChecking no\n")
            f.write("# ---- end azure tools ---\n")


def clean_known_hosts():
    """
    Clean up all the known ssh hosts so fingerprints are real nice and up to date.
    """
    check_for_strict_hosts()
    logger.info("Cleaning up your known ssh hosts")
    # put github's keys on the nice list
    with open(os.path.expanduser("~/.ssh/known_hosts"), "w") as f:
        f.write(GITHUB_HOSTS.strip() + "\n")
    # add everyone's keys to the nice list
    logger.debug("Adding everyone's ssh fingerpint to your list")
    add_known_host = os.path.join(os.path.dirname(__file__), "add_known_host.sh")
    for host in expand_nodes(find_all_nodes()):
        try:
            bash(f"{add_known_host} {host}", silent_stderr=True)
        except:
            logging.error(f"Couldn't add host {host} to known hosts.")


def _pdsh_unsafe(cmd: str, hosts: str, silent_stderr: bool = False):
    """
    Runs the command on all hosts, and returns the output, ala pdsh.

    Hosts should be in '-w' format, e.g. 'hpc-pg0-[1,3-5]'.

    This version differs from the public "pdsh" version in that it
    does not perform ssh checks.
    """
    TIMEOUT = 60
    logger.debug(f"Running global pdsh (hosts: {hosts}) with cmd {cmd}")
    # note -u timeout flag only marks the cmd as a failure on the launcher side.
    # the command may continue to run indefinitely on the worker. see
    # `man pdsh` LIMITATIONS section.
    # however, we DO need the timeout on this side so we don't hang just because
    # one worker is having a bad day.
    return bash(
        f"pdsh -R ssh -u {TIMEOUT} -w '{hosts}' {cmd}", silent_stderr=silent_stderr
    )


def pdsh(cmd: str, hosts: str, silent_stderr: bool = False):
    check_for_strict_hosts()
    return _pdsh_unsafe(cmd, hosts, silent_stderr=silent_stderr)


def main():
    argparse = get_argparse()
    _args = argparse.parse_args()  # for verbose mode
    clean_known_hosts()


if __name__ == "__main__":
    main()
