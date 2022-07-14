#!/usr/bin/env python3

import logging
import re
import subprocess
from typing import List

from slurm import expand_nodes, find_idle_nodes, remove_nodes
from ssh import pdsh
from utils import get_argparse

logger = logging.getLogger("nvidia_smi")

# number of acceptable uncorrectable errors
UNCORRECT_THRESH = 0
# number of unacceptable correctable errors
CORRECT_THRESH = 10000


def collapse_space(s: str):
    return re.sub(r" +", " ", s)


def disconnected_check(hosts: str) -> List[str]:
    # this one is a little awkward compared to others bc we only want stderr
    logger.info(f"Checking {hosts} for disconnected GPUs")
    cmd = f"pdsh -R ssh -u 5 -w '{hosts}' nvidia-smi -q -d 'ECC'"
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
    lines = output.strip().split("\n")
    bad_nodes = []
    for line in lines:
        if "exited with exit code 15" in line:
            hostname = line.split(": ")[1]
            logger.error(f"Disconnected GPU on {hostname}.")
            bad_nodes.append(hostname)
    return bad_nodes


def ecc_check(hosts: str) -> List[str]:
    """
    Checks the given hosts for ECC issues.
    """
    logger.info(f"Running ECC checks on {hosts}")
    raw_output = pdsh("nvidia-smi -q -d 'ECC'", hosts)
    lines = raw_output.split("\n")

    bad_hosts = set()
    in_volatile = {}
    for line in lines:
        hostname = line.split(":")[0]
        if hostname not in in_volatile:
            in_volatile[hostname] = False
        if "Volatile" in line:
            in_volatile[hostname] = True
        elif "Aggregate" in line:
            in_volatile[hostname] = False
        if in_volatile[hostname] and ("Correctable" in line or "Uncorrectable" in line):
            niceline = (
                collapse_space(line).replace(hostname + ": ", "").replace(" : ", ": ")
            )
            if "N/A" in line:
                logger.warning(f"{hostname}: value N/A")
                continue
            value = int(line.split(": ")[-1])
            if "Correctable" in line and value > CORRECT_THRESH:
                logger.warning(f"{hostname}: ecc high correctables: {niceline}")
            elif "Uncorrectable" in line and value > UNCORRECT_THRESH:
                logger.critical(f"{hostname}: ecc high uncorrectables: {niceline}")
                bad_hosts.add(hostname)
    if not bad_hosts:
        logger.info("All nodes pass ECC checks.")
    return list(bad_hosts)


def inforom_check(hosts: str) -> List[str]:
    logger.info(f"Running InfoROM checks on {hosts}")
    raw_output = pdsh("nvidia-smi", hosts, silent_stderr=True)
    lines = raw_output.strip().split("\n")
    bad_nodes = []
    for line in lines:
        if "WARNING: infoROM is corrupted" in line:
            host = line.split(": ")[0]
            logger.warning(line.replace("WARNING: ", ""))
            # bad_nodes.append(host)  # intentionally commented as inforom is a warning

    # this is never fatal so just warn
    return bad_nodes


def mig_check(hosts: str) -> List[str]:
    logger.info(f"Running MIG checks on {hosts}")
    raw_output = pdsh("nvidia-smi -L", hosts)

    lines = raw_output.strip().split("\n")
    bad_hosts = set()
    for line in lines:
        hostname = line.split(":")[0]
        if " MIG " in line:
            bad_hosts.add(hostname)
            logger.error(f"{hostname}: MIG issue: {line}")
    if not bad_hosts:
        logger.info("All nodes pass MIG tests")
    return list(bad_hosts)


def all_nvidia_smi_check(hosts):
    failed_disconnect = disconnected_check(hosts)
    # skip hosts that already failed gpu disconnected, to prevent log spam
    hosts = remove_nodes(hosts, failed_disconnect)
    failed_ecc = ecc_check(hosts)
    failed_mig = mig_check(hosts)
    failed_inforom = inforom_check(hosts)

    return list(set(failed_disconnect + failed_ecc + failed_mig + failed_inforom))


def main():
    parser = get_argparse()
    parser.add_argument("hosts", nargs="?", default=None, help="Nodes to run this on")
    args = parser.parse_args()
    if args.hosts is None:
        args.hosts = find_idle_nodes()

    all_nvidia_smi_check(args.hosts)


if __name__ == "__main__":
    main()
