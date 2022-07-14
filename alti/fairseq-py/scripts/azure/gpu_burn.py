#!/usr/bin/env python3

"""
Script for checking the health of a node via GPU-burn.
"""

import logging
import os
from multiprocessing import Pool
from typing import List

from slurm import expand_nodes, find_idle_nodes
from ssh import pdsh
from utils import bash, get_argparse, get_script_dir

logger = logging.getLogger("gpu_burn")


def gpu_burn(hosts: str) -> List[str]:
    """
    Runs GPU burn on all provided hosts.

    hosts should be in -w form, e.g. "hpc-pg0-[1,3-5]"

    Returns a list of the nodes containing at least 1 bad gpu.
    """
    logger.info(f"Running GPU burn test on {hosts}")
    raw_output = pdsh(f"{get_script_dir()}/_gpuburn/wrapper", hosts, silent_stderr=True)
    output = raw_output.strip().replace("\t", "").split("\n")
    bad_hosts = set()
    for line in output:
        hostname, *result = line.split(":")
        result = ":".join(result)
        if "FAULTY" in line:
            logger.critical(f"{hostname}: gpu-burn fault: {result}")
            bad_hosts.add(line.split(":")[0])
        else:
            logger.debug(f"{hostname}: gpu-burn okay: {result}")

    bad_hosts = list(bad_hosts)
    if not bad_hosts:
        logger.info("All nodes pass gpu_burn")
    return bad_hosts


def _run_slurm_individual(hostname):
    path = os.path.join(get_script_dir(), "_gpuburn", "wrapper")
    output = bash(
        "srun --error /dev/null --time 00:01:00 --exclusive --gpus-per-node 8 "
        f"-c 96 -w {hostname} {path}"
    )
    output = output.strip().replace("\t", "")
    return hostname, output


def slurm_gpu_burn(hosts: str) -> List[str]:
    expanded = expand_nodes(hosts)
    bad_hosts = set()
    with Pool(processes=len(expanded)) as pool:
        for hostname, status in pool.map(_run_slurm_individual, expanded):
            for line in status.split("\n"):
                if "FAULTY" in line:
                    logging.critical(f"{hostname}: gpu-burn fault: {line}")
                    bad_hosts.add(hostname)
                else:
                    logging.debug(f"{hostname}: gpu-burn okay: {line}")
    return bad_hosts


def main():
    parser = get_argparse()
    parser.add_argument(
        "hosts",
        default=None,
        nargs="?",
        help="Which nodes to GPU burn. Defaults to all idle nodes.",
    )
    parser.add_argument(
        "--slurm", action="store_true", help="If true, use slurm as a backend"
    )
    args = parser.parse_args()
    if args.hosts is None:
        args.hosts = find_idle_nodes()
    bad_hosts = slurm_gpu_burn(args.hosts) if args.slurm else gpu_burn(args.hosts)
    for bad_host in bad_hosts:
        print(f"{bad_host}:failed gpuburn")
    if not bad_hosts:
        logging.info("All tested hosts pass gpu-burn.")


if __name__ == "__main__":
    main()
