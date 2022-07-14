#!/usr/bin/env python3

import logging
import os
import random
import sys
from multiprocessing import Pool
from typing import Dict, List

from slurm import expand_nodes, find_idle_nodes, is_slurmable
from utils import bash, get_argparse, get_script_dir

logger = logging.getLogger(__name__)

# minimum threshold to be considered okay by our tests
THRESHOLD = 179


def _run_pairing(host1, host2):
    logger.debug(f"Running infinband pairing {host1} and {host2}")
    script_dir = get_script_dir()
    pairing = f"{host1},{host2}"
    fname = f".nccl_output_{pairing}"
    try:
        bash(
            f"bash {script_dir}/nccl_tests/run_nccl_allreduce.sh 2 {fname} -w {pairing} --wait"
        )
        with open(fname) as f:
            bandwidth_line = f.read().split("\n")[-5]
            bandwidth = bandwidth_line.split()[-2]
        logger.debug(f"Infiniband pairing {host1},{host2} got bandwidth {bandwidth}")
        return float(bandwidth), host1, host2
    except Exception:
        # something went wrong with the test. note the failure by saying bandwidth
        # was terrible
        return 0, host1, host2
    finally:
        # clean up even if we control-c
        if os.path.exists(fname):
            os.remove(fname)


def measure_infiniband(hosts: str) -> Dict[str, float]:
    """
    Measures the minimum infiniband from utilizing a host.
    """
    logger.debug(f"Running infiniband tests on {hosts}")
    expanded = expand_nodes(hosts)

    # make sure we're skipping any drained nodes
    # as fixmyazure may have drained them earlier.
    expanded = [n for n in expanded if is_slurmable(n)]
    random.shuffle(expanded)

    if len(expanded) < 3:
        raise ValueError("Infiniband can only be measured with 3+ hosts")

    rotated = expanded[1:] + expanded[:1]

    max_bw = {}
    with Pool(processes=len(expanded)) as pool:
        for bw, host1, host2 in pool.starmap(_run_pairing, zip(expanded, rotated)):
            max_bw[host1] = max(max_bw.get(host1, 0), bw)
            max_bw[host2] = max(max_bw.get(host2, 0), bw)

    for key, value in max_bw.items():
        if value < THRESHOLD:
            logger.error(f"{key}: max bandwidth {value} below threshold {THRESHOLD}")
        else:
            logger.debug(f"{key}: max bandwidth was {value}")
    return max_bw


def find_bad_infiniband_hosts(hosts: str) -> List[str]:
    logger.info(f"Running NCCL tests on {hosts}")
    max_bw = measure_infiniband(hosts)
    bad_hosts = [h for h, bw in max_bw.items() if bw < THRESHOLD]
    if not bad_hosts:
        logger.info(f"All hosts pass NCCL tests.")
    return bad_hosts


def main():
    parser = get_argparse()
    parser.add_argument(
        "hosts", default=None, nargs="?", help="Which nodes to check for infiniband."
    )
    args = parser.parse_args()
    if args.hosts is None:
        args.hosts = find_idle_nodes()

    for host in find_bad_infiniband_hosts(args.hosts):
        print("{host}: bad infiniband")


if __name__ == "__main__":
    main()
