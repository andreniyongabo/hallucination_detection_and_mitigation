#!/usr/bin/env python3

"""
The global health check script.
"""
import logging
import os
from enum import Enum

import blocklist
import gpu_burn
import nccl
import nvidia_smi
import ssh
import update_hosts
from gather_diagnostics import DEFAULT_BLOB_PATH, gather_diagnostics_remote
from slurm import (
    drain,
    find_all_nodes,
    find_drained_nodes,
    find_idle_nodes,
    remove_nodes,
    undrain_kill_task,
)
from ssh import pdsh
from utils import get_argparse

logger = logging.getLogger("fixmyazure")


class ItemToFix:
    BLOCKLIST = "blocklist"
    NVIDIA_SMI = "nvidia_smi"
    GPU_BURN = "gpu_burn"
    INFINIBAND = "infiniband"


def drain_and_report(host, reason, blob_path):
    drain(host, reason)
    return gather_diagnostics_remote(host, blob_path)


def all_health_checks(mode=None, hosts=None, skip="", blob_path=DEFAULT_BLOB_PATH):
    if mode is None and not hosts:
        raise ValueError("Must specify mode or hosts")
    if mode and hosts:
        raise ValueError("Can only specify one of mode or hosts")

    if hosts:
        pass
    elif mode == "idle":
        undrain_kill_task()
        hosts = find_idle_nodes()
    elif mode == "all":
        hosts = find_all_nodes()
    elif mode == "drained":
        hosts = find_drained_nodes()
    logger.info(f"Running all possible health checks on {hosts}.")

    ssh.check_for_strict_hosts()
    update_hosts.update_all_nodes()

    reprovision_nodes = {}
    skip = skip.split(",")

    if ItemToFix.BLOCKLIST not in skip:
        # bad hardware checks
        blocklist_hosts = blocklist.find_bad_hosts(hosts)
        for bad_host in blocklist_hosts:
            reprovision_nodes[bad_host] = drain_and_report(
                bad_host, reason="Blocklist", blob_path=blob_path
            )
        hosts = remove_nodes(hosts, blocklist_hosts)

    if ItemToFix.NVIDIA_SMI not in skip:
        # Nvidia SMI checks
        nvidiasmi_hosts = nvidia_smi.all_nvidia_smi_check(hosts)
        for bad_host in nvidiasmi_hosts:
            reprovision_nodes[bad_host] = drain_and_report(
                bad_host, reason="Failed nvidiasmi checks", blob_path=blob_path
            )
        hosts = remove_nodes(hosts, nvidiasmi_hosts)

    if ItemToFix.GPU_BURN not in skip:
        # gpu burn checks
        gpuburn_hosts = gpu_burn.gpu_burn(hosts)
        for bad_host in gpuburn_hosts:
            reprovision_nodes[bad_host] = drain_and_report(
                bad_host, reason="Failed GPU burn", blob_path=blob_path
            )
        hosts = remove_nodes(hosts, gpuburn_hosts)

    if ItemToFix.INFINIBAND not in skip:
        try:
            ib_hosts = nccl.find_bad_infiniband_hosts(hosts)
            for bad_host in ib_hosts:
                reprovision_nodes[bad_host] = drain_and_report(
                    bad_host, reason="Bad infiniband", blob_path=blob_path
                )
            hosts = remove_nodes(hosts, ib_hosts)
        except ValueError:
            # didn't get enough hosts to check
            logger.warning(
                "Skipping infiniband check! You should add more nodes to the checks."
            )

    logger.info("Finished running health checks.")
    return reprovision_nodes


def main():
    parser = get_argparse()
    parser.add_argument(
        "--hosts",
        help="Run on specific hosts. Can be in the slurm format: hpc-pg0-[1,5-10]",
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="idle",
        choices=["idle", "all", "drained"],
        help='Running mode. Selecting "idle" checks only health of idle nodes. '
        "Ignored if --hosts is given.",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Health check steps (blocklist, nvidia_smi, gpu_burn, infiniband) to skip, separated by comma",
    )
    parser.add_argument(
        "--blob-path",
        default=DEFAULT_BLOB_PATH,
        help="save diagnostics to the given blob directory",
    )

    args = parser.parse_args()
    if args.hosts:
        bad_nodes = all_health_checks(
            hosts=args.hosts, skip=args.skip, blob_path=args.blob_path
        )
    else:
        bad_nodes = all_health_checks(
            mode=args.mode, skip=args.skip, blob_path=args.blob_path
        )
    if bad_nodes:
        print("All diagnostics:")
    for bad_node, diagnostics in bad_nodes.items():
        print(f"{bad_node}: {diagnostics}")


if __name__ == "__main__":
    main()
