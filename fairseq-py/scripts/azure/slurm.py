#!/usr/bin/env python3

"""
SLURM utilities.
"""

import logging
import re
from typing import Dict, List

from utils import bash, get_argparse

logger = logging.getLogger("slurm")


# useful for finding info like the cluster name.
# TODO: not portable!!!
SLURM_CONF = "/etc/slurm/slurm.conf"


def remove_nodes(hoststr: str, to_remove: List[str]) -> str:
    """
    Takes a merged node list (hpc-pg0-[1,2,3]) and strip nodes from it.

    >>> remove_nodes('hpc-pg0-[1-5]', ['hpc-pg0-2'])
    ... 'hpc-pg0-[1,3,4,5]'
    """
    assert not isinstance(to_remove, str), "to_remove should be a list of hostnames."
    full_list = expand_nodes(hoststr)
    to_remove = set(to_remove)
    shortlist = [h for h in full_list if h not in to_remove]
    return merge_node_names(shortlist)


def find_all_nodes():
    """
    Finds all reachable nodes in the cluster.
    """
    # pull out the hpc-pg0- prefix from everyone
    prefix = "hpc-pg0-"

    results = bash("sinfo --Node").strip()
    lines = results.split("\n")
    nodes = []
    for line in lines[1:]:
        node, _cnt, _partition, state = line.split()
        if not node.startswith(prefix):
            logger.warning(f"Node {node} doesn't start with prefix {prefix}")
            continue
        number = node[len(prefix) :]
        if "~" not in state and "down" not in state and "*" not in state:
            nodes.append(node)
        else:
            logger.error(f"Node {node} appears unreachable.")
    return merge_node_names(nodes)


def find_unreachable_nodes():
    """
    Find nodes that slurm thinks are unreachable.
    """
    raise NotImplementedError()


def _escape_reason(reason):
    # just aggressively remove risk
    return reason.replace("'", "").replace('"', "").replace(" ", "_")


def undrain(hostname, reason="No reason provided"):
    logger.warning(f'Undraining {hostname} because "{reason}"')
    reason = _escape_reason(reason)
    return bash(f"sudo scontrol update node={hostname} state=undrain reason={reason}")


def drain(hostname, reason="No reason provided"):
    logger.warning(f'Draining {hostname} because "{reason}"')
    reason = _escape_reason(reason)
    return bash(f"sudo scontrol update node={hostname} state=drain reason={reason}")


def get_cluster_name():
    with open(SLURM_CONF) as f:
        for line in f:
            if line.startswith("ClusterName"):
                return line.strip().split("=")[1][1:-1]


def find_drained_nodes():
    """
    Find all the drained nodes.
    """
    output = bash("sinfo")
    lines = output.split("\n")
    drain = [line for line in lines if " drain " in line]
    if len(drain) >= 1:
        assert len(drain) == 1
        return drain[0].split()[5]
    else:
        return ""


def parse_slurm_properties(raw_text: str) -> List[Dict[str, str]]:
    """
    Parses common SLURM property format.
    """
    subsets = raw_text.rstrip().split("\n\n")
    output = []
    for subset in subsets:
        results = dict(re.findall(r"(\w*)=(.*\[.*@.*\]|\".*?\"|\S*)", subset))
        output.append(results)
    return output


def _get_state(node):
    for line in bash("sinfo --Node").strip().split("\n"):
        if line.startswith(node):
            _node, _cnt, _part, state = line.split()
            return state


def is_slurmable(node):
    """
    Determines if a node is eligible to run jobs on SLURM.

    Unreachable and drained nodes return false.
    """
    state = _get_state(node)
    if "drain" in state:
        return False
    if "~" in state:
        return False
    if "down" in state:
        return False
    assert "idle" in state or "alloc" in state, f"What state is {state}?"
    return True


def find_idle_nodes():
    """
    Find all the idle nodes.
    """
    output = bash("sinfo")
    lines = output.split("\n")
    idle = [line for line in lines if " idle " in line]
    drain = [line for line in lines if " drain " in line]
    if len(drain) >= 1:
        assert len(drain) == 1
        drained = drain[0].split()[5]
        logger.warning(
            f"Hosts {drained} are drained and excluded from list of idle hosts."
        )

    assert len(idle) == 1
    return idle[0].split()[5]


def expand_nodes(nodes):
    """Expand from "hpc-pg0-[1,2]" to ["hpc-pg0-1", "hpc-pg0-2"]"""
    return bash(f"scontrol show hostnames {nodes}").strip().split("\n")


def merge_node_names(hosts: List[str]) -> str:
    """Gets the short form of an expanded list of nodes."""
    return bash(f"scontrol show hostlistsorted {','.join(hosts)}").strip()


def get_summary():
    drained = set(expand_nodes(find_drained_nodes()))
    nodes = parse_slurm_properties(bash("scontrol show node"))
    for node in nodes:
        if "NodeName" in node and "Reason" in node:
            name = node["NodeName"]
            reason = node["Reason"]
            if name in drained:
                yield name, reason


def print_summary():
    for name, reason in get_summary():
        print(f"{name}: {reason}")


def get_job_state(jobid):
    return parse_slurm_properties(bash(f"scontrol show job={jobid}"))[0]["JobState"]


def undrain_kill_task():
    # Jobs which don't report finished in time will be drained automatically by
    # slurm. Sometimes this can be because they're unhealthy, but sometimes it's just
    # random. This undrains anything that was marked by kill task failed, so that
    # normal health checks can be run.
    for name, reason in get_summary():
        if "Kill task failed" in reason:
            logging.info(f"Undraining {name} because SLURM drained it for {reason}")
            undrain(name, "kill task recovery")


def main():
    argparse = get_argparse()
    argparse.add_argument(
        "cmd", choices=("summary", "drain", "undrain"), help="cmd to run"
    )
    argparse.add_argument("hosts", nargs="?", help="hosts to drain/undrain")
    argparse.add_argument("--reason", default="No reason given")

    args = argparse.parse_args()
    if args.cmd == "summary":
        print_summary()
    elif args.cmd == "drain":
        drain(args.hosts, reason=args.reason)
    elif args.cmd == "undrain":
        undrain(args.hosts, reason=args.reason)
    else:
        raise ValueError(f"Unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
