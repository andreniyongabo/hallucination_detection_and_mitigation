#!/usr/bin/env python3

"""
This script is used to automate replacing a node
this _usually_ happens when there is some sort of dram (ecc) failure

takes one argument: the name of the node
Does several steps:
 - drains the node in slurm
 - terminates the node in cycle cloud
 - asks it to start again
"""

import json
import logging
import os
import sys
import time
from typing import Optional
from urllib.parse import urlparse

from gather_diagnostics import gather_diagnostics_remote
from slurm import drain, expand_nodes, find_all_nodes
from ssh import clean_known_hosts
from update_hosts import update_all_nodes
from utils import bash, get_argparse

CYCLE_CLOUD = "/data/users/common/bin/cyclecloud"
CLUSTER = "xlmg_east_us_oct_2021"
DRY_RUN = True
SLEEP_TIME = 10.0

logger = logging.getLogger("cyclecloud")


class States:
    FAILED = "Failed"
    STARTED = "Started"
    ALLOCATING = "Allocation"
    TERMINATED = "Terminated"


class FailedAllocationError(RuntimeError):
    """
    Represents some sort of failed state (with a reason)
    """

    pass


def cycle_cloud(cmd):
    """
    Runs the cyclecloud command.
    """
    return bash(f"{CYCLE_CLOUD} {cmd}").strip()


def check_cycle_config():
    if os.path.exists(os.path.expanduser("~/.cycle/config.ini")):
        logger.debug("Found the cyclecloud config.")
        return True
    else:
        logger.fatal("Could not find the cyclecloud config.")
        raise RuntimeError(
            "Cyclecloud CLI is not initialized. Please run:\n\n"
            f"    $ {CYCLE_CLOUD} initialize\n\n"
            "    CycleServer URL: 20.98.242.41\n"
            "    Detected untrusted certificate.  Allow?: yes\n"
            "    ..."
        )


def terminate_node(hostname, cluster=CLUSTER):
    """
    Terminate a node and block until it successfully terminates.

    Returns the final node attributes.
    """
    logger.info(f"Terminating {hostname}")
    cycle_cloud(f"terminate_node {cluster} {hostname}")
    return wait_for_state(hostname, States.TERMINATED, cluster=cluster)


def start_node(hostname, cluster=CLUSTER, max_tries=3):
    tries_left = max_tries
    while True:
        # always terminate a node before starting it, in case it's in a failed state
        drain(hostname, reason="Replacing node")
        terminate_node(hostname, cluster)

        logger.info(f"Attempting to start {hostname} ({tries_left} tries left)")
        cycle_cloud(f"start_node {cluster} {hostname}")
        try:
            result = wait_for_state(
                hostname, States.STARTED, cluster=cluster, min_confirmations=2
            )
            return result
        except FailedAllocationError as fae:
            tries_left -= 1
            if tries_left <= 0:
                logger.critical(
                    f"{hostname}: Failed for the last time. Giving up "
                    f"(last message: {fae})."
                )
                raise


def wait_for_state(hostname, goal_state, min_confirmations=1, cluster=CLUSTER):
    """
    Loop idly until the node shows the state we desire, or there is a failure.

    min_confirmations is the number of times we must see the node in the
    desired state. This stems from sometimes state changing rapidly from
    "Started" to "Failed".

    Returns the file states
    """

    # observed that sometimes a node may briefly say "Started" followed by
    # a failed, causing an early exit
    confirmations = min_confirmations
    last_state = last_status = last_reason = ""
    while True:
        attrs = get_node_attributes(hostname, cluster=cluster)
        state = attrs["State"]
        status = attrs["Status"]
        reason = attrs.get("StatusMessage", "")
        if status == States.FAILED:
            logger.error(
                f"{hostname}: failed cloud state state={state}, "
                f"status={status}, reason={reason}"
            )
            raise FailedAllocationError(attrs["StatusMessage"])
        else:
            msg = (
                f"{hostname}: cyclecloud status state={state}, "
                f"status={status}, reason={reason}"
            )
            if state != last_state or status != last_status or reason != last_reason:
                logger.info(msg)
            else:
                logger.debug(msg)
            last_state, last_status, last_reason = state, status, reason

        if state == goal_state:
            confirmations -= 1
            logger.debug(
                f"{hostname}: confirmation of {goal_state}, {confirmations} left"
            )
        else:
            confirmations = min_confirmations

        if state == goal_state and confirmations <= 0:
            return attrs

        logger.debug("Sleeping")
        time.sleep(SLEEP_TIME)


def get_node_attributes(hostname: Optional[str] = None, cluster=CLUSTER):
    """
    Fetches the attributes of node(s), including state.

    If hostname is provided, returns a Dict with only that node's attrs.
    If hostname is None (default), returns a dict[host, attrs]

    Empirically, fetching attr for all nodes is the same speed as just one.
    """
    if hostname:
        logger.debug(f"Fetching node state for {hostname}")
        results = cycle_cloud(f"show_nodes --long -c {cluster} {hostname}")
    else:
        logger.debug("Fetching all node states")
        results = cycle_cloud(f"show_nodes --long -c {cluster}")

    entries = results.split("\n\n")
    results = {}
    for entry in entries:
        result = {}
        elements = entry.split("\n")
        for element in elements:
            idx = element.index(" = ")
            key, value = element[:idx], element[idx + 3 :]
            try:
                # parse it if we can easily
                value = json.loads(value)
            except ValueError:
                pass
            result[key] = value
        results[result["Name"]] = result

    if hostname:
        assert len(results) == 1
        return results[hostname]
    else:
        return results


def replace_host(hostname, cluster=CLUSTER):
    check_cycle_config()
    logger.info(f"Replacing host {hostname}")
    retval = start_node(hostname, cluster=cluster)
    drain(hostname, reason="Node replaced")
    return retval


def main():
    parser = get_argparse()
    parser.add_argument("hosts", default=None, help="Which nodes to replace")
    parser.add_argument("-c", "--cluster", default=CLUSTER, help="Cyclecloud cluster")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation dialogue")
    parser.add_argument(
        "--skip-diagnostics", action="store_true", help="Skip gathering diagnostics"
    )
    args = parser.parse_args()

    check_cycle_config()

    hosts = expand_nodes(args.hosts)

    if not args.yes:
        response = input(
            f"About to replace {len(hosts)} hosts ({args.hosts}). Continue? [Yy] "
        )
        if response.strip().lower() != "y":
            logger.error("User failed to confirm.")
            sys.exit(1)
    else:
        logger.debug("Supplied --yes on the commandline. Skipping confirmation.")

    for hostname in hosts:
        attrs = get_node_attributes(hostname, cluster=args.cluster)
        state = attrs["State"]
        status = attrs["Status"]
        reason = attrs.get("StatusMessage", "")
        logger.info(
            f"{hostname}: cycle status state={state}, status={status}, reason={reason}"
        )
        if not args.skip_diagnostics:
            try:
                diag_blob_path = gather_diagnostics_remote(hostname)
                logger.warning(
                    "Diagnostics uploaded (Please share this URL with Microsoft via "
                    f"Teams): {diag_blob_path}"
                )
            except RuntimeError:
                raise RuntimeError(
                    "Unable to gather diagnostics! Please gather diagnostics manually "
                    "and then rerun this script with --skip-diagnostics"
                )
        replace_host(hostname)

    update_all_nodes()
    clean_known_hosts()
    logger.info(f"Nodes {args.hosts} have been replaced.")


if __name__ == "__main__":
    main()
