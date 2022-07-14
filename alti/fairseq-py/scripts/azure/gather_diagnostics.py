#!/usr/bin/env python3

"""
This script runs diagnostics and uploads the results to blob storage.
"""

import logging
import os
import subprocess
import tempfile
from urllib.parse import urlparse

from blob import azcopy_local_file_to_blob_dir, get_blob_path_as_a_directory
from find_host import find_my_hostname
from slurm import get_cluster_name
from utils import bash, get_argparse

logger = logging.getLogger("diag")
DEFAULT_BLOB_PATH = "https://fairacceleastus.blob.core.windows.net/public/?sv=2020-08-04&ss=b&srt=sco&sp=rwdlactfx&se=2023-10-06T11:23:33Z&st=2021-10-06T03:23:33Z&spr=https&sig=s6aw4Ca4Ohbr7LQ%2BG9s58PEyYJsbXHjs%2Fc%2BuoTvzTUo%3D"


def gather_diagnostics_local(blob_path=None):
    """
    Gather diagnostics from the local host and return a path to the generated
    .tar.gz file. If blob_path is given, then upload to blob storage and return
    that URL instead.
    """
    gather_diagnostics_sh = os.path.join(
        os.path.dirname(__file__), "_gather_diagnostics.sh"
    )
    tar_gz_path = bash(f"bash {gather_diagnostics_sh}").strip()
    if not tar_gz_path.endswith(".tar.gz"):
        raise RuntimeError("Unable to gather diagnostics. Please gather manually.")

    if blob_path is not None:
        blob_path = get_blob_path_as_a_directory(blob_path)
        base_filename = os.path.basename(tar_gz_path)
        my_hostname = find_my_hostname()
        new_filename = f"{get_cluster_name()}_{my_hostname}_{base_filename}"
        new_abs_path = os.path.join(tempfile.gettempdir(), new_filename)
        bash(f"sudo mv {tar_gz_path} {new_abs_path}")
        tar_gz_path = new_abs_path

        azcopy_local_file_to_blob_dir(tar_gz_path, blob_path)
        o = urlparse(blob_path)
        o = o._replace(
            path=os.path.join(o.path, os.path.basename(tar_gz_path)),
            query="",  # remove any credentials
        )
        return o.geturl()
    else:
        return tar_gz_path


def gather_diagnostics_remote(hostname, blob_path):
    """Run diagnostics on a remote host and return the blob URL."""
    logger.info(f"Gathering diagnostics for {hostname}")
    diag_py_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gather_diagnostics.py"
    )
    cmd = f"ssh {hostname} \"python3 {diag_py_path} --blob-path '{blob_path}'\""
    output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT).decode()
    for line in output.splitlines():
        # check for errors
        if (
            line.startswith("Final Job Status:")
            and line != "Final Job Status: Completed"
        ):
            raise RuntimeError(output)
        # parse blob upload URL
        if "Diagnostics uploaded to: " in line:
            url = line.partition("Diagnostics uploaded to: ")[2]
            logger.info(f"Diagnostics for {hostname}: {url}")
            return url
    # not sure what happened, return full output
    return RuntimeError(output)


def main():
    parser = get_argparse()
    parser.add_argument(
        "--blob-path",
        default=DEFAULT_BLOB_PATH,
        help="save diagnostics to the given blob directory",
    )
    parser.add_argument("host", nargs="?")
    args = parser.parse_args()

    if args.host is None:
        uploaded_path = gather_diagnostics_local(args.blob_path)
    else:
        uploaded_path = gather_diagnostics_remote(args.host, args.blob_path)
    logger.info(f"Diagnostics uploaded to: {uploaded_path}")


if __name__ == "__main__":
    main()
