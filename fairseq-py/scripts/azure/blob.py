#!/usr/bin/env python3

"""
Helper functions for working with Azure blob storage.
"""

import logging
import os
from urllib.parse import urlparse

from utils import bash

AZCOPY = "/data/users/common/bin/azcopy"

logger = logging.getLogger("blob")


def get_blob_path_as_a_directory(blob_path):
    """
    In many cases we want to ensure that a given blob URL is treated as a directory.
    For example, suppose we get an input blob URL that looks like this:

        https://fairacceleastus.blob.core.windows.net/public?sv=...

    In most cases we want to treat `public` as a directory, but this requires
    that there's a trailing slash (i.e., `.../public/?sv=...`).

    This function can be used to ensure there's a trailing slash in the URL path.
    """
    o = urlparse(blob_path)
    if not o.path.endswith("/"):
        o = o._replace(path=o.path + "/")
    return o.geturl()


def azcopy_local_file_to_blob_dir(local_file_path, blob_path):
    o = urlparse(blob_path)
    if not o.path.endswith("/"):
        raise ValueError(
            f"expected {blob_path} to be a directory, with a trailing slash"
        )
    output = bash(f"{AZCOPY} cp {local_file_path} '{blob_path}'")
    logger.warning(output)
