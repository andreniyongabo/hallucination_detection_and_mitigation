# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import re
import subprocess
import tempfile
import typing as tp
from contextlib import contextmanager

import numpy as np

from examples.nllb.mining.global_mining.data_utils import DataConfig
from examples.nllb.mining.global_mining.embedding_utils import Embedding
from examples.nllb.mining.nllb_lib.utils import ensure_dir

logger = logging.getLogger("mining_utils")


def iso3_to_iso2(iso3: str) -> str:
    iso3_to_iso2 = {
        "eng": "en",
        "amh": "am",
        "hau": "ha",
        "ibo": "ig",
        "lin": "ln",
        "lug": "lg",
        "luo": "luo",
        "nya": "ny",
        "orm": "om",
        "sna": "sn",
        "som": "so",
        "swh": "sw",
        "wol": "wo",
        "xho": "xh",
        "yor": "yo",
        "zul": "zu",
    }
    if iso3 not in iso3_to_iso2:
        logger.warning(f"No mapping for lang {iso3}")
        return iso3
    return iso3_to_iso2[iso3]


def tokenization_type(lang: str, token_lang_file: str):
    # TODO: move away from sed file
    # note: current file accepts ISO3
    proc = subprocess.run(
        f"echo {lang} | sed -f {token_lang_file}",
        capture_output=True,
        shell=True,
        check=True,
    )
    return proc.stdout.decode("utf-8").strip()


def get_cached_line_count(
    lang: str,
    data_cfg: DataConfig,
    shard: tp.Optional[int] = None,
) -> int:
    """
    the xxx.nl file contains the number of lines for each shard of that lang. Sum this up.
    If you ask for a specific shard, return just the number for that shard.
    """
    nl_file = os.path.join(
        data_cfg.data_shard_dir, f"{data_cfg.bname}.{data_cfg.shard_type}.{lang}.xxx.nl"
    )
    assert os.path.isfile(nl_file), f"ERROR: {nl_file} missing"
    count = 0
    with open(nl_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if shard is not None and shard == idx:
                return int(line)
            count += int(line)
    return count


def get_cached_num_parts(
    lang: str,
    data_cfg: DataConfig,
) -> int:
    """
    the xxx.nl file contains the number of lines for each shard of that lang. Get number of shards
    from that.
    """
    nl_file = os.path.join(
        data_cfg.data_shard_dir, f"{data_cfg.bname}.{data_cfg.shard_type}.{lang}.xxx.nl"
    )
    assert os.path.isfile(nl_file), f"ERROR: {nl_file} missing"
    count = 0
    with open(nl_file, "r", encoding="utf-8") as _:
        count += 1
    return count


def get_faiss_index_type(
    lang: str,
    data_cfg: DataConfig,
) -> str:
    nb_sent = get_cached_line_count(data_cfg=data_cfg, lang=lang)
    if nb_sent > 500000000:
        return "OPQ64,IVF262144,PQ64"
    elif nb_sent > 100000000:
        return "OPQ64,IVF131072,PQ64"
    elif nb_sent > 10000000:
        return "OPQ64,IVF65536,PQ64"
    elif nb_sent > 4000000:
        return "OPQ64,IVF32768,PQ64"
    elif nb_sent > 700000:
        return "OPQ64,IVF16384,PQ64"
    elif nb_sent > 250000:
        return "OPQ64,IVF8192,PQ64"
    elif nb_sent > 160000:
        return "OPQ64,IVF4096,PQ64"
    elif nb_sent > 80000:
        return "OPQ64,IVF2048,PQ64"
    return "OPQ64,IVF1024,PQ64"


@contextmanager
def FakeEmbedName(true_filename: str, file_index: str, lang: str):
    """ "
    index.py does some gymnastics expecting specific file names, cheat a bit
    by creating a symlink that looks correct. Moving to pure py should get
    rid of this.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        embedding_basename = os.path.join(tmp_dir, f"dummy_emb-{file_index:03d}")
        embedding_file = f"{embedding_basename}.{lang}.000"
        os.symlink(
            true_filename,  # iteration_value is the name of the input embedding
            embedding_file,
        )
        yield (embedding_basename, embedding_file)


def extract_shard_id(filename: str, default: int = 0) -> int:
    """
    extract shard index from the input file name.
    """
    m = re.search("\.([0-9]{3})\.", filename)
    if m is None:
        return default
    return int(m[1])
