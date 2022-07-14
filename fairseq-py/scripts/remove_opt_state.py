#!/usr/bin/env python
import argparse
import functools
import time
from glob import glob
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional

import torch

from fairseq.file_io import load_and_pop_last_optimizer_state
from fairseq.utils import remove_prefix


def remove_opt_state_from_ckpt_files(
    pth_prefix: str,
    save_dir: Optional[str] = None,
    save_prefix: str = "checkpoint_eval",
    nproc: Optional[int] = None,
    resume_failed: Optional[bool] = False,
) -> None:
    """
    Write checkpoint files to disk without the last_optimizer_state key using `load_and_pop_last_optimizer_state`
        Args::

            pth_prefix: we will process glob(f'{pth_prefix}-*.pt')
            save_dir: defaults to pth-prefix/..
            save_prefix: results written to {save_dir}/{save_prefix}{shard_info}
            nproc: how many processes to split the work across
            resume_failed: do not rewrite checkpoints that have already been written.

        Usage::

            alias co="python scripts/remove_opt_state.py"
            # MOE
            SD=/large_experiments/xlmg/models/sshleifer/expert_drop/
            co $SD/m2m.dl12.d2048.moe_w0.01.ngpu32/checkpoint_8_40000 --save-dir $SD/m2m_32e_eval/checkpoint_eval_40000/
            # One Path
            co $SD/m2m.dl12.d2048.moe_w0.01.ngpu32/checkpoint_8_40000-shard0.pt
            # sharded checkpoints
            co $SD/checkpoint_8_40000

    """
    tstart = time.time()
    parent, name = Path(pth_prefix).parent, Path(pth_prefix).name
    is_moe = Path(f"{pth_prefix}-shared.pt").exists()
    if save_dir is None:
        save_dir = parent
    if Path(pth_prefix).exists() and str(pth_prefix).endswith("pt"):
        print(
            f"Since {pth_prefix} exists, we assume this is a single dense checkpoint."
        )
        dense_ckpt = load_and_pop_last_optimizer_state(pth_prefix)
        save_path = (
            f"{save_dir}/{name[:-3]}_eval.pt"
            if save_prefix == "checkpoint_eval"
            else f"{save_dir}/{save_prefix}.pt"
        )
        Path(save_path).parent.mkdir(exist_ok=True, parents=True)
        torch.save(dense_ckpt, save_path)
        print(f"Done. Saved 1 file to {save_path}, {time.time() - tstart:.1f} seconds")
    elif not is_moe:  # Many dense checkpoints
        if pth_prefix.endswith(".pt"):
            pth_prefix = pth_prefix[:-3]
        if pth_prefix.endswith("-"):
            pth_prefix = pth_prefix[:-1]
        all_ckpt_paths = glob(f"{pth_prefix}-*.pt")
        if resume_failed:
            ckpt_paths = [
                p
                for p in all_ckpt_paths
                if not Path(get_dest_path(p, pth_prefix, save_prefix)).exists()
            ]
        else:
            ckpt_paths = all_ckpt_paths
        full_save_prefix = str(Path(save_dir).joinpath(save_prefix))
        Path(full_save_prefix).parent.mkdir(exist_ok=True)
        print(
            f"processing {len(ckpt_paths)} shards to {full_save_prefix} on {nproc or 1} workers"
        )

        if nproc is None:
            _process_sharded_dense_checkpoints(pth_prefix, full_save_prefix, ckpt_paths)
        else:
            with Pool(nproc) as p:
                func = functools.partial(
                    _process_sharded_dense_checkpoints, pth_prefix, full_save_prefix
                )
                p.map(func, ckpt_paths)
        print(
            f"Done. Processed {len(ckpt_paths)} files, {time.time() - tstart:.1f} seconds"
        )
    else:
        shared_dest = f"{save_dir}/{save_prefix}-shared.pt"
        Path(shared_dest).parent.mkdir(exist_ok=True, parents=True)
        if resume_failed and Path(shared_dest).exists():
            pass
        else:
            shared = load_and_pop_last_optimizer_state(f"{pth_prefix}-shared.pt")
            torch.save(shared, shared_dest)
        expert_files = glob(
            f"{pth_prefix}-rank-*.pt"
        )  # rank-[0-9]+ doesnt work for some reason
        assert expert_files, f"No files matching {pth_prefix}-rank-*.pt"

        if resume_failed:
            p = f"{save_dir}/{save_prefix}-rank"
            ranks_to_process = [
                i
                for i in list(range(len(expert_files)))
                if not Path(f"{p}-{i}.pt").exists()
            ]
        else:
            ranks_to_process = list(range(len(expert_files)))
        print(
            f"processing {len(ranks_to_process)} expert files on {nproc or 1} workers"
        )
        if nproc is None:
            _process_expert_files(pth_prefix, save_dir, save_prefix, ranks_to_process)
        else:
            with Pool(nproc) as p:
                func = functools.partial(
                    _process_expert_files, pth_prefix, save_dir, save_prefix
                )
                p.map(func, ranks_to_process)

        print(
            f"Done. Processed {len(ranks_to_process)} files, {time.time() - tstart:.1f} seconds"
        )


def _process_sharded_dense_checkpoints(
    pth_prefix: str, save_prefix: str, paths: List[str]
):
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        save_path = get_dest_path(p, pth_prefix, save_prefix)
        ckpt = load_and_pop_last_optimizer_state(p)
        torch.save(ckpt, save_path)
        print(f"saved {save_path}")


def get_dest_path(src_path, pth_prefix, save_prefix):
    shard_info = remove_prefix(str(src_path), pth_prefix)
    if save_prefix.endswith("-"):
        remove_prefix(shard_info, "-")
    save_path = f"{save_prefix}{shard_info}"
    return save_path


def _process_expert_files(
    pth_prefix: str, save_dir: str, save_prefix: str, ranks_to_process: List[int]
):
    if isinstance(ranks_to_process, int):
        ranks_to_process = [ranks_to_process]
    assert isinstance(ranks_to_process, list), f"expected list got {ranks_to_process}"
    for expert_rank in ranks_to_process:
        efile = f"{pth_prefix}-rank-{expert_rank}.pt"
        expert = load_and_pop_last_optimizer_state(efile)
        save_path = f"{save_dir}/{save_prefix}-rank-{expert_rank}.pt"
        torch.save(expert, save_path)
        print(f"saved expert {expert_rank}")


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("pth_prefix", type=str)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-prefix", type=str, default="checkpoint_eval")
    parser.add_argument("--nproc", type=int, default=None)
    parser.add_argument("--resume-failed", "--rf", action="store_true", default=False)
    return parser


if __name__ == "__main__":
    args = get_argparser().parse_args()
    remove_opt_state_from_ckpt_files(
        args.pth_prefix,
        args.save_dir,
        save_prefix=args.save_prefix,
        nproc=args.nproc,
        resume_failed=args.resume_failed,
    )
