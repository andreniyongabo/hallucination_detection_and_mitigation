import argparse
import logging
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

from omegaconf import OmegaConf

from examples.nllb.modeling.filtering.dataset import Dataset


def get_mined_datasets(mini_mine_root: str):
    root = Path(mini_mine_root)
    datasets = defaultdict(dict)

    # mini-mine5
    for path in glob(
        str(root / "mini-mine5" / "bitexts.mined" / "mm5_p5.*.OPQ64*.bitext.tsv.gz")
    ):
        _, filename = os.path.split(path)
        direction = filename.split(".")[1]
        datasets[direction]["mini-mine.5"] = Dataset(tsv=path)

    # mini-mine3
    for path in glob(str(root / "mini-mine3" / "bitexts.final" / "*.bitextf.tsv.gz")):
        _, filename = os.path.split(path)
        direction = filename.split(".")[0]
        # only add if not present in newer versions
        if direction not in datasets:
            datasets[direction]["mini-mine.3"] = Dataset(tsv=path)
    return dict(datasets)


def get_primary_datasets(paths):
    datasets = defaultdict(dict)  # direction -> corpus -> paths
    for path in paths:
        direction_directories = glob(str(Path(path) / "*-*"))
        for direction_directory in direction_directories:
            _, direction = os.path.split(direction_directory)
            src, tgt = direction.split("-")
            src_gz_glob = glob(f"{direction_directory}/*.{src}.gz")
            src_glob = glob(f"{direction_directory}/*.{src}")
            seen_fbseed = None  # We only want to save the latest fbseed version
            for src_path in src_gz_glob + src_glob:
                _, src_filename = os.path.split(src_path)
                if src_path.endswith(".gz"):
                    src_filename = src_filename[:-3]
                corpus_name = src_filename[: src_filename.rfind(".")]
                if "EXCLUDE" in src_path or corpus_name in args.exclude_corpora:
                    logging.debug(f"Excluding {src_path}")
                    continue
                tgt_filename = f"{corpus_name}.{tgt}"
                if src_path.endswith(".gz"):
                    tgt_filename += ".gz"
                tgt_path = Path(direction_directory) / tgt_filename
                if not os.path.isfile(tgt_path):
                    logging.warning(
                        f"Skipping {src_path}: the corresponding {tgt} file is missing"
                    )
                    continue

                # Special handling code for fbseed. We only want to save the latest
                # version so if we've already seen an fbseed corpus for this direction,
                # check if this one is newer and if so replace it. Otherwise, we should
                # not have duplicate corpora.
                if "fbseed" in src_path:
                    if seen_fbseed is not None and seen_fbseed > corpus_name:
                        continue
                    seen_fbseed = corpus_name
                    corpus_name = "fbseed"
                else:
                    assert (
                        corpus_name not in datasets[direction]
                    ), f"duplicated direction {direction} for corpus {corpus_name}"
                datasets[direction][corpus_name] = Dataset(src=src_path, tgt=tgt_path)
    return dict(datasets)


def main(args):
    data_path = Path(args.components_conf_path) / "data"
    with open(data_path / "train_primary.yaml", "wt") as fout:
        fout.write(
            OmegaConf.to_yaml(
                get_primary_datasets(args.primary_train_paths), sort_keys=True,
            )
        )
    with open(data_path / "train_mined.yaml", "wt") as fout:
        fout.write(
            OmegaConf.to_yaml(get_mined_datasets(args.mini_mine_root), sort_keys=True,)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mini-mine-root", default="/large_experiments/nllb/mmt/mining/data",
    )
    parser.add_argument(
        "--primary-train-paths",
        nargs="*",
        default=["/large_experiments/nllb/mmt/data/bitexts/mtdata/corpora"],
    )
    parser.add_argument(
        "--components-conf-path",
        help="Root directory where configs are to be stored. "
        "Typically $FAIRSEQ_PY/examples/nllb/modeling/components_conf .",
    )
    parser.add_argument(
        "--exclude-corpora", nargs="*", default=["jw300.tok"],
    )
    args = parser.parse_args()
    main(args)
