#!/usr/bin/env python
"""Helper script to compare two argparse.Namespace objects."""

import argparse
from argparse import Namespace
from typing import Dict  # noqa

UNINTERESTING_KEYS = [
    "distributed_training.distributed_init_method",
    "checkpoint.save_dir",
    "common.tensorboard_logdir",
    "distributed_training.distributed_port",
]


def get_cfg(filename, pattern):
    namespace_lines = []
    with open(filename) as fin:
        for line in fin:
            if pattern in line:
                tokens = line.strip().split("|")
                if len(tokens) >= 4:
                    for entry in tokens:
                        entry = entry.strip()
                        if entry.startswith("{"):
                            namespace_lines.append(entry)

    if len(namespace_lines) == 0:
        raise ValueError(
            f"Could not find pattern {pattern} in {filename}. Pass a different --pattern."
        )
    print(f"Found {len(namespace_lines)} namespaces")
    return eval(namespace_lines[-1])


def flatten_cfg(cfg):
    flattened_cfg = {}
    for k, v in cfg.items():
        if isinstance(v, Namespace):
            for inner_k, inner_v in v.__dict__.items():
                flattened_cfg[k + "." + inner_k] = inner_v
        elif isinstance(v, Dict):
            for inner_k, inner_v in v.items():
                flattened_cfg[k + "." + inner_k] = inner_v
        else:
            flattened_cfg[k] = v
    return flattened_cfg


def main(args):
    namespace1 = get_cfg(args.trainlog1, pattern=args.pattern)
    namespace2 = get_cfg(args.trainlog2, pattern=args.pattern)

    flat_cfg1 = flatten_cfg(namespace1)
    flat_cfg2 = flatten_cfg(namespace2)

    k1 = flat_cfg1.keys()
    k2 = flat_cfg2.keys()

    def print_keys(ks, ns1, ns2=None):
        for k in ks:
            if ns2 is None:
                print("{}\t{}".format(k, getattr(ns1, k, None)))
            else:
                print(
                    "{}\t{}\t{}".format(k, getattr(ns1, k, None), getattr(ns2, k, None))
                )

    print("Keys unique to namespace 1:")
    print_keys(k1 - k2, flat_cfg1)

    print("Keys unique to namespace 2:")
    print_keys(k2 - k1, flat_cfg2)

    print("Keys with different values:")
    for k in k1 & k2:
        if k in UNINTERESTING_KEYS:
            continue
        if flat_cfg1[k] != flat_cfg2[k]:
            print(f"Key:{k}, {flat_cfg1[k]}, {flat_cfg2[k]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("trainlog1")
    parser.add_argument("trainlog2")
    parser.add_argument("--pattern", default="quantization_config_path", type=str)
    main(parser.parse_args())
