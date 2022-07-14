# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
import string
import typing as tp


def sizes(data):
    return [len(sentence) for sentence in data]


POPULATION = string.ascii_letters + string.digits


def make_sentence() -> tp.List[str]:
    length = random.randint(10, 50)
    return random.choices(
        population=POPULATION, k=length, weights=range(1, len(POPULATION) + 1)
    )


def make_data(length=1000, out_file=None) -> tp.List[tp.List[str]]:
    data = (
        [make_sentence() for _ in range(0, length)]
        # add all the symbols at least once
        + [list(string.ascii_letters), list(string.digits)]
    )
    if out_file is not None:
        with open(out_file, "w", encoding="utf-8") as out:
            for s in data:
                print(" ".join(s), file=out)

    return data
