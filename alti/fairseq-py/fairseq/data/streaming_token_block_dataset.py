# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import numpy as np
import torch


class StreamingTokenBlockDataset(torch.utils.data.IterableDataset):
    """View an IterableDataset of tokens as a 1D tensor and chunk into blocks.

    This dataset can only be iterated over once.

    Args:
        dataset (~torch.utils.data.IterableDataset): dataset to chunk
        block_size (int): maximum block size
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
        drop_last (bool, optional): drop the last item (default: False)
        padding_idx (int, optional): index to use for padding symbols
            (required if *drop_last* is ``False``)
        shuffle_buffer_size (int, optional): buffer this many items and shuffle
            using the provided *seed*; default value is 1, so no shuffling is
            performed. This can be adjusted dynamically after initialization,
            but only before iteration has begun.
        seed (int, optional): seed for shuffling
    """

    def __init__(
        self,
        dataset: torch.utils.data.IterableDataset,
        block_size: int,
        break_mode: str = "none",
        drop_last: Optional[bool] = False,
        padding_idx: Optional[int] = None,
        shuffle_buffer_size: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.block_size = block_size
        self.break_mode = break_mode
        self.drop_last = drop_last
        self.padding_idx = padding_idx
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.block_iterator = (
            yield_token_blocks if break_mode == "none" else yield_doc_blocks
        )

        if not drop_last and padding_idx is None:
            raise ValueError("padding_idx is required when drop_last is False")

        assert shuffle_buffer_size >= 1
        if shuffle_buffer_size > 1 and seed is None:
            raise ValueError("seed is required when shuffle_buffer_size > 1")

        # if break_mode != "none": raise NotImplementedError

        self._started_iteration = False

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

    def set_shuffle_buffer_size(self, new_shuffle_buffer_size):
        assert not self._started_iteration
        self.shuffle_buffer_size = new_shuffle_buffer_size

    def __iter__(self):
        assert not self._started_iteration
        self.started_iteration = True

        block_itr = self.block_iterator(
            self.dataset,
            self.block_size,
            self.drop_last,
            self.padding_idx,
        )

        if self.seed is not None:
            # add a random offset (2273) to the given seed to decouple this RNG
            # from any other RNG instances elsewhere
            rng = np.random.default_rng(2273 + self.seed)
        else:
            rng = None

        buffer = []

        def get_next_item_and_replace_in_buffer(replacement_item):
            # return a random item from the buffer and replace with a new item
            nonlocal rng
            idx = rng.integers(len(buffer)) if rng is not None else 0
            item = buffer[idx]
            if replacement_item is not None:
                buffer[idx] = replacement_item
            else:
                buffer.pop(idx)
            return item

        for block in block_itr:
            if len(buffer) < self.shuffle_buffer_size:
                # initially fill the buffer to the requested size
                buffer.append(block)
            else:
                # return random block from the buffer and replace with new block
                yield get_next_item_and_replace_in_buffer(block)

        # clear buffer of any remaining items
        while buffer:
            yield get_next_item_and_replace_in_buffer(None)


def yield_doc_blocks(iterable, block_size, drop_last, padding_idx):
    """Mimics sample-break-mode complete"""
    cur_block = []
    cur_block_ids = []
    cur_block_remain = block_size
    for idx, item in enumerate(iterable):
        if item.numel() > block_size:
            # truncate right side
            item = item[:block_size]

        if item.numel() > cur_block_remain:
            padding = cur_block[-1].new_full((cur_block_remain,), padding_idx)
            cur_block.append(padding)
            block = torch.cat(cur_block)
            yield {
                "ids": torch.LongTensor(cur_block_ids),
                "block": block,
            }

            cur_block = []
            cur_block_ids = []
            cur_block_remain = block_size

        cur_block.append(item)
        cur_block_ids.append(idx)
        cur_block_remain -= item.numel()
        assert cur_block_remain >= 0

    if not drop_last and len(cur_block) > 0:
        if cur_block_remain > 0:
            padding = cur_block[-1].new_full((cur_block_remain,), padding_idx)
            cur_block.append(padding)
        block = torch.cat(cur_block)
        assert block.numel() == block_size
        yield {
            "ids": torch.LongTensor(cur_block_ids),
            "block": block,
        }


def yield_token_blocks(iterable, block_size, drop_last, padding_idx):
    """Sample break mode = None. (Pre-Training default)."""
    cur_block = []
    cur_block_ids = []
    cur_block_remain = block_size
    for idx, item in enumerate(iterable):
        cur_block_ids.append(idx)
        while item.numel() > 0:
            num_to_take = min(item.numel(), cur_block_remain)

            cur_block.append(item[:num_to_take])
            item = item[num_to_take:]  # remainder

            cur_block_remain -= num_to_take
            assert cur_block_remain >= 0

            if cur_block_remain == 0:
                block = torch.cat(cur_block)
                assert block.numel() == block_size
                yield {
                    "ids": torch.LongTensor(cur_block_ids),
                    "block": block[:block_size],
                }

                cur_block = []
                cur_block_ids = []
                cur_block_remain = block_size

    if not drop_last and len(cur_block) > 0:
        if cur_block_remain > 0:
            padding = cur_block[-1].new_full((cur_block_remain,), padding_idx)
            cur_block.append(padding)
        block = torch.cat(cur_block)
        assert block.numel() == block_size
        yield {
            "ids": torch.LongTensor(cur_block_ids),
            "block": block,
        }