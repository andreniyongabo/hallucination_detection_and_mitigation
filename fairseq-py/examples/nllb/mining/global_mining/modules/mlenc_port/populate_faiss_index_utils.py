import dataclasses
import math
from pathlib import Path

import faiss
import numpy as np

from examples.nllb.mining.global_mining.embedding_utils import Embedding
from examples.nllb.mining.global_mining.modules.indexing.train_index import index_to_gpu


# CheckpointSummary stores a summary of key information required for the checkpoints
@dataclasses.dataclass
class CheckpointSummary:
    """CheckpointSummary is an object that stores a 'snapshot'/saved checkpoint of the progress so far in the population process

    Note: The "original index" refers to the original index provided by the config
    Attributes:
        partial_idx (faiss.Index) - Stores the original index with the partial embedding shard that's been populated onto the original index so far. Stored as faiss.index.
        partial_idx_file (Path) - Stores the original index with the partial embedding shard that's been populated onto the original index so far. Stored as faiss.index written to a file.
        idx_size_before_populating_embedding (int) - This is the size of the original index. That is, it's the size of the original index, before we have even started populating the embedding onto it. Specifically, it's calculated as faiss.read_index(str(self.config.index)).ntotal.
        is_partial_file_valid (bool) - boolean to indicate if partial_idx_file is valid or not. The partial_idx_file is considered valid if the index was written to it without interruption - ie the job did not interrupt while doing faiss.write
        is_partial_idx_valid (bool) - boolean to indicate if partial_idx is valid or not. The partial_idx is considered valid if the chunk is added to the idx without interruption - ie the job did not interrupt while doing partial_idx.add(chunk)

    What's the diff between partial_idx and partial_idx_file?
        They both represent the same thing - the partial index (ie. the original index with the partial embedding shard that's been populated onto the original index so far)
        At the end of each iteration in the for loop (within add_embedding_to_index function), these two values should be exactly the same.
        The only difference is the format of storing the index: as a faiss.Index vs as written to a file.
    """

    partial_idx: faiss.Index
    partial_idx_file: Path
    idx_size_before_populating_embedding: int
    is_partial_file_valid: bool
    is_partial_idx_valid: bool


# Reads embeddings from the given file and add them to the index
def add_embedding_to_index(
    checkpoint_summary: CheckpointSummary,
    embeddings_file: Path,
    dim: int,
    dtype=np.float32,
    gpu: bool = True,
) -> faiss.Index:
    assert checkpoint_summary is not None, f"checkpoint_summary must not be None"
    embedding = Embedding(embeddings_file, dim, dtype=dtype)
    assert isinstance(checkpoint_summary.partial_idx, faiss.Index)
    partial_idx = checkpoint_summary.partial_idx
    n_total_start = partial_idx.ntotal
    if gpu:
        partial_idx = index_to_gpu(partial_idx)

    with embedding.open_for_read(mode="memory") as data:
        chunk_size = 2**14  # Speed gains are marginal beyond 10k embeddings

        # Below, we calculate how much of the embedding is already populated onto the index
        # checkpointed_embedding_starting_row is the starting row to continue on from (every row before this we've already completed/checkpointed)
        checkpointed_embedding_starting_row = (
            partial_idx.ntotal - checkpoint_summary.idx_size_before_populating_embedding
        )

        # length_of_remaining_embedding is the length of the remaining embedding that still needs to be populated onto index
        length_of_remaining_embedding = (
            len(embedding) - checkpointed_embedding_starting_row
        )

        # embedding_iterator tracks total rows added to index from embedding, after the checkpointed_embedding_starting_row
        embedding_iterator = 0

        if length_of_remaining_embedding > 0:
            num_chunks = math.ceil(length_of_remaining_embedding / chunk_size)

            for chunk in np.array_split(
                data[checkpointed_embedding_starting_row:], num_chunks
            ):

                assert (
                    checkpoint_summary.is_partial_idx_valid
                    and checkpoint_summary.is_partial_file_valid
                ), f"Both partial idx and partial idx file must be valid before proceeding to work on current checkpoint"

                embedding_iterator += chunk.shape[0]

                # fp16 currently not supported by FAISS
                if dtype == np.float16:
                    chunk = chunk.astype(np.float32)
                faiss.normalize_L2(chunk)

                # Add chunk to partial index (and while this happens, keep is_partial_idx_valid as False)
                checkpoint_summary.is_partial_idx_valid = False
                partial_idx.add(chunk)
                checkpoint_summary.is_partial_idx_valid = True

                # Write partial index to the partial index file (and while this happens, keep is_partial_file_valid as False)
                # checkpoint_summary.is_partial_file_valid = False
                # faiss.write_index(
                #     partial_idx,
                #     str(partial_idx_file),
                # )
                # checkpoint_summary.is_partial_file_valid = True

    # Write partial index to the partial index file (and while this happens, keep is_partial_file_valid as False)
    checkpoint_summary.is_partial_file_valid = False
    if gpu:
        partial_idx = faiss.index_gpu_to_cpu(partial_idx)
    faiss.write_index(
        partial_idx,
        str(checkpoint_summary.partial_idx_file),
    )
    checkpoint_summary.is_partial_file_valid = True

    assert (
        partial_idx.ntotal == n_total_start + embedding_iterator
    ), f"population with {embeddings_file} didn't succeed"

    return partial_idx
