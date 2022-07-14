# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import typing as tp

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from examples.nllb.mining.global_mining.mining_utils import extract_shard_id
from examples.nllb.mining.global_mining.modules.preprocess.encode_to_npy import (
    EncodeToNPY,
)


class HFTextEncoder(EncodeToNPY):
    """
    1. load a pre-trained model located in the HuggingFace Hub
    2. tokenize and encode input using the tokenizer specified in HuggingFace model config
    3. perform model-specific pooling method over sentence tokens (e.g. mean/cls/max)
    4. send embeddings to specified output file
    """

    def __init__(
        self,
        _name: str,
        encoder_model: str,
        outfile_prefix: str,
        input_file: str,
        output_dir: str,
        input_file_idx: int,
        pooling_method: str = "cls",
        normalize: bool = False,
        fp16_storage: bool = False,
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            normalize=normalize,
            fp16_storage=fp16_storage,
        )

        # load model and respective tokenizer from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_model)
        self.model = AutoModel.from_pretrained(encoder_model)
        assert pooling_method in [
            "mean",
            "max",
            "cls",
        ], "Please provide pooling option from: [mean, max, cls]"
        self.pooling_method = pooling_method
        # ensure all nodes are used and batch-norm set for evaluation
        self.model.eval()

    def name_output_file(self) -> str:
        shard_idx = extract_shard_id(self.input_file, default=self.input_file_idx)

        return os.path.abspath(
            os.path.join(
                self.output_dir,
                f"{self.outfile_prefix}.{shard_idx:03d}",
            )
        )

    def encode_to_np(
        self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]
    ) -> np.ndarray:
        # tokenize sentences and return pytorch tensor (pt)
        encoded_sents = self.tokenizer(
            [s for (_, s) in lines_with_number],
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # compute embeddings
        with torch.no_grad():
            model_output = self.model(**encoded_sents)

        token_embeddings = model_output[0]

        # extract sentence embeddings using specified pooling method and convert to numpy
        sentence_embeddings = (
            self.pooler(
                token_embeddings, encoded_sents["attention_mask"], self.pooling_method
            )
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )  # faiss requires float32

        return sentence_embeddings

    def pooler(
        self,
        token_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        pooling_method: str = "cls",
    ) -> torch.Tensor:
        if pooling_method == "max":
            token_embeddings[attention_mask == 0] = float("-inf")
            return torch.max(token_embeddings, 1)[0]

        if pooling_method == "mean":
            token_embeddings[attention_mask == 0] = 0
            return torch.sum(token_embeddings, 1) / torch.count_nonzero(
                token_embeddings, dim=1
            )
        # default to cls
        return token_embeddings[:, 0]

    def __exit__(self, _exc_type, _exc_value, _traceback):
        return None
