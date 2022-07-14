# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import asyncio
import glob
import os

# from cc_net import jsonql, text_normalizer
import shlex
import subprocess
import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import hydra
import kenlm  # type: ignore
import pandas as pd  # type: ignore
import sentencepiece  # type: ignore
import wandb
from omegaconf import MISSING, DictConfig, OmegaConf

from examples.nllb.mining.global_mining.modules.preprocess.multiproc_line_processor import (
    MultiprocLineProcessorCallback,
    MultiprocLineProcessorConfig,
    MultiprocLineProcessorModule,
)
from examples.nllb.mining.monolingual.utils import (
    slurm_tmp_maybe,
    sort,
    text_normalizer,
)
from examples.nllb.mining.nllb_lib import utils
from examples.nllb.mining.nllb_lib.nllb_module import DistributedRequirements

LMDescriptor = Union[Dict[str, Path], Union[Path, str]]


def pp(log_score, length):
    return 10.0 ** (-log_score / length)


class Transformer:
    def __call__(self, x):
        if x is None:
            return
        return self.do(x)


class SentencePiece(Transformer):  # (jsonql.Transformer):
    # Sentence Pieces model have to be read back from disk.
    warning_when_pickling = True

    def __init__(
        self,
        model: Path,
        field: str,
        output_field: str = "tokenized",
        normalize: bool = False,
    ):
        super().__init__()
        self.model = model
        self.field = field
        self.output_field = output_field
        self.normalize = normalize
        self.sp: sentencepiece.SentencePieceProcessor = None

    def _prepare(self):
        if self.sp is not None:
            return
        self.sp = sentencepiece.SentencePieceProcessor()
        self.sp.load(str(self.model))
        return self

    def do(self, document: dict) -> dict:
        text = document[self.field]
        if self.normalize:
            text = text_normalizer.normalize(text)
        tokenized = self.sp.encode_as_pieces(text)
        document[self.output_field] = " ".join(tokenized)
        return document


class MultiSentencePiece(Transformer):  # (jsonql.Transformer):
    warning_when_pickling = True

    def __init__(
        self,
        models: Union[Path, Dict[str, Path]],
        field: str,
        output_field: str = "tokenized",
        normalize: bool = False,
    ):
        super().__init__()
        self.field = field
        self.output_field = output_field
        self.normalize = normalize
        self._prefetch: Sequence[str] = []

        if isinstance(models, Path):
            self.models = {
                m.name.split(".")[0]: m for m in models.parent.glob(models.name)
            }
        else:
            self.models = models
            self._prefetch = list(models.keys())
        self.sp: Dict[str, sentencepiece.SentencePieceProcessor] = {}

    def _prepare(self) -> None:
        for lang in self._prefetch:
            assert (
                self.get_sp(lang) is not None
            ), f"No model found for {lang} at {self.models.get(lang)}."

    def get_sp(self, lang) -> Optional[sentencepiece.SentencePieceProcessor]:
        sp = self.sp.get(lang)
        if sp is not None:
            return sp
        if lang not in self.models:
            return None

        start_load = time.time()
        print(f"Loading {self.models[lang]}...")
        sp = sentencepiece.SentencePieceProcessor()
        sp.load(str(self.models[lang]))
        self.sp[lang] = sp
        load_time = time.time() - start_load
        print(f"Loaded {self.models[lang]} (took {load_time / 60:.1f}min)")
        return sp

    def do(self, document: dict) -> Optional[dict]:
        text = document[self.field]
        if self.normalize:
            text = text_normalizer.normalize(text)
        sp = self.get_sp(document.get("language"))
        if sp is None:
            return document
        tokenized = sp.encode_as_pieces(text)
        document[self.output_field] = " ".join(tokenized)
        return document


class DocLM(Transformer):  # (jsonql.Transformer):
    def __init__(
        self,
        models: Union[Path, Dict[str, Path]],
        field: str,
        output_field: str = "perplexity",
        newline: str = "\n",
        normalize: bool = True,
        load_method: int = 2,
    ):
        super().__init__()
        self.field = field
        self.output_field = output_field
        self.newline = newline
        self.normalize = normalize
        self._prefetch: Sequence[str] = []
        self.lm_config = kenlm.Config()
        # This is the default settings
        # POPULATE will mmap the models and populate the pages.
        # Maybe that's not the best way when the models are on a network disk.
        # TODO: try copying models file, try READ or PARALLEL_READ
        self.lm_config.load_method = load_method

        if isinstance(models, Path):
            self.models = {
                m.name.split(".")[0]: m for m in models.parent.glob(models.name)
            }
        else:
            self.models = models
            self._prefetch = list(models.keys())
        self.lm: Dict[str, kenlm.Model] = {}
        self.n_lines = 0

    def _prepare(self) -> None:
        for lang in self._prefetch:
            assert (
                self.get_lm(lang) is not None
            ), f"No model found for {lang} at {self.models.get(lang)}."

    def get_lines(self, document: dict) -> List[str]:
        lang = document.get("language")
        if not lang:
            return []
        if lang not in self.models:
            return []

        content = document.get(self.field)
        if not content:
            return []

        lines = content.split(self.newline)
        self.n_lines += len(lines)
        return lines

    def get_lm(self, lang: Optional[str]) -> Optional[kenlm.Model]:
        if lang is None:
            return None
        lm = self.lm.get(lang)
        if lm is not None:
            return lm
        model = self.models.get(lang)
        if model is None:
            return None
        start_load = time.time()
        print(f"Loading {self.models[lang]}...")
        lm = kenlm.Model(str(model), self.lm_config)
        self.lm[lang] = lm
        load_time = time.time() - start_load
        print(f"Loaded {self.models[lang]} (took {load_time / 60:.1f}min)")

        return lm

    def do(self, document: dict) -> dict:
        lines = self.get_lines(document)
        model = self.get_lm(document.get("language"))
        if not lines or not model:
            return document

        doc_log_score, doc_length = 0, 0
        for line in lines:
            if self.normalize:
                line = text_normalizer.normalize(line)
            log_score = model.score(line)
            length = len(line.split()) + 1
            doc_log_score += log_score
            doc_length += length

        document[self.output_field] = round(pp(doc_log_score, doc_length), 1)
        return document

    def summary(self):
        delay = time.time() - self.start_time
        h = delay / 3600
        s = self.n_lines / delay

        summ = super().summary()
        summ.append(f"Processed {self.n_lines:_} lines in {h:.2}h ({s:.1} lines/s).")
        return summ


class SentencesLM(DocLM):
    """Returns the score of each individual paragraph."""

    def do(self, document: dict) -> Optional[dict]:  # type: ignore
        lines = self.get_lines(document)
        model = self.get_lm(document.get("language"))
        if not lines or not model:
            return None

        sentences = []
        # always one line
        for line in lines:
            if self.normalize:
                line = text_normalizer.normalize(line)
            log_score = model.score(line)
            length = len(line.split()) + 1

            new_element = {}
            new_element[self.output_field] = pp(log_score, length)
            new_element["line"] = line

            return new_element


class PerplexityBucket(Transformer):  # (jsonql.Transformer):
    def __init__(
        self, cutoff_csv: Path, percentile_head: int = 30, percentile_tail: int = 60
    ):
        super().__init__()
        self.cutoff_csv = cutoff_csv
        self.percentile_head = percentile_head
        self.percentile_tail = percentile_tail
        self.cutoffs: Dict[str, Tuple[float, float]] = {}

    def _prepare(self) -> None:
        cutoffs = pd.read_csv(self.cutoff_csv, index_col=0)
        self.cutoffs = {
            l: (cutoffs[l][self.percentile_head], cutoffs[l][self.percentile_tail])
            for l in cutoffs.columns
        }

    def get_bucket(self, doc: dict) -> str:
        perplexity = doc.get("perplexity", -1)
        lang = doc.get("language")
        if lang not in self.cutoffs or perplexity < 0:
            return "all"

        pp_head, pp_tail = self.cutoffs[lang]
        if perplexity < pp_head:
            return "head"
        if perplexity < pp_tail:
            return "middle"
        return "tail"

    def do(self, doc: dict) -> dict:
        doc["bucket"] = self.get_bucket(doc)
        return doc


class DropKeys(Transformer):  # (jsonql.Transformer):
    def __init__(self, *keys):
        super().__init__()
        self.keys = keys

    def do(self, document: dict) -> Optional[dict]:
        if not document:
            return None

        for key in self.keys:
            document.pop(key, None)
        return document


class RemoveSmall(Transformer):  # (jsonql.Transformer):
    def __init__(self, field, min_len):
        super().__init__()
        self.field = field
        self.min_len = min_len
        self.removed = 0

    def do(self, document: dict) -> Optional[dict]:
        if not document:
            return None

        content = document.get(self.field)
        if not content or len(content) < self.min_len:
            self.removed += 1
            return None
        return document

    def summary(self):
        r, n = self.removed, self.processed
        ratio = r / n if n else 0
        return [f"Removed {r} small documents out of {n} ({ratio:.1%})"]


# python run_kenlm.py --models /checkpoint/guw/cc_clean/lm_sp/en.arpa.bin --sentences --file /checkpoint/mortimer/merge/cc200xl.dedup.100-199/partial_dedup.0-10.xz  --output /checkpoint/angelafan/lm_score/cc200xl.dedup.100-199/partial_dedup.0-10.xz

# boilerplate after this to run in multiproc+slurm nllb modules


@dataclass
class KenLMConfig:
    launcher: DictConfig = MISSING

    max_score: float = MISSING
    lm_model: str = MISSING
    spm_model: str = MISSING
    glob: str = MISSING
    output_prefix: str = MISSING
    local_tmp_dir: str = MISSING
    spm_normalize: bool = True
    lm_normalize: bool = False
    buffer_size: int = 10_000
    requirements: DistributedRequirements = DistributedRequirements()


class KenLMLineProcessor(MultiprocLineProcessorCallback):
    def __init__(
        self,
        # set by LineProcessorModule
        outfile_prefix: str,
        input_file: str,
        input_file_idx: int,
        output_dir: str,
        offset_start: tp.Optional[int],
        offset_end: tp.Optional[int],
        merging: bool,
        # set by the config
        config: KenLMConfig,
        _version: str,  # used to bump config version and invalidate cache
    ) -> None:
        super().__init__(
            outfile_prefix=outfile_prefix,
            input_file=input_file,
            input_file_idx=input_file_idx,
            output_dir=output_dir,
            offset_start=offset_start,
            offset_end=offset_end,
            merging=merging,
        )

        self.config = config

        self.pp_field = "perplexity"
        # "/checkpoint/guw/cc_clean/lm_sp/en.sp.model"
        self.spm_model = MultiSentencePiece(
            {"eng": Path(config.spm_model)},
            field="field",
            output_field="tokenized",
            normalize=True,
        )
        self.lm = SentencesLM(
            {"eng": Path(config.lm_model)},
            "tokenized",
            output_field=self.pp_field,
            normalize=False,
        )
        # prefetch models at init...
        if not self.spm_model.get_sp("eng"):
            raise Exception(config.spm_model)
        if not self.lm.get_lm("eng"):
            raise Exception(config.lm_model)

        infile = Path(input_file)

        if not merging:

            tmp_dir = slurm_tmp_maybe(Path(config.local_tmp_dir)) / infile.stem
            tmp_dir.mkdir(parents=True, exist_ok=True)

            out = Path(config.output_prefix).stem

            self.tmp_output_file = tmp_dir / (
                f"{out}.{input_file_idx:03d}.{offset_start}_{offset_end}.tsv"
            )

    def __enter__(self):
        self._outf = self.tmp_output_file.open("w", encoding="utf-8")
        return self

    def __exit__(self, *exc):
        self._outf.close()

    def process_lines(self, lines_with_number: tp.Iterator[tp.Tuple[int, str]]) -> None:
        for _idx, line in lines_with_number:
            metadata = line.split("\t")
            text = metadata[-1]
            doc = {"field": text, "language": "eng", "perplexity": 0}
            doc = self.spm_model(doc)
            pp = self.lm(doc)[self.pp_field]
            if pp < self.config.max_score:
                columns = metadata[:-1] + [f"{pp:.5f}", text]
                print(*columns, sep="\t", file=self._outf)

    def final_result(self) -> str:
        print(f"finished processing to: {self.tmp_output_file}")
        return self.tmp_output_file

    def merge_results(self, splits: tp.List[Path]) -> str:
        outfile = Path(self.config.output_prefix).with_suffix(
            f".{self.input_file_idx:03d}.xz"
        )

        # we assume inputs are already sorted, so filtered output will still be sorted
        subprocess.run(
            utils.bash_pipefail(
                shlex.join(["cat"] + [str(f) for f in splits]),
                " ".join(["xz", ">", shlex.quote(str(outfile))]),
            ),
            shell=True,
            check=True,
        )

        return outfile


async def multiproc_kenlm(config: KenLMConfig):

    launcher = hydra.utils.instantiate(config.launcher)

    # glob
    shards = glob.glob(config.glob)

    out_dir = Path(config.output_prefix).parent

    file_processor = MultiprocLineProcessorModule(
        config=MultiprocLineProcessorConfig(
            line_processor=DictConfig(
                {
                    # this will eventually initialize KenLMLineProcessor above
                    "_target_": f"examples.nllb.mining.monolingual.kenlm_pipeline.KenLMLineProcessor",
                    "config": config,
                    "_version": "0.0",
                }
            ),
            custom_name=f"kenLM-{Path(config.output_prefix).stem}",
            output_dir=str(out_dir),
            outfile_prefix="",  # TODO from config?
            shards=shards,
            buffer_size=config.buffer_size,
            requirements=DistributedRequirements(**config.requirements),
            tmp_dir=config.local_tmp_dir,
        )
    )

    await launcher.schedule(file_processor)


@hydra.main(config_path="conf", config_name="kenlm_pipeline")
def main(config: KenLMConfig) -> None:
    run = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=OmegaConf.to_container(config),
    )
    run.config.update({"cwd": os.getcwd()})
    run.name = f"kenlm.{run.name}"
    asyncio.run(multiproc_kenlm(config))


if __name__ == "__main__":
    main()
