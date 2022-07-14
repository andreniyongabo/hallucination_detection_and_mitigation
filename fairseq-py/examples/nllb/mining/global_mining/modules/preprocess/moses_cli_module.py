# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import subprocess
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from omegaconf import MISSING

from examples.nllb.mining.global_mining.mining_utils import tokenization_type
from examples.nllb.mining.nllb_lib import nllb_module, utils

logger = logging.getLogger("moses_preprocess")


@dataclass
class MosesPreprocessConfig:
    # core conf
    lang: str = MISSING
    shards: tp.List[str] = MISSING
    output_dir: str = MISSING
    lowercase: bool = True
    normalize_punctuation: bool = True
    remove_non_printing_chars: bool = False
    deescape_special_chars: bool = False
    token_lang_file: str = (
        "/data/home/schwenk/projects/mlenc/tasks/laser3/map_token_lang.22h1.sed"
    )


class MosesPreprocessModule(nllb_module.NLLBModule):
    """
    Module to run the moses processing perl scripts on a set of text files
    """

    def __init__(self, config: MosesPreprocessConfig = MosesPreprocessConfig()):
        super().__init__(config, MosesPreprocessConfig)

        N = len(self.config.shards)
        self.punc_lang = tokenization_type(
            self.config.lang, self.config.token_lang_file
        )
        logger.info(f"Preprocess {self.config.lang} ({self.punc_lang}), {N} files")
        self.output_dir = Path(self.config.output_dir).resolve()
        self.output_dir.mkdir(exist_ok=True)

    def array(self):
        return self.config.shards

    def requirements(self):
        return nllb_module.DistributedRequirements(
            nodes=1,
            tasks_per_node=1,
            gpus_per_node=0,
            cpus_per_task=4,
            # TODO tune
            timeout_min=60 * 24,
        )

    def run(
        self,
        iteration_value: tp.Optional[tp.Any] = None,
        iteration_index: int = 0,
    ) -> Path:
        input_file = Path(iteration_value)  # type: ignore
        assert input_file.exists()
        # TODO: find a way to allow the caller to specify output
        output_file = self.output_dir / input_file.with_suffix(f".moses").name

        cmds = [utils.open_file_cmd(input_file)]
        moses_dir = resolve_moses_dir()

        assert Path(moses_dir).exists(), f"moses_dir not found: {moses_dir}"
        if self.config.remove_non_printing_chars:
            cmds.append(f"perl {moses_dir}/remove-non-printing-char.perl")
        if self.config.normalize_punctuation:
            cmds.append(
                f"perl {moses_dir}/normalize-punctuation.perl -l {self.punc_lang}"
            )
        if self.config.deescape_special_chars:
            cmds.append(f"perl {moses_dir}/deescape-special-chars.perl")
        if self.config.lowercase:
            cmds.append(f"perl {moses_dir}/lowercase.perl")

        command = utils.bash_pipefail(*cmds)
        logger.info(f"moses command: ${command}")
        try:
            subprocess.run(
                f"{command} > {output_file}",
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            if output_file.is_file():
                output_file.unlink()
            logger.error(
                f"ERROR during encoding of {input_file}; removed dirty output.",
                exc_info=e,
            )
            raise e

        return output_file

    def name(self):
        return (
            f"moses_cli.{self.config.lang}.{len(self.config.shards)}.{self.sha_key()}"
        )

    def version(cls):
        return "0.2"


def resolve_moses_dir() -> Path:
    return Path(__file__).resolve().parents[4] / "modeling/preprocessing/moses"
