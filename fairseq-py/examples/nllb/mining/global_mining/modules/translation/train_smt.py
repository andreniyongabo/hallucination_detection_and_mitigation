import dataclasses
import typing as tp
import subprocess
import hydra
import omegaconf
import logging
from pathlib import Path

from examples.nllb.mining import nllb_lib
from examples.nllb.mining.nllb_lib import utils

log = logging.getLogger(__name__)


@dataclasses.dataclass()
class TrainSmtConfig:
    src: str
    tgt: str = "eng"
    src_moses: str = ""
    tgt_moses: str = ""

    # TODO convert to Path
    template_file: str = ""

    output_dir: str = "/large_experiments/nllb/mmt/smt/"
    train_data: str = "/large_experiments/nllb/mmt/data/bitexts/raw.v2/${.src}-${.tgt}"
    dev_data: str = "/large_experiments/nllb/mmt/flores101/dev/{lang}.dev"
    test_data: str = "/large_experiments/nllb/mmt/flores101/devtest/{lang}.devtest"
    monolingual_corpus: str = "${.train_data}/${.tgt}"

    moses_dir: str = "/private/home/guw/github/mosesdecoder"


def instantiate_moses_config(train_smt_config, template_file: Path) -> str:
    try:
        kwargs = dataclasses.asdict(train_smt_config)
    except:
        kwargs = omegaconf.OmegaConf.to_container(train_smt_config)

    # We only modify the first occurence.
    # the config format already has variables,
    # but they just need to be define inside the file
    config = template_file.read_text()
    for k, v in kwargs.items():
        config = config.replace("{" + k.upper() + "}", str(v), 1)

    return config


class TrainSmtModule(nllb_lib.NLLBModule):
    def __init__(self, config: TrainSmtConfig):
        if not config.src_moses:
            config.src_moses = moses_lang(config.src)
        if not config.tgt_moses:
            config.tgt_moses = moses_lang(config.tgt)

        # config = omegaconf.OmegaConf._promote(TrainSmtConfig, config)
        super().__init__(config)
        omegaconf.OmegaConf.resolve(self.config)
        self.output_dir = Path(self.config.output_dir) / f"{config.src}-{config.tgt}"
        self.output_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / "data"
        self.data_dir.mkdir(exist_ok=True, parents=True)

    def run(self, iteration_value: None = None, iteration_index: int = 0) -> Path:
        template_file_str = self.config.template_file
        if template_file_str:
            template_file = Path(template_file_str)
        else:
            template_file = Path(__file__).resolve().parent / "moses-config.lowercase"
        config_file = self.output_dir / Path(template_file).name
        config_file.write_text(instantiate_moses_config(self.config, template_file))
        moses = Path(self.config.moses_dir).resolve()
        assert moses.is_dir()

        self.prepare_data()
        subprocess.run(
            [moses / "scripts/ems/experiment.perl", "-config", config_file, "-exec"],
            check=True,
        )

        model, run_id = self.find_model()
        bleu_file = self.output_dir / "evaluation" / f"opus.multi-bleu.{run_id}"
        bleu = float(bleu_file.read_text().split("BLEU = ")[1].split(",")[0])

        log.info(f"Trained model {model} for a bleu score of {bleu}")
        return model

    def prepare_data(self):
        # TODO: make the test set more configurable
        src, tgt = self.config.src, self.config.tgt
        test_data = Path(self.config.test_data).resolve()

        for lang in (src, tgt):
            utils.symlink(
                self.data_dir / f"dev.{lang}",
                Path(self.config.dev_data.format(lang=lang)),
            )
            utils.symlink(
                self.data_dir / f"devtest.{lang}",
                Path(self.config.test_data.format(lang=lang)),
            )

        lang_pair = "-".join(sorted([src, tgt]))
        src_files, tgt_files = find_bitext_files(Path(self.config.train_data), src, tgt)

        # Dezip the training files
        # TODO: if the file is already text, replace by symlink
        # TODO: is this required by experiment.perl ?
        if not (self.data_dir / f"train.{src}").exists():
            with open(self.data_dir / f"train.{src}", "w") as o:
                for line in utils.read_files(src_files):
                    print(line, end="", file=o)

        if not (self.data_dir / f"train.{tgt}").exists():
            with open(self.data_dir / f"train.{tgt}", "w") as o:
                for line in utils.read_files(tgt_files):
                    print(line, end="", file=o)

    def find_model(self) -> tp.Tuple[Path, int]:
        model_dir = self.output_dir / "model"
        # Moses will create one new .ini file per config file
        candidates = sorted(model_dir.glob("moses.ini.*"))
        assert candidates, f"No moses model found in {model_dir} !"
        model = candidates[-1]
        run_id = int(model.suffix.strip("."))
        return model, run_id


def find_bitext_files(dir: Path, src: str, tgt: str):
    src_files = {f.name.split(".")[0]: f for f in dir.glob(f"*.{src}.gz")}
    tgt_files = {f.name.split(".")[0]: f for f in dir.glob(f"*.{tgt}.gz")}
    valid = [n for n in src_files if n in tgt_files]
    return ([src_files[n] for n in valid], [tgt_files[n] for n in valid])


LANG_MAPPING = {
    "bel": "ru",
    "pus": "ps",
    "ga": "gle",
    "kk": "kaz",
    "ky": "kir",
    "mn": "mon",
}


def moses_lang(lang: str) -> str:
    # TODO: use the lang mapping from Onur
    return LANG_MAPPING.get(lang, lang[:2])


# TODO make this module compatible with launch_module.py and remove this
def main() -> None:
    config = TrainSmtConfig(
        src="bel",
        tgt="eng",
        monolingual_corpus="/private/home/guw/nllb/mmt/smt/data/bel-eng/train.eng",
    )
    module = TrainSmtModule(config)
    cache = nllb_lib.FileCache("/large_experiments/nllb/evaluation/evaluation_cache")
    result = module()
    print(result)
    cache.save_cache(module, result)


if __name__ == "__main__":
    main()
