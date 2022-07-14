# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import gzip
import hydra
import os
import subprocess
import logging
import json
from typing import List, Tuple
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pycountry import languages
from hydra.utils import get_original_cwd


def read_file(infile: str, cfg: DictConfig) -> List[str]:
    """
    read in text or gzipped file and sample if specified
    """
    text = []
    opener = gzip.open if infile.endswith(".gz") else open
    with opener(infile, "rt", errors="surrogateescape") as fin:
        for line_index, line in enumerate(fin):
            if cfg.max_tokens and len(line.split()) > cfg.max_tokens:
                continue
            elif cfg.sample_size and line_index == cfg.sample_size:
                break
            text.append(line.strip())
    return text


def run_moses_preprocessing(
    input_text: List[str], moses_config: DictConfig, outfile: str
) -> None:
    """
    run suite of moses preprocessing scripts listed in the moses config
    and redirect output to the appropriate outfile.
    note: moses punctuation normalisation is handled by the binarization
    """
    directory = moses_config.directory
    commands = " | ".join(
        f"perl {directory}/{script}" for script in moses_config.scripts
    )
    logging.info(f"running tokenisation command: {commands} > {outfile}")
    subprocess.run(
        [f"{commands} > {outfile}"], text=True, input="\n".join(input_text), shell=True
    )


def convert_to_iso3(lang: str) -> str:
    """
    convert language code to ISO 639-3 format
    """
    return lang if len(lang) == 3 else languages.get(alpha_2=lang).alpha_3


def get_lang_pairs(cfg: DictConfig, use_iso3: bool = True) -> List[Tuple[str, str]]:
    """
    return a list of src-tgt pairs
    """
    pairs = []
    for src in cfg.src_langs:
        for tgt in cfg.tgt_langs:
            if src == tgt:
                continue
            src = src if not use_iso3 else convert_to_iso3(src)
            tgt = tgt if not use_iso3 else convert_to_iso3(tgt)
            pairs.append([src, tgt])
    return pairs


def process_data(
    cfg: DictConfig,
    infile: str,
    outfile: str,
):
    """
    read in text and then pass through moses tokenization
    """
    if not Path(outfile).exists() and Path(infile).exists():
        text = read_file(infile, cfg.sampling_config)
        run_moses_preprocessing(text, cfg.moses_config, outfile)


def create_yaml(
    config_directory: str,
    corpus: str,
    src: str,
    tgt: str,
    source_path: str,
    target_path: str,
) -> str:
    """
    create yaml file for source and target paths in corpus
    language directions are standardised in ISO 639-3 format
    """
    yaml = f"""values:
    {corpus}:
        s3_paths:
            source: N/A
            target: N/A
        local_paths:
            source: {source_path}
            target: {target_path}
            is_gzip: false
    """
    src = convert_to_iso3(src)
    tgt = convert_to_iso3(tgt)
    direction = "-".join(sorted([src, tgt]))
    for split in ["train_corpora", "valid_corpora", "test_corpora"]:
        output_directory = f"{config_directory}/{split}/{direction}"
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        with open(f"{output_directory}/{corpus}.yaml", "w") as outfile:
            outfile.write(yaml)
    return f"{src}-{tgt}/{corpus}"


def fetch_corpora_data(
    src: str, tgt: str, data_cfg: DictConfig, cfg: DictConfig
) -> List[str]:
    yaml_paths = []
    src, tgt = sorted([src, tgt])
    dir = "-".join([src, tgt])
    out_dir = Path(f"{data_cfg.out_dir}", "raw", "{dir}")
    for corpus in data_cfg.corpora:
        src_in = f"{data_cfg.in_dir}/{dir}/{corpus}.{src}"
        tgt_in = f"{data_cfg.in_dir}/{dir}/{corpus}.{tgt}"
        src_out = f"{out_dir}/{corpus}.{src}"
        tgt_out = f"{out_dir}/{corpus}.{tgt}"
        if Path(src_in).exists() and Path(tgt_in).exists():
            if not out_dir.exists():
                out_dir.mkdir(parents=True)
            process_data(cfg, src_in, src_out)
            process_data(cfg, tgt_in, tgt_out)
            yaml_paths.append(
                create_yaml(cfg.yaml_directory, corpus, src, tgt, src_out, tgt_out)
            )
    return yaml_paths


def extract_baseline_data(
    cfg: DictConfig,
) -> List[str]:
    yaml_paths = []
    data_cfg = cfg.baseline_config
    lang_pairs = get_lang_pairs(cfg)
    for src, tgt in lang_pairs:
        yaml_paths.extend(fetch_corpora_data(src, tgt, data_cfg, cfg))
    return yaml_paths


def create_data_config(cfg: DictConfig, yaml_paths: List[str]) -> None:
    """
    generate and serialize data config based on stored yaml files
    """
    conf = OmegaConf.create(
        {
            "source_vocab_config": cfg.source_vocab_config,
            "target_vocab_config": cfg.target_vocab_config,
            "binarization_config": cfg.binarization_config,
            "executor_config": cfg.executor_config,
        }
    )
    conf["hydra"] = {"searchpath": [f"file://{cfg.yaml_directory}"]}
    splits = ["train_corpora", "valid_corpora", "test_corpora"]
    conf["defaults"] = []
    for split in splits:
        split_dict = {split: []}
        for yaml_path in yaml_paths:
            split_dict[split].append(yaml_path)
        conf["defaults"].append(split_dict)
    orig_dir = get_original_cwd()  # don't store in hydra working directory
    OmegaConf.save(config=conf, f=f"{orig_dir}/{cfg.generated_data_config}")


def run_binarization(cfg: DictConfig):
    """
    make call to binarization service using command-line
    """
    cmd = [
        "python",
        f"{cfg.scripts.prepare_data}",
        "--data-config",
        os.path.relpath(
            get_original_cwd(),
            os.path.dirname(cfg.scripts.prepare_data),
        )
        + "/"
        + cfg.generated_data_config,
        "--output-dir",
        f"{cfg.baseline_config.out_dir}/bin",
    ]
    logging.info(f"calling binarization with command: {cmd}")
    subprocess.run(cmd, cwd="{cfg.dirs.fairseq_gshard}")


def create_json_task(src: str, tgt: str) -> str:
    """
    fill in json template for translation task
    """
    task = {
        "type": "translation",
        "id": 0,
        "sample": 1.0,
        "src": f"{src}",
        "tgt": f"{tgt}",
    }
    return task


def create_json(cfg: DictConfig) -> None:
    """
    populate json config containing src/tgt data for each task
    """
    lang_pairs = get_lang_pairs(cfg)
    train_tasks = []
    bin_dir = f"{cfg.baseline_config.out_dir}/bin/data_bin/shard000"
    assert Path(bin_dir).exists()
    for src, tgt in lang_pairs:
        dir = "-".join(sorted([src, tgt]))
        train_tasks.append(
            create_json_task(
                f"{bin_dir}/{dir}/train.{dir}.{src}",
                f"{bin_dir}/{dir}/train.{dir}.{tgt}",
            )
        )
    train_tasks.append(
        create_json_task(
            cfg.seed_data.mono_data,
            cfg.seed_data.mono_data,
        )
    )
    train_tasks.append(
        create_json_task(
            cfg.seed_data.bi_data_src,
            cfg.seed_data.bi_data_tgt,
        )
    )
    array_dict = {
        "src_vocab": cfg.spm_vocab,
        "tgt_vocab": cfg.spm_vocab,
        "train": train_tasks,
    }
    with open(f"{get_original_cwd()}/{cfg.json_training_config}", "w") as outf:
        json.dump(array_dict, outf, indent=4)


def run_train_command(cfg: DictConfig) -> None:
    """
    make call to fairseq-train
    """
    student_teacher_config = cfg.seed_data.student_teacher_pairs
    pairs = get_lang_pairs(cfg)
    for src, tgt in pairs:
        sorted_pair = "-".join(sorted([src, tgt]))
        student_teacher_config += f",distil:{sorted_pair}"
    params = []
    for key, value in cfg.train_config.items():
        if key == "params_to_store_true":
            for param in cfg.train_config[key]:
                params.append(f"--{param}")
        else:
            params.append(f"--{key} {value}")
    cmd = [
        "python",
        f"{cfg.scripts.train}",
        f"{get_original_cwd()}/{cfg.json_training_config}",
        "--student-teacher-config",
        f'"{student_teacher_config}"',
    ] + params
    logging.info(f"running teacher-student train with command: {cmd}")
    subprocess.run(cmd)


def extract_encoder(cfg: DictConfig, checkpoint: str) -> str:
    """
    extract encoder from saved checkpoint object
    """
    output = checkpoint.replace(".pt", ".enc.pt")
    if not Path(output).exists():
        logging.info(f"extracting encoder from {checkpoint}")
        cmd = " ".join(
            [
                f"python {cfg.scripts.extract_encoder}",
                f"--config {get_original_cwd()}/{cfg.json_training_config}",
                f"--input {checkpoint}",
                f"--output {output}",
            ]
        )
        logging.info(f"running extraction command: {cmd}")
        subprocess.run(cmd, shell=True)
    return output


def evaluate_encoder(
    cfg: DictConfig,
    enc: str,
    corpus: str = "flores101",
    margin: str = "distance",
    out_dir: str = None,
) -> None:
    """
    evaluate extracted laser encoder on corpus using xsim
    """
    if not out_dir:
        # default to saving in same directory as checkpoint
        out_dir = os.path.dirname(enc)
    logging.info(f"XSIM {margin} evaluation on {corpus} into {out_dir}")
    langs = ",".join(cfg.src_langs)
    for part in ["dev", "test"]:
        outf = f"{out_dir}/{os.path.basename(enc)}.{corpus}.{margin}.{part}"
        if Path(outf).exists():
            continue
        cmd = " ".join(
            [
                f"python -u {cfg.scripts.xsim_eval}",
                f"--corpus {corpus} --corpus-part {part}",
                f"--filter-langs {langs} --margin {margin}",
                f"--spm-model {cfg.source_vocab_config.pretrained.model_file}",
                f"--encoder {enc} > {outf}",
            ]
        )
        logging.info(f"- evaluating {enc} on {corpus}/{part}")
        subprocess.run(cmd, shell=True)


def run_preprocessing(cfg: DictConfig) -> None:
    yaml_paths = extract_baseline_data(cfg)
    create_data_config(cfg, yaml_paths)
    run_binarization(cfg)
    create_json(cfg)


def run_evaluation(cfg: DictConfig) -> None:
    assert cfg.checkpoint
    encoder = extract_encoder(cfg, cfg.checkpoint)
    evaluate_encoder(cfg, encoder)


@hydra.main(config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    """
    teacher-student training.
    """
    if cfg.mode == "preprocess":
        run_preprocessing(cfg)
    elif cfg.mode == "train":
        run_train_command(cfg)
    elif cfg.mode == "eval":
        run_evaluation(cfg)
    else:
        print("Usage: teacher_student.py --mode=[preprocess|train|eval]")


if __name__ == "__main__":
    main()
