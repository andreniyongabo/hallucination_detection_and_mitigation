# GPT-2 BPE model trained on RoBERTa
import os

IS_AZURE = os.path.exists("/shared/home")
IS_AWS = os.path.exists("/fsx")
PATH_TO_ROBERTA_DICT = "/private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/dict.txt"
if IS_AZURE:
    PATH_TO_ROBERTA_DICT = (
        "/data/xlmg/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/dict.txt"
    )
if IS_AWS:
    PATH_TO_ROBERTA_DICT = "/fsx/punitkoura/data/xlmg/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin/dict.txt"


def moe_bpe_config(path, bpe="gpt2", **extra_overrides) -> dict:
    overrides = {"bpe": bpe}
    overrides.update(extra_overrides)
    return {
        "model_path": path,
        "dict_path": PATH_TO_ROBERTA_DICT,
        "extra_args": ["--is-moe"],
        "model_overrides": overrides,
    }


def dense_bpe_config(
    path, bpe="gpt2", fsdp=False, use_sharded_state=False, **extra_overrides
) -> dict:
    overrides = {"bpe": bpe, "use_fused_softmax": False}

    overrides.update(extra_overrides)
    extra_args = []
    if use_sharded_state:
        assert fsdp
        extra_args.extend(["--use-sharded-state"])

    return {
        "model_path": path,
        "dict_path": PATH_TO_ROBERTA_DICT,
        "extra_args": extra_args,
        "model_overrides": overrides,
    }


def gptz_sharded_config(path) -> dict:
    return {
        "model_path": path,
        "extra_args": [
            "--use-sharded-state",
            "--memory-efficient-fp16",
        ],
        "model_overrides": GPTZ_OVERRIDES_AZURE if IS_AZURE else GPTZ_OVERRIDES,
    }


GPTZ_OVERRIDES = {
    "bpe": "hf_byte_bpe",
    "bpe_merges": "/large_experiments/xlmg/data/gptz/tokenizers/gpt2-merges.txt",
    "merges_filename": "/large_experiments/xlmg/data/gptz/tokenizers/gpt2-merges.txt",
    "bpe_vocab": "/large_experiments/xlmg/data/gptz/tokenizers/gpt2-vocab.json",
    "vocab_filename": "/large_experiments/xlmg/data/gptz/tokenizers/gpt2-vocab.json",
    "bpe_add_prefix_space": True,
    "specify_arch": True,
}
GPTZ_OVERRIDES_AZURE = {
    "bpe": "hf_byte_bpe",
    "bpe_merges": "/data/xlmg/gptz/tokenizers/gpt2-merges.txt",
    "merges_filename": "/data/xlmg/gptz/tokenizers/gpt2-merges.txt",
    "bpe_vocab": "/data/xlmg/gptz/tokenizers/gpt2-vocab.json",
    "vocab_filename": "/data/xlmg/gptz/tokenizers/gpt2-vocab.json",
    "bpe_add_prefix_space": True,
    "specify_arch": True,
}
