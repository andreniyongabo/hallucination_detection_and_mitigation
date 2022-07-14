#!/usr/bin/env python
import glob
import json
import os
import pickle
import subprocess
import tempfile
import time
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import torch
from fairscale.nn import FullyShardedDataParallel as FSDP

from fairseq import distributed_utils, options, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import fsdp_enable_wrap, fsdp_wrap
from fairseq.distributed.utils import get_global_world_size
from fairseq.models import BaseFairseqModel
from fairseq.utils import print_r0

from .model_configs import *


def pickle_dump(obj, pickle_path):
    with open(pickle_path, "wb") as f:
        pickle.dump(obj, f)


def pickle_load(pickle_path):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def get_model_last_valid_info_from_train_log(model_path):
    """Reads the valid log info from the model train file"""
    try:
        model_dir = os.path.dirname(model_path)
        train_log_file = os.path.join(model_dir, "train.log")
        if not os.path.exists(train_log_file):
            return None

        cmd = f"/bin/grep valid_num_updates {train_log_file} | /bin/tail -1"
        log_line = subprocess.check_output(cmd, shell=True).decode("ascii")
        log_split = log_line.split("| INFO | valid |")
        if len(log_split) != 2:
            return None

        log_info = json.loads(log_split[1].strip())
        log_info["time_stamp"] = log_split[0].strip()
        log_info["log_file"] = train_log_file

        return log_info
    except Exception as e:
        return None


def get_lm_config(model_name, **kwargs):
    # see if we are using some custom model_configs
    model_configs = kwargs.get("model_configs", None)  # we can pass model_configs
    if model_configs is None or len(model_configs) == 0:
        # search for model_name in different config groups/classes of models
        model_config_groups = get_model_config_groups()

        model_class = getattr(kwargs, "model_class", None)
        model_configs = model_config_groups.get(model_class, None)

        if model_configs is None:
            model_configs_candidates = []
            for cfg_name, config in model_config_groups.items():
                if model_name in config:
                    model_configs_candidates.append((cfg_name, config))
            if len(model_configs_candidates) == 0:
                raise ValueError(
                    f"--model-name={model_name} is not found in the model configs!"
                )
            elif len(model_configs_candidates) > 1:
                model_classes_names = ",".join([x[0] for x in model_configs])
                raise ValueError(
                    f"--model-name={model_name} is available in more than 1 model classes: {model_classes_names}. Please specify --model-class!"
                )
            model_configs = model_configs_candidates[0][1]
    else:
        assert isinstance(
            model_configs, dict
        ), f"Model configs must be a dict with {{'model_X': [some model config here]}}, not {type(model_configs)} - {model_configs}"

    moe_eval_capacity_token_fraction = kwargs.get(
        "moe_eval_capacity_token_fraction", None
    )
    if model_name in model_configs:
        model_config = model_configs[model_name]
        # Override batch_size, batch_size valid. We will determine them later (likely to 1)
        # If we don't set them None, moe_layer will warn and pad because of expected_bsz logic.
        model_config["model_overrides"]["batch_size"] = None
        model_config["model_overrides"]["batch_size_valid"] = None
        if moe_eval_capacity_token_fraction is not None:
            model_config["model_overrides"][
                "moe_eval_capacity_token_fraction"
            ] = moe_eval_capacity_token_fraction
        distributed_port = int(kwargs.get("distributed_port", 0)) # Some models require only 1 gpu so no need to run on distributed env (as far as I remember distributed-port=0 disables it).

        model_config["extra_args"] = model_config.get("extra_args", [])
        model_config["extra_args"] += [
            "--fp16",
            "--distributed-port",
            str(distributed_port),
        ]

        # Set distributed world size
        model_config["distributed_world_size"] = get_global_world_size()

    elif os.path.exists(model_name):
        model_config = {"model_path": model_name}
    else:
        raise ValueError(f"unknown --model-name={model_name}")

    if not model_config.get("enabled", True):
        print_r0("skipping disabled model: " + model_name)
        return

    # Read and save validation info
    model_path = model_config["model_path"]
    model_train_log_info = get_model_last_valid_info_from_train_log(model_path)
    if model_train_log_info is not None:
        print_r0("model_pretraining_valid_info=" + json.dumps(model_train_log_info))

    eval_lm_input_args = (
        [
            os.path.dirname(model_config["model_path"]),  # data argument
        ]
        + model_config.get("model_parallel_args", [])
        + model_config.get("extra_args", [])
    )
    parser = options.get_eval_lm_parser()
    args = options.parse_args_and_arch(parser, eval_lm_input_args)

    fairseq_cfg = convert_namespace_to_omegaconf(args)

    num_training_updates = kwargs.get("num_training_updates", -1)
    if fairseq_cfg.common_eval.is_moe and num_training_updates > 0:
        model_path = model_config["model_path"]
        model_dir = os.path.dirname(model_path)
        in_pt = None
        for in_pt in glob.glob(f"{model_dir}/*_{num_training_updates}-shared.pt"):
            break
        assert (
            "{}-shared.pt".format(num_training_updates) in in_pt
        ), "Cannot MoE locate checkpoints at update {}".format(num_training_updates)
        model_path = os.path.join(model_dir, in_pt[: -len("-shared.pt")] + ".pt")
        model_config["model_path"] = model_path
        print("* update model path to {}".format(model_path))

    # If --fsdp is passed, also set [--ddp-backend fully_sharded] in extra_args
    if kwargs.get("fsdp", False):
        model_config["extra_args"] += ["--ddp-backend", "fully_sharded"]

    return fairseq_cfg, model_config


def load_lm_and_run_func(func, model_name, **kwargs):
    fairseq_cfg, model_config = get_lm_config(model_name, **kwargs)
    print_r0(
        f"distributed_training.distributed_port={fairseq_cfg.distributed_training.distributed_port}"
    )

    return_value_path = Path(tempfile.mkstemp()[1])

    distributed_utils.call_main(
        main=_load_lm_and_run_func,
        cfg=fairseq_cfg,
        config=model_config,
        return_value_path=return_value_path,
        func=func,
        **kwargs,
    )

    if fairseq_cfg.distributed_training.device_id == 0:
        return_value = pickle_load(return_value_path)
        return_value_path.unlink()
    else:
        return_value = None
    return return_value


@contextmanager
def maybe_enable_fsdp(cfg, is_moe: bool, fsdp: bool, use_sharded_state: bool = False):
    if fsdp:
        with fsdp_enable_wrap(
            cfg.distributed_training, is_moe=is_moe, use_sharded_state=use_sharded_state
        ):
            yield
    else:
        yield
    return


def load_and_get_model(
    fairseq_cfg,
    config,
    skip_prepare_for_inference=False,
    post_build_model_hook=None,
    fsdp=False,
):
    is_moe = getattr(fairseq_cfg.common_eval, "is_moe", False)
    load_sharded = fairseq_cfg.distributed_training.use_sharded_state
    r = distributed_utils.get_global_rank()
    if is_moe:
        suffix = f"-rank-{r}"
    elif load_sharded:
        suffix = f"-shard{r}"
    else:
        suffix = ""

    def default_post_build_model_hook(model, task):
        # fsdp_wrap will be a no-op if not using FSDP
        return fsdp_wrap(model)

    with maybe_enable_fsdp(
        fairseq_cfg, is_moe=is_moe, fsdp=fsdp, use_sharded_state=load_sharded
    ):
        return get_model(
            model_path=config["model_path"],
            dict_path=config.get("dict_path"),
            suffix=suffix,
            is_moe=is_moe,
            skip_prepare_for_inference=skip_prepare_for_inference or fsdp,
            post_build_model_hook=post_build_model_hook
            or default_post_build_model_hook,
            **config.get("model_overrides", {}),
        )


def _load_lm_and_run_func(fairseq_cfg, config, return_value_path, func, **kwargs):
    start_time = time.monotonic()
    utils.import_user_module(
        Namespace(**{k: v for k, v in kwargs.items()})
    )  # HACK to make sure --user-dir loads in slurm runs
    model = load_and_get_model(fairseq_cfg, config, fsdp=kwargs.get("fsdp", False))
    load_time = time.monotonic() - start_time
    print_r0(f"model_loading_time={load_time:.1f} seconds")
    model.half()
    model.cuda()
    to_cuda_time = time.monotonic() - start_time
    print_r0(f"model_loading_time_cuda={to_cuda_time:.1f} seconds")
    model.eval()  # disable dropout
    max_tokens = get_or_infer_max_tokens(model, **kwargs)
    model.cfg.dataset.max_tokens = max_tokens
    original_max_positions = model.max_positions
    if kwargs["max_positions"] == 0:
        model.max_positions = max_tokens
    elif kwargs["max_positions"] is not None:
        model.max_positions = kwargs["max_positions"]
    if original_max_positions != model.max_positions:
        print_r0(
            f"Changing max_positions from {original_max_positions} to {model.max_positions}"
        )

    return_value = func(model=model, **kwargs)
    rank = fairseq_cfg.distributed_training.device_id
    if rank == 0:
        pickle_dump(return_value, return_value_path)


def get_model(model_path, dict_path=None, suffix="", bpe="empty", **kwargs):
    # assert bpe != "empty", "bpe must be set in model_overrides of the model in model configurations"
    # assert model_path.exists() # this fails for moe models since multiple checkpoints are loaded
    # check if model directory exists
    assert os.path.exists(
        os.path.dirname(model_path)
    ), f"Model path dir {os.path.dirname(model_path)} does not exist!"

    model_path = Path(model_path)
    model_name_or_path = str(model_path.parent)
    checkpoint_file = model_path.name
    data_name_or_path = "."
    if dict_path is not None:
        dict_path = Path(dict_path)
        assert dict_path.exists()
        # HACK: The setup_task method will look in the data_dir for dict.txt
        # https://github.com/pytorch/fairseq/blob/dea66cc294a18dd4d9e59aa0af8d51f951e83884/fairseq/tasks/language_modeling.py#L141
        data_name_or_path = str(dict_path.parent)

    model = BaseFairseqModel.from_pretrained(
        model_name_or_path=model_name_or_path,
        checkpoint_file=checkpoint_file,
        data_name_or_path=data_name_or_path,
        suffix=suffix,
        # If the criterion in the checkpoint is set to
        # vocab_parallel_cross_entropy, then the output layer will skip the
        # gather_from_model_parallel_region call, which is necessary.
        # So here we override the criterion to force the output layer to call
        # gather_from_model_parallel_region.
        criterion="cross_entropy",
        moe_disable_padding=False,
        bpe=bpe,
        **kwargs,
    )
    print_r0("Loaded model")

    override_task(model)

    return model


def override_task(model):
    """This methods ovverrides the task for the models in some cases including:
    1) seq2seq_lm for bart_base, bart_large when the task is language_modeling, multilingual_language_modeling
    2) language_modeling_inference_for_streaming_trained_models when the model is trained with streaming_language_modeling

    Args:
        model ([type]): Model to override the task for

    """

    # HACK: override task for some seq2seq models
    if (
        model.cfg.model._name in ["bart_base", "bart_large"]
        and model.cfg.task._name
        in ["language_modeling", "multilingual_language_modeling"]
        and model.cfg.criterion._name == "seq2seq_lm"
    ):
        from fairseq import tasks

        task_config = model.cfg.task
        task_config._name = "seq2seq_lm"
        new_task = tasks.setup_task(task_config, criterion_args=model.cfg.criterion)
        model.task = new_task
        model.setup_task()
    elif model.cfg.task._name == "streaming_language_modeling":
        # HACK: Use special version of the language_modeling task for inference with the models
        # trained with streaming_language_modeling.
        # TO DO: More elegant solution would be to decouple the training and inference logic for the tasks

        from omegaconf import OmegaConf

        from fairseq import tasks

        task_config = OmegaConf.create(
            tasks.TASK_DATACLASS_REGISTRY[
                "language_modeling_inference_for_models_trained_with_streaming"
            ]()
        )
        task_config._name = (
            "language_modeling_inference_for_models_trained_with_streaming"
        )
        task_config.data = model.cfg.task.data
        task_config.vocab_filename = model.cfg.task.vocab_filename
        task_config.merges_filename = model.cfg.task.merges_filename

        new_task = tasks.setup_task(task_config, criterion_args=model.cfg.criterion)
        model.task = new_task
        model.cfg.task = task_config
        model.setup_task()


def get_or_infer_max_tokens(model, **kwargs):
    if "max_tokens" in kwargs:
        model.cfg.dataset.max_tokens = kwargs["max_tokens"]
    if (
        model.cfg.dataset.max_tokens is not None
        or model.cfg.dataset.batch_size is not None
    ):
        return model.cfg.dataset.max_tokens
    return infer_max_tokens_before_oom(model)


def convert_max_positions_to_int(max_positions):
    if isinstance(max_positions, int):
        return max_positions
    elif isinstance(
        max_positions, tuple
    ):  # For seq2seq models where it's a tuple for encoder and decoder
        # TODO: Ideally we could take the sum because tokens are spread across encoder and decoder.
        # However it can impose a constraint on how the tokens are split w.r.t. encoder_resp.
        return min(max_positions)


def infer_max_tokens_before_oom(model, n_to_stop=8192) -> int:
    """Run the model on more and more tokens until it OOMs or succeeds with n_to_stop_tokens"""
    # NOTE(SS): This only works for dense models, for MOE models it runs in lockstep rather than DDP
    def is_max_tokens_oom(max_tokens):
        try:
            max_positions = convert_max_positions_to_int(model.max_positions)
            dummy_sample = model.decode(
                [42] * (max_positions - 2)
            )  # Minus 2 for special tokens in seq2seq
            local_bsz = int(max_tokens / max_positions)
            input_texts = [
                dummy_sample for _ in range(local_bsz)
            ] * distributed_utils.get_global_world_size()
            model.score(input_texts, batch_size=local_bsz, batch_by_size=False)
            return False
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise e
            return True

    assert model.cfg.dataset.max_tokens is None
    assert model.cfg.dataset.batch_size is None
    print_r0("Inferring max tokens for model...")
    candidate_max_tokens = convert_max_positions_to_int(model.max_positions)
    while not is_max_tokens_oom(candidate_max_tokens):
        candidate_max_tokens *= 2
        if candidate_max_tokens > (n_to_stop * 2):
            break
    max_tokens = candidate_max_tokens // 2
    print_r0(f"Setting max_tokens to {max_tokens}")
    return max_tokens


def run_with_oom_catch(func, **kwargs):
    """Return results of func(**kwargs) or catches OOM exceptions and returns None.

    Useful to run the OOM catch in a compartimented function to release tensors when exiting the local scope.
    Catching OOM in the outer scope will indeed prevent the tensors that exist at the time of the error to be released.
    """
    try:
        return func(**kwargs)
    except RuntimeError as e:
        if "CUDA out of memory" not in str(e):
            raise e
        return None


def run_with_adaptative_max_tokens(model, func, **kwargs):
    """Runs func(model, **kwargs) or lower max_tokens and try again when OOM occurs. func should never return None."""
    results = run_with_oom_catch(func, **kwargs)
    if results is not None:
        return results
    else:
        print_r0(
            f"OOM: max_tokens={model.cfg.dataset.max_tokens} ==> max_tokens={model.cfg.dataset.max_tokens//2}"
        )
        model.cfg.dataset.max_tokens //= 2
        return run_with_adaptative_max_tokens(model, func, **kwargs)
