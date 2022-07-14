# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from omegaconf import II
from torch import nn

from examples.few_shot import models
from examples.few_shot.gpt3_eval import (
    iterate_over_tasks,
    load_task_template_calibrator_predictor,
)
from examples.few_shot.tasks import get_all_tasks, get_tasks_by_group, is_task_group
from fairseq.data import (
    BaseWrapperDataset,
    IdDataset,
    ListDataset,
    MultiCorpusSampledDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PadDataset,
    SliceTokensDataset,
    TokenBlockDataset,
)
from fairseq.dataclass import ChoiceEnum
from fairseq.distributed import fsdp_wrap
from fairseq.tasks import register_task
from fairseq.tasks.language_modeling import LanguageModelingConfig, LanguageModelingTask

PREFIX_INIT_METHOD_CHOICES = ChoiceEnum(
    [
        # initialize with end-of-sentence symbol
        "eos",
        # use the mean of the first input tokens
        "first_input",
        # use the mean of the embed table
        "mean_embed",
        # use the mean of the embed table with some added noise
        "mean_embed_with_noise",
        # use the first few (most common) embeddings in the embed table
        "most_common_embed",
        # randomly initialize
        "random",
        # init with zeros (mostly for testing)
        "zeros",
    ]
)


logger = logging.getLogger(__name__)


def build_example_proportional_sampler(x, maximum):
    probs = []
    for dataset in x:
        probs.append(min(len(x[dataset]), maximum))
    probs = [x / sum(probs) for x in probs]
    return lambda y: y[np.random.choice(len(x), 1, p=probs)[0]]


@dataclass
class PromptTuningConfig(LanguageModelingConfig):
    def positional_args(self):
        # base class includes "data", which we remove, use --downstream-task instead
        return []

    downstream_task: str = field(
        default="", metadata={"help": "name of downstream task (e.g., cb, copa, ...)"}
    )
    num_few_shot_samples: int = field(
        default=-1,
        metadata={"help": "number of few shot samples to use; defaults to -1 (all)"},
    )
    uniform_few_shot_sampling: bool = field(
        default=False,
        metadata={"help": "take the same number of candidates per class when sampling"},
    )
    cheat: bool = field(
        default=False,
        metadata={
            "help": "train on the full train set and use the evaluation data as valid"
        },
    )

    model_name: str = field(
        default="?",
        metadata={"help": "name of model (e.g., 124M) or path to checkpoint to tune"},
    )
    num_prefix_tokens: int = field(
        default=-1, metadata={"help": "number of tokens to add as prefix"}
    )
    prefix_init_method: PREFIX_INIT_METHOD_CHOICES = field(
        default="most_common_embed",
        metadata={"help": "method for initializing prefix tokens"},
    )
    prefix_with_positional_embed: bool = field(
        default=False,
        metadata={"help": "add positional embeddings to the prefix tokens"},
    )
    finetune_model_weights: bool = field(
        default=False,
        metadata={
            "help": "finetune the model weights in addition to the prefix weights "
            "(by default we freeze model weights)"
        },
    )
    loss_for_label_tokens_only: bool = field(
        default=False,
        metadata={"help": "fill target with pads to only have loss on label words"},
    )
    # inherited from LanguageModelingConfig
    # seed: int = field(
    #     default=II("common.seed"), metadata={"help": "seed for selecting few shot samples"}
    # )
    memory_efficient_fp16: bool = II("common.memory_efficient_fp16")
    checkpoint_activations: bool = field(
        default=False, metadata={"help": "checkpoint activations at each layer"}
    )
    max_train_samples_per_task: int = field(
        default=-1, metadata={"help": "limit the maximum number of samples for a task"}
    )
    example_proportional_sampling: int = field(
        default=0,
        metadata={
            "help": "Sample datasets based on number of examples, with upper bound"
        },
    )
    freeze_up_to_layer: Optional[int] = II("distributed_training.freeze_up_to_layer")


@contextlib.contextmanager
def get_embedding_layer(model):
    if isinstance(model, FSDP):
        with model.summon_full_params(recurse=False, volatile=True):
            yield model.decoder.embed_tokens
    else:
        yield model.decoder.embed_tokens
    return


@torch.no_grad()
def init_prefix(cfg: PromptTuningConfig, embed_tokens, dictionary):
    offset = dictionary.nspecial  # ignore special symbols
    prefix = nn.Parameter(
        embed_tokens[offset : offset + cfg.num_prefix_tokens].clone().detach(),
        requires_grad=True,
    )
    if cfg.prefix_init_method == "eos":
        prefix.fill_(dictionary.eos())
    elif cfg.prefix_init_method == "first_input":
        raise NotImplementedError
    elif cfg.prefix_init_method in {"mean_embed", "mean_embed_with_noise"}:
        prefix[:, :] = embed_tokens[offset:].mean(dim=0, keepdim=True)
        if cfg.prefix_init_method == "mean_embed_with_noise":
            embedding_dim = embed_tokens.size(1)
            prefix[:, :].add_(torch.randn_like(prefix) * (embedding_dim**-0.5))
    elif cfg.prefix_init_method == "most_common_embed":
        pass
    elif cfg.prefix_init_method == "random":
        embedding_dim = embed_tokens.size(1)
        nn.init.normal_(prefix, mean=0, std=embedding_dim**-0.5)
    elif cfg.prefix_init_method == "zeros":
        nn.init.zeros_(prefix)
    else:
        raise ValueError(f"Unrecognized --prefix-init-method={cfg.prefix_init_method}")

    return prefix


class PromptTuningModelWrapper(nn.Module):
    def __init__(
        self, cfg: PromptTuningConfig, module: nn.Module, task: "PromptTuningTask"
    ):
        super().__init__()
        self.cfg = cfg
        self.module = module
        self.task = task

        if self.cfg.num_prefix_tokens > 0:
            with get_embedding_layer(module) as embed_tokens:
                self.register_parameter(
                    "prefix",
                    init_prefix(self.cfg, embed_tokens.weight, self.task.dictionary),
                )

    def extra_repr(self):
        return f"prefix={self.prefix.size() if self.prefix else 0}"

    def __getattr__(self, name):
        """Forward missing attributes to wrapped module."""
        try:
            # defer to nn.Module's logic
            return super().__getattr__(name)
        except AttributeError:
            # forward to the wrapped module
            return getattr(self.module, name)

    def state_dict(self, *args, **kwargs):
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.module.load_state_dict(*args, **kwargs)

    def forward(self, src_tokens, **kwargs):
        if self.task.args.prefix_init_method == "first_input":
            # TODO implement and test
            raise NotImplementedError

        if self.task.args.num_prefix_tokens > 0:
            with get_embedding_layer(self.module) as embed_tokens:
                token_embeddings = self.module.decoder.embed_tokens(src_tokens)
            token_embeddings = torch.cat(
                (
                    self.prefix.unsqueeze(0).repeat(token_embeddings.size(0), 1, 1),
                    token_embeddings,
                ),
                dim=1,
            )
            kwargs["token_embeddings"] = token_embeddings
            # left-pad tokens so that positional embeddings are set properly
            src_tokens = nn.functional.pad(
                src_tokens,
                (self.task.args.num_prefix_tokens, 0),
                value=(
                    # eos is a placeholder that will receive positional embeddings
                    self.task.dictionary.eos()
                    if self.task.args.prefix_with_positional_embed
                    # pad is a placeholder that will not receive positional embeddings
                    else self.task.dictionary.pad()
                ),
            )
            # but we still allow the self-attention to attend to prefix tokens
            kwargs["self_attn_padding_mask"] = src_tokens.eq(self.task.dictionary.pad())
            kwargs["self_attn_padding_mask"][
                :, : self.task.args.num_prefix_tokens
            ] = False

        net_output = self.module(src_tokens, **kwargs)

        # remove prefix tokens from output
        if self.task.args.num_prefix_tokens > 0:
            net_output = (
                net_output[0][:, self.task.args.num_prefix_tokens :, :],
                *net_output[1:],
            )

        return net_output


class Pad3DDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset)
        self.pad_idx = pad_idx

    def collater(self, samples):
        batch = torch.nn.utils.rnn.pad_sequence(
            [sample.t() for sample in samples],
            batch_first=True,
            padding_value=self.pad_idx,
        )
        if batch.dim() == 3:
            batch = batch.transpose(1, 2)
        return batch.reshape(-1, batch.size(-1))


@register_task("prompt_tuning", dataclass=PromptTuningConfig)
class PromptTuningTask(LanguageModelingTask):
    def __init__(
        self, cfg: PromptTuningConfig, dictionary, output_dictionary=None, **kwargs
    ):
        super().__init__(args=cfg, dictionary=None, output_dictionary=None, **kwargs)
        if cfg.sample_break_mode != "eos":
            raise NotImplementedError(
                "only --sampling-break-mode=eos is supported for prompt tuning task"
            )

    def _load_model_from_hub(self, cfg: PromptTuningConfig):
        # read model config from examples.few_shot.models
        eval_lm_cfg, model_config = models.get_lm_config(cfg.model_name)
        model_config["model_overrides"][
            "checkpoint_activations"
        ] = cfg.checkpoint_activations

        def post_build_model_hook(model, task):
            if not cfg.finetune_model_weights:
                assert cfg.num_prefix_tokens > 0, (
                    "either --num-prefix-tokens must be greater than 0, or you need "
                    "--finetune-model-weights; else there would be no learnable parameters"
                )
                for p in model.parameters():
                    p.requires_grad = False

            # this is normally done in DistributedFairseqModel
            if cfg.memory_efficient_fp16:
                model = model.half()

            return fsdp_wrap(model)

        # load model
        hub_model = models.load_and_get_model(
            eval_lm_cfg,
            model_config,
            skip_prepare_for_inference=True,
            post_build_model_hook=post_build_model_hook,
            fsdp=False,  # wrap using this hook
        )
        model = hub_model.models[0]

        return hub_model

    def build_model(self, *unused):
        # reload model so that we wrap properly with FSDP
        self.hub_model = self._load_model_from_hub(self.args)
        self.dictionary = self.hub_model.task.dictionary
        self.output_dictionary = self.hub_model.task.output_dictionary
        model = self.hub_model.models[0]

        # add prompt parameters and apply one final FSDP wrapper
        if self.args.num_prefix_tokens > 0:
            # Resume from checkpoint doesn't work right now with PromptTuningModelWrapper and fsdp models
            model = PromptTuningModelWrapper(
                cfg=self.args, module=self.hub_model.models[0], task=self
            )
            model = fsdp_wrap(model, flatten_parameters=False)

        return model

    def build_criterion(self, cfg):
        assert (
            cfg._name == "prompt_tuning"
        ), "--task=prompt_tuning requires --criterion=prompt_tuning"
        return super().build_criterion(cfg)

    def has_sharded_data(self, split):
        return False

    def _generate_dataset(self, task, predictor, template, samples, split):
        # Expand samples with subproblems (e.g., MultiRC)
        samples = [
            subproblem
            for sample in samples
            for subproblem in (
                sample.subproblems if sample.has_subproblems else [sample]
            )
        ]

        # HACK: empty train_samples in task to reuse get_prompts
        # without adding training samples in prompts
        task._train_samples = []

        def tokenize_sentence(sentence):
            _, sentence = self.hub_model.get_sentence_and_language(sentence)
            sentence = self.hub_model.tokenize(sentence)
            lines = [
                self.hub_model.apply_bpe(line) for line in sentence.splitlines() if line
            ]
            sentence = " </s> ".join(lines)
            return self.hub_model.binarize(sentence)

        def common_prefix_len(tensor1, tensor2):
            max_len = min(len(tensor1), len(tensor2))
            diff_mask = tensor1[:max_len] != tensor2[:max_len]
            prefix_len = diff_mask.nonzero(as_tuple=True)[0][0]
            return prefix_len

        src, tgt, labels, src_sizes = [], [], [], []
        prompts_with_mask, _, _ = predictor.get_prompts(samples)

        for prompt_with_mask, sample in zip(prompts_with_mask, samples):
            toks_with_mask = tokenize_sentence(prompt_with_mask)
            correct_cands = set(sample.correct_candidates)

            src_i, tgt_i, labels_i = [], [], []
            if task.has_candidates and split != "train": # Only use other candidates for validation, to compute validation accuracy
                candidates = sample.candidates
            else:
                # Generation task
                candidates = set(sample.correct_candidates)
            for candidate in candidates:
                label = 1 if candidate in correct_cands else 0

                candidate = template.verbalize(sample, candidate)
                prompt_without_mask = prompt_with_mask.replace("<mask>", candidate)
                toks_without_mask = tokenize_sentence(prompt_without_mask)

                src_toks = toks_without_mask[:-1]
                tgt_toks = toks_without_mask[1:]

                # fill target with pads to only have loss on label words
                if self.args.loss_for_label_tokens_only:
                    prefix_len = common_prefix_len(toks_with_mask, toks_without_mask)
                    tgt_toks = tgt_toks.clone()
                    tgt_toks[: prefix_len - 1] = self.target_dictionary.pad()

                src_i.append(src_toks)
                tgt_i.append(tgt_toks)
                labels_i.append(label)

            src.append(self._pad_batch(src_i))
            tgt.append(self._pad_batch(tgt_i))
            labels.append(torch.tensor(labels_i, dtype=torch.int))
            src_sizes.append(src[-1].size(-1))

        # pad_idx = self.source_dictionary.pad()
        input_dataset = ListDataset(src, np.array(src_sizes))
        tgt_dataset = ListDataset(tgt, np.array(src_sizes))
        labels_dataset = ListDataset(labels)
        return input_dataset, tgt_dataset, labels_dataset

    def _pad_batch(self, tensor_list):
        return torch.nn.utils.rnn.pad_sequence(
            [x.t() for x in tensor_list],
            batch_first=True,
            padding_value=self.source_dictionary.pad(),
        )

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        assert (
            self.args.data is None
        ), "data argument is not compatible with --task=prompt_tuning"
        assert self.args.downstream_task != "", "--downstream-task is required"
        split_name = split
        if "valid_" in split or "eval_" in split or "test_" in split:
            downstream_tasks = ["_".join(split.split("_")[1:])]
            split = split.split("_")[0]
        else:
            downstream_tasks = self.args.downstream_task.split(",")

        # Retrieve all the task if the downstream_tasks has a task group name
        downstream_tasks_expanded = []
        for task_name in downstream_tasks:
            all_tasks = get_all_tasks()
            if task_name in all_tasks:
                downstream_tasks_expanded.append(task_name)
            elif is_task_group(task_name):
                downstream_tasks_expanded.extend(get_tasks_by_group(task_name))
        downstream_tasks = downstream_tasks_expanded

        if split in {"valid", "eval", "test"}:
            assert (
                len(downstream_tasks) == 1
            ), 'Please specify "valid_{task_name}" for --valid-subset when using multiple tasks'

        total_samples = 0
        input_dataset, tgt_dataset, labels_dataset = {}, {}, {}
        for task_name, template_name, _, __, language, *__ in iterate_over_tasks(
            downstream_tasks, **self.args
        ):
            (
                task,
                template,
                calibrator,
                predictor,
            ) = load_task_template_calibrator_predictor(
                model=self.hub_model,
                task_name=task_name,
                template_name=template_name,
                predictor_name="clmprompting",
                language=language,
                nb_few_shot_samples=self.args.num_few_shot_samples,
                uniform_sampling=self.args.uniform_few_shot_sampling,
                seed=self.args.seed,
                use_full_train_set=self.args.cheat,
            )

            if split == "train":
                if self.args.max_train_samples_per_task > 0:
                    samples = task.train_samples[: self.args.max_train_samples_per_task]
                else:
                    samples = task.train_samples
            elif split == "valid":
                if self.args.cheat:
                    logger.warning(
                        "CHEATING MODE: training on full train set and using eval data for validation"
                    )
                    samples = task.eval_samples
                else:
                    samples = task.valid_samples
            elif split in {"eval", "test"}:
                logger.warning(
                    "using evaluation data, this risks overfitting if used for training or model selection!"
                )
                samples = task.eval_samples
            else:
                raise ValueError(f"unknown dataset split: {split}")

            input_list, tgt_list, labels_list = self._generate_dataset(
                task, predictor, template, samples, split
            )
            input_dataset[task_name] = input_list
            tgt_dataset[task_name] = tgt_list
            labels_dataset[task_name] = labels_list
            total_samples += len(samples)

            logger.info(
                'loaded split "{}" of {} task with {:,} samples'.format(
                    split, task_name, len(samples)
                )
            )

        fixed_pad_length = None
        if self.args.pad_to_fixed_length:
            # This probably works, but not tested:
            # fixed_pad_length = self.args.tokens_per_sample
            # sizes = torch.full_like(sizes, fill_value=fixed_pad_length)
            raise NotImplementedError

        if self.args.pad_to_fixed_bsz:
            raise NotImplementedError

        if self.args.add_bos_token:
            raise NotImplementedError

        datasets = OrderedDict()
        for key in input_dataset.keys():
            datasets[key] = NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "input_tokens": Pad3DDataset(
                        input_dataset[key], pad_idx=self.source_dictionary.pad()
                    ),
                    "target": Pad3DDataset(
                        tgt_dataset[key], pad_idx=self.source_dictionary.pad()
                    ),
                    "is_positive_example": Pad3DDataset(
                        labels_dataset[key], pad_idx=-1
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(input_dataset[key], reduce=True),
                },
                sizes=[input_dataset[key].sizes],
            )

        if len(datasets) == 1:
            self.datasets[split_name] = datasets.popitem(last=False)[1]
        else:
            assert split == "train"
            logger.info(
                "Combining multiple tasks together for training with a total of {:,} samples".format(
                    total_samples
                )
            )
            if self.args.example_proportional_sampling == 0:
                self.datasets[split] = MultiCorpusSampledDataset(
                    datasets
                )  # Currently only doing uniform sampling
            else:
                logger.info(
                    f"Using Example proportional Sampling with an upper bound of {self.args.example_proportional_sampling}"
                )
                example_proportional_sampler = build_example_proportional_sampler(
                    datasets, maximum=self.args.example_proportional_sampling
                )
                self.datasets[split] = MultiCorpusSampledDataset(
                    datasets, sampling_func=example_proportional_sampler
                )

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        """Generate batches for inference."""
        dataset = TokenBlockDataset(
            src_tokens,
            src_lengths,
            block_size=None,  # ignored for "eos" break mode
            pad=self.source_dictionary.pad(),
            eos=self.source_dictionary.eos(),
            break_mode="eos",
        )
        input_dataset = SliceTokensDataset(dataset, right_slice=-1)
        tgt_dataset = SliceTokensDataset(dataset, left_slice=1)
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(
                        input_dataset,
                        pad_idx=self.source_dictionary.pad(),
                        left_pad=False,
                    ),
                    "src_lengths": NumelDataset(input_dataset, reduce=False),
                },
                "target": PadDataset(
                    tgt_dataset,
                    pad_idx=self.source_dictionary.pad(),
                    left_pad=False,
                ),
            },
            sizes=[np.array(src_lengths)],
        )
