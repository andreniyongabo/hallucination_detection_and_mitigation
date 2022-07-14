# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, field

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.cross_entropy import CrossEntropyCriterion, CrossEntropyCriterionConfig


@dataclass
class PromptTuningCriterionConfig(CrossEntropyCriterionConfig):
    add_negative_ce_loss: bool = field(
        default=False, metadata={"help": "add a negative cross entropy loss for negative labels"}
    )
    report_accuracy: bool = field(
        default=False, metadata={"help": "report accuracy"}
    )


@register_criterion("prompt_tuning", dataclass=PromptTuningCriterionConfig)
class PromptTuningCriterion(CrossEntropyCriterion):
    def __init__(self, task, add_negative_ce_loss, report_accuracy):
        super().__init__(task, sentence_avg=False)
        self.add_negative_ce_loss = add_negative_ce_loss
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        should_compute_loss_for_negatives = (
            self.add_negative_ce_loss
            or self.report_accuracy
        )

        input_tokens = sample["input_tokens"]
        target = sample["target"]
        pos_examples = sample["is_positive_example"].eq(1)
        if not should_compute_loss_for_negatives:
            input_tokens = input_tokens[pos_examples.view(-1)]
            target = target[pos_examples.view(-1)]

        net_output = model(input_tokens)
        loss_per_instance = self.compute_loss(model, net_output, target)

        if should_compute_loss_for_negatives:
            loss_per_instance = loss_per_instance.view_as(pos_examples)
            loss = loss_per_instance[pos_examples].sum()
            if self.add_negative_ce_loss:
                loss -= loss_per_instance[~pos_examples].sum()
        else:
            loss = loss_per_instance.sum()

        # TODO only count label tokens if --loss-for-label-tokens-only
        sample_size = target.ne(self.padding_idx).sum()

        logging_output = {
            "loss": loss.detach(),
            "ntokens": sample_size,
            "nsentences": target.size(0),
            "sample_size": sample_size,
        }

        if self.report_accuracy:
            assert should_compute_loss_for_negatives
            predictions = loss_per_instance.min(dim=1).values
            labels = loss_per_instance[pos_examples]
            logging_output["accuracy_numer"] = predictions.eq(labels).sum().detach()
            logging_output["accuracy_denom"] = predictions.numel()

        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, target):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss = F.nll_loss(lprobs, target.contiguous().view(-1), ignore_index=self.padding_idx, reduction="none")
        return loss.view_as(target).sum(dim=-1)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

        if any("accuracy_numer" in log for log in logging_outputs):
            accuracy_numer = sum(log.get("accuracy_numer", 0) for log in logging_outputs)
            accuracy_denom = sum(log.get("accuracy_denom", 0) for log in logging_outputs)
            metrics.log_scalar("_acc_numer", accuracy_numer)
            metrics.log_scalar("_acc_denom", accuracy_denom)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    (100.0 * meters["_acc_numer"].sum / meters["_acc_denom"].sum).item(),
                    2
                )
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
