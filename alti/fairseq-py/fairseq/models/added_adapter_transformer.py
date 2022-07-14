# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import checkpoint_utils, utils
from fairseq.distributed import utils as dist_utils, fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Linear
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerNorm,
)
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor
import logging

logger = logging.getLogger(__name__)


class Adapter(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout_module):
        super().__init__()
        self.dropout_module = dropout_module
        self.layer_norm = LayerNorm(embed_dim)
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.dropout_module(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x


@register_model("added_adapter_transformer")
class AddedAdapterTransformerModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    def half(self):
        self.encoder.base().half()
        self.decoder.base().half()
        return super().half()

    def eval(self):
        self.encoder.base().eval()
        self.decoder.base().eval()
        return super().eval()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument(
            "--base-model",
            type=str,
            help="checkpoint providing transformer model for which to learn adapters",
        )
        parser.add_argument(
            "--adapter-hidden-dim",
            type=int,
            default=256,
            help="checkpoint providing transformer model for which to learn adapters",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0,
            help="dropout probablility (for adapter FCs only)",
        )
        parser.add_argument(
            "--moe-base-model",
            action="store_true",
            help="base model is mixture-of-experts model (arbitrary rank file supplied to --base-model)",
        )
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # pass through MoE-related model overrides
        base_model_overrides = {}
        for argname in (
            "world_size",
            "moe_eval_capacity_token_fraction",
            "use_moe_pad_mask",
        ):
            if hasattr(args, argname):
                base_model_overrides[argname] = getattr(args, argname)

        base_model_state = checkpoint_utils.load_checkpoint_to_cpu(
            args.base_model,
            is_moe=getattr(args, "moe_base_model", False),
            arg_overrides=base_model_overrides,
        )
        base_model_cfg = base_model_state["cfg"]

        # TODO: fix tutel installation
        base_model_cfg.model.use_tutel_moe = False

        base_model = task.build_model(base_model_cfg.model)
        base_model.load_state_dict(base_model_state["model"])
        if torch.cuda.is_available():
            base_model.to(torch.cuda.current_device())
        for param in base_model.parameters():
            param.requires_grad = False

        encoder = cls.build_encoder(args, base_model.encoder)
        decoder = cls.build_decoder(args, base_model.decoder)
        return cls(args, encoder, decoder)

    @classmethod
    def build_encoder(cls, args, base_encoder):
        return AddedAdapterTransformerEncoder(args, base_encoder)

    @classmethod
    def build_decoder(cls, args, base_decoder):
        return AddedAdapterTransformerDecoder(args, base_decoder)

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class AddedAdapterTransformerEncoder(FairseqEncoder):
    def __init__(self, args, base_encoder):
        self.args = args
        super().__init__(base_encoder.dictionary)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self._base_encoder = (base_encoder,)
        self.embed_dim = base_encoder.embed_tokens.embedding_dim
        self.paddding_idx = base_encoder.padding_idx

        self.adapters = nn.ModuleList()
        for _ in self.base().layers:
            self.adapters.append(
                Adapter(
                    self.embed_dim,
                    args.adapter_hidden_dim,
                    self.dropout_module,
                ),
            )

    def base(self):
        return self._base_encoder[0]

    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        return self.forward_scriptable(
            src_tokens,
            src_lengths,
            return_all_hiddens,
            token_embeddings,
        )

    # TorchScript doesn't support super() method so that the scriptable Subclass
    # can't access the base class model in Torchscript.
    # Current workaround is to add a helper function with different name and
    # call the helper function from scriptable Subclass.
    def forward_scriptable(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.base().padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()

        x, encoder_embedding = self.base().forward_embedding(
            src_tokens, token_embeddings
        )

        # account for padding while computing the representation
        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        # encoder layers
        l_aux = []
        for idx, layer in enumerate(self.base().layers):
            passed_src_tokens = (
                src_tokens
                if getattr(self.base().args, "pass_tokens_transformer_layer", False)
                else None
            )
            x, l_aux_i = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                tokens=passed_src_tokens,
            )

            adapter = self.adapters[idx]
            x = adapter(x) + x

            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            l_aux.append(l_aux_i)

        if self.base().layer_norm is not None:
            x = self.base().layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
            "l_aux": l_aux,
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if len(encoder_out["encoder_out"]) == 0:
            new_encoder_out = []
        else:
            new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                encoder_out["encoder_padding_mask"][0].index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                encoder_out["encoder_embedding"][0].index_select(0, new_order)
            ]

        if len(encoder_out["src_tokens"]) == 0:
            src_tokens = []
        else:
            src_tokens = [(encoder_out["src_tokens"][0]).index_select(0, new_order)]

        if len(encoder_out["src_lengths"]) == 0:
            src_lengths = []
        else:
            src_lengths = [(encoder_out["src_lengths"][0]).index_select(0, new_order)]

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": src_tokens,  # B x T
            "src_lengths": src_lengths,  # B x 1
        }

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.base().embed_positions is None:
            return self.base().max_source_positions
        return min(
            self.base().max_source_positions, self.base().embed_positions.max_positions
        )


class AddedAdapterTransformerDecoder(FairseqIncrementalDecoder):
    def __init__(self, args, base_decoder):
        self.args = args
        super().__init__(base_decoder.dictionary)

        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )

        self._base_decoder = (base_decoder,)
        self.embed_dim = base_decoder.embed_tokens.embedding_dim
        self.padding_idx = base_decoder.padding_idx

        self.adapters = nn.ModuleList()
        for _ in self.base().layers:
            self.adapters.append(
                Adapter(
                    self.embed_dim,
                    args.adapter_hidden_dim,
                    self.dropout_module,
                ),
            )

    def base(self):
        return self._base_decoder[0]

    def reorder_incremental_state_scripting(
        self,
        incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor,
    ):
        return self.base().reorder_incremental_state_scripting(
            incremental_state, new_order
        )

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            token_embeddings=token_embeddings,
            self_attn_padding_mask=self_attn_padding_mask,
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        token_embeddings: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            token_embeddings=token_embeddings,
            self_attn_padding_mask=self_attn_padding_mask,
        )

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        token_embeddings: Optional[Tensor] = None,
        self_attn_padding_mask: Optional[Tensor] = None,
    ):
        """
        A scriptable subclass of this class has an extract_features method and calls
        super().extract_features, but super() is not supported in torchscript. A copy
        of this function is made to be used in the subclass instead.
        """
        if alignment_layer is None:
            alignment_layer = self.base().num_layers - 1

        # compute self-attention padding mask (involves device-to-host transfer,
        # so put it at the top of the forward)
        if self_attn_padding_mask is None and (
            self.base().cross_self_attention
            or prev_output_tokens.device.type == "xla"
            or prev_output_tokens.eq(self.padding_idx).any()
        ):
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # embed tokens and positions
        x, _ = self.base().forward_embedding(
            prev_output_tokens, token_embeddings, incremental_state
        )

        if incremental_state is None and not full_context_alignment:
            self_attn_mask = self.base().buffered_future_mask(x)
        else:
            self_attn_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        if encoder_out is None:
            l_aux = []
        else:
            l_aux = encoder_out["l_aux"] if "l_aux" in encoder_out else []

        for idx, layer in enumerate(self.base().layers):
            prev_output_tokens = (
                prev_output_tokens
                if getattr(self.base().args, "pass_tokens_transformer_layer", False)
                else None
            )
            x, layer_attn, _, l_aux_i = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
                tokens=prev_output_tokens,
            )

            adapter = self.adapters[idx]
            x = adapter(x) + x

            l_aux.append(l_aux_i)
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.base().layer_norm is not None:
            x = self.base().layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.base().project_out_dim is not None:
            x = self.base().project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "l_aux": l_aux}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.base().adaptive_softmax is None:
            # project back to size of vocabulary
            return self.base().output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.base().embed_positions is None:
            return self.base().max_target_positions
        return min(
            self.base().max_target_positions, self.base().embed_positions.max_positions
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


@register_model_architecture("added_adapter_transformer", "added_adapter_transformer")
def base_architecture(args):
    return args
