from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from flash_attn.utils.generation import GenerationMixin
from torch import Tensor

from ..modules.dc import (
    ChunkLayer,
    DeChunkLayer,
    DeChunkState,
    RoutingModule,
    RoutingModuleOutput,
    RoutingModuleState,
)
from ..modules.dm import (
    ChunkAttnScoreModule,
    ChunkAttnScoreState,
    DeChunkAttnScoreLayer,
    DeChunkAttnScoreState,
)
from ..modules.isotropic import Isotropic, IsotropicInferenceParams
from ..modules.utils import ste_func
from .config_hmnet import HMNetConfig, HMNetTrainConfig


@dataclass
class HMNetState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HMNetState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None
    chunk_attn_score_state: Optional[ChunkAttnScoreState] = None
    dechunk_attn_score_state: Optional[DeChunkAttnScoreState] = None


class HMNet(nn.Module):
    def __init__(
        self,
        config: HMNetConfig,
        stage_idx: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]

        assert isinstance(arch_layout, list), f"Wrong arch_layout: {arch_layout}"
        if len(arch_layout) == 3:
            self.is_innermost = False
        elif len(arch_layout) == 1:
            self.is_innermost = True
        else:
            raise NotImplementedError

        if self.is_innermost:
            self.main_network = Isotropic(
                config=config.decoder_hnet_config(),
                stage_idx=stage_idx,
                pos_idx=0,  # Assuming the innermost network is at position 0
                **factory_kwargs,
            )
        else:
            self.encoder = Isotropic(
                config=config.encoder_hnet_config(),
                stage_idx=stage_idx,
                pos_idx=0,
                **factory_kwargs,
            )
            self.decoder = Isotropic(
                config=config.decoder_hnet_config(),
                stage_idx=stage_idx,
                pos_idx=2,
                **factory_kwargs,
            )
            self.main_network = HMNet(
                config=config,
                stage_idx=stage_idx + 1,
                **factory_kwargs,
            )
            self.routing_module = RoutingModule(self.d_model, **factory_kwargs)
            self.chunk_layer = ChunkLayer()
            self.dechunk_layer = DeChunkLayer(self.d_model)
            self.dechunk_attn_score_layer = DeChunkAttnScoreLayer(
                window_size=config.decoder_attn_cfg.window_size[stage_idx]
            )
            self.chunk_attn_score_module = ChunkAttnScoreModule(
                self.d_model, **factory_kwargs
            )

            self.residual_proj = nn.Linear(
                self.d_model, self.d_model, device=device, dtype=torch.float32
            )
            nn.init.zeros_(self.residual_proj.weight)
            self.residual_proj.weight._no_reinit = True
            self.residual_func = lambda out, residual, p: out * ste_func(p) + residual

        if stage_idx > 0 and self.d_model - config.d_model[stage_idx - 1] > 0:
            self.pad_dimension = nn.Parameter(
                torch.zeros(
                    self.d_model - config.d_model[stage_idx - 1], **factory_kwargs
                )
            )
        else:
            self.pad_dimension = None

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        if self.is_innermost:
            hmnet_state = HMNetState(
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                )
            )
        else:
            device = self.residual_proj.weight.device
            hmnet_state = HMNetState(
                encoder_state=self.encoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                routing_module_state=self.routing_module.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                main_network_state=self.main_network.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                dechunk_state=self.dechunk_layer.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                decoder_state=self.decoder.allocate_inference_cache(
                    batch_size, max_seqlen, dtype=dtype
                ),
                chunk_attn_score_state=self.chunk_attn_score_module.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                dechunk_attn_score_state=self.dechunk_attn_score_layer.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
            )
        return hmnet_state

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        mask=None,
        inference_params: HMNetState | None = None,
        **mixer_kwargs,
    ):
        assert mask is not None or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        if inference_params is None:
            inference_params = HMNetState(main_network_state=None)
        else:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is provided"

        D = hidden_states.shape[-1]
        EARLY_DIMS = hidden_states.shape[:-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (hidden_states, self.pad_dimension.expand(*EARLY_DIMS, -1)), dim=-1
            )

        if self.is_innermost:
            hidden_states = self.main_network(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                mask=mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, [], []

        hidden_states = self.encoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.encoder_state,
            **mixer_kwargs,
        )

        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module(
            hidden_states,
            cu_seqlens=cu_seqlens,
            mask=mask,
            inference_params=inference_params.routing_module_state,
        )
        hidden_states, next_cu_seqlens, next_max_seqlen, next_mask = self.chunk_layer(
            hidden_states, bpred_output.boundary_mask, cu_seqlens, mask=mask
        )

        hidden_states, prev_boundary_predictions, prev_chunk_attn_scores = (
            self.main_network(
                hidden_states,
                cu_seqlens=next_cu_seqlens,
                max_seqlen=next_max_seqlen,
                mask=next_mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
        )

        chunk_attn_logit = self.chunk_attn_score_module(
            hidden_states, inference_params.chunk_attn_score_state
        )
        chunk_attn_score = chunk_attn_logit.softmax(dim=-1)

        hidden_states = self.dechunk_layer(
            hidden_states,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            next_cu_seqlens,
            mask=mask,
            inference_params=inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype),
            residual,
            bpred_output.selected_probs,
        ).to(hidden_states.dtype)

        score_mod, block_mask = self.dechunk_attn_score_layer(
            boundary_mask=bpred_output.boundary_mask,
            chunk_attn_score=chunk_attn_score,
            mask=mask,
            inference_params=inference_params.dechunk_attn_score_state,
        )

        hidden_states = self.decoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.decoder_state,
            block_mask=block_mask,
            score_mod=score_mod,
            **mixer_kwargs,
        )

        hidden_states = hidden_states[..., :D]

        return (
            hidden_states,
            [bpred_output, *prev_boundary_predictions],
            [chunk_attn_logit, *prev_chunk_attn_scores],
        )

    def step(self, hidden_states: torch.Tensor, inference_params: HMNetState):
        D = hidden_states.shape[-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (
                    hidden_states,
                    self.pad_dimension.expand(*hidden_states.shape[:-1], -1),
                ),
                dim=-1,
            )

        if self.is_innermost:
            hidden_states = self.main_network.step(
                hidden_states, inference_params.main_network_state
            )
            hidden_states = hidden_states[..., :D]
            return hidden_states, [], []

        hidden_states = self.encoder.step(hidden_states, inference_params.encoder_state)
        hidden_states_for_residual = hidden_states.to(
            dtype=self.residual_proj.weight.dtype
        )
        residual = self.residual_proj(hidden_states_for_residual)

        bpred_output = self.routing_module.step(
            hidden_states, inference_params.routing_module_state
        )
        hidden_states_inner = self.chunk_layer.step(
            hidden_states, bpred_output.boundary_mask
        )

        if hidden_states_inner.shape[0] > 0:
            hidden_states_inner, prev_boundary_predictions, prev_chunk_attn_scores = (
                self.main_network.step(
                    hidden_states_inner, inference_params.main_network_state
                )
            )
        else:
            prev_boundary_predictions = []
            prev_chunk_attn_scores = []

        chunk_attn_logit = self.chunk_attn_score_module.step(
            hidden_states_inner, inference_params.chunk_attn_score_state
        )
        chunk_attn_score = chunk_attn_logit.softmax(dim=-1)

        hidden_states = self.dechunk_layer.step(
            hidden_states_inner,
            bpred_output.boundary_mask,
            bpred_output.boundary_prob,
            inference_params.dechunk_state,
        )

        hidden_states = self.residual_func(
            hidden_states.to(dtype=residual.dtype),
            residual,
            bpred_output.selected_probs,
        ).to(hidden_states.dtype)

        score_mod, block_mask = self.dechunk_attn_score_layer.step(
            boundary_mask=bpred_output.boundary_mask,
            chunk_attn_score=chunk_attn_score,
            inference_params=inference_params.dechunk_attn_score_state,
        )
        hidden_states = self.decoder.step(
            hidden_states,
            inference_params.decoder_state,
            block_mask=block_mask,
            score_mod=score_mod,
        )

        hidden_states = hidden_states[..., :D]

        return (
            hidden_states,
            [bpred_output, *prev_boundary_predictions],
            [chunk_attn_logit, *prev_chunk_attn_scores],
        )


class HMNetForClassification(nn.Module):
    def __init__(
        self,
        model_config: HMNetConfig,
        num_classes: int,
    ):
        super().__init__()
        self.backbone = HMNet(model_config, stage_idx=0)
        self.classifier = torch.nn.Linear(model_config.d_model[0], num_classes)
        self.num_classes = num_classes

    def forward(self, x: Tensor, mask: Tensor | None = None):
        x, chunk_pred, attn_pred = self.backbone(x, mask=mask)
        if mask is not None:
            seq_lengths = mask.sum(dim=1) - 1
            batch_indices = torch.arange(x.size(0), device=x.device)
            x = x[batch_indices, seq_lengths, :]
        else:
            x = x[:, -1, :]
        y = self.classifier(x)
        return y, chunk_pred, attn_pred


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    bpred_output: list[RoutingModuleOutput]
    chunk_attn_logit_output: list[Tensor]
    inference_params: HMNetState | None


class HMNetForCausalLM(nn.Module, GenerationMixin):
    def __init__(
        self,
        config: HMNetConfig,
        device=None,
        dtype=None,
    ) -> None:
        self.config = config

        vocab_size = self.config.vocab_size
        d_embed = self.config.d_model[0]
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        # We consider the HNet as a map (B, L, D[0]) -> (B, L, D[0])
        # Thus, the embedding is defined outside of the HNet.
        self.embeddings = nn.Embedding(vocab_size, d_embed, **factory_kwargs)

        self.backbone = HMNet(
            config=config,
            # We pass in the stage_idx as an HNet needs to know what
            # depth of the hierarchy it is in.
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_embed, vocab_size, bias=False, **factory_kwargs)
        self._tie_weights()

    def _tie_weights(self):
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embeddings.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.backbone.allocate_inference_cache(
            batch_size, max_seqlen, dtype=dtype, **kwargs
        )

    def forward(
        self,
        input_ids,
        mask=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        """
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        hidden_states = self.embeddings(input_ids)

        B, L, D = hidden_states.shape

        assert (
            position_ids is None
        ), "Position ids are not supported for HNet due to the subsampling hierarchical structure"

        if mask is None:
            # Absent a mask, we assume we are running in packed mode
            assert (
                inference_params is None
            ), "Inference params are not supported in packed mode"
            hidden_states = hidden_states.flatten(0, 1)
            cu_seqlens = torch.arange(B + 1, device=hidden_states.device) * L
            max_seqlen = torch.tensor(L, dtype=torch.int, device=hidden_states.device)
        else:
            cu_seqlens = None
            max_seqlen = None

        hidden_states, bpred_output, chunk_attn_logit_output = self.backbone(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params,
            **mixer_kwargs,
        )

        hidden_states = hidden_states.view(B, L, D)

        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=lm_logits,
            bpred_output=bpred_output,
            chunk_attn_logit_output=chunk_attn_logit_output,
            inference_params=inference_params,
        )

    def step(self, input_ids, inference_params):
        B = input_ids.shape[0]
        assert (
            B == 1
        ), "HNetForCausalLM step currently only supports batch size 1 -- need to handle different-size lengths for each sample"

        hidden_states = self.embeddings(input_ids)

        hidden_states, bpred_output, chunk_attn_logit_output = self.backbone.step(
            hidden_states, inference_params
        )
        logits = self.lm_head(hidden_states)

        return CausalLMOutput(
            logits=logits,
            bpred_output=bpred_output,
            chunk_attn_logit_output=chunk_attn_logit_output,
            inference_params=inference_params,
        )
