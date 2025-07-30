from copy import copy
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn as nn
from hnet.models.config_hnet import HNetConfig
from hnet.modules.dc import (
    ChunkLayer,
    DeChunkLayer,
    DeChunkState,
    RoutingModule,
    RoutingModuleState,
)
from hnet.modules.isotropic import IsotropicInferenceParams

from ..modules.dm import (
    CausalBlockMask,
    CausalBlockMaskState,
    MaskingModule,
    MaskingModuleState,
)
from ..modules.isotropic import Isotropic
from ..modules.utils import ste_func


@dataclass
class HMNetState:
    encoder_state: Optional[IsotropicInferenceParams] = None
    routing_module_state: Optional[RoutingModuleState] = None
    main_network_state: Optional[Union["HMNetState", IsotropicInferenceParams]] = None
    dechunk_state: Optional[DeChunkState] = None
    decoder_state: Optional[IsotropicInferenceParams] = None
    masking_state: Optional[MaskingModuleState] = None
    causal_block_mask_state: Optional[CausalBlockMaskState] = None


class HMNet(nn.Module):
    def __init__(
        self,
        config: HNetConfig,
        stage_idx: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            config=config,
            stage_idx=stage_idx,
            device=device,
            dtype=dtype,
        )
        factory_kwargs = {"device": device, "dtype": dtype}

        self.stage_idx = stage_idx
        self.d_model = config.d_model[stage_idx]
        self.window_size = config.attn_cfg.window_size[stage_idx]

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
                config=config,
                stage_idx=stage_idx,
                pos_idx=0,  # Assuming the innermost network is at position 0
                **factory_kwargs,
            )
        else:
            self.encoder = Isotropic(
                config=config,
                stage_idx=stage_idx,
                pos_idx=0,
                **factory_kwargs,
            )
            self.decoder = Isotropic(
                config=config,
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
            self.causal_block_mask = CausalBlockMask(self.window_size)
            self.masking_module = MaskingModule(self.d_model, **factory_kwargs)

            # do the residual in fp32
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
        """
        Allocate the inference cache for the HNet.

        Arguments:
            batch_size: int. The number of sequences in the batch.
            max_seqlen: int. The maximum sequence length in the batch.
            dtype: torch.dtype. The dtype of the inference cache.

        The structure of the inference cache is as follows:
            - [encoder state]
            - [routing module state]
            - [main network state]
            - [dechunk state]
            - [decoder state]
        It is thus a list of length 5.
        """
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
                masking_state=self.masking_module.allocate_inference_cache(
                    batch_size, max_seqlen, device, dtype=dtype
                ),
                causal_block_mask_state=self.causal_block_mask.allocate_inference_cache(
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
                (hidden_states, self.pad_dimension.expand(EARLY_DIMS + (-1,))), dim=-1
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

        hidden_states, prev_boundary_predictions, prev_mask_predictions = (
            self.main_network(
                hidden_states,
                cu_seqlens=next_cu_seqlens,
                max_seqlen=next_max_seqlen,
                mask=next_mask,
                inference_params=inference_params.main_network_state,
                **mixer_kwargs,
            )
        )

        mask_prediction = self.masking_module(
            hidden_states, inference_params.masking_state
        )

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

        block_mask, score_mod = self.causal_block_mask(
            boundary_mask=bpred_output.boundary_mask,
            block_score=mask_prediction,
        )
        mixer_kwargs_for_decoder = copy.deepcopy(mixer_kwargs)
        mixer_kwargs_for_decoder.update(
            {
                "block_mask": block_mask,
                "score_mod": score_mod,
            }
        )
        hidden_states = self.decoder(
            hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            mask=mask,
            inference_params=inference_params.decoder_state,
            **mixer_kwargs_for_decoder,
        )

        hidden_states = hidden_states[..., :D]

        return (
            hidden_states,
            [bpred_output, *prev_boundary_predictions],
            [mask_prediction, *prev_mask_predictions],
        )

    def step(self, hidden_states: torch.Tensor, inference_params: HMNetState):
        D = hidden_states.shape[-1]

        if self.pad_dimension is not None:
            hidden_states = torch.cat(
                (
                    hidden_states,
                    self.pad_dimension.expand(hidden_states.shape[:-1] + (-1,)),
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
            hidden_states_inner, prev_boundary_predictions, prev_mask_predictions = (
                self.main_network.step(
                    hidden_states_inner, inference_params.main_network_state
                )
            )
        else:
            prev_boundary_predictions = []
            prev_mask_predictions = []

        mask_prediction = self.masking_module.step(
            hidden_states_inner, inference_params.masking_state
        )

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
        block_mask, score_mod = self.causal_block_mask.step(
            boundary_mask=bpred_output.boundary_mask,
            block_score=mask_prediction,
            inference_params=inference_params.causal_block_mask_state,
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
            [mask_prediction, *prev_mask_predictions],
        )
