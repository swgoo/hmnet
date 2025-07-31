from functools import partial

import torch
from flash_attn.ops.triton.layer_norm import RMSNorm
from hnet.modules.block import Block as HNetBlock
from hnet.modules.mlp import SwiGLU
from mamba_ssm.modules.mamba2 import Mamba2
from torch import nn

from .mha import CausalMaskMHA


class Mamba2Wrapper(Mamba2):
    """
    Mamba2 wrapper class that has the same inference interface as the CausalMHA class.
    """

    def forward(self, hidden_states, inference_params=None, **kwargs):
        kwargs = {
            "seqlen": kwargs.get("seqlen", None),
            "seq_idx": kwargs.get("seq_idx", None),
            "cu_seqlens": kwargs.get("cu_seqlens", None),
        }
        return super().forward(
            hidden_states, inference_params=inference_params, **kwargs
        )

    def step(self, hidden_states, inference_params, **kwargs):
        # Don't use _get_states_from_cache because we want to assert that they exist
        conv_state, ssm_state = inference_params.key_value_memory_dict[
            self.layer_idx
        ]  # init class of Mamba2 accepts layer_idx
        result, conv_state, ssm_state = super().step(
            hidden_states, conv_state, ssm_state
        )

        # Update the state cache in-place
        inference_params.key_value_memory_dict[self.layer_idx][0].copy_(conv_state)
        inference_params.key_value_memory_dict[self.layer_idx][1].copy_(ssm_state)
        return result


def create_block(
    arch,
    d_model,
    d_intermediate=None,
    ssm_cfg=dict(),
    attn_cfg=dict(),
    norm_epsilon=1e-5,
    layer_idx=None,
    residual_in_fp32=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}

    # Mixer
    if arch in ("t", "T"):
        mixer_cls = partial(
            CausalMaskMHA, **attn_cfg, **factory_kwargs, layer_idx=layer_idx
        )
    elif arch in ("m", "M"):
        mixer_cls = partial(
            Mamba2Wrapper, **ssm_cfg, **factory_kwargs, layer_idx=layer_idx
        )
    else:
        raise NotImplementedError

    # MLP
    if arch in ("T", "M"):
        mlp_cls = partial(
            SwiGLU,
            d_intermediate=d_intermediate,
            **factory_kwargs,
        )
    elif arch in ("t", "m"):
        mlp_cls = nn.Identity
    else:
        raise NotImplementedError

    # Normalization
    norm_cls = partial(RMSNorm, eps=norm_epsilon, **factory_kwargs)

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        residual_in_fp32=residual_in_fp32,
    )
    return block


class Block(HNetBlock):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        inference_params=None,
        masking_score=None,
        mixer_kwargs=None,
    ):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )

        if mixer_kwargs is None:
            mixer_kwargs = {}
        hidden_states = self.mixer(
            hidden_states,
            inference_params=inference_params,
            masking_score=masking_score,
            **mixer_kwargs,
        )

        if self.mlp is not None:
            hidden_states, residual = self.norm2(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def step(self, hidden_states, inference_params, residual=None, **kwargs):
        hidden_states, residual = self.norm1(
            hidden_states,
            residual=residual,
            prenorm=True,
            residual_in_fp32=self.residual_in_fp32,
        )
        hidden_states = self.mixer.step(hidden_states, inference_params, **kwargs)
        if self.mlp is not None:
            hidden_states, residual = self.norm2(
                hidden_states,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
            )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
