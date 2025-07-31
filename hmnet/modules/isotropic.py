import re

import torch.nn as nn
from flash_attn.ops.triton.layer_norm import RMSNorm
from hnet.modules.isotropic import Isotropic as HNetIsotropic
from hnet.models.config_hnet import HNetConfig
from .block import create_block
from hnet.modules.utils import get_seq_idx
from copy import deepcopy


class Isotropic(HNetIsotropic):
    def __init__(
        self,
        config: HNetConfig,
        pos_idx: int,
        stage_idx: int,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(config, pos_idx, stage_idx, **factory_kwargs)

        arch_layout = config.arch_layout
        for _ in range(stage_idx):
            arch_layout = arch_layout[1]
        arch_layout = arch_layout[pos_idx]
        layout_parse = re.findall(r"([mMtT])(\d+)", arch_layout)

        layers = []
        layer_idx = 0
        self.arch_full = []
        for arch, n_layer in layout_parse:
            assert arch in ("m", "M", "t", "T")
            assert n_layer.isdigit()
            layers += [
                create_block(
                    arch,
                    self.d_model,
                    d_intermediate=config.d_intermediate[self.stage_idx],
                    ssm_cfg=self.ssm_cfg,
                    attn_cfg=self.attn_cfg,
                    layer_idx=(layer_idx + i),
                    **factory_kwargs,
                )
                for i in range(int(n_layer))
            ]
            self.arch_full.extend([arch for _ in range(int(n_layer))])
            layer_idx += int(n_layer)

        self.layers = nn.ModuleList(layers)

        self.rmsnorm = RMSNorm(self.d_model, eps=1e-5, **factory_kwargs)

    def forward(
        self,
        hidden_states,
        cu_seqlens=None,
        max_seqlen=None,
        masking_score=None,
        mask=None,
        inference_params=None,
        **mixer_kwargs,
    ):
        assert (mask is not None) or (
            cu_seqlens is not None and max_seqlen is not None
        ), "Either mask or cu_seqlens and max_seqlen must be provided"

        attn_mixer_kwargs = deepcopy(mixer_kwargs)
        ssm_mixer_kwargs = deepcopy(mixer_kwargs)
        if mask is not None:
            packed = False
            assert (
                hidden_states.dim() == 3
            ), "Hidden states must be (B, L, D) in unpacked mode"
        else:
            attn_mixer_kwargs.update(
                {"cu_seqlens": cu_seqlens.int(), "max_seqlen": max_seqlen}
            )
            ssm_mixer_kwargs.update(
                {"seq_idx": get_seq_idx(cu_seqlens, device=hidden_states.device)}
            )
            packed = True

        residual = None
        for layer, arch in zip(self.layers, self.arch_full):
            if arch in ("m", "M"):
                layer_mixer_kwargs = ssm_mixer_kwargs
                if hidden_states.dim() == 2:
                    hidden_states = hidden_states.unsqueeze(0)
                    residual = None if residual is None else residual.unsqueeze(0)
            elif arch in ("t", "T"):
                layer_mixer_kwargs = attn_mixer_kwargs
                if hidden_states.dim() == 3 and packed:
                    hidden_states = hidden_states.squeeze(0)
                    residual = None if residual is None else residual.squeeze(0)
            else:
                # Currently supporting only Mamba2 and MHA
                raise NotImplementedError

            hidden_states, residual = layer(
                hidden_states,
                residual,
                masking_score=masking_score,
                inference_params=inference_params,
                mixer_kwargs=layer_mixer_kwargs,
            )

        # Setting prenorm=False ignores the residual
        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )

        if hidden_states.dim() == 3 and packed:
            hidden_states = hidden_states.squeeze(0)

        if inference_params is not None:
            # here we also explicitly assume the mask is all True
            assert mask.shape[0] == 1, "seqlen_offset handling assumes batch size 1"
            inference_params.seqlen_offset += hidden_states.shape[1]

        return hidden_states

    def step(self, hidden_states, inference_params, **kwargs):
        """
        Assumes hidden_states is (B, 1, D). Steps each of the layers in order, and then steps the main model.
        """
        residual = None

        for layer in self.layers:
            hidden_states, residual = layer.step(
                hidden_states, inference_params, residual=residual, **kwargs
            )

        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )
        inference_params.seqlen_offset += 1

        return hidden_states
