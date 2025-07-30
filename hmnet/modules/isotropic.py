import re

import torch.nn as nn
from flash_attn.ops.triton.layer_norm import RMSNorm
from hnet.modules.isotropic import Isotropic as HNetIsotropic
from hnet.models.config_hnet import HNetConfig
from .block import create_block


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

    def step(self, hidden_states, inference_params, block_mask=None, score_mod=None):
        """
        Assumes hidden_states is (B, 1, D). Steps each of the layers in order, and then steps the main model.
        """
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer.step(
                hidden_states, inference_params, residual=residual
            )

        hidden_states = self.rmsnorm(
            hidden_states, residual=residual, prenorm=False, residual_in_fp32=True
        )
        inference_params.seqlen_offset += 1

        return hidden_states
