from dataclasses import dataclass, field
from typing import Any, List
from hnet.models.config_hnet import SSMConfig, AttnConfig, HNetConfig


@dataclass
class HMNetConfig:
    arch_layout: List[Any] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    encoder_ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    encoder_attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    decoder_ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    decoder_attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False

    @classmethod
    def from_dict(cls, data: dict) -> "HMNetConfig":
        cfg = dict(data)  # shallow copy

        # pop & instantiate nested configs
        cfg["encoder_ssm_cfg"] = SSMConfig(**cfg.pop("encoder_ssm_cfg", {}))
        cfg["encoder_attn_cfg"] = AttnConfig(**cfg.pop("encoder_attn_cfg", {}))
        cfg["decoder_ssm_cfg"] = SSMConfig(**cfg.pop("decoder_ssm_cfg", {}))
        cfg["decoder_attn_cfg"] = AttnConfig(**cfg.pop("decoder_attn_cfg", {}))

        return cls(**cfg)

    def encoder_hnet_config(self):
        return HNetConfig(
            arch_layout=self.arch_layout,
            d_model=self.d_model,
            d_intermediate=self.d_intermediate,
            ssm_cfg=self.encoder_ssm_cfg,
            attn_cfg=self.encoder_attn_cfg,
            vocab_size=self.vocab_size,
            tie_embeddings=self.tie_embeddings,
        )

    def decoder_hnet_config(self):
        return HNetConfig(
            arch_layout=self.arch_layout,
            d_model=self.d_model,
            d_intermediate=self.d_intermediate,
            ssm_cfg=self.decoder_ssm_cfg,
            attn_cfg=self.decoder_attn_cfg,
            vocab_size=self.vocab_size,
            tie_embeddings=self.tie_embeddings,
        )
