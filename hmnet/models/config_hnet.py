from dataclasses import dataclass, field
from typing import Any, List, Union


@dataclass
class AttnConfig:

    num_heads: List = field(default_factory=list)
    rotary_emb_dim: List = field(default_factory=list)
    window_size: List = field(default_factory=list)


@dataclass
class SSMConfig:

    d_conv: int = 4
    expand: int = 2
    d_state: int = 128
    chunk_size: int = 256


@dataclass
class HNetConfig:
    arch_layout: List[Any] = field(default_factory=list)
    d_model: List[int] = field(default_factory=list)
    # intermediate dimension for the FFNs (0 indicates no FFN)
    d_intermediate: List[int] = field(default_factory=list)
    vocab_size: int = 256
    ssm_cfg: SSMConfig = field(default_factory=SSMConfig)
    attn_cfg: AttnConfig = field(default_factory=AttnConfig)
    tie_embeddings: bool = False

    @classmethod
    def from_dict(cls, cfg) -> "HNetConfig":
        cfg["ssm_cfg"] = SSMConfig(**cfg.pop("ssm_cfg", {}))
        cfg["attn_cfg"] = AttnConfig(**cfg.pop("attn_cfg", {}))

        return cls(**cfg)
