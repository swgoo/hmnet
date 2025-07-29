from hmnet.modules.isotropic import Isotropic
from hnet.models.config_hnet import HNetConfig, AttnConfig, SSMConfig
import torch

def test_isotropic_block():
    attn_config = AttnConfig()
    ssm_config = SSMConfig()
    config = HNetConfig(
        attn_cfg=attn_config,
        ssm_cfg=ssm_config,
        arch_layout= ["M2", ["T2"], "M2"],
        d_model=[640, 1280],
        d_intermediate=[1280, 2560],
        vocab_size=256,
        tie_embeddings=False
    )

    isotropic_layer = Isotropic(
        config=config,
        pos_idx=0,
        stage_idx=0,
        device='cuda',
        dtype=torch.float32
    )