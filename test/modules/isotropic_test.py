from hmnet.modules.isotropic import Isotropic
from hnet.models.config_hnet import HNetConfig, AttnConfig, SSMConfig
import torch


def test_isotropic_block():
    attn_config = AttnConfig(
        num_heads=[16, 16], rotary_emb_dim=[32, 48], window_size=[-1, 10]
    )
    ssm_config = SSMConfig()
    config = HNetConfig(
        attn_cfg=attn_config,
        ssm_cfg=ssm_config,
        arch_layout=["T2", ["T2"], "T2"],
        d_model=[640, 1280],
        d_intermediate=[1280, 2560],
        vocab_size=256,
        tie_embeddings=False,
    )

    isotropic_layer = Isotropic(
        config=config, pos_idx=0, stage_idx=0, device="cuda", dtype=torch.float32
    )

    x = torch.randn(2, 1200, 640).to("cuda")  # (batch_size, seq_len, d_model)
    mask = torch.ones(2, 1200).to("cuda")  # (batch_size, seq_len)
    output = isotropic_layer(x, mask=mask)
    assert output.shape == (2, 1200, 640)  # Output shape
