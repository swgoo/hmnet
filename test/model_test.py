import torch
from hmnet.models.hmnet import HMNet, HMNetState
from hnet.models.config_hnet import HNetConfig, SSMConfig, AttnConfig


def test_hmnet():
    ssm_config = SSMConfig(chunk_size=256, d_conv=4, d_state=128, expand=2)
    attn_config = AttnConfig(
        num_heads=[16, 16], rotary_emb_dim=[4, 4], window_size=[1023, -1]
    )
    config = HNetConfig(
        d_model=[64, 128],
        vocab_size=256,
        tie_embeddings=True,
        ssm_cfg=ssm_config,
        attn_cfg=attn_config,
        arch_layout=["t1", ["T1"], "t1"],
        d_intermediate=[128, 256],
    )
    model = HMNet(config=config, device="cuda", stage_idx=0).to(
        "cuda", dtype=torch.bfloat16
    )

    # Test forward pass
    batch_size = 2
    seqlen = 5
    input_tensor = torch.randn(
        batch_size, seqlen, config.d_model[0], device="cuda", dtype=torch.bfloat16
    )
    mask = torch.ones(batch_size, seqlen, dtype=torch.bool, device="cuda")
    output = model.forward(hidden_states=input_tensor, mask=mask)

    assert output.shape == (batch_size, seqlen, config.d_model[0])

    # Test inference cache allocation
    inference_cache = model.allocate_inference_cache(batch_size, seqlen)
    assert isinstance(inference_cache, HMNetState)
