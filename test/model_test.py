import torch
from hmnet.models.config_hmnet import HMNetConfig
from hmnet.models.hmnet import HMNet
from hnet.models.config_hnet import SSMConfig, AttnConfig
from tqdm import tqdm


@torch.no_grad()
def test_hmnet():
    ssm_config = SSMConfig(chunk_size=256, d_conv=4, d_state=128, expand=2)
    attn_config = AttnConfig(
        num_heads=[16, 16], rotary_emb_dim=[4, 4], window_size=[1023, -1]
    )
    config = HMNetConfig(
        d_model=[64, 128],
        vocab_size=256,
        tie_embeddings=True,
        encoder_attn_cfg=attn_config,
        decoder_attn_cfg=attn_config,
        encoder_ssm_cfg=ssm_config,
        decoder_ssm_cfg=ssm_config,
        arch_layout=["m1T1", ["t1"], "T1m1"],
        d_intermediate=[128, 256],
    )

    # Test forward pass
    batch_size = 2
    seqlen = 5

    # torch.cuda.empty_cache()
    model = HMNet(config=config, device="cuda", stage_idx=0)

    input_tensor = torch.randn(
        batch_size,
        seqlen,
        config.d_model[0],
        device="cuda",
    )
    mask = torch.ones(batch_size, seqlen, dtype=torch.bool, device="cuda")
    output = model(hidden_states=input_tensor, mask=mask)

    assert output[0].shape == (batch_size, seqlen, config.d_model[0])


@torch.no_grad()
def test_step():
    ssm_config = SSMConfig(chunk_size=256, d_conv=8, d_state=128, expand=2)
    attn_config = AttnConfig(
        num_heads=[16, 16], rotary_emb_dim=[4, 4], window_size=[1023, -1]
    )
    config = HMNetConfig(
        d_model=[64, 128],
        vocab_size=256,
        tie_embeddings=True,
        encoder_attn_cfg=attn_config,
        decoder_attn_cfg=attn_config,
        encoder_ssm_cfg=ssm_config,
        decoder_ssm_cfg=ssm_config,
        arch_layout=["T1", ["t1"], "T1"],
        d_intermediate=[128, 256],
    )
    model = HMNet(config=config, device="cuda", stage_idx=0).to("cuda")

    # Test step method
    batch_size = 1
    seqlen = 20
    input_tensor = torch.randn(
        batch_size,
        seqlen,
        config.d_model[0],
        device="cuda",
        dtype=torch.float32,
    )
    inference_params = model.allocate_inference_cache(batch_size, seqlen + 100)

    mask = torch.ones(batch_size, seqlen, device="cuda", dtype=torch.bool)
    output = model.forward(
        hidden_states=input_tensor, mask=mask, inference_params=inference_params
    )
    for _ in tqdm(range(2)):

        current_emb = torch.randn(
            batch_size, 1, config.d_model[0], device="cuda", dtype=torch.float32
        )
        output = model.step(
            hidden_states=current_emb, inference_params=inference_params
        )

    assert output[0].shape == (batch_size, 1, config.d_model[0])
