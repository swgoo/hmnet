import torch
from hmnet.modules.block import create_block


def test_create_block():
    block = create_block(
        arch="t",
        d_model=640,
        d_intermediate=256,
        attn_cfg={"num_heads": 8},
        layer_idx=0,
        device="cuda",
        dtype=torch.float32,
    )
    x = torch.rand(2, 1200, 640).to("cuda")  # (batch_size, seq_len, d_model)
    masking_score = (torch.ones(2, 1200, 1200) * 0.3).to(
        "cuda"
    )  # (batch_size, seq_len, seq_len)
    hs, res = block.forward(
        x,
        residual=x,
        inference_params=None,
        mixer_kwargs={"masking_score": masking_score},
    )
    assert hs.shape == (2, 1200, 640)
    assert res.shape == (2, 1200, 640)

    # assert block is not None
    assert hasattr(block, "forward")
    assert hasattr(block, "step")


import torch
from torch import nn
from hmnet.modules.dm import MaskingModule, MaskingModuleState, DeChunkMaskLayer


def test_masking_module():
    d_model = 64
    batch_size = 2
    seqlen = 5
    max_seqlen = 10
    max_batch_size = 4

    module = MaskingModule(d_model=d_model, device="cpu", dtype=torch.float32)
    inference_params = MaskingModuleState(
        max_seqlen=max_seqlen, max_batch_size=max_batch_size
    )
    x = torch.randn(batch_size, seqlen, d_model)
    block_score = module.forward(x)
    assert block_score.shape == (batch_size, seqlen, seqlen)
    assert torch.all(block_score.sum(-1) > 0.999)
    assert torch.all(block_score.sum(-1) < 1.001)
    mask = torch.ones(batch_size, seqlen, seqlen)
    boundary_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    cbm_module = DeChunkMaskLayer(window_size=5)
    x = torch.randn(batch_size, seqlen, d_model)
    mask_score = cbm_module.forward(
        boundary_mask=boundary_mask, block_score=block_score, mask=mask
    )
    assert mask_score.shape == (batch_size, seqlen, seqlen)


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
    masking_score = (torch.ones(2, 1200, 1200, requires_grad=True) * 0.3).to(
        "cuda"
    )  # (batch_size, seq_len, seq_len)
    mask = torch.ones(2, 1200).to("cuda")  # (batch_size, seq_len)
    output = isotropic_layer.forward(x, mask=mask, masking_score=masking_score)
    assert output.shape == (2, 1200, 640)  # Output shape


from hmnet.modules.mha import CausalMaskMHA
import torch


def test_causal_block_mask_mha():
    cbmha = CausalMaskMHA(
        d_model=640,
        num_heads=8,
    ).cuda()
    masking_score = (torch.ones(2, 1200, 1200) * 0.3).to(
        "cuda"
    )  # (batch_size, seq_len, seq_len)

    x = torch.ones(2, 1200, 640).to("cuda")  # (batch_size, seq_len, d_model)
    # Simulate a checkpoint loading scenario

    output = cbmha(x, masking_score=masking_score)
    assert output.shape == (2, 1200, 640)  # Output shape should match the input shape


def test_window_tril_score():
    window_size = 3
    full_mask = torch.ones(5, 5)
    window_mask = (
        torch.tril(full_mask).bool()
        * ~torch.tril(full_mask, diagonal=-window_size).bool()
    )

    expected_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(window_mask, expected_mask)
