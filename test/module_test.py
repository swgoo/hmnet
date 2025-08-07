import torch
from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig

from hmnet.modules.block import create_block
from hmnet.modules.dm import DeChunkMaskLayer, MaskingModule, MaskingModuleState
from hmnet.modules.isotropic import Isotropic
from hmnet.modules.mha import CausalMaskMHA


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
    hs, res = block(
        x,
        residual=x,
        inference_params=None,
        masking_score=masking_score,
    )
    assert hs.shape == (2, 1200, 640)
    assert res.shape == (2, 1200, 640)


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


def test_causal_mask_mha():
    cbmha = CausalMaskMHA(
        d_model=640,
        num_heads=8,
    ).cuda()

    x = torch.ones(2, 1200, 640).to("cuda")  # (batch_size, seq_len, d_model)
    # Simulate a checkpoint loading scenario
    for _ in range(100):
        masking_score = (torch.rand(2, 1200, 1200)).to(
            "cuda"
        )  # (batch_size, seq_len, seq_len)
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


def test_masking_module():

    masking_module = MaskingModule(
        d_model=64,
        device="cuda",
        dtype=torch.float32,
    )
    batch_size = 2
    seqlen = 5
    x = torch.randn(batch_size, seqlen, 64).to("cuda")
    block_score = masking_module.forward(x)
    assert block_score.shape == (batch_size, seqlen, seqlen)


def test_masking_module_step():
    masking_module = MaskingModule(
        d_model=64,
        device="cuda",
        dtype=torch.float32,
    )
    batch_size = 1
    seqlen = 5
    n_step = 3
    max_seqlen = seqlen + n_step
    x = torch.randn(batch_size, seqlen, 64).to("cuda")
    inference_params = masking_module.allocate_inference_cache(
        batch_size=batch_size, max_seqlen=max_seqlen, dtype=torch.float32, device="cuda"
    )
    masking_module.forward(x, inference_params=inference_params)
    empty_query = torch.randn(0, 1, 64).to("cuda")
    query = torch.randn(1, 1, 64).to("cuda")

    for _ in range(n_step):
        block_score = masking_module.step(query, inference_params)
        block_score = masking_module.step(empty_query, inference_params)

    assert block_score.shape == (batch_size, 1, seqlen + n_step)


def test_dechunk_mask_layer():
    batch_size = 2
    seqlen = 5
    window_size = 3
    boundary_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    mask = torch.ones(batch_size, seqlen, seqlen, dtype=torch.float32)
    block_score = torch.rand(batch_size, seqlen, seqlen)

    de_chunk_mask_layer = DeChunkMaskLayer(window_size=window_size)
    mask_score = de_chunk_mask_layer.forward(
        boundary_mask=boundary_mask, block_score=block_score, mask=mask
    )
    assert mask_score.shape == (batch_size, seqlen, seqlen)


def test_dechunk_mask_layer_step():
    batch_size = 1
    seqlen = 5
    window_size = 3
    n_step = 3
    boundary_mask = torch.tensor(
        [
            [1, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    mask = torch.ones(batch_size, seqlen, seqlen, dtype=torch.float32)
    block_score = torch.rand(batch_size, seqlen, seqlen)

    de_chunk_mask_layer = DeChunkMaskLayer(window_size=window_size)
    inference_params = de_chunk_mask_layer.allocate_inference_cache(
        batch_size=batch_size,
        max_seqlen=seqlen + 10,
        device="cpu",
        dtype=torch.float32,
    )

    mask_score = de_chunk_mask_layer.forward(
        boundary_mask=boundary_mask,
        block_score=block_score,
        mask=mask,
        inference_params=inference_params,
    )

    empty_query = torch.randn(0, 1, 64).to("cpu")
    query = torch.randn(1, 1, 64).to("cpu")

    block_score = torch.rand(
        batch_size,
        1,
        seqlen,
    ).to("cpu")
    for _ in range(n_step):
        boundary_mask = torch.tensor([False])
        mask_score = de_chunk_mask_layer.step(
            boundary_mask, block_score, inference_params
        )
        boundary_mask = torch.tensor([True])
        block_score = torch.concat(
            [block_score, torch.rand(batch_size, 1, 1).to("cpu")], dim=-1
        )
        mask_score = de_chunk_mask_layer.step(
            boundary_mask=boundary_mask,
            block_score=block_score,
            inference_params=inference_params,
        )

    assert mask_score.shape == (batch_size, 1, seqlen + 2 * n_step)
