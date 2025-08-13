import torch
from hmnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig

from hmnet.modules.block import create_block
from hmnet.modules.dm import (
    DeChunkAttnScoreLayer,
    ChunkAttnScoreModule,
    ChunkAttnScoreState,
)
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
    # masking_score = (torch.ones(2, 1200, 1200) * 0.3).to(
    #     "cuda"
    # )  # (batch_size, seq_len, seq_len)
    hs, res = block(
        x,
        residual=x,
        inference_params=None,
        # mask_score=masking_score,
    )
    assert hs.shape == (2, 1200, 640)
    assert res.shape == (2, 1200, 640)


def test_masking_module():
    d_model = 64
    batch_size = 2
    seqlen = 5
    max_seqlen = 10
    max_batch_size = 4

    module = ChunkAttnScoreModule(d_model=d_model, device="cuda", dtype=torch.float32)
    inference_params = ChunkAttnScoreState(
        max_seqlen=max_seqlen, max_batch_size=max_batch_size
    )
    x = torch.randn(batch_size, seqlen, d_model, device="cuda")
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
        device="cuda",
    )
    cbm_module = DeChunkAttnScoreLayer(window_size=5)
    x = torch.randn(batch_size, seqlen, d_model)
    mask_score = cbm_module.forward(
        boundary_mask=boundary_mask, chunk_attn_score=block_score, mask=mask
    )


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
    output = isotropic_layer.forward(x, mask=mask)
    assert output.shape == (2, 1200, 640)  # Output shape


def test_causal_mask_mha():
    cbmha = CausalMaskMHA(
        d_model=640,
        num_heads=8,
    ).cuda()

    x = torch.ones(2, 1200, 640).to("cuda")  # (batch_size, seq_len, d_model)
    # Simulate a checkpoint loading scenario
    for _ in range(100):
        output = cbmha(x)
    assert output.shape == (2, 1200, 640)  # Output shape should match the input shape


def test_masking_module_step():
    masking_module = ChunkAttnScoreModule(
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


def test_dechunk_mask_layer_2d():
    batch_size = 3
    seqlen = 5
    window_size = 3
    boundary_mask = torch.tensor(
        [
            [1, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    mask = torch.ones(batch_size, seqlen, seqlen, dtype=torch.float32, device="cuda")
    block_score = torch.rand(batch_size, seqlen, seqlen, device="cuda")

    de_chunk_mask_layer = DeChunkAttnScoreLayer(window_size=window_size)
    mask_score = de_chunk_mask_layer.forward(
        boundary_mask=boundary_mask, chunk_attn_score=block_score, mask=mask
    )


def test_dechunk_mask_layer_1d():
    batch_size = 2
    seqlen = 1
    window_size = 3
    boundary_mask = torch.tensor(
        [
            [1],
            [1],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    mask = torch.ones(batch_size, seqlen, seqlen, dtype=torch.float32, device="cuda")
    block_score = torch.rand(batch_size, seqlen, seqlen, device="cuda")

    de_chunk_mask_layer = DeChunkAttnScoreLayer(window_size=window_size)
    mask_score = de_chunk_mask_layer.forward(
        boundary_mask=boundary_mask, chunk_attn_score=block_score, mask=mask
    )


def test_dechunk_mask_layer_step():
    batch_size = 1
    seqlen = 5
    chunk_seqlen = 2
    window_size = 3
    n_step = 3
    boundary_mask_prefill = torch.tensor(
        [
            [1, 0, 1, 0, 0],
        ],
        dtype=torch.bool,
        device="cuda",
    )
    mask = torch.ones(batch_size, seqlen, seqlen, dtype=torch.float32, device="cuda")
    chunk_attn_score = torch.rand(batch_size, chunk_seqlen, chunk_seqlen, device="cuda")

    de_chunk_mask_layer = DeChunkAttnScoreLayer(window_size=window_size)
    inference_params = de_chunk_mask_layer.allocate_inference_cache(
        batch_size=batch_size,
        max_seqlen=seqlen + 10,
        device="cuda",
        dtype=torch.float32,
    )

    de_chunk_mask_layer.forward(
        boundary_mask=boundary_mask_prefill,
        chunk_attn_score=chunk_attn_score,
        mask=mask,
        inference_params=inference_params,
    )
    cur_chunk_seqlen = chunk_seqlen
    result_boundary_mask = boundary_mask_prefill.squeeze(0)
    for _ in range(n_step):
        boundary_mask = torch.tensor([False]).to("cuda")
        result_boundary_mask = torch.cat([result_boundary_mask, boundary_mask])
        chunk_attn_score = torch.rand(
            0,
            1,
            cur_chunk_seqlen,
        ).to("cuda")

        _ = de_chunk_mask_layer.step(boundary_mask, chunk_attn_score, inference_params)
        boundary_mask = torch.tensor([True]).to("cuda")
        result_boundary_mask = torch.cat([result_boundary_mask, boundary_mask])
        cur_chunk_seqlen += 1
        chunk_attn_score = torch.rand(
            batch_size,
            1,
            cur_chunk_seqlen,
        ).to("cuda")
        _ = de_chunk_mask_layer.step(
            boundary_mask=boundary_mask,
            chunk_attn_score=chunk_attn_score,
            inference_params=inference_params,
        )

    assert torch.equal(
        result_boundary_mask, inference_params.last_boundary_mask.squeeze(0)
    )
