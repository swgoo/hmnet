import pytest
import torch

# from src.module import *


@pytest.fixture
def fixture_example() -> int:
    return 42


def test_dataset(fixture_example):
    assert fixture_example == 42


def test_attn_score():
    boundary_mask = torch.tensor(
        [[1, 0, 1, 0], [1, 1, 0, 0]], dtype=torch.bool
    )  # (B, L)
    block_score = torch.tensor(
        [[[0.1, 0.2]], [[0.4, 0.5]]], dtype=torch.float32
    )  # (B, num_query_chunks, num_key_chunks)
    # block_score = torch.tensor(
    #     [[[0.1, 0.2], [0.3, 0.4]], [[0.4, 0.5], [0.6, 0.7]]], dtype=torch.float32
    # )  # (B, num_query_chunks, num_key_chunks)
    plug_back_idx = torch.cumsum(boundary_mask, dim=1) - 1  # (B, L)
    # block_mask_score: (B, L, L) or (B, 1, L)?
    if block_score.size(1) == 1:
        # Inference mode: block_score shape is [B, 1, num_key_chunks]
        # Gather along dim=2 to map key indices
        block_mask_score = torch.gather(
            block_score,
            dim=2,
            index=plug_back_idx.unsqueeze(1).expand(-1, 1, plug_back_idx.size(1)),
        )
        assert block_mask_score.shape == (2, 1, 4)  # (B, L, L
    else:
        # Prefill mode: block_score shape is [B, num_query_chunks, num_key_chunks]
        # First gather along query dimension using plug_back_idx
        tmp = torch.gather(
            block_score,
            dim=1,
            index=plug_back_idx.unsqueeze(-1).expand(
                -1, plug_back_idx.size(1), block_score.size(2)
            ),
        )
        # Then gather along key dimension using plug_back_idx
        block_mask_score = torch.gather(
            tmp,
            dim=2,
            index=plug_back_idx.unsqueeze(1).expand(
                -1, tmp.size(1), plug_back_idx.size(1)
            ),
        )
        assert block_mask_score.shape == (2, 4, 4)  # (B, L, L)


def test_ckpt_loading():
    import torch

    ckpt = torch.load("../ckpts/hnet_2stage_XL.pt", map_location="cpu")
    for k, v in ckpt.items():
        print(k, v.shape)
