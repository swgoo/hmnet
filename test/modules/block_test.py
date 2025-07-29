import torch
from hmnet.modules.block import create_block


def test_create_block():
    block = create_block(
        arch='t',
        d_model=640,
        d_intermediate=256,
        attn_cfg={'num_heads': 8},
        layer_idx=0,
        device='cuda',
        dtype=torch.float32
    )
    seq = torch.rand(2, 1200, 640).to('cuda')  # (batch_size, seq_len, d_model)
    hs, res = block(seq)
    assert hs.shape == (2, 1200, 640)  # Output shape
    assert res.shape == (2, 1200, 640)  # Residual shape

    assert block is not None
    assert hasattr(block, 'forward')  # Ensure it has a forward method
    assert hasattr(block, 'step')  # Ensure it has a step method