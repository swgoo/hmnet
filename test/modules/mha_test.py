from hmnet.modules.mha import CausalBlockMaskMHA
import torch


def test_causal_block_mask_mha():
    cbmha = CausalBlockMaskMHA(
        d_model=640,
        num_heads=8,
    ).cuda()

    x = torch.randn(2, 1200, 640).to("cuda")  # (batch_size, seq_len, d_model)
    # Simulate a checkpoint loading scenario

    output = cbmha(x)
    assert output.shape == (2, 1200, 640)  # Output shape should match the input shape
