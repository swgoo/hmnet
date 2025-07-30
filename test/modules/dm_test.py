import torch
from torch import nn
def test_window_tril_score():
    window_size = 3
    full_mask = torch.ones(5, 5)
    window_mask = torch.tril(full_mask).bool() * ~torch.tril(full_mask, diagonal=-window_size).bool()

    expected_mask = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1]
    ], dtype=torch.bool)
    assert torch.equal(window_mask, expected_mask)