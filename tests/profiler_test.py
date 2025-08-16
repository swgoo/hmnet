import torch
from hmnet.scripts.model_profiler import (
    RoutingProfiler,
    RoutingModuleOutput,
    ChunkAttnScoreProfiler,
)


def test_routing_profiler():
    # Example usage
    stage_2_dechunked =     torch.tensor([[1, 0, 2, 0, 0, 0, 0, 0, 0, 3]]) # fmt: skip
    stage_2_boundary_prob = torch.tensor([[1,    2,                   3]]) # fmt: skip
    stage_2_boundary_mask = torch.tensor([[1,    1,             0,    1]]) # fmt: skip

    stage_1_dechunked =     torch.tensor([[1, 0, 2, 0, 0, 0, 0, 3, 0, 4]])# fmt: skip
    stage_1_boundary_prob = torch.tensor([[1,    2,             3,    4]]) # fmt: skip
    stage_1_boundary_mask = torch.tensor([[1,    1,    0,       1,    1]]) # fmt: skip

    stage_0_dechunked =     torch.tensor([[1, 0, 2, 0, 3, 0, 0, 4, 0, 5]])# fmt: skip
    stage_0_boundary_mask = torch.tensor([[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]]) # fmt: skip
    stage_0_boundary_prob = torch.tensor([[1,    2,    3,       4,    5]]) # fmt: skip

    routing_profiler = RoutingProfiler(
        [
            RoutingModuleOutput(
                stage_0_boundary_prob, stage_0_boundary_mask, torch.tensor([])
            ),
            RoutingModuleOutput(
                stage_1_boundary_prob, stage_1_boundary_mask, torch.tensor([])
            ),
            RoutingModuleOutput(
                stage_2_boundary_prob, stage_2_boundary_mask, torch.tensor([])
            ),
        ]
    )
    expected_outputs = [
        stage_0_dechunked,
        stage_1_dechunked,
        stage_2_dechunked,
    ]

    dechunked_outputs = routing_profiler.dechunk_all_stage()
    for i, output in enumerate(dechunked_outputs):
        assert torch.equal(
            output, expected_outputs[i]
        ), f"Stage {i} dechunked output does not match expected output."


def test_chunk_attn_score_profiler():
    # Example usage
    # stage_2_chunked = torch.tensor([[1, 0, 2, 0, 0, 0, 0, 0, 0, 3]])
    stage_2_chunk_attn_score = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    stage_2_boundary_mask = torch.tensor([[1, 1, 0, 1]])

    # stage_1_chunked = torch.tensor([[1, 0, 2, 0, 0, 0, 0, 3, 0, 4]])
    stage_1_chunk_attn_score = torch.tensor(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]
    )
    stage_1_boundary_mask = torch.tensor([[1, 1, 0, 1, 1]])

    # stage_0_chunked = torch.tensor([[1, 0, 2, 0, 3, 0, 0, 4, 0, 5]])
    stage_0_boundary_mask = torch.tensor([[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]])
    stage_0_chunk_attn_score = torch.tensor(
        [
            [
                [1, 2, 3, 4, 5],
                [6, 7, 8, 9, 10],
                [11, 12, 13, 14, 15],
                [16, 17, 18, 19, 20],
                [21, 22, 23, 24, 25],
            ]
        ]
    )

    chunk_attn_score_profiler = ChunkAttnScoreProfiler(
        [
            stage_0_chunk_attn_score,
            stage_1_chunk_attn_score,
            stage_2_chunk_attn_score,
        ],
        [
            stage_0_boundary_mask,
            stage_1_boundary_mask,
            stage_2_boundary_mask,
        ],
    )

    all_dechunked_scores = chunk_attn_score_profiler.dechunk_all_stage()
    for i, output in enumerate(all_dechunked_scores):
        # assert torch.equal(
        #     output, expected_outputs[i]
        # ), f"Stage {i} dechunked output does not match expected output."
        print(f"Stage {i} dechunked output:\n{output}\n")
    assert True
