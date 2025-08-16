import torch
from hmnet.modules.profiler import (
    RoutingProfiler,
    RoutingModuleOutput,
    ChunkAttnScoreProfiler,
)


def test_routing_profiler():
    stage_2_dechunked =     torch.tensor([[1, 1, 2, 2, 2, 2, 2, 2, 2, 3]]) # fmt: skip
    stage_2_boundary_prob = torch.tensor([[1,    2,             0,    3]]).unsqueeze(-1).expand(-1,-1,2) # fmt: skip
    stage_2_boundary_mask = torch.tensor([[1,    1,             0,    1]]).bool() # fmt: skip

    stage_1_dechunked =     torch.tensor([[1, 1, 2, 2, 2, 2, 2, 3, 3, 4]])# fmt: skip
    stage_1_boundary_prob = torch.tensor([[1,    2,    0,       3,    4]]).unsqueeze(-1).expand(-1,-1,2) # fmt: skip
    stage_1_boundary_mask = torch.tensor([[1,    1,    0,       1,    1]]).bool() # fmt: skip

    stage_0_dechunked =     torch.tensor([[1, 1, 2, 2, 3, 3, 3, 4, 4, 5]])# fmt: skip
    stage_0_boundary_mask = torch.tensor([[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]]).bool() # fmt: skip
    stage_0_boundary_prob = torch.tensor([[1, 0, 2, 0, 3, 0, 0, 4, 0, 5]]).unsqueeze(-1).expand(-1,-1,2) # fmt: skip

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
    stage_2_dechunked = torch.tensor(
        [
            [
                [1, 1, 2, 2, 2, 2, 2, 2, 2, 3],
                [1, 1, 2, 2, 2, 2, 2, 2, 2, 3],
                [4, 4, 5, 5, 5, 5, 5, 5, 5, 6],
                [4, 4, 5, 5, 5, 5, 5, 5, 5, 6],
                [4, 4, 5, 5, 5, 5, 5, 5, 5, 6],
                [4, 4, 5, 5, 5, 5, 5, 5, 5, 6],
                [4, 4, 5, 5, 5, 5, 5, 5, 5, 6],
                [4, 4, 5, 5, 5, 5, 5, 5, 5, 6],
                [4, 4, 5, 5, 5, 5, 5, 5, 5, 6],
                [7, 7, 8, 8, 8, 8, 8, 8, 8, 9],
            ]
        ]
    )
    stage_2_chunk_attn_score = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
    stage_2_boundary_mask = torch.tensor([[1, 1, 0, 1]])

    stage_1_dechunked = torch.tensor(
        [
            [
                [1, 1, 2, 2, 2, 2, 2, 3, 3, 4],
                [1, 1, 2, 2, 2, 2, 2, 3, 3, 4],
                [5, 5, 6, 6, 6, 6, 6, 7, 7, 8],
                [5, 5, 6, 6, 6, 6, 6, 7, 7, 8],
                [5, 5, 6, 6, 6, 6, 6, 7, 7, 8],
                [5, 5, 6, 6, 6, 6, 6, 7, 7, 8],
                [5, 5, 6, 6, 6, 6, 6, 7, 7, 8],
                [9, 9, 10, 10, 10, 10, 10, 11, 11, 12],
                [9, 9, 10, 10, 10, 10, 10, 11, 11, 12],
                [13, 13, 14, 14, 14, 14, 14, 15, 15, 16],
            ]
        ]
    )
    stage_1_chunk_attn_score = torch.tensor(
        [[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]
    )
    stage_1_boundary_mask = torch.tensor([[1, 1, 0, 1, 1]])

    stage_0_dechunked = torch.tensor(
        [
            [
                [1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
                [1, 1, 2, 2, 3, 3, 3, 4, 4, 5],
                [6, 6, 7, 7, 8, 8, 8, 9, 9, 10],
                [6, 6, 7, 7, 8, 8, 8, 9, 9, 10],
                [11, 11, 12, 12, 13, 13, 13, 14, 14, 15],
                [11, 11, 12, 12, 13, 13, 13, 14, 14, 15],
                [11, 11, 12, 12, 13, 13, 13, 14, 14, 15],
                [16, 16, 17, 17, 18, 18, 18, 19, 19, 20],
                [16, 16, 17, 17, 18, 18, 18, 19, 19, 20],
                [21, 21, 22, 22, 23, 23, 23, 24, 24, 25],
            ]
        ]
    )
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

    expected_outputs = [
        stage_0_dechunked,
        stage_1_dechunked,
        stage_2_dechunked,
    ]

    all_dechunked_scores = chunk_attn_score_profiler.dechunk_all_stage()
    for i, output in enumerate(all_dechunked_scores):
        assert torch.equal(
            output, expected_outputs[i]
        ), f"Stage {i} dechunked output does not match expected output."
        # print(f"Stage {i} dechunked output:\n{output}\n")
