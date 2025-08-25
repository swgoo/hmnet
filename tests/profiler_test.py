import torch
from hmnet.modules.profiler import (
    RoutingProfiler,
    RoutingModuleOutput,
    ChunkAttnScoreProfiler,
)
from pytest import fixture


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

    dechunked_outputs = RoutingProfiler.dechunk_all_stage(
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
    for i, output in enumerate(dechunked_outputs):
        assert torch.equal(
            output, expected_outputs[i]
        ), f"Stage {i} dechunked output does not match expected output."


def test_chunk_attn_score_profiler(routing_outputs):
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

    all_dechunked_scores = ChunkAttnScoreProfiler.dechunk_all_stage(
        [
            stage_0_chunk_attn_score,
            stage_1_chunk_attn_score,
            stage_2_chunk_attn_score,
        ],
        routing_outputs,
    )

    expected_outputs = [
        stage_0_dechunked,
        stage_1_dechunked,
        stage_2_dechunked,
    ]

    for i, output in enumerate(all_dechunked_scores):
        assert torch.equal(
            output, expected_outputs[i]
        ), f"Stage {i} dechunked output does not match expected output."
        # print(f"Stage {i} dechunked output:\n{output}\n")


@fixture
def routing_outputs():
    stage_2_boundary_mask = torch.tensor([[1, 1, 0, 1]])

    stage_1_boundary_mask = torch.tensor([[1, 1, 0, 1, 1]])

    stage_0_boundary_mask = torch.tensor([[1, 0, 1, 0, 1, 0, 0, 1, 0, 1]])

    return [
        RoutingModuleOutput(
            boundary_prob=None,
            boundary_mask=stage_0_boundary_mask,
            selected_probs=None,
        ),
        RoutingModuleOutput(
            boundary_prob=None,
            boundary_mask=stage_1_boundary_mask,
            selected_probs=None,
        ),
        RoutingModuleOutput(
            boundary_prob=None,
            boundary_mask=stage_2_boundary_mask,
            selected_probs=None,
        ),
    ]


def test_chunk_attn_step():
    attn_score = torch.tensor(
        [
            [11, 12, 13, 14, 15],
            [21, 22, 23, 24, 25],
            [31, 32, 33, 34, 35],
            [41, 42, 43, 44, 45],
            [51, 52, 53, 54, 55],
        ]
    ).unsqueeze(0)
    boundary_mask = torch.tensor([[1, 0, 0, 1, 1]])
    chunked_attn_score = ChunkAttnScoreProfiler._chunk_step(attn_score, boundary_mask)
    assert torch.equal(
        chunked_attn_score, torch.tensor([[[11, 14, 15], [41, 44, 45], [51, 54, 55]]])
    )


def test_chunk_attn_all(routing_outputs):
    attn_score = torch.tensor(
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
    chunked_attn_scores = ChunkAttnScoreProfiler.chunk_all_stage(
        attn_score, routing_outputs
    )
    pass
