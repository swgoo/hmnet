from hmnet.modules.dc import RoutingModuleOutput
from torch import Tensor
import torch


class RoutingProfiler:
    def __init__(self, routing_outputs: list[RoutingModuleOutput]):
        self.routing_outputs = routing_outputs
        for i in range(len(routing_outputs)):
            routing_outputs[i].boundary_prob = routing_outputs[i].boundary_prob[..., -1]

    def dechunk_all_stage(self):
        result = []
        for ro in reversed(self.routing_outputs):
            result = [self._dechunk_step(pb, ro.boundary_mask) for pb in result]
            result.append(self._dechunk_step(ro.boundary_prob, ro.boundary_mask))
        return reversed(result)

    def _dechunk_step(self, boundary_prob: Tensor, boundary_mask: Tensor) -> Tensor:
        B, L = boundary_prob.shape
        # remove boundary_prob < 0.5
        if boundary_prob.size(-1) == boundary_mask.size(-1):
            token_idx = (
                torch.arange(boundary_prob.size(-1), device=boundary_prob.device)[
                    None, :
                ]
                + (~boundary_mask).long() * L
            )
            seq_sorted_indices = torch.argsort(token_idx, dim=1)
            boundary_prob = torch.gather(
                boundary_prob, dim=1, index=seq_sorted_indices[:, :L]
            )

        plug_back_idx = torch.cumsum(boundary_mask.long(), dim=-1) - 1  # (B, L)
        return torch.gather(
            boundary_prob,  # (B, L)
            dim=-1,
            index=plug_back_idx,
        )


class ChunkAttnScoreProfiler:
    def __init__(self, chunked_attn_scores: list[Tensor], boundary_masks: list[Tensor]):
        self.chunked_attn_scores = chunked_attn_scores  # (B, L, L)
        self.boundary_masks = boundary_masks  # (B, S) bool S > L

    def _dechunk_step(self, chunked_attn: Tensor, boundary_mask: Tensor) -> Tensor:
        plug_back_idx = torch.cumsum(boundary_mask.long(), dim=-1) - 1  # (B, L)
        selected_queries = torch.gather(
            chunked_attn,
            dim=-2,
            index=plug_back_idx.unsqueeze(-1).expand(
                -1, plug_back_idx.size(-1), chunked_attn.size(-1)
            ),
        )
        out = torch.gather(
            selected_queries,
            dim=-1,
            index=plug_back_idx.unsqueeze(-2).expand(
                -1, selected_queries.size(-2), plug_back_idx.size(-1)
            ),
        )
        return out

    def dechunk_all_stage(self):
        result = []
        iter_list = list(zip(self.chunked_attn_scores, self.boundary_masks))
        iter_list = reversed(iter_list)
        for chunked_attn, boundary_mask in iter_list:
            result = [self._dechunk_step(pb, boundary_mask) for pb in result]
            result.append(self._dechunk_step(chunked_attn, boundary_mask))
        return reversed(result)
