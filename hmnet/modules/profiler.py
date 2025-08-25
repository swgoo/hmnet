from hmnet.modules.dc import RoutingModuleOutput
from torch import Tensor
import torch


class RoutingProfiler:
    @classmethod
    def dechunk_all_stage(cls, routing_outputs: list[RoutingModuleOutput]):
        for i in range(len(routing_outputs)):
            routing_outputs[i].boundary_prob = routing_outputs[i].boundary_prob[..., -1]
        result = []
        for ro in reversed(routing_outputs):
            result = [cls._dechunk_step(pb, ro.boundary_mask) for pb in result]
            result.append(cls._dechunk_step(ro.boundary_prob, ro.boundary_mask))
        return list(reversed(result))

    @classmethod
    def _dechunk_step(cls, boundary_prob: Tensor, boundary_mask: Tensor) -> Tensor:
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
    @classmethod
    def _dechunk_step(cls, chunked_attn: Tensor, boundary_mask: Tensor) -> Tensor:
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

    @classmethod
    def dechunk_all_stage(
        cls,
        chunked_attn_scores: list[Tensor],
        routing_outputs: list[RoutingModuleOutput],
    ):
        result = []
        iter_list = list(zip(chunked_attn_scores, routing_outputs))
        iter_list = reversed(iter_list)
        for chunked_attn, routing_output in iter_list:
            result = [
                cls._dechunk_step(pb, routing_output.boundary_mask) for pb in result
            ]
            result.append(cls._dechunk_step(chunked_attn, routing_output.boundary_mask))
        return list(reversed(result))

    @classmethod
    def _chunk_step(cls, attn_score: Tensor, boundary_mask: Tensor):
        """
        Chunk a single step of attention scores.
        attn_score: (B, L, L)
        boundary_mask: (B, S)
        """
        boundary_mask = boundary_mask.bool()
        num_tokens = boundary_mask.sum(dim=-1)
        next_max_seqlen = int(num_tokens.max())

        device = attn_score.device
        L = attn_score.shape[-2]
        token_idx = (
            torch.arange(L, device=device)[None, :] + (~boundary_mask).long() * L
        )
        seq_sorted_indices = torch.argsort(token_idx, dim=1)

        selected_queries = torch.gather(
            attn_score,
            dim=-2,
            index=seq_sorted_indices[:, :next_max_seqlen, None].expand(
                -1, -1, attn_score.shape[-1]
            ),
        )
        next_attn_score = torch.gather(
            selected_queries,
            dim=-1,
            index=seq_sorted_indices[:, None, :next_max_seqlen].expand(
                -1, selected_queries.size(-2), -1
            ),
        )
        return next_attn_score

    @classmethod
    def chunk_all_stage(
        cls,
        target_attn_score: Tensor,
        routing_outputs: list[RoutingModuleOutput],
    ):
        """
        Chunk all stages of attention scores.
        attn_score: (B, L, L)
        routing_outputs: list[RoutingModuleOutput]
        """
        result = []
        for routing_output in routing_outputs:
            cur_dechunked_attn = target_attn_score if not result else result[-1]
            result.append(
                cls._chunk_step(cur_dechunked_attn, routing_output.boundary_mask)
            )
        return result
