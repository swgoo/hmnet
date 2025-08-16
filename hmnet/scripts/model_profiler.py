import re
from typing import Iterable
from hmnet.modules.dc import RoutingModuleOutput
from torch import Tensor
import torch


class RoutingProfiler:
    def __init__(self, routing_outputs: list[RoutingModuleOutput]):
        self.routing_outputs = routing_outputs

    def dechunk_all_stage(self):
        result = []
        for ro in reversed(self.routing_outputs):
            result = [self._dechunk_step(pb, ro.boundary_mask) for pb in result]
            result.append(self._dechunk_step(ro.boundary_prob, ro.boundary_mask))
        return reversed(result)

    def _dechunk_step(self, boundary_prob: Tensor, boundary_mask: Tensor) -> Tensor:
        """
        Place values from boundary_prob into positions marked True in boundary_mask.
        boundary_prob: (B, L) float tensor (invalid entries < 0, e.g. -1)
        boundary_mask: (B, S) bool tensor
        returns: (B, S) float tensor with fill -1 for non-assigned positions.
        Example:
        boundary_mask [1,1,0,1,0,0,1,0,0,0,1]
        boundary_prob [0.1,0.2,0.3,0.4,0.5,-1,-1,-1]
        -> [0.1,0.2,-1,0.3,-1,-1,0.4,-1,-1,-1,0.5]
        """
        # assert boundary_prob.shape[-1] <= boundary_mask.shape[-1]
        # normalize dimensions to (B, ...)
        if boundary_prob.dim() == 1:
            boundary_prob = boundary_prob.unsqueeze(0)
        if boundary_mask.dim() == 1:
            boundary_mask = boundary_mask.unsqueeze(0)

        B = boundary_mask.size(0)
        S = boundary_mask.size(1)
        out = torch.full(
            (B, S),
            fill_value=0.0,
            dtype=boundary_prob.dtype,
            device=boundary_prob.device,
        )

        for i in range(B):
            mask_idx = torch.nonzero(boundary_mask[i], as_tuple=False).squeeze(1)
            if mask_idx.numel() == 0:
                continue
            probs = boundary_prob[i]
            # consider probs >= 0 as valid (sentinel values like -1 are skipped)
            # valid = probs[probs >= 0]
            n = min(mask_idx.numel(), probs.numel())
            if n > 0:
                out[i, mask_idx[:n]] = probs[:n]
        return out


class ChunkAttnScoreProfiler:
    def __init__(self, chunked_attn_scores: list[Tensor], boundary_masks: list[Tensor]):
        self.chunked_attn_scores = chunked_attn_scores  # (B, L, L)
        self.boundary_masks = boundary_masks  # (B, S) bool S > L

    def _dechunk_step(self, chunked_attn: Tensor, boundary_mask: Tensor) -> Tensor:
        """
        Dechunk the attention scores based on the boundary mask.
        chunked_attn: (B, L, L) float tensor
        boundary_mask: (B, S) bool tensor
        returns: (B, S, S) float tensor with fill -1 for non-assigned positions.
        """
        # B, L, _ = chunked_attn.shape
        # S = boundary_mask.size(-1)
        # out = torch.full((B, S, S), fill_value=0.0, dtype=chunked_attn.dtype)

        # for i in range(B):
        #     mask_idx = torch.nonzero(boundary_mask[i], as_tuple=False).squeeze(1)
        #     if mask_idx.numel() == 0:
        #         continue
        #     out[i][:, mask_idx] = chunked_attn[i]
        #     out[i][mask_idx] = chunked_attn[i][:, mask_idx]

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
