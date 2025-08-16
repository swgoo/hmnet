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
    def __init__(self, chunked_outputs: Iterable[Tensor]):
        self.chunked_outputs = chunked_outputs

    def profile(self):
        # Implement profiling logic here
        pass
