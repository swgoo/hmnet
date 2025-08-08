from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn


@dataclass
class ChunkAttnScoreState:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_memory: Tensor | None = None
    last_query: Tensor | None = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.key_memory is not None:
            self.key_memory.zero_()
        if self.last_query is not None:
            self.last_query.zero_()


def _update_k_cache(k: Tensor, inference_params: ChunkAttnScoreState):
    """k: (batch_size, seqlen, d_model) tensor"""
    # Pre-allocate memory for key-values for inference.
    d_model = k.shape[-1]
    if inference_params.key_memory is None:
        k_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            d_model,
            dtype=k.dtype,
            device=k.device,
        )
        inference_params.key_memory = k_cache
    else:
        k_cache = inference_params.key_memory
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + k.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + k.shape[1]
    assert batch_end <= k_cache.shape[0]
    assert sequence_end <= k_cache.shape[1]
    assert k_cache is not None
    k_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = k
    return k_cache[batch_start:batch_end, :sequence_end, ...]


class ChunkAttnScoreModule(nn.Module):

    def __init__(
        self,
        d_model,
        softmax_scale=None,
        device=None,
        dtype=None,
    ):
        self.d_model = d_model
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / np.sqrt(d_model)
        )

        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.Wqk = nn.Linear(d_model, 2 * d_model, bias=False, **factory_kwargs)

    def allocate_inference_cache(
        self, batch_size, max_seqlen, device, dtype=None
    ) -> ChunkAttnScoreState:
        dtype = self.Wqk.weight.dtype if dtype is None else dtype
        device = self.Wqk.weight.device if device is None else device
        return ChunkAttnScoreState(
            max_seqlen=max_seqlen,
            max_batch_size=batch_size,
            key_memory=torch.empty(
                batch_size,
                max_seqlen,
                self.d_model,
                device=device,
                dtype=dtype,
            ),
        )

    def _update_k_cache(self, k, inference_params: ChunkAttnScoreState):
        """Update the key cache with the new k tensor."""
        return _update_k_cache(k, inference_params)

    def forward(
        self,
        hidden_states: Tensor,
        inference_params: ChunkAttnScoreState | None = None,
        **kwargs,
    ):

        q, k = self._project(hidden_states)
        attn_score = self._attention_score(q, k)

        if inference_params is not None:
            self._update_k_cache(k, inference_params)
            inference_params.seqlen_offset += int(k.shape[-2])
            inference_params.last_query = q[:, -1, :]

        return attn_score

    def _project(self, hidden_states: Tensor):
        """Project hidden states to query and key tensors."""
        qk = self.Wqk(hidden_states)
        qk = rearrange(qk, "... (two d) -> ... two d", two=2, d=self.d_model)
        q, k = qk[:, :, 0], qk[:, :, 1]
        return q, k

    def _attention_score(self, q, k):
        attn_score = einsum(
            F.normalize(q, dim=-1),
            F.normalize(k, dim=-1),
            "... l d, ... m d -> ... l m",
        )
        attn_score = torch.softmax(attn_score * self.softmax_scale, dim=-1)
        return attn_score

    def step(self, hidden_states: Tensor, inference_params: ChunkAttnScoreState):
        if hidden_states.shape[0] > 0:
            q, k = self._project(hidden_states)
            k_cache = self._update_k_cache(k, inference_params)
            inference_params.last_query = q.squeeze(-2)
            inference_params.seqlen_offset += 1
        else:
            q = inference_params.last_query.unsqueeze(-2)
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + q.shape[0]
            k_cache = inference_params.key_memory[
                batch_start:batch_end, : inference_params.seqlen_offset, :
            ]

        chunk_attn_score = self._attention_score(
            q.to(device=hidden_states.device),
            k_cache.to(device=hidden_states.device),
        )
        return chunk_attn_score


@dataclass
class DeChunkAttnScoreState:
    last_mask_score: Tensor  # (batch_size, seqlen)
    last_boundary_mask: Tensor  # (batch_size, seqlen)


class DeChunkAttnScoreLayer(nn.Module):

    def __init__(self, window_size: int, threshold=0.5):
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkAttnScoreState(
            last_mask_score=torch.zeros(
                batch_size, max_seqlen, device=device, dtype=dtype
            ),
            last_boundary_mask=torch.zeros(
                batch_size, max_seqlen, device=device, dtype=torch.bool
            ),
        )

    def forward(
        self,
        boundary_mask: Tensor,
        chunk_attn_score: Tensor,
        cu_seqlens=None,
        mask=None,
        inference_params: DeChunkAttnScoreState | None = None,
    ) -> Tensor:

        if inference_params is None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is not provided"
            assert boundary_mask[
                :, 0
            ].all(), "First token must be a boundary if running prefill"

        if cu_seqlens is not None:
            raise NotImplementedError(
                "CausalBlockMask does not support cu_seqlens yet. Please use mask instead."
            )

        chunk_attn_score = torch.clamp(chunk_attn_score, min=1e-4, max=1.0 - 1e-4)

        plug_back_idx = torch.cumsum(boundary_mask.to(torch.int64), dim=1) - 1  # (B, L)
        selected_queries = torch.gather(
            chunk_attn_score,
            dim=-2,
            index=plug_back_idx.unsqueeze(-1).expand(
                -1, plug_back_idx.size(-1), chunk_attn_score.size(-1)
            ),
        )
        mask_score = torch.gather(
            selected_queries,
            dim=-1,
            index=plug_back_idx.unsqueeze(-1).expand(
                -1, selected_queries.size(-2), plug_back_idx.size(-1)
            ),
        )
        if inference_params is not None:
            inference_params.last_mask_score = mask_score[..., -1, :]
            inference_params.last_boundary_mask = boundary_mask

        return mask_score

    def step(
        self,
        boundary_mask: Tensor,
        chunk_attn_score: Tensor,
        inference_params: DeChunkAttnScoreState,
    ) -> Tensor:
        prev_mask = inference_params.last_mask_score.unsqueeze(-2)  # (B, 1, seq_len)
        current_mask_score = torch.cat(
            [prev_mask, prev_mask[..., -1:]], dim=-1
        )  # (B, 1, seq_len+1)

        current_boundary_mask = torch.concat(
            [
                inference_params.last_boundary_mask,
                boundary_mask.unsqueeze(-1),
            ],
            dim=-1,
        )

        if boundary_mask.sum() > 0:
            current_block_score = torch.zeros_like(
                current_boundary_mask,
                dtype=chunk_attn_score.dtype,
                device=chunk_attn_score.device,
            ).unsqueeze(-2)
            current_block_score[..., : chunk_attn_score.shape[-1]] = chunk_attn_score

            plug_back_idx = (
                torch.cumsum(current_boundary_mask.to(torch.int64), dim=-1) - 1
            )
            current_mask_score = torch.gather(
                current_block_score,
                dim=-1,
                index=plug_back_idx.unsqueeze(-2),
            )

        inference_params.last_mask_score = current_mask_score[..., -1, :]
        inference_params.last_boundary_mask = current_boundary_mask

        return current_mask_score
