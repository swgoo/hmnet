from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import nn


@dataclass
class MaskingModuleState:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_memory: torch.Tensor | None = None
    last_query: torch.Tensor | None = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.key_memory is not None:
            self.key_memory.zero_()
        if self.last_query is not None:
            self.last_query.zero_()


def _update_k_cache(k: torch.Tensor, inference_params: MaskingModuleState):
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


class MaskingModule(nn.Module):

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
    ) -> MaskingModuleState:
        dtype = self.Wqk.weight.dtype if dtype is None else dtype
        device = self.Wqk.weight.device if device is None else device
        return MaskingModuleState(
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

    def _update_k_cache(self, k, inference_params: MaskingModuleState):
        """Update the key cache with the new k tensor."""
        return _update_k_cache(k, inference_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        inference_params: MaskingModuleState | None = None,
        **kwargs,
    ):

        qk = self.Wqk(hidden_states)
        qk = rearrange(qk, "... (two d) -> ... two d", two=2, d=self.d_model)
        q, k = qk[:, :, 0], qk[:, :, 1]
        attn_score = self.attention_score(q, k)

        if inference_params is not None:
            self._update_k_cache(k, inference_params)
            inference_params.seqlen_offset += int(k.shape[1])
            inference_params.last_query = q[:, -1, :]

        return attn_score

    def attention_score(self, q, k):
        attn_score = einsum(
            F.normalize(q, dim=-1),
            F.normalize(k, dim=-1),
            "... l d, ... m d -> ... l m",
        )
        attn_score = torch.softmax(attn_score * self.softmax_scale, dim=-1)
        return attn_score

    def step(self, hidden_states: torch.Tensor, inference_params: MaskingModuleState):
        if hidden_states.shape[0] > 0:
            qk = self.Wqk(hidden_states)
            qk = rearrange(qk, "... (two d) -> ... two d", two=2, d=self.d_model)
            q, k = qk[:, :, 0], qk[:, :, 1]
            k_cache = self._update_k_cache(k, inference_params)
            attn_score = self.attention_score(
                q, k_cache.to(device=hidden_states.device)
            )
            inference_params.last_query = q.squeeze(1)
            inference_params.seqlen_offset += 1
        else:
            k_empty = torch.empty(
                1,
                0,
                self.d_model,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            k_cache = self._update_k_cache(k_empty, inference_params)
            q_last = inference_params.last_query
            attn_score = self.attention_score(
                q_last.unsqueeze(1).to(hidden_states.device),
                k_cache.to(device=hidden_states.device),
            )
        return attn_score


@dataclass
class DeChunkMaskState:
    """
    The state of the dechunk mask.

    Contains
        - [last_value] (batch_size, seqlen) tensor. The last value of the batch element
    """

    last_masking_score: torch.Tensor  # (batch_size, seqlen)
    last_boundary_mask: torch.Tensor  # (batch_size, seqlen)


class DeChunkMaskLayer(nn.Module):

    def __init__(self, window_size: int, threshold=0.5):
        super().__init__()
        self.window_size = window_size
        self.threshold = threshold

    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return DeChunkMaskState(
            last_masking_score=torch.zeros(
                batch_size, max_seqlen, device=device, dtype=dtype
            ),
            last_boundary_mask=torch.zeros(
                batch_size, max_seqlen, device=device, dtype=torch.bool
            ),
        )

    def forward(
        self,
        boundary_mask,
        block_score: torch.Tensor,
        cu_seqlens=None,
        mask=None,
        inference_params: DeChunkMaskState | None = None,
    ) -> torch.Tensor:
        """
        boundary_mask: [batch, seq_len] tensor, 1 if boundary (same chunk
        with next 0s), idx = 0 is always 1
        block_score: [batch, num_query_chunks, num_key_chunks] tensor of probabilities
        cu_seqlens: [batch,] tensor of cumulative sequence lengths
        mask: [batch, seq_len] tensor, True if token is valid
        inference_params: CausalBlockMaskState, state of the causal block mask
        """

        if inference_params is None:
            assert (
                mask is not None
            ), "Mask must be provided if inference_params is not provided"
            assert boundary_mask[
                :, 0
            ].all(), "First token must be a boundary if running prefill"

        block_score = torch.clamp(block_score, min=1e-4, max=1.0 - 1e-4)

        if cu_seqlens is not None:
            raise NotImplementedError(
                "CausalBlockMask does not support cu_seqlens yet. Please use mask instead."
            )

        plug_back_idx = torch.cumsum(boundary_mask.to(torch.int64), dim=1) - 1  # (B, L)
        if block_score.size(1) == 1:
            # Inference mode: block_score shape is [B, 1, num_key_chunks]
            # Gather along dim=2 to map key indices
            block_mask_score = torch.gather(
                block_score,
                dim=2,
                index=plug_back_idx.unsqueeze(1).expand(-1, 1, plug_back_idx.size(1)),
            )
        else:
            # Prefill mode: block_score shape is [B, num_query_chunks, num_key_chunks]
            # First gather along query dimension using plug_back_idx
            tmp = torch.gather(
                block_score,
                dim=1,
                index=plug_back_idx.unsqueeze(-1).expand(
                    -1, plug_back_idx.size(1), block_score.size(2)
                ),
            )
            # Then gather along key dimension using plug_back_idx
            block_mask_score = torch.gather(
                tmp,
                dim=2,
                index=plug_back_idx.unsqueeze(1).expand(
                    -1, tmp.size(1), plug_back_idx.size(1)
                ),
            )

        # block_mask = self.create_chunked_causal_block_mask(
        #     block_score=block_mask_score,
        # )
        # score_mod = self.create_score_mod(block_score=block_mask_score)

        if inference_params is not None:
            inference_params.last_masking_score = block_mask_score[:, -1, :]
            inference_params.last_boundary_mask = boundary_mask

        return block_mask_score

    def step(
        self, boundary_mask, block_score, inference_params: DeChunkMaskState
    ) -> torch.Tensor:
        """
        boundary_mask: [batch,] boolean tensor
        block_score: [batch, 1, num_key_chunks] 확률 텐서
        inference_params: CausalBlockMaskState, 캐싱된 상태

        Returns:
            block_mask, score_mod: flex_attention에 사용될 마스크와 스코어 수정 함수
        """
        B = boundary_mask.shape[0]

        seq_len = inference_params.last_masking_score.shape[-1]

        current_masking_score = torch.zeros(
            B, 1, seq_len + 1, device=block_score.device, dtype=block_score.dtype
        )

        current_masking_score[:, :, :seq_len] = (
            inference_params.last_masking_score.unsqueeze(1)
        )
        current_masking_score[:, :, -1] = inference_params.last_masking_score[:, -1]

        last_boundary_mask = inference_params.last_boundary_mask
        cur_boundary_mask = torch.concat(
            [
                last_boundary_mask,
                boundary_mask.unsqueeze(1),
            ],
            dim=1,
        )

        if boundary_mask.sum() > 0:

            plug_back_idx = torch.cumsum(cur_boundary_mask.to(torch.int64), dim=1) - 1

            current_block_score = torch.zeros_like(
                plug_back_idx, device=block_score.device, dtype=block_score.dtype
            ).unsqueeze(1)
            current_block_score[:, :, : block_score.size(2)] = block_score[:, :, :]

            current_masking_score = torch.gather(
                current_block_score,
                dim=-1,
                index=plug_back_idx.unsqueeze(-2),
            )

        inference_params.last_masking_score = current_masking_score[:, -1, :]
        inference_params.last_boundary_mask = cur_boundary_mask

        return current_masking_score
