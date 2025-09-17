from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor, nn
from torch.nn.attention.flex_attention import create_block_mask

from .utils import ste_func


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

    @property
    def last_key(self):
        if self.key_memory is not None:
            batch_start = self.batch_size_offset
            batch_end = batch_start + self.key_memory.shape[0]
            return self.key_memory[batch_start:batch_end, : self.seqlen_offset, :]
        return None


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
        window_size,
        n_chunk_select=16,
        softmax_scale=None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.n_chunk_select = n_chunk_select
        self.softmax_scale = (
            softmax_scale if softmax_scale is not None else 1.0 / np.sqrt(d_model)
        )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True

    def allocate_inference_cache(
        self, batch_size, max_seqlen, device, dtype=None
    ) -> ChunkAttnScoreState:
        dtype = self.q_proj_layer.weight.dtype if dtype is None else dtype
        device = self.q_proj_layer.weight.device if device is None else device
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

    def _project(self, hidden_states: Tensor):
        """Project hidden states to query and key tensors."""
        q = self.q_proj_layer(hidden_states)
        k = self.k_proj_layer(hidden_states)
        return q, k

    def _attention_logit(self, q, k):
        attn_score = einsum(
            F.normalize(q, dim=-1),
            F.normalize(k, dim=-1),
            "... l d, ... m d -> ... l m",
        )
        return attn_score * self.softmax_scale

    def _apply_soft_mask(self, logits: Tensor, mask: Tensor, neg_value: float = -1e4):
        mask_float = mask.to(dtype=logits.dtype)
        inv_mask_float = 1.0 - mask_float
        return logits + neg_value * inv_mask_float

    def forward(
        self,
        hidden_states: Tensor,
        mask: Tensor,
        inference_params: ChunkAttnScoreState | None = None,
    ):
        q, k = self._project(hidden_states)
        mask = mask.bool()
        if inference_params is not None:
            self._update_k_cache(k, inference_params)
            inference_params.seqlen_offset += int(q.shape[-2])
            inference_params.last_query = q[:, -1, :]
        logits = self._attention_logit(q, k)
        mask_2d = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        logits.masked_fill_(~mask_2d, -1e9)
        causal_mask = torch.ones_like(logits).tril().bool()
        logits.masked_fill_(~causal_mask, -1e9)
        causal_without_window = torch.ones_like(logits).tril(-self.window_size).bool()
        long_range_logits = logits.masked_fill(~causal_without_window, -1e9)
        num_top_k = min(logits.shape[-1], self.n_chunk_select)
        top_k = torch.topk(long_range_logits, k=num_top_k, dim=-1, sorted=False).indices
        causal_with_window = causal_mask & (~causal_without_window)

        top_mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, top_k, True)
        top_mask = (top_mask & causal_mask & mask_2d) | (causal_with_window & mask_2d)
        return self._apply_soft_mask(logits, top_mask)

    def step(self, hidden_states: Tensor, inference_params: ChunkAttnScoreState):
        if hidden_states.shape[0] > 0:
            q, k = self._project(hidden_states)
            k_cache = self._update_k_cache(k, inference_params)
            inference_params.last_query = q.squeeze(-2)
            inference_params.seqlen_offset += 1
        else:
            q = inference_params.last_query.unsqueeze(-2)
            k_cache = inference_params.last_key
        logits = self._attention_logit(q, k_cache)
        kv_len = logits.shape[-1]
        causal_without_window = torch.ones_like(
            logits, device=logits.device, dtype=torch.bool
        )
        if self.window_size > 0 and kv_len > self.window_size:
            causal_without_window[..., : -self.window_size] = False
        long_range_logits = logits.masked_fill(~causal_without_window, -1e9)

        sliding_window = torch.zeros_like(logits, dtype=torch.bool)
        if self.window_size > 0:
            sliding_window[..., -self.window_size :] = True
        else:
            sliding_window[..., :] = True
        num_top_k = min(kv_len, self.n_chunk_select)
        top_k = torch.topk(long_range_logits, k=num_top_k, dim=-1, sorted=False).indices
        top_mask = torch.zeros_like(logits, dtype=torch.bool)
        top_mask.scatter_(-1, top_k, True)
        top_mask = top_mask | sliding_window
        logits = logits.masked_fill(~top_mask, -1e9)
        return logits


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
    ):

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

        plug_back_idx = torch.cumsum(boundary_mask.long(), dim=-1) - 1  # (B, L)
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
            index=plug_back_idx.unsqueeze(-2).expand(
                -1, selected_queries.size(-2), plug_back_idx.size(-1)
            ),
        )
        if inference_params is not None:
            inference_params.last_mask_score = mask_score[..., -1, :]
            inference_params.last_boundary_mask = boundary_mask

        return self.create_masks(mask_score)

    def step(
        self,
        boundary_mask: Tensor,
        chunk_attn_score: Tensor,
        inference_params: DeChunkAttnScoreState,
    ):
        current_boundary_mask = torch.concat(
            [
                inference_params.last_boundary_mask,
                boundary_mask.unsqueeze(-1),
            ],
            dim=-1,
        )

        prev_mask_score = inference_params.last_mask_score.unsqueeze(
            -2
        )  # (B, 1, seq_len)
        current_mask_score = torch.cat(
            [prev_mask_score, prev_mask_score[..., -1:]], dim=-1
        )  # (B, 1, seq_len+1)

        if boundary_mask.sum() > 0:
            current_mask_score[..., : chunk_attn_score.shape[-1]] = chunk_attn_score
            plug_back_idx = torch.cumsum(current_boundary_mask.long(), dim=-1) - 1
            current_mask_score = torch.gather(
                current_mask_score,
                dim=-1,
                index=plug_back_idx.unsqueeze(-2),
            )

        inference_params.last_mask_score = current_mask_score[..., -1, :]
        inference_params.last_boundary_mask = current_boundary_mask

        return self.create_masks(current_mask_score)

    def _create_chunk_causal_window_mask(self, mask):
        batch_size = mask.shape[0]
        q_len = mask.shape[-2]
        kv_len = mask.shape[-1]
        device = mask.device

        if q_len == 1:
            causal_mask = torch.ones(1, kv_len, device=device, dtype=torch.bool)
            if self.window_size > 0:
                window_mask = torch.zeros(1, kv_len, device=device, dtype=torch.bool)
                start = max(0, kv_len - int(self.window_size))
                if start < kv_len:
                    window_mask[:, start:] = True
            else:
                window_mask = causal_mask
        elif q_len != 1 and kv_len != 1:
            q_idx = torch.arange(q_len, device=device).unsqueeze(-1)
            kv_idx = torch.arange(kv_len, device=device).unsqueeze(-2)
            causal_mask = q_idx >= kv_idx
            if self.window_size > 0:
                diff = q_idx - kv_idx
                window_mask = (diff >= 0) & (diff < self.window_size)
            else:
                window_mask = causal_mask
        else:
            raise ValueError(
                "Invalid combination of q_seq_len, kv_seq_len, and window_size"
            )
        final_mask = (mask & causal_mask) | window_mask.unsqueeze(0)

        return create_block_mask(
            lambda b, h, q_idx, kv_idx: final_mask[b, q_idx, kv_idx],
            B=batch_size,
            H=1,
            Q_LEN=q_len,
            KV_LEN=kv_len,
        )

    def _create_score_mod(self, mask_score):
        inv_mask = (mask_score < self.threshold).float()

        def score_fn(score, b, h, q_idx, kv_idx):
            return score - 1e-9 * inv_mask[b, q_idx, kv_idx]

        return score_fn

    def create_masks(self, mask_score: Tensor):
        if mask_score is None:
            return None, None
        elif mask_score.dtype == torch.bool:
            mask = mask_score
            score_mod = None
        else:
            mask = mask_score > self.threshold
            score_mod = self._create_score_mod(mask_score)
        block_mask = self._create_chunk_causal_window_mask(mask)
        return score_mod, block_mask
