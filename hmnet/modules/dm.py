from re import I
import select
from einops import rearrange
from hypothesis import infer
from torch import nn
import torch
import numpy as np
from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch.nn.functional as F


@dataclass
class MaskingModuleState:
    max_seqlen: int
    max_batch_size: int
    seqlen_offset: int = 0
    batch_size_offset: int = 0
    key_memory: torch.Tensor | None = None

    def reset(self, max_seqlen, max_batch_size):
        self.max_seqlen = max_seqlen
        self.max_batch_size = max_batch_size
        self.seqlen_offset = 0
        if self.key_memory is not None:
            self.key_memory.zero_()


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
        q, k = rearrange(qk, "... (two d) -> ... two d", two=2)

        if inference_params is None:
            attn_score = self.attention_score(q, k)
        else:
            k_cache = self._update_k_cache(k, inference_params)
            attn_score = self.attention_score(q, k_cache)

        return attn_score

    def attention_score(self, q, k):
        attn_score = torch.einsum(
            "bld,bmd->blm", F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        )
        attn_score = torch.softmax(attn_score * self.softmax_scale, dim=-1)
        return attn_score

    def step(self, hidden_states: torch.Tensor, inference_params: MaskingModuleState):
        attn_score = self.forward(
            hidden_states=hidden_states,
            inference_params=inference_params,
        )
        inference_params.seqlen_offset += 1
        return attn_score


def calculate_chunk_boundaries_from_mask(boundary_mask, device=None):
    """
    boundary_mask에서 청크 경계를 계산
    boundary_mask: [seq_len] 텐서, 1이면 경계 (직전 0들과 같은 청크)
    """
    if device is None:
        device = (
            boundary_mask.device
            if hasattr(boundary_mask, "device")
            else torch.device("cpu")
        )

    boundary_mask = torch.tensor(boundary_mask, device=device, dtype=torch.bool)
    seq_len = len(boundary_mask)

    # 청크 경계 위치 찾기 (1인 위치들)
    boundary_positions = torch.where(boundary_mask)[0]

    # 청크 시작 위치들 계산
    chunk_starts = [0]  # 첫 번째 청크는 항상 0에서 시작
    if len(boundary_positions) > 0:
        # 경계 다음 위치들이 새로운 청크의 시작
        chunk_starts.extend((boundary_positions + 1).tolist())

    # 청크 끝 위치들 계산
    chunk_ends = boundary_positions.tolist() + [seq_len - 1]

    # 청크 길이들 계산
    chunk_lengths = []
    for i in range(len(chunk_starts)):
        if i < len(chunk_ends):
            chunk_lengths.append(chunk_ends[i] - chunk_starts[i] + 1)

    # 청크 경계들 (cumsum)
    chunk_boundaries = [0] + [
        sum(chunk_lengths[: i + 1]) for i in range(len(chunk_lengths))
    ]

    return (
        torch.tensor(chunk_boundaries, device=device, dtype=torch.long),
        chunk_lengths,
    )


class CausalBlockMask(nn.Module):

    def __init__(self, device=None, dtype=None):
        super().__init__()

    def forward(self, seq_len, boundary_mask, block_score, threshold=0.5):
        block_mask = self.create_chunked_causal_block_mask(
            seq_len=seq_len,
            boundary_mask=boundary_mask,
            block_score=block_score,
            threshold=threshold,
            device=block_score.device,
        )
        score_mod = self.create_chunked_score_mod(
            boundary_mask=boundary_mask,
            block_score=block_score,
            device=block_score.device,
        )
        return block_mask, score_mod

    def create_chunked_causal_block_mask(
        self,
        seq_len: int,
        boundary_mask: list,
        block_score: torch.Tensor,
        threshold: float = 0.5,
        device: torch.device = None,
    ):
        """
        청킹된 시퀀스를 위한 causal block mask 생성 (확률 기반)
        boundary_mask: [seq_len] 리스트/텐서, 1이면 경계 (직전 0들과 같은 청크)
        block_score: [num_chunks, num_chunks] 확률 텐서
        threshold: 블록 허용 임계값 (기본 0.5)
        """
        if device is None:
            device = torch.device("cpu")

        # boundary_mask에서 청크 경계 계산
        chunk_boundaries, chunk_lengths = calculate_chunk_boundaries_from_mask(
            boundary_mask, device
        )

        # block_score를 텐서로 변환하고 임계값으로 이진화
        block_selection_tensor = (block_score > threshold).to(device)

        num_chunks = len(chunk_lengths)

        def causal_chunked_mask(b, h, q_idx, kv_idx):
            """
            텐서 연산으로만 구성된 마스크 함수 (조건문 없음)
            """
            # causal mask: kv_idx <= q_idx 조건
            causal_condition = kv_idx <= q_idx

            # 각 위치가 어느 청크에 속하는지 계산
            q_expanded = q_idx.expand(num_chunks)
            kv_expanded = kv_idx.expand(num_chunks)

            # 청크 경계 비교
            q_ge_start = q_expanded >= chunk_boundaries[:-1]  # [num_chunks]
            q_lt_end = q_expanded < chunk_boundaries[1:]  # [num_chunks]
            q_in_chunk = q_ge_start & q_lt_end  # [num_chunks]

            kv_ge_start = kv_expanded >= chunk_boundaries[:-1]  # [num_chunks]
            kv_lt_end = kv_expanded < chunk_boundaries[1:]  # [num_chunks]
            kv_in_chunk = kv_ge_start & kv_lt_end  # [num_chunks]

            # 청크 조합별 허용 여부 계산 (확률 > threshold)
            chunk_pairs = q_in_chunk.unsqueeze(1) & kv_in_chunk.unsqueeze(
                0
            )  # [num_chunks, num_chunks]
            allowed_pairs = (
                chunk_pairs & block_selection_tensor
            )  # [num_chunks, num_chunks]
            chunk_allowed = torch.any(allowed_pairs)

            # 최종 마스크: causal 조건 AND 청크 허용 조건
            return causal_condition & chunk_allowed

        # create_block_mask 사용
        block_mask = create_block_mask(
            causal_chunked_mask,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
        )

        return block_mask

    def create_chunked_score_mod(
        self,
        boundary_mask: list,
        block_score: torch.Tensor,
        device: torch.device = None,
    ):
        """
        청킹된 시퀀스를 위한 score_mod 함수 생성
        boundary_mask: [seq_len] 리스트/텐서, 1이면 경계 (직전 0들과 같은 청크)
        block_score: [num_chunks, num_chunks] 확률 텐서
        """
        if device is None:
            device = torch.device("cpu")

        # boundary_mask에서 청크 경계 계산
        chunk_boundaries, chunk_lengths = calculate_chunk_boundaries_from_mask(
            boundary_mask, device
        )

        # block_score를 텐서로 변환
        block_score_tensor = block_score.to(device)

        num_chunks = len(chunk_lengths)

        def chunked_score_mod(score, b, h, q_idx, kv_idx):
            """
            청크별 확률을 적용하는 score_mod 함수
            """
            # 각 위치가 어느 청크에 속하는지 계산
            q_expanded = q_idx.expand(num_chunks)
            kv_expanded = kv_idx.expand(num_chunks)

            # 청크 경계 비교
            q_ge_start = q_expanded >= chunk_boundaries[:-1]  # [num_chunks]
            q_lt_end = q_expanded < chunk_boundaries[1:]  # [num_chunks]
            q_in_chunk = q_ge_start & q_lt_end  # [num_chunks]

            kv_ge_start = kv_expanded >= chunk_boundaries[:-1]  # [num_chunks]
            kv_lt_end = kv_expanded < chunk_boundaries[1:]  # [num_chunks]
            kv_in_chunk = kv_ge_start & kv_lt_end  # [num_chunks]

            # 해당하는 청크 조합의 확률값 찾기
            q_chunk_idx = torch.argmax(q_in_chunk.float())
            kv_chunk_idx = torch.argmax(kv_in_chunk.float())

            # 해당 청크 조합의 확률값 가져오기
            prob_score = block_score_tensor[q_chunk_idx, kv_chunk_idx]

            # 원래 점수에 확률을 곱해서 반환
            return score * prob_score

        return chunked_score_mod
