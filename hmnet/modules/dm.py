from re import I
import select
from torch import nn
import torch
import numpy as np
from hmnet.models.mixer_seq import HNetForCausalLM
from hnet.models.config_hnet import HNetConfig
from hnet.modules.dc import mamba_chunk_scan_combined, get_seq_idx
from dataclasses import dataclass
from einops import rearrange, repeat 
from hnet.modules.dc import DeChunkState  

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
import torch.nn.functional as F

@dataclass
class MaskingModuleState:
    """
    The state of the routing module.

    Contains
        - [has_seen_tokens] (batch_size,) bool tensor. Whether that batch element has processed any tokens yet.
        - [last_hidden_state] (batch_size, d_model) tensor. The last hidden state of the batch element (used for boundary prediction).
    """

    has_seen_tokens: torch.Tensor  # (batch_size,)
    last_hidden_state: torch.Tensor  # (batch_size, d_model)

@dataclass
class MaskingModuleOutput:
    """
    The output of the MaskModule.
    Contains
        - [attention_mask] (batch_size, num_boundaries, num_boundaries) tensor. The attention mask for the current batch.
        - [attention_probs] (batch_size, num_boundaries, num_boundaries) tensor. The attention probabilities for the current batch.
        - [selected_probs] (batch_size, num_boundaries) tensor. The selected probabilities for the current batch.
    """
    attention_mask: torch.Tensor  # (batch_size, num_boundaries, num_boundaries)
    attention_probs: torch.Tensor  # (batch_size, num_boundaries, num_boundaries)
    selected_probs: torch.Tensor

class MaskingModule(nn.Module):

    def __init__(
        self,
        d_model,
        device=None,
        dtype=None,
    ):
        self.d_model = d_model
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.q_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        self.k_proj_layer = nn.Linear(d_model, d_model, bias=False, **factory_kwargs)
        with torch.no_grad():
            self.q_proj_layer.weight.copy_(torch.eye(d_model))
            self.k_proj_layer.weight.copy_(torch.eye(d_model))
        self.q_proj_layer.weight._no_reinit = True
        self.k_proj_layer.weight._no_reinit = True
    
    def allocate_inference_cache(self, batch_size, max_seqlen, device, dtype=None):
        return MaskingModuleState(
            has_seen_tokens=torch.zeros(batch_size, device=device, dtype=torch.bool),
            last_hidden_state=torch.zeros(
                batch_size, self.d_model, device=device, dtype=dtype
            ),
        )

    def forward(self, hidden_states: torch.Tensor, boundary_mask: torch.Tensor, prev_attention_mask: torch.Tensor) -> MaskingModuleOutput:
        """
        MaskModule의 forward 메서드입니다.
        주어진 hidden_states와 boundary_mask를 사용하여 attention mask를 생성합니다.

        Args:
            hidden_states (torch.Tensor): 입력 hidden states. Shape: (B, N, D).
            boundary_mask (torch.Tensor): 경계 위치를 나타내는 마스크. Shape: (B, L).
            prev_attention_mask (torch.Tensor): 이전 attention mask. Shape: (..., N, N).

        Returns:
            torch.Tensor: 생성된 attention mask. Shape: (..., L, L).
            torch.Tensor: attention probabilities.
        """
        hidden_states = hidden_states[..., :self.d_model]  # Ensure hidden_states has the correct shape

        q = self.q_proj_layer(hidden_states)
        k = self.k_proj_layer(hidden_states)

        attention_mask = torch.einsum("bld,bmd->blm", q, k)
        attention_mask = attention_mask / np.sqrt(self.d_model)
        attention_probs = torch.softmax(attention_mask, dim=-1)


        attention_probs = torch.stack(((1- attention_probs), attention_probs), dim=-1)
        selected_idx = torch.argmax(attention_probs, dim=-1)
        attention_mask = selected_idx == 1
        selected_probs = attention_probs.gather(-1, selected_idx.unsqueeze(-1))
        
        return MaskingModuleOutput(
            attention_mask=attention_mask,
            attention_probs=attention_probs,
            selected_probs=selected_probs
        )

    def restore_attention_mask(self, attention_mask: torch.Tensor, boundary_mask: torch.Tensor) -> torch.Tensor:
        """
        압축된 attention_mask를 원래 시퀀스 길이로 복원합니다.
        이 메서드는 경계 사이의 값을 복사하는 방식으로 마스크를 확장합니다.

        Args:
            attention_mask (torch.Tensor): 복원할 어텐션 마스크. 
                                           Shape: (..., num_boundaries, num_boundaries).
            boundary_mask (torch.Tensor): 경계 위치를 나타내는 마스크. 
                                          Shape: (B, L).

        Returns:
            torch.Tensor: 원래 시퀀스 길이로 복원된 어텐션 마스크. 
                          Shape: (..., L, L).
        """
        restore_indices = torch.cumsum(boundary_mask, dim=1) - 1

        original_ndim = attention_mask.ndim
        num_boundaries = attention_mask.shape[-1]
        
        view_shape = restore_indices.shape + (1,) * (original_ndim - 2)
        permute_dims = (0,) + tuple(range(2, original_ndim - 1)) + (1, original_ndim - 1)
        
        indices_dim1 = restore_indices.view(view_shape).permute(permute_dims)
        indices_dim1 = indices_dim1.expand(attention_mask.shape[:-1] + (restore_indices.shape[-1],))

        expanded_mask = torch.gather(attention_mask, -1, indices_dim1)

        indices_dim2 = restore_indices.view(view_shape).permute(permute_dims).transpose(-2, -1)
        indices_dim2 = indices_dim2.expand(expanded_mask.shape[:-2] + (restore_indices.shape[-1], expanded_mask.shape[-1]))
        
        restored_mask = torch.gather(expanded_mask, -2, indices_dim2)

        return restored_mask

    def step(self, hidden_states, inference_params):
        raise NotImplementedError("MaskModule does not support step method. Use forward method instead.")
        """
        MaskModule의 step 메서드입니다.
        주어진 hidden_states를 사용하여 attention mask를 생성하고 상태를 업데이트합니다.

        Args:
            hidden_states (torch.Tensor): 입력 hidden states. Shape: (B, N, D).
            inference_params (MaskModuleState): MaskModule의 상태 정보.

        Returns:
            MaskModuleState: 업데이트된 상태 정보.
        """
        batch_size = hidden_states.shape[0]
        if inference_params is None:
            inference_params = self.allocate_inference_cache(batch_size, hidden_states.shape[1], hidden_states.device, hidden_states.dtype)

        attention_mask, attention_probs = self.forward(hidden_states, inference_params.has_seen_tokens, None)
        
        inference_params.has_seen_tokens = torch.ones(batch_size, device=hidden_states.device, dtype=torch.bool)
        inference_params.last_hidden_state = hidden_states[:, -1, :]

        return inference_params
    

def calculate_chunk_boundaries_from_mask(boundary_mask, device=None):
    """
    boundary_mask에서 청크 경계를 계산
    boundary_mask: [seq_len] 텐서, 1이면 경계 (직전 0들과 같은 청크)
    """
    if device is None:
        device = boundary_mask.device if hasattr(boundary_mask, 'device') else torch.device('cpu')
    
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
    chunk_boundaries = [0] + [sum(chunk_lengths[:i+1]) for i in range(len(chunk_lengths))]
    
    return torch.tensor(chunk_boundaries, device=device, dtype=torch.long), chunk_lengths

def create_chunked_causal_block_mask(
    seq_len: int,
    boundary_mask: list,
    block_score: torch.Tensor,
    threshold: float = 0.5,
    device: torch.device = None
):
    """
    청킹된 시퀀스를 위한 causal block mask 생성 (확률 기반)
    boundary_mask: [seq_len] 리스트/텐서, 1이면 경계 (직전 0들과 같은 청크)
    block_score: [num_chunks, num_chunks] 확률 텐서
    threshold: 블록 허용 임계값 (기본 0.5)
    """
    if device is None:
        device = torch.device('cpu')
    
    # boundary_mask에서 청크 경계 계산
    chunk_boundaries, chunk_lengths = calculate_chunk_boundaries_from_mask(boundary_mask, device)
    
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
        q_lt_end = q_expanded < chunk_boundaries[1:]      # [num_chunks]
        q_in_chunk = q_ge_start & q_lt_end                # [num_chunks]
        
        kv_ge_start = kv_expanded >= chunk_boundaries[:-1]  # [num_chunks]
        kv_lt_end = kv_expanded < chunk_boundaries[1:]      # [num_chunks]
        kv_in_chunk = kv_ge_start & kv_lt_end               # [num_chunks]
        
        # 청크 조합별 허용 여부 계산 (확률 > threshold)
        chunk_pairs = q_in_chunk.unsqueeze(1) & kv_in_chunk.unsqueeze(0)  # [num_chunks, num_chunks]
        allowed_pairs = chunk_pairs & block_selection_tensor              # [num_chunks, num_chunks]
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
        device=device
    )
    
    return block_mask

def create_chunked_score_mod(
    boundary_mask: list,
    block_score: torch.Tensor,
    device: torch.device = None
):
    """
    청킹된 시퀀스를 위한 score_mod 함수 생성
    boundary_mask: [seq_len] 리스트/텐서, 1이면 경계 (직전 0들과 같은 청크)
    block_score: [num_chunks, num_chunks] 확률 텐서
    """
    if device is None:
        device = torch.device('cpu')
    
    # boundary_mask에서 청크 경계 계산
    chunk_boundaries, chunk_lengths = calculate_chunk_boundaries_from_mask(boundary_mask, device)
    
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
        q_lt_end = q_expanded < chunk_boundaries[1:]      # [num_chunks]
        q_in_chunk = q_ge_start & q_lt_end                # [num_chunks]
        
        kv_ge_start = kv_expanded >= chunk_boundaries[:-1]  # [num_chunks]
        kv_lt_end = kv_expanded < chunk_boundaries[1:]      # [num_chunks]
        kv_in_chunk = kv_ge_start & kv_lt_end               # [num_chunks]
        
        # 해당하는 청크 조합의 확률값 찾기
        q_chunk_idx = torch.argmax(q_in_chunk.float())
        kv_chunk_idx = torch.argmax(kv_in_chunk.float())
        
        # 해당 청크 조합의 확률값 가져오기
        prob_score = block_score_tensor[q_chunk_idx, kv_chunk_idx]
        
        # 원래 점수에 확률을 곱해서 반환
        return score * prob_score
    
    return chunked_score_mod

def chunked_flex_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    boundary_mask: list,
    block_score: torch.Tensor,
    threshold: float = 0.5,
    use_score_mod: bool = True,
    scale: float = None
):
    """
    청킹된 FlexAttention 실행 (확률 기반 block_mask와 score_mod)
    boundary_mask: [seq_len] 리스트/텐서, 1이면 경계 (직전 0들과 같은 청크)
    block_score: [num_chunks, num_chunks] 확률 텐서
    threshold: 블록 허용 임계값
    use_score_mod: score_mod 사용 여부
    """
    batch_size, num_heads, seq_len, head_dim = query.shape
    
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # 확률 기반 block mask 생성
    block_mask = create_chunked_causal_block_mask(
        seq_len=seq_len,
        boundary_mask=boundary_mask,
        block_score=block_score,
        threshold=threshold,
        device=query.device
    )
    
    # score_mod 함수 생성
    score_mod = None
    if use_score_mod:
        score_mod = create_chunked_score_mod(
            boundary_mask=boundary_mask,
            block_score=block_score,
            device=query.device
        )
    
    # FlexAttention 실행
    output = flex_attention(
        query, key, value,
        block_mask=block_mask,
        score_mod=score_mod,
        scale=scale
    )
    
    return output