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
       
