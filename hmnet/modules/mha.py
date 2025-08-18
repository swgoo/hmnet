import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from .rotary import RotaryEmbedding
from torch.nn.attention.flex_attention import flex_attention, create_block_mask


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class CausalMaskMHA(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        qkv_proj_bias=False,
        out_proj_bias=False,
        window_size=-1,  # -1 for global Attention
        softmax_scale=None,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_interleaved=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale
        self.rotary_emb_dim = rotary_emb_dim
        assert window_size == -1 or window_size > 0, "window_size must be == -1 or > 0"
        self.window_size = window_size
        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        qkv_dim = self.head_dim * (3 * self.num_heads)

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        self.Wqkv = nn.Linear(d_model, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(
            d_model, d_model, bias=out_proj_bias, **factory_kwargs
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert (
            self.layer_idx is not None
        ), "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        block_mask=None,
        score_mod=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        assert q.dtype in [torch.float16, torch.bfloat16, torch.float32]
        assert kv.dtype in [torch.float16, torch.bfloat16, torch.float32]
        if cu_seqlens is not None or max_seqlen is not None:
            raise NotImplementedError(
                "Flex attention with variable length sequences is not implemented"
            )

        k, v = kv.unbind(dim=2)  # Each: (B, S, H, D)

        # Reshape for flex_attention: (B, H, S, D)
        # Apply flex_attention
        out = flex_attention(
            q.transpose(-3, -2),
            k.transpose(-3, -2),
            v.transpose(-3, -2),
            block_mask=block_mask,
            score_mod=score_mod,
            scale=self.softmax_scale,
        )

        # Reshape back: (B, S, H, D)
        out = out.transpose(-3, -2)
        return out

    def _create_window_causal_mask(
        self,
        batch_size: int,
        q_len: int,
        kv_len: int,
    ):

        def mask_fn(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            if q_len == 1:
                if self.window_size == -1:
                    return torch.ones_like(kv_idx, dtype=torch.bool)
                else:
                    start_idx = max(0, kv_len - 1 - self.window_size)
                    return kv_idx >= start_idx
            else:
                if self.window_size == -1:
                    return q_idx >= kv_idx
                else:
                    return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= self.window_size)

        return create_block_mask(
            mask_fn,
            B=batch_size,
            H=1,
            Q_LEN=q_len,
            KV_LEN=kv_len,
        )

    def forward(
        self,
        x,
        block_mask=None,
        score_mod=None,
        cu_seqlens=None,
        max_seqlen=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
            masking_score: Optional attention patterns
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths.
            max_seqlen: int. Maximum sequence length in the batch.
            inference_params: for generation.
        """
        if cu_seqlens is not None or max_seqlen is not None:
            raise NotImplementedError(
                "BlockMaskingMHA does not support cu_seqlens or max_seqlen"
            )
        if inference_params is not None:
            assert cu_seqlens is None and max_seqlen is None

        kwargs = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen, **kwargs}
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = (
            inference_params.max_seqlen if inference_params is not None else None
        )

        qkv = self.Wqkv(x)
        qkv = rearrange(
            qkv,
            "... (three h d) -> ... three h d",
            three=3,
            h=self.num_heads,
            d=self.head_dim,
        )

        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(
                qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
            )

        q, kv = (
            qkv[..., 0, :, :],
            qkv[..., 1:, :, :],
        )  # q: (B, S, H, D), kv: (B, S, 2, H, D)

        if inference_params is not None:
            kv = self._update_kv_cache(kv, inference_params)

        if block_mask is None:
            block_mask = self._create_window_causal_mask(
                batch_size=q.shape[0],
                q_len=q.shape[1],
                kv_len=kv.shape[1],
            )

        context = self._attention(
            q,
            kv,
            block_mask=block_mask,
            score_mod=score_mod,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out

    def step(
        self,
        x,
        inference_params,
        **kwargs,
    ):
        return self.forward(x, inference_params=inference_params, **kwargs)
