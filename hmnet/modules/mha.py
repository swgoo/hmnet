import torch
import torch.nn as nn
from einops import rearrange
from hnet.modules.isotropic import IsotropicInferenceParams
from hnet.modules.mha import _update_kv_cache
from hnet.modules.rotary import RotaryEmbedding
from torch.nn.attention.flex_attention import create_block_mask, flex_attention


class FlexAttention(nn.Module):
    """Implement causal cross-attention using FlexAttention with block mask support.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
    """

    def __init__(
        self,
        softmax_scale=None,
    ):
        super().__init__()
        self.softmax_scale = softmax_scale

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        block_mask=None,
        score_mod=None,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        """Implements the multihead softmax attention using FlexAttention.
        Arguments
        ---------
            q: The tensor containing the query. (B, S_q, H, D)
            kv: The tensor containing the key and value. (B, S_kv, 2, H, D)
            block_mask: Optional BlockMask for structured attention patterns
        Returns:
        --------
            out: (B, S, H, D)
        """
        assert q.dtype in [torch.float16, torch.bfloat16, torch.float32]
        assert kv.dtype in [torch.float16, torch.bfloat16, torch.float32]
        if cu_seqlens is not None or max_seqlen is not None:
            raise NotImplementedError(
                "Flex attention with variable length sequences is not implemented"
            )

        k, v = kv.unbind(dim=2)  # Each: (B, S, H, D)

        # Reshape for flex_attention: (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply flex_attention
        out = flex_attention(
            q,
            k,
            v,
            block_mask=block_mask,
            score_mod=score_mod,
            scale=self.softmax_scale,
        )

        # Reshape back: (B, S, H, D)
        out = out.transpose(1, 2)

        return out


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


def causal_mask_fn(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


class CausalBlockMaskMHA(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        qkv_proj_bias=False,
        out_proj_bias=False,
        window_size=-1,
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
        if window_size != -1:
            print(
                "Warning: window_size is not used in CausalBlockMaskMHA, "
                "it is only for compatibility with other MHA implementations."
            )

        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        kv_dim = self.head_dim * (3 * self.num_heads)

        if self.rotary_emb_dim > 0:
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        self.Wqkv = nn.Linear(d_model, kv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.inner_attn = FlexAttention(softmax_scale=softmax_scale)
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

    def forward(
        self,
        x,
        block_mask=None,
        score_mod=None,
        cu_seqlens=None,
        max_seqlen=None,
        inference_params: IsotropicInferenceParams | None = None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim)
            block_mask: Optional BlockMask for structured attention patterns
            score_mod: Optional score modifiers for attention
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
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )

        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(
                qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
            )

        q, kv = qkv[:, :, 0], qkv[:, :, 1:]

        if block_mask is None:
            block_mask = create_block_mask(
                causal_mask_fn,
                B=qkv.shape[0],
                H=self.num_heads,
                Q_LEN=qkv.shape[1],
                KV_LEN=qkv.shape[1],
            )

        if inference_params is None:
            context = self.inner_attn(
                q, kv, block_mask=block_mask, score_mod=score_mod, **kwargs
            )
        else:
            kv_cache = self._update_kv_cache(kv, inference_params)
            context = self.inner_attn(
                q,
                kv_cache,
                block_mask=block_mask,
                score_mod=score_mod,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out

    def step(self, x, inference_params, block_mask=None, score_mod=None, **kwargs):
        return self.forward(
            x,
            block_mask=block_mask,
            score_mod=score_mod,
            inference_params=inference_params,
        )
