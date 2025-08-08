import torch
import torch.nn as nn
from einops import rearrange
from hnet.modules.isotropic import IsotropicInferenceParams
from hnet.modules.mha import _update_kv_cache
from hnet.modules.rotary import RotaryEmbedding
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)
from .utils import ste_func


class FlexAttention(nn.Module):

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
        masking_threshold=0.5,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale
        self.rotary_emb_dim = rotary_emb_dim
        assert window_size >= -1, "window_size must be >= -1"
        self.window_size = window_size
        self.num_heads = num_heads
        assert self.d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = self.d_model // num_heads
        self.masking_threshold = masking_threshold
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
        mask_score: torch.Tensor | None = None,
        cu_seqlens=None,
        max_seqlen=None,
        inference_params: IsotropicInferenceParams | None = None,
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

        q, kv = qkv[:, :, 0], qkv[:, :, 1:]
        mask_score = mask_score.to(device=x.device) if mask_score is not None else None
        if inference_params is None:
            score_mod, block_mask = self.create_masks(mask_score, q, kv)
            context = self.inner_attn(
                q, kv, block_mask=block_mask, score_mod=score_mod, **kwargs
            )
        else:
            kv_cache = self._update_kv_cache(kv, inference_params)
            score_mod, block_mask = self.create_masks(mask_score, q, kv_cache)
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

    def create_masks(self, mask_score, q, kv):
        batch_size, q_len, kv_len = q.shape[0], q.shape[1], kv.shape[1]

        if mask_score is None:
            block_mask = create_block_mask(
                self.create_window_causal_mask(self.window_size),
                B=batch_size,
                H=self.num_heads,
                Q_LEN=q_len,
                KV_LEN=kv_len,
            )
            return None, block_mask

        if mask_score.dtype == torch.bool:
            masking = mask_score
            score_mod = None
        else:
            masking = mask_score > self.masking_threshold
            masking_confidence = torch.where(
                mask_score > self.masking_threshold, mask_score, 1 - mask_score
            )
            masking_confidence = ste_func(
                masking_confidence, threshold=self.masking_threshold
            )
            score_mod = self.create_score_mod(masking_confidence)

        block_mask = create_block_mask(
            self.create_chunked_causal_mask(masking, window_size=self.window_size),
            B=batch_size,
            H=1,
            Q_LEN=q_len,
            KV_LEN=kv_len,
        )

        return score_mod, block_mask

    def step(
        self, x, inference_params, masking_score: torch.Tensor | None = None, **kwargs
    ):
        return self.forward(
            x,
            mask_score=masking_score,
            inference_params=inference_params,
        )

    def create_chunked_causal_mask(self, mask, window_size: int = -1):
        seq_len = mask.shape[-2]
        device = mask.device

        if not hasattr(self, "_mask_cache"):
            self._mask_cache = {}

        cache_key = (seq_len, window_size, device)
        if cache_key not in self._mask_cache:
            causal = torch.tril(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
            )
            if window_size >= 0:
                small_tril = torch.tril(
                    torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                    diagonal=-window_size,
                )
                window_mask = causal & ~small_tril
            else:
                window_mask = causal
            self._mask_cache[cache_key] = window_mask

        window_mask = self._mask_cache[cache_key]  # [seq_len, seq_len] bool
        final_mask = mask & window_mask.unsqueeze(0)  # [batch, seq_len, seq_len]

        def mask_fn(b, h, q_idx, kv_idx):
            return final_mask[b, q_idx, kv_idx]

        return mask_fn

    def create_score_mod(self, mask_score):
        def score_mod(score, b, h, q_idx, kv_idx):
            return score * mask_score[b, q_idx, kv_idx]

        return score_mod

    def create_window_causal_mask(self, window_size):
        if window_size == -1:

            def window_causal_mask_Fn(b, h, q_idx, kv_idx):
                return q_idx >= kv_idx

        elif window_size >= 0:

            def window_causal_mask_Fn(b, h, q_idx, kv_idx):
                return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size)

        else:
            raise ValueError("window_size must be -1 or >= 0")

        return window_causal_mask_Fn
