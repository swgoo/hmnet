import torch
import torch.nn as nn
from einops import rearrange
from hnet.modules.isotropic import IsotropicInferenceParams
from hnet.modules.mha import _update_kv_cache
from hnet.modules.rotary import RotaryEmbedding
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from .utils import STE


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
        masking_score: torch.Tensor | None = None,
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
            qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )

        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(
                qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
            )

        q, kv = qkv[:, :, 0], qkv[:, :, 1:]

        score_mod = None
        if masking_score is None:
            block_mask = create_block_mask(
                self.create_window_causal_mask(self.window_size),
                B=qkv.shape[0],
                H=self.num_heads,
                Q_LEN=qkv.shape[1],
                KV_LEN=qkv.shape[1],
            )
        elif masking_score.dtype == torch.bool:
            block_mask = self.create_chunked_causal_block_mask(
                masking_score, window_size=self.window_size
            )
        else:
            masking = masking_score > self.masking_threshold
            block_mask = self.create_chunked_causal_block_mask(
                masking, window_size=self.window_size
            )
            masking_score = torch.stack(((1 - masking_score), masking_score), dim=-1)
            masking_confidence = masking_score.max(dim=-1).values
            masking_confidence = STE.apply(None, masking_confidence)
            score_mod = self.create_score_mod(masking_confidence)

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

    def step(
        self, x, inference_params, masking_score: torch.Tensor | None = None, **kwargs
    ):
        return self.forward(
            x,
            masking_score=masking_score,
            inference_params=inference_params,
        )

    def create_chunked_causal_block_mask(self, mask, window_size: int = -1):
        """
        Create a block mask for flex_attention based on block_score and threshold.

        Args:
            mask: [batch, seq_len, seq_len] boolean tensor

        Returns:
            block_mask: BlockMask for flex_attention
        """
        # Apply causal mask (lower triangular)
        batch_size, seq_len, _ = mask.shape
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=mask.device))

        # Combine with causal constraint
        final_mask = mask * causal_mask.unsqueeze(0)

        if window_size >= 0:
            window_mask = (
                torch.tril(torch.ones_like(final_mask)).bool()
                * ~torch.tril(torch.ones_like(final_mask), diagonal=-window_size).bool()
            )
            final_mask = final_mask * window_mask.unsqueeze(0)

        # Create block mask for flex_attention
        # flex_attention expects a function that returns True for positions to attend to
        def mask_fn(b, h, q_idx, kv_idx):
            return final_mask[b, q_idx, kv_idx] > 0

        block_mask = create_block_mask(
            mask_fn, B=batch_size, H=1, Q_LEN=seq_len, KV_LEN=seq_len
        )

        return block_mask

    def create_score_mod(self, masking_score):
        """
        Create a score modification function for flex_attention.

        Args:
            masking_score: [batch, seq_len, seq_len] tensor of probabilities

        Returns:
            score_mod: Function for flex_attention score modification
        """

        def score_mod(score, b, h, q_idx, kv_idx):
            # Get the block score for this position
            return score * masking_score[b, q_idx, kv_idx]

        return score_mod

    def create_window_causal_mask(self, window_size):
        def window_causal_mask_Fn(b, h, q_idx, kv_idx):
            if window_size == -1:
                return q_idx >= kv_idx
            elif window_size >= 0:
                mask = (q_idx - kv_idx) < window_size
                return mask >= 0
            else:
                raise ValueError("window_size must be -1 or >= 0")

        return window_causal_mask_Fn
