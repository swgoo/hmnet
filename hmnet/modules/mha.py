# Copyright (c) 2023, Tri Dao.

import torch
import torch.nn as nn
from einops import rearrange

from flash_attn import (
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

from torch.nn.attention.flex_attention import flex_attention, BlockMask
from hnet.modules.isotropic import IsotropicInferenceParams

from hnet.modules.rotary import RotaryEmbedding


class FlashCausalSelfAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
    """

    def __init__(
        self,
        softmax_scale=None,
        window_size=(-1, -1),
    ):
        super().__init__()
        assert (
            flash_attn_varlen_qkvpacked_func is not None
        ), "FlashAttention is not installed"
        self.softmax_scale = softmax_scale
        self.window_size = window_size

    def forward(self, qkv, cu_seqlens=None, max_seqlen=None):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value.
                If cu_seqlens is None and max_seqlen is None, then qkv has shape (B, S, 3, H, D).
                If cu_seqlens is not None and max_seqlen is not None, then qkv has shape
                (total, 3, H, D), where total is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into qkv.
            max_seqlen: int. Maximum sequence length in the batch.
        Returns:
        --------
            out: (total, H, D) if cu_seqlens is not None and max_seqlen is not None,
                else (B, S, H, D).
        """
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda
        if cu_seqlens is not None:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            return flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_seqlen,
                softmax_scale=self.softmax_scale,
                causal=True,
                window_size=(self.window_size, -1),
            )
        else:
            return flash_attn_qkvpacked_func(
                qkv,
                softmax_scale=self.softmax_scale,
                causal=True,
                window_size=(self.window_size, -1),
            )


class FlexAttention(nn.Module):
    """Implement causal self-attention using FlexAttention with block mask support.
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
        assert max_seqlen is not None
        assert isinstance(max_seqlen, int)

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


class FlashCausalCrossAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        window_size: The window size to use for the attention.
    """

    def __init__(
        self,
        softmax_scale=None,
        window_size=(-1, -1),
    ):
        super().__init__()
        assert (
            flash_attn_varlen_kvpacked_func is not None
        ), "FlashAttention is not installed"
        assert flash_attn_kvpacked_func is not None, "FlashAttention is not installed"
        self.softmax_scale = softmax_scale
        self.window_size = window_size

    def forward(
        self,
        q,
        kv,
        cu_seqlens=None,
        max_seqlen=None,
        cu_seqlens_k=None,
        max_seqlen_k=None,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q: The tensor containing the query. (B, Sq, H, D)
            kv: The tensor containing the key and value. (B, Sk, 2, H_k, D)
            causal: if passed, will override self.causal
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into q.
            max_seqlen: int. Maximum sequence length in the batch of q.
            cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into kv.
            max_seqlen_k: int. Maximum sequence length in the batch of k and v.
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda and kv.is_cuda
        if cu_seqlens is not None:
            assert cu_seqlens.dtype == torch.int32
            assert max_seqlen is not None
            assert isinstance(max_seqlen, int)
            assert cu_seqlens_k is not None
            assert cu_seqlens_k.dtype == torch.int32
            assert max_seqlen_k is not None
            assert isinstance(max_seqlen_k, int)
            return flash_attn_varlen_kvpacked_func(
                q,
                kv,
                cu_seqlens,
                cu_seqlens_k,
                max_seqlen,
                max_seqlen_k,
                softmax_scale=self.softmax_scale,
                causal=True,
                window_size=(self.window_size, -1),
            )
        else:
            assert kv.shape[0] == q.shape[0] and kv.shape[4] == q.shape[3]
            return flash_attn_kvpacked_func(
                q,
                kv,
                softmax_scale=self.softmax_scale,
                causal=True,
                window_size=(self.window_size, -1),
            )


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


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


class CausalMHA(nn.Module):
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
        """
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale
        self.rotary_emb_dim = rotary_emb_dim

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
        self.inner_attn = FlashCausalSelfAttention(
            softmax_scale=softmax_scale,
            window_size=window_size,
        )
        self.inner_cross_attn = FlashCausalCrossAttention(
            softmax_scale=softmax_scale,
            window_size=window_size,
        )
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

    def _apply_rotary_update_kvcache_attention(self, q, kv, inference_params):
        """
        Fast path that combine 3 steps: apply rotary to Q and K, update kv cache, and apply attention.
        q: (batch_size, seqlen_q, nheads, head_dim)
        kv: (batch_size, seqlen_k, 2, nheads_kv, head_dim)
        """
        assert inference_params is not None and inference_params.seqlen_offset > 0
        if self.rotary_emb_dim > 0:
            assert self.rotary_emb.scale is None, "This code path does not support xPos"
            self.rotary_emb._update_cos_sin_cache(
                inference_params.max_seqlen, device=q.device, dtype=q.dtype
            )
            rotary_cos, rotary_sin = (
                self.rotary_emb._cos_cached,
                self.rotary_emb._sin_cached,
            )
        else:
            rotary_cos, rotary_sin = None, None
        batch = q.shape[0]
        kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
        cache_seqlens = (
            inference_params.lengths_per_sample[:batch]
            if inference_params.lengths_per_sample is not None
            else inference_params.seqlen_offset
        )
        context = flash_attn_with_kvcache(
            q,
            kv_cache[:, :, 0],
            kv_cache[:, :, 1],
            kv[:, :, 0],
            kv[:, :, 1],
            rotary_cos=rotary_cos,
            rotary_sin=rotary_sin,
            cache_seqlens=cache_seqlens,
            softmax_scale=self.softmax_scale,
            causal=True,
            rotary_interleaved=(
                self.rotary_emb.interleaved if self.rotary_emb_dim > 0 else False
            ),
        )
        return context

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        if (
            inference_params.seqlen_offset == 0
            or flash_attn_with_kvcache is None
            or not self.use_flash_attn
        ):
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self.inner_cross_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache = inference_params.key_value_memory_dict[self.layer_idx][:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=True,
            )

    def forward(
        self,
        x,
        cu_seqlens=None,
        max_seqlen=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            # assert self.rotary_emb_dim == 0
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
        if (
            inference_params is None
            or inference_params.seqlen_offset == 0
            or (self.rotary_emb_dim == 0 or self.rotary_emb_dim % 16 != 0)
        ):
            if self.rotary_emb_dim > 0:
                # qkv = self.rotary_emb(
                #     qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                # )
                qkv = self.rotary_emb(qkv, seqlen_offset=seqlen_offset, **kwargs)
            if inference_params is None:
                context = self.inner_attn(qkv, **kwargs)
            else:
                context = self._update_kvcache_attention(
                    qkv[:, :, 0], qkv[:, :, 1:], inference_params
                )
        else:
            context = self._apply_rotary_update_kvcache_attention(
                qkv[:, :, 0], qkv[:, :, 1:], inference_params
            )
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out

    def step(self, x, inference_params):
        return self.forward(x, inference_params=inference_params)


class BlockMaskingMHA(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        qkv_proj_bias=False,
        out_proj_bias=False,
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

        # Apply rotary embeddings if configured
        if self.rotary_emb_dim > 0:
            qkv = self.rotary_emb(
                qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
            )

        # Split qkv into query and key-value
        q, kv = qkv[:, :, 0], qkv[:, :, 1:]

        if inference_params is None:
            # Regular forward pass during training
            context = self.inner_attn(
                q, kv, block_mask=block_mask, score_mod=score_mod, **kwargs
            )
        else:
            # Inference mode with KV caching
            if self.layer_idx is None:
                raise ValueError("Generation requires layer_idx in the constructor")

            # Update KV cache
            kv_cache = self._update_kv_cache(kv, inference_params)

            # Get the current batch size
            batch = q.shape[0]

            # Get sequence length information for the current batch
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )

            # Use the cached KV values for attention
            context = self.inner_attn(
                q,
                kv_cache,
                block_mask=block_mask,
                score_mod=score_mod,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        # Project output
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out

    def step(self, x, inference_params, block_mask=None, score_mod=None):
        """Simplified forward for step-by-step generation"""
        return self.forward(
            x,
            block_mask=block_mask,
            score_mod=score_mod,
            inference_params=inference_params,
        )
