import torch
from hmnet.models.config_hmnet import HMNetConfig
from hmnet.models.hmnet import HMNet
from hmnet.models.config_hnet import SSMConfig, AttnConfig
from tqdm import tqdm


@torch.no_grad()
def test_hmnet():
    ssm_config = SSMConfig(chunk_size=256, d_conv=4, d_state=128, expand=2)
    attn_config = AttnConfig(
        num_heads=[16, 16], rotary_emb_dim=[4, 4], window_size=[1023, -1]
    )
    config = HMNetConfig(
        d_model=[64, 128],
        vocab_size=256,
        tie_embeddings=True,
        encoder_attn_cfg=attn_config,
        decoder_attn_cfg=attn_config,
        encoder_ssm_cfg=ssm_config,
        decoder_ssm_cfg=ssm_config,
        arch_layout=["m1T1", ["t1"], "T1m1"],
        d_intermediate=[128, 256],
    )

    # Test forward pass
    batch_size = 2
    seqlen = 5

    # torch.cuda.empty_cache()
    model = HMNet(config=config, device="cuda", stage_idx=0)
    for _ in range(10):
        input_tensor = torch.randn(
            batch_size,
            seqlen,
            config.d_model[0],
            device="cuda",
        )
        mask = torch.ones(batch_size, seqlen, dtype=torch.bool, device="cuda")
        output = model(hidden_states=input_tensor, mask=mask)

        assert output[0].shape == (batch_size, seqlen, config.d_model[0])


@torch.no_grad()
def test_step():
    ssm_config = SSMConfig(chunk_size=256, d_conv=8, d_state=128, expand=2)
    attn_config = AttnConfig(
        num_heads=[16, 16], rotary_emb_dim=[4, 4], window_size=[3, -1]
    )
    config = HMNetConfig(
        d_model=[64, 128],
        vocab_size=256,
        tie_embeddings=True,
        encoder_attn_cfg=attn_config,
        decoder_attn_cfg=attn_config,
        encoder_ssm_cfg=ssm_config,
        decoder_ssm_cfg=ssm_config,
        arch_layout=["T1", ["t1"], "T1"],
        d_intermediate=[128, 256],
    )
    model = HMNet(config=config, device="cuda", stage_idx=0).to("cuda")

    # Test step method
    batch_size = 1
    seqlen = 10
    input_tensor = torch.randn(
        batch_size,
        seqlen,
        config.d_model[0],
        device="cuda",
        dtype=torch.float32,
    )
    inference_params = model.allocate_inference_cache(batch_size, seqlen + 150)

    mask = torch.ones(batch_size, seqlen, device="cuda", dtype=torch.bool)
    output = model.forward(
        hidden_states=input_tensor, mask=mask, inference_params=inference_params
    )
    for _ in tqdm(range(30)):

        current_emb = torch.randn(
            batch_size, 1, config.d_model[0], device="cuda", dtype=torch.float32
        )
        output = model.step(
            hidden_states=current_emb, inference_params=inference_params
        )

    assert output[0].shape == (batch_size, 1, config.d_model[0])


def test_block_diag():
    context_lens = [2, 3]
    qa_len = 2
    blocks = [torch.ones(1, 1, dtype=torch.bool)]  # BOS
    blocks += [torch.ones(l, l, dtype=torch.bool).tril() for l in context_lens]
    blocks.append(torch.ones(qa_len, qa_len, dtype=torch.bool).tril())
    blocks.append(torch.ones(1, 1, dtype=torch.bool))  # EOS
    causal_attn_mask = torch.block_diag(*blocks)
    # QA 행들에서 선택된 context 블록만 attend 가능
    total_ctx_len = sum(context_lens)
    qa_row_slice = slice(total_ctx_len, total_ctx_len + qa_len)
    ctx_start = sum(context_lens[:1])
    ctx_end = ctx_start + context_lens[1]
    # 초기 값은 False → 선택된 context 범위 열만 True
    causal_attn_mask[qa_row_slice, ctx_start:ctx_end] = True
    causal_attn_mask[:, 0] = True  # BOS 토큰은 모두 attend 가능
    causal_attn_mask[-1, :] = True  # EOS 토큰은 모두 attend 가능
    print(causal_attn_mask)
    pass


def test_auxiliary_loss():
    pass


def _compute_auxiliary_loss(
    chunk_attn_logit: torch.Tensor,
    attn_label: torch.Tensor,
    pad_mask: torch.Tensor,
):
    # chunk_attn_score: (S,B,L,L) or (B,L,L)
    # attn_label: (B,L,L)
    # pad_mask: (B,L)

    preds = chunk_attn_logit.float()
    targets = attn_label.bool()

    if preds.dim() == 3:
        preds = preds.unsqueeze(0)  # (1,B,L,L)
    if targets.dim() == 3:
        targets = targets.unsqueeze(0)  # (1,B,L,L)
    targets = targets.repeat(preds.size(0), 1, 1, 1)

    pad_mask = pad_mask.bool()
    valid_mask = (
        (pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(-2))
        .unsqueeze(0)
        .repeat(preds.size(0), 1, 1, 1)
    )
    L = pad_mask.size(1)
    causal_mask = (
        torch.ones(L, L, dtype=torch.bool, device=pad_mask.device)
        .tril()
        .reshape(1, 1, L, L)
    )
    valid_mask = valid_mask & causal_mask  # (1,B,L,L)
    valid_zeros = valid_mask & ~targets.bool()
    loss = self.aux_criterion(preds[valid_zeros], targets[valid_zeros].float())

    valid_ones = valid_mask & targets.bool()
    for i, ws in enumerate(self.config.decoder_attn_cfg.window_size[:-1]):
        causal_without_sliding_window = torch.ones(
            L, L, dtype=torch.bool, device=pad_mask.device
        ).tril(-ws)
        valid_ones[i] = valid_ones[i] & causal_without_sliding_window

    valid_preds_ones = preds[valid_ones]
    num_top_10_pct: int = max(1, valid_ones.sum().item() // 10)  # 최소 1개
    num_bottom_50_pct: int = max(1, valid_ones.sum().item() // 2)
    preds_top_10_pct_ones = valid_preds_ones.topk(num_top_10_pct).values
    loss += self.aux_criterion(
        preds_top_10_pct_ones,
        torch.ones_like(preds_top_10_pct_ones, dtype=torch.float32),
    )
    preds_bottom_50_pct_ones = valid_preds_ones.topk(
        num_bottom_50_pct, largest=False
    ).values
    loss += self.aux_criterion(
        preds_bottom_50_pct_ones,
        torch.zeros_like(preds_bottom_50_pct_ones, dtype=torch.float32),
    )

    return loss
