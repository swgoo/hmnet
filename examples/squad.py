import argparse
import json
from math import e
import random
from dataclasses import dataclass
from pathlib import Path

import lightning as L
from omegaconf import OmegaConf
import requests
import torch
import torch.nn.functional as F  # 추가
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import ByteTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from hmnet.models.config_hmnet import HMNetConfig
from hmnet.models.hmnet import CausalLMOutput, HMNetForCausalLM
from hmnet.models.tokenizer import ByteTokenizer
from hmnet.modules.dc import RoutingModuleOutput
from hmnet.modules.profiler import ChunkAttnScoreProfiler


@dataclass
class SquadExample:
    context: str
    question: str
    answer: str
    is_impossible: bool
    answer_start: int

    def __str__(self) -> str:
        return f"Context: {self.context}\nQuestion: {self.question}\nAnswer: {self.answer}\nIs Impossible: {self.is_impossible}\nAnswer Start: {self.answer_start}"


def load_SQuAD_json(squad_data_path=Path("data/squad.json")) -> list[dict]:
    if not squad_data_path.exists():
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
        response = requests.get(url)
        with open(squad_data_path, "wb") as f:
            f.write(response.content)

    with open(squad_data_path) as f:
        data = json.load(f)["data"]
    return data


def get_SQuAD_examples(data: dict) -> list[SquadExample]:

    squad_examples: list[SquadExample] = []
    max_question_len = 128
    max_answer_len = 64

    for item in data:
        for paragraph in item["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                if len(question) > max_question_len:
                    continue
                is_impossible = qa["is_impossible"]
                if is_impossible:
                    answers = qa["plausible_answers"]
                else:
                    answers = qa["answers"]
                for ans in answers:
                    ans_text = ans["text"]
                    if len(ans_text) > max_answer_len:
                        continue
                    squad_examples.append(
                        SquadExample(
                            context,
                            question,
                            ans_text,
                            is_impossible,
                            ans["answer_start"],
                        )
                    )

    return squad_examples


def eda_squad(squad_examples: list[SquadExample]):
    print(f"Number of examples: {len(squad_examples)}")
    context_lengths = [len(ex.context) for ex in squad_examples]
    print(f"Average context length: {sum(context_lengths) / len(context_lengths)}")
    print(f"Max context length: {max(context_lengths)}")
    print(f"Min context length: {min(context_lengths)}")
    # std
    print(
        f"Context length std: {torch.std(torch.tensor(context_lengths).float()).item()}"
    )

    question_lengths = [len(ex.question) for ex in squad_examples]
    print(f"Average question length: {sum(question_lengths) / len(question_lengths)}")
    print(f"Max question length: {max(question_lengths)}")
    print(f"Min question length: {min(question_lengths)}")
    # std
    print(
        f"Question length std: {torch.std(torch.tensor(question_lengths).float()).item()}"
    )

    answer_lengths = [len(ex.answer) for ex in squad_examples if not ex.is_impossible]
    print(f"Average answer length: {sum(answer_lengths) / len(answer_lengths)}")
    print(f"Max answer length: {max(answer_lengths)}")
    print(f"Min answer length: {min(answer_lengths)}")
    # std
    print(
        f"Answer length std: {torch.std(torch.tensor(answer_lengths).float()).item()}"
    )


@dataclass
class QAInputIDs:
    context: ByteTensor
    question: ByteTensor
    answer: ByteTensor


def crop_context(
    context_text: str, answer_start: int, answer_text: str, max_len: int
) -> str:
    """
    Returns a substring of context_text of at most max_len chars that contains the answer.
    If impossible (answer_text == ""), returns the leading max_len chars.
    """
    if max_len <= 0:
        return ""
    if not answer_text:  # impossible case
        return context_text[:max_len]

    answer_end = answer_start + len(answer_text)
    if answer_start < 0 or answer_end > len(context_text):
        # Fallback: invalid span, just truncate
        return context_text[:max_len]

    # If answer itself longer than max_len, just return its head
    if len(answer_text) >= max_len:
        return answer_text[:max_len]

    # Try to center answer in the window
    remaining = max_len - len(answer_text)
    left_budget = remaining // 2
    right_budget = remaining - left_budget

    window_start = max(0, answer_start - left_budget)
    window_end = min(len(context_text), answer_end + right_budget)

    # Adjust if we are short due to boundaries
    current_len = window_end - window_start
    if current_len < max_len:
        need = max_len - current_len
        # Try extend left then right
        extend_left = min(need, window_start)
        window_start -= extend_left
        need -= extend_left
        if need:
            window_end = min(len(context_text), window_end + need)

    return context_text[window_start:window_end]


def generate_qa_input_ids(squad_examples: list[SquadExample]) -> list[QAInputIDs]:
    tokenizer = ByteTokenizer()
    max_len = 512
    input_ids_list: list[QAInputIDs] = []
    for ex in squad_examples:
        cropped_context = crop_context(ex.context, ex.answer_start, ex.answer, max_len)
        if ex.is_impossible:
            answer_text = "there is no answer"
        else:
            answer_text = ex.answer

        context_ids = tokenizer.encode([cropped_context])[0]["input_ids"]
        question_ids = tokenizer.encode([ex.question])[0]["input_ids"]
        answer_ids = tokenizer.encode([answer_text])[0]["input_ids"]

        input_ids_list.append(
            QAInputIDs(
                context=torch.tensor(context_ids, dtype=torch.uint8),
                question=torch.tensor(question_ids, dtype=torch.uint8),
                answer=torch.tensor(answer_ids, dtype=torch.uint8),
            )
        )

    return input_ids_list


class QAGeneratorDataset(IterableDataset):
    def __init__(self, input_ids_list, seed=42, samples_per_epoch=5000):
        self.data: list[QAInputIDs] = input_ids_list
        self.seed = seed
        self._rng = torch.Generator().manual_seed(seed)
        self._len = len(input_ids_list)
        self.samples_per_epoch = samples_per_epoch  # 한 epoch 당 생성할 합성 샘플 수
        self.tokenizer = ByteTokenizer()

    def __len__(self):
        return self.samples_per_epoch

    def _sample_triplet(self):
        # 3개 context/QA 예제 랜덤 선택
        indices = torch.randint(0, self._len, (3,), generator=self._rng)
        context_lens = [self.data[i].context.size(0) for i in indices]

        selected_qa_index = torch.randint(0, 3, (1,), generator=self._rng).item()
        qa_example = self.data[indices[selected_qa_index]]
        qa = torch.cat([qa_example.question, qa_example.answer], dim=0)
        question_len = qa_example.question.size(0)
        answer_len = qa_example.answer.size(0)
        qa_len = question_len + answer_len

        context_concat = torch.cat([self.data[i].context for i in indices], dim=0)

        # 블록 대각 causal (각 context 및 QA)
        blocks = [torch.ones(1, 1, dtype=torch.bool)]  # BOS
        blocks += [torch.ones(l, l, dtype=torch.bool).tril() for l in context_lens]
        blocks.append(torch.ones(qa_len, qa_len, dtype=torch.bool).tril())
        blocks.append(torch.ones(1, 1, dtype=torch.bool))  # EOS
        causal_attn_mask = torch.block_diag(
            *blocks
        )  # shape: (sum_ctx + qa_len, sum_ctx + qa_len)

        # QA 행들에서 선택된 context 블록만 attend 가능
        total_ctx_len = sum(context_lens)
        qa_row_slice = slice(total_ctx_len, total_ctx_len + qa_len)
        ctx_start = sum(context_lens[:selected_qa_index])
        ctx_end = ctx_start + context_lens[selected_qa_index]
        # 초기 값은 False → 선택된 context 범위 열만 True
        causal_attn_mask[qa_row_slice, ctx_start:ctx_end] = True
        causal_attn_mask[:, 0] = True  # BOS 토큰은 모두 attend 가능
        causal_attn_mask[-1, :] = True  # EOS 토큰은 모두 attend 가능

        # 최종 토큰 시퀀스 (context들 + QA)
        tokens = torch.cat([context_concat, qa], dim=0)
        tokens = self.tokenizer.add_special_tokens(tokens, bos=True, eos=True)

        # answer mask 1d
        answer_mask = torch.zeros(tokens.size(0), dtype=torch.bool)
        a_slice = slice(total_ctx_len + question_len, total_ctx_len + qa_len)
        answer_mask[a_slice] = True

        return tokens, causal_attn_mask, answer_mask

    def __iter__(self):
        # samples_per_epoch 만큼 샘플 생성
        for _ in range(self.samples_per_epoch):
            yield self._sample_triplet()


def collate_fn(batch):
    seqs, label_masks, answer_masks = zip(*batch)
    lengths = torch.tensor([s.size(0) for s in seqs])
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    max_len = padded_seqs.size(1)

    padded_labels = torch.stack(
        [
            F.pad(m, (0, max_len - m.size(0), 0, max_len - m.size(0)), value=0.0)
            for m in label_masks
        ]
    )  # (B,L,L)

    lengths = lengths.to(padded_seqs.device)
    input_ids_mask = torch.arange(max_len, device=padded_seqs.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)

    answer_masks = torch.stack(
        [F.pad(m, (0, max_len - m.size(0)), value=False) for m in answer_masks]
    )  # (B,L)

    return {
        "input_ids": padded_seqs,
        "attn_label": padded_labels,
        "answer_mask": answer_masks,
        "mask": input_ids_mask,
    }


class QADataModule(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
        )


class HMNetForSQuAD(HMNetForCausalLM, L.LightningModule):
    def __init__(
        self,
        config: HMNetConfig,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        pad_token_id: int = 0,
        aux_loss_weight: float = 1.0,
    ):
        super().__init__(config=config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.pad_token_id = pad_token_id
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.aux_criterion = torch.nn.BCEWithLogitsLoss()
        self.aux_loss_weight = aux_loss_weight

    def _compute_loss(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        answer_mask: torch.Tensor,
        pad_mask: torch.Tensor,
    ):
        # logits: (B, L, E)
        shift_logits = logits[:, :-1, :].contiguous()  # (B, L-1, E)
        shift_labels = input_ids[:, 1:].contiguous()  # (B, L-1)

        shift_mask = pad_mask[:, 1:] & answer_mask[:, 1:]
        shift_labels = shift_labels.masked_fill(~shift_mask, self.pad_token_id)
        loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return loss

    def _compute_auxiliary_loss(
        self,
        routing_outputs: list[RoutingModuleOutput],
        chunked_attn_logits: list[torch.Tensor],
        attn_label: torch.Tensor,
        pad_mask: torch.Tensor,
    ):
        chunked_attn_labels = ChunkAttnScoreProfiler.chunk_all_stage(
            attn_label, routing_outputs
        )
        chunked_pad_masks = ChunkAttnScoreProfiler.chunk_all_stage(
            pad_mask.bool().unsqueeze(-1) & pad_mask.bool().unsqueeze(-2),
            routing_outputs,
        )

        total_loss = torch.tensor(0.0, device=pad_mask.device)
        for logit, al, pm, ws in zip(
            chunked_attn_logits,
            chunked_attn_labels,
            chunked_pad_masks,
            self.config.decoder_attn_cfg.window_size,
        ):
            causal_mask_without_sliding_window = torch.ones_like(pm).tril(-ws)
            valid_mask = pm & causal_mask_without_sliding_window

            valid_zeros = valid_mask & ~al.bool()
            total_loss += self.aux_criterion(
                logit[valid_zeros], al[valid_zeros].float()
            )
            valid_ones = valid_mask & al.bool()
            num_top_10_pct: int = max(1, valid_ones.sum().item() // 10)  # 최소 1개
            num_bottom_50_pct: int = max(1, valid_ones.sum().item() // 2)
            valid_preds_ones = logit[valid_ones]
            if valid_preds_ones.numel() == 0:
                continue
            preds_top_10_pct_ones = valid_preds_ones.topk(num_top_10_pct).values
            total_loss += self.aux_criterion(
                preds_top_10_pct_ones,
                torch.ones_like(preds_top_10_pct_ones, dtype=torch.float32),
            )
            if valid_ones.sum().item() > 1:
                preds_bottom_50_pct_ones = valid_preds_ones.topk(
                    num_bottom_50_pct, largest=False
                ).values
                total_loss += self.aux_criterion(
                    preds_bottom_50_pct_ones,
                    torch.zeros_like(preds_bottom_50_pct_ones, dtype=torch.float32),
                )

        return (
            total_loss
            if total_loss.isfinite()
            else torch.tensor(0.0, device=pad_mask.device)
        )

    # def _compute_auxiliary_loss(
    #     self,
    #     chunk_attn_logit: torch.Tensor,
    #     attn_label: torch.Tensor,
    #     pad_mask: torch.Tensor,
    # ):
    #     # chunk_attn_score: (S,B,L,L) or (B,L,L)
    #     # attn_label: (B,L,L)
    #     # pad_mask: (B,L)

    #     preds = chunk_attn_logit.float()
    #     targets = attn_label.bool()

    #     if preds.dim() == 3:
    #         preds = preds.unsqueeze(0)  # (S,B,L,L)
    #     targets = targets.unsqueeze(0).repeat(preds.size(0), 1, 1, 1)  # (S,B,L,L)

    #     pad_mask = pad_mask.bool()
    #     valid_mask = (
    #         (pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(-2))
    #         .unsqueeze(0)
    #         .repeat(preds.size(0), 1, 1, 1)
    #     )
    #     L = pad_mask.size(1)
    #     causal_mask = torch.ones(L, L, dtype=torch.bool, device=pad_mask.device).tril()
    #     valid_mask = valid_mask & causal_mask  # (S,B,L,L)
    #     valid_zeros = valid_mask & ~targets.bool()
    #     loss = self.aux_criterion(preds[valid_zeros], targets[valid_zeros].float())

    #     valid_ones = valid_mask & targets.bool()
    #     for i, ws in enumerate(self.config.decoder_attn_cfg.window_size[:-1]):
    #         causal_without_sliding_window = torch.ones(
    #             L, L, dtype=torch.bool, device=pad_mask.device
    #         ).tril(-ws)
    #         valid_ones[i] = valid_ones[i] & causal_without_sliding_window

    #     valid_preds_ones = preds[valid_ones]
    #     num_top_10_pct: int = max(1, valid_ones.sum().item() // 10)  # 최소 1개
    #     num_bottom_50_pct: int = max(1, valid_ones.sum().item() // 2)
    #     preds_top_10_pct_ones = valid_preds_ones.topk(num_top_10_pct).values
    #     loss += self.aux_criterion(
    #         preds_top_10_pct_ones,
    #         torch.ones_like(preds_top_10_pct_ones, dtype=torch.float32),
    #     )
    #     preds_bottom_50_pct_ones = valid_preds_ones.topk(
    #         num_bottom_50_pct, largest=False
    #     ).values
    #     loss += self.aux_criterion(
    #         preds_bottom_50_pct_ones,
    #         torch.zeros_like(preds_bottom_50_pct_ones, dtype=torch.float32),
    #     )

    #     return loss

    def training_step(self, batch, batch_idx: int):
        input_ids = batch["input_ids"]
        attn_label = batch["attn_label"]
        answer_mask = batch["answer_mask"]
        pad_mask = batch.get("mask", None)

        outputs: CausalLMOutput = self(input_ids, mask=pad_mask)
        dechunk_attn_scores = ChunkAttnScoreProfiler.dechunk_all_stage(
            outputs.chunk_attn_logit_output, outputs.bpred_output
        )

        loss = self._compute_loss(
            outputs.logits, input_ids.long(), pad_mask, answer_mask
        )
        aux_loss = self._compute_auxiliary_loss(
            routing_outputs=outputs.bpred_output,
            chunked_attn_logits=outputs.chunk_attn_logit_output,
            attn_label=attn_label,
            pad_mask=pad_mask,
        )
        total = loss + self.aux_loss_weight * aux_loss

        self.log("lm_loss", loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("aux_loss", aux_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("train_loss", total, prog_bar=True, on_step=True, on_epoch=True)
        return total

    def validation_step(self, batch, batch_idx: int):
        input_ids = batch["input_ids"]
        attn_2d = batch.get("attn_label", None)
        pad_mask = batch.get("mask", None)
        answer_mask = batch.get("answer_mask", None)
        outputs: CausalLMOutput = self(input_ids, attention_mask=attn_2d, mask=pad_mask)
        logits = outputs.logits
        loss = self._compute_loss(logits, input_ids.long(), pad_mask, answer_mask)
        loss_aux = self._compute_auxiliary_loss(
            attn_label=attn_2d,
            pad_mask=pad_mask,
            chunked_attn_logits=outputs.chunk_attn_logit_output,
            routing_outputs=outputs.bpred_output,
        )
        self.log("val_lmloss", loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_aux_loss", loss_aux, prog_bar=False, on_step=False, on_epoch=True)
        loss = loss + loss_aux
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def predict_step(self, batch, batch_idx: int):
        input_ids = batch["input_ids"]
        attn_2d = batch.get("attn_label", None)
        pad_mask = batch.get("mask", None)
        outputs = self(input_ids, attention_mask=attn_2d, mask=pad_mask)
        logits = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        return {"logits": logits, "input_ids": input_ids}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


def random_split_list(data: list, val_size: float = 0.1, seed: int = 42):
    """
    data 리스트를 랜덤 셔플 후 (1 - val_size) / val_size 로 분할.
    """
    if not 0.0 < val_size < 1.0:
        raise ValueError("val_size must be between 0 and 1.")
    rng = random.Random(seed)
    data_copy = data[:]  # 원본 보존
    rng.shuffle(data_copy)
    n_val = max(1, int(len(data_copy) * val_size))
    val = data_copy[:n_val]
    train = data_copy[n_val:]
    return train, val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HMNet model")
    parser.add_argument(
        "--model_config",
        type=str,
        required=False,
        help="Path to model config file",
        default="configs/hmnet_3stage_XL.yaml",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to model file",
        default="ckpts/hmnet_3stage_XL_from_2stage_XL.pt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Batch size for training",
        default=4,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        required=False,
        help="Learning rate for training",
        default=1e-4,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        required=False,
        help="Number of training epochs",
        default=1_000_000,
    )

    args = parser.parse_args()

    model_cfg = OmegaConf.load(args.model_config)
    default_model_cfg = OmegaConf.structured(HMNetConfig)
    merged_model_cfg = OmegaConf.merge(default_model_cfg, model_cfg)
    model_config: HMNetConfig = OmegaConf.to_object(merged_model_cfg)

    model = HMNetForSQuAD(
        config=model_config,
        lr=args.learning_rate,
        weight_decay=0.0,
        pad_token_id=0,
        aux_loss_weight=1.0,
    )
    if Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path), strict=False)

    squad_data: list[dict] = load_SQuAD_json()
    train_data, val_data = random_split_list(squad_data, val_size=0.1, seed=42)
    train_squad_examples = get_SQuAD_examples(train_data)
    val_squad_examples = get_SQuAD_examples(val_data)

    eda_squad(train_squad_examples)
    train_input_ids_list = generate_qa_input_ids(train_squad_examples)
    train_dataset = QAGeneratorDataset(train_input_ids_list, samples_per_epoch=5000)

    val_input_ids_list = generate_qa_input_ids(val_squad_examples)
    val_dataset = QAGeneratorDataset(val_input_ids_list, samples_per_epoch=500)

    data_module = QADataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="hmnet-squad-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=args.num_epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
        precision="bf16",
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path="checkpoints/hmnet-squad-epoch=13-val_loss=0.25.ckpt",
    )
