# %%
import os
import json
import torch

from dataclasses import dataclass

os.chdir("/root/workspace/hmnet-archive")

with open("data/squad.json") as f:
    data = json.load(f)

data = data["data"]


@dataclass
class SquadExample:
    context: str
    question: str
    answer: str
    is_impossible: bool
    answer_start: int

    def __str__(self) -> str:
        return f"Context: {self.context}\nQuestion: {self.question}\nAnswer: {self.answer}\nIs Impossible: {self.is_impossible}\nAnswer Start: {self.answer_start}"


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
                        context, question, ans_text, is_impossible, ans["answer_start"]
                    )
                )

# %% text length statistics
print(f"Number of examples: {len(squad_examples)}")
context_lengths = [len(ex.context) for ex in squad_examples]
print(f"Average context length: {sum(context_lengths) / len(context_lengths)}")
print(f"Max context length: {max(context_lengths)}")
print(f"Min context length: {min(context_lengths)}")
# std
print(f"Context length std: {torch.std(torch.tensor(context_lengths).float()).item()}")

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
print(f"Answer length std: {torch.std(torch.tensor(answer_lengths).float()).item()}")


# %%
import torch
from torch import ByteTensor
from hmnet.models.tokenizer import ByteTokenizer


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

# %%
import torch
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F  # 추가


class QAGeneratorDataset(IterableDataset):
    def __init__(self, input_ids_list, seed=42, samples_per_epoch=1000):
        self.data: list[QAInputIDs] = input_ids_list
        self.seed = seed
        self.rng = torch.Generator().manual_seed(seed)
        self._len = len(input_ids_list)
        self.samples_per_epoch = samples_per_epoch  # 한 epoch 당 생성할 합성 샘플 수
        self.tokenizer = ByteTokenizer()

    def __len__(self):
        return self.samples_per_epoch

    def _sample_triplet(self):
        # 3개 context/QA 예제 랜덤 선택
        indices = torch.randint(0, self._len, (3,), generator=self.rng)
        context_lens = [self.data[i].context.size(0) for i in indices]

        # 하나 선택하여 (question + answer) 붙인 QA 시퀀스 구성
        selected_qa_index = torch.randint(0, 3, (1,), generator=self.rng).item()
        qa_example = self.data[indices[selected_qa_index]]
        qa = torch.cat([qa_example.question, qa_example.answer], dim=0)
        qa_len = qa.size(0)

        # context 전체 (세 개 이어붙임)
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

        return tokens, causal_attn_mask

    def __iter__(self):
        # samples_per_epoch 만큼 샘플 생성
        for _ in range(self.samples_per_epoch):
            yield self._sample_triplet()


def collate_fn(batch):
    # batch: list[(tokens_1d, attn_mask_2d)]
    seqs, masks = zip(*batch)

    lengths = torch.tensor([s.size(0) for s in seqs])
    # 1D 토큰 시퀀스 패딩 (B, L)
    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)

    max_len = padded_seqs.size(1)

    # 2D 마스크 패딩: 각 (Li, Li) -> (max_len, max_len)
    padded_masks = torch.stack(
        [
            F.pad(m, (0, max_len - m.size(0), 0, max_len - m.size(0)), value=False)
            for m in masks
        ]
    )  # (B, L, L) bool

    # input_ids 용 1D padding mask (B, L) - True where token is real (not padding)
    lengths = lengths.to(padded_seqs.device)
    input_ids_mask = torch.arange(max_len, device=padded_seqs.device).unsqueeze(
        0
    ) < lengths.unsqueeze(1)

    return {
        "input_ids": padded_seqs,
        "attention_mask": padded_masks,  # (B, L, L) bool for causal attention
        "mask": input_ids_mask,  # (B, L) bool for padding / token-level attention
    }


dataset = QAGeneratorDataset(input_ids_list, samples_per_epoch=512)
loader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn, shuffle=False)

# %%
