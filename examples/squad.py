import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import lightning as L
import requests
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from hmnet.models.config_hmnet import HMNetConfig
from hmnet.models.hmnet import CausalLMOutput, HMNetForCausalLM
from hmnet.models.tokenizer import ByteTokenizer

byte_tokenizer = ByteTokenizer()


def load_SQuAD_json(
    squad_data_path=Path("data/squad.json"),
    squad_dev_data_path=Path("data/squad_dev.json"),
):
    if not squad_data_path.exists():
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json"
        response = requests.get(url)
        with open(squad_data_path, "wb") as f:
            f.write(response.content)
    if not squad_dev_data_path.exists():
        url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
        response = requests.get(url)
        with open(squad_dev_data_path, "wb") as f:
            f.write(response.content)

    with open(squad_data_path) as f:
        train_data = json.load(f)["data"]
    with open(squad_dev_data_path) as f:
        dev_data = json.load(f)["data"]
    return {"train": train_data, "dev": dev_data}


@dataclass
class SQuADExample:
    qa_id: str
    context: Tensor
    question: Tensor
    answer: Tensor
    is_impossible: bool
    answer_start: int

    @classmethod
    def get_SQuAD_examples(
        cls, data: dict, max_question_len=128, max_answer_len=64
    ) -> list["SQuADExample"]:

        squad_examples: list["SQuADExample"] = []

        for item in data:
            for paragraph in item["paragraphs"]:
                context: str = paragraph["context"]
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
                            SQuADExample(
                                qa_id=qa["id"],
                                context=byte_tokenizer.encode([context])[0],
                                question=byte_tokenizer.encode([question])[0],
                                answer=byte_tokenizer.encode([ans_text])[0],
                                is_impossible=is_impossible,
                                answer_start=ans["answer_start"],
                            )
                        )

        return squad_examples


@dataclass
class QAInput:
    input_ids: torch.Tensor
    answer_masks: torch.Tensor
    pad_masks: torch.Tensor
    qa_ids: list[str]

    @classmethod
    def get_qa_inputs(cls, squad_examples: list[SQuADExample]) -> list["QAInput"]:
        result: list["QAInput"] = []
        for ex in squad_examples:
            if ex.is_impossible:
                answer_ids = byte_tokenizer.encode(["no answer"])[0]
            else:
                answer_ids = ex.answer

            input_ids = torch.cat([ex.context, ex.question, answer_ids], dim=0)
            input_ids = byte_tokenizer.add_special_tokens(input_ids, bos=True, eos=True)
            answer_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            answer_mask[-(answer_ids.size(0) + 1) :] = True  # answer + EOS

            result.append(
                QAInput(
                    input_ids=input_ids,
                    answer_masks=answer_mask,
                    pad_masks=torch.ones_like(input_ids, dtype=torch.bool),
                    qa_ids=[ex.qa_id],
                )
            )

        return result


class QADataset(Dataset):
    def __init__(self, qa_inputs: list[QAInput]):
        self.data: list[QAInput] = qa_inputs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class QATripletDataset(IterableDataset):
    def __init__(
        self, squad_examples: list[SQuADExample], seed=42, samples_per_epoch=5000
    ):
        self.data: list[SQuADExample] = squad_examples
        self.seed = seed
        self._rng = torch.Generator().manual_seed(seed)
        self._len = len(squad_examples)
        self.samples_per_epoch = samples_per_epoch
        self.max_context_length = 512

        for ex in self.data:
            ex.context = self._crop_context(
                ex.context, ex.answer_start, ex.answer.size(0)
            )

    def _crop_context(
        self, context_ids: Tensor, answer_start: int, answer_len: int
    ) -> Tensor:
        # answer_start 위치가 정확하지는 않아서, max_len을 충분히 크게 잡아야 함
        """
        Crops the context_ids to a maximum length of max_len, centering the answer_text if present.
        """
        assert self.max_context_length > 0, "max_context_length must be positive"
        assert answer_len >= 0, "answer_len must be non-negative"
        assert answer_start >= 0, "answer_start must be non-negative"
        assert answer_len <= context_ids.size(
            0
        ), "answer_len must be less than or equal to context_ids.size(0)"

        answer_end = answer_start + answer_len
        if answer_end > len(context_ids):
            return context_ids[: self.max_context_length]

        if answer_len >= self.max_context_length:
            return context_ids[: self.max_context_length]

        window_start = max(0, answer_start - self.max_context_length // 2)
        window_end = min(len(context_ids), answer_start + self.max_context_length // 2)

        return context_ids[window_start:window_end]

    def __len__(self):
        return self.samples_per_epoch

    def _sample_triplet(self):
        indices = torch.randint(0, self._len, (3,), generator=self._rng)
        contexts = [self.data[i].context for i in indices]
        context_lens = [c.size(0) for c in contexts]

        selected_qa_index = torch.randint(0, 3, (1,), generator=self._rng).item()
        qa_example = self.data[indices[selected_qa_index]]
        qa = torch.cat([qa_example.question, qa_example.answer], dim=0)

        tokens = torch.cat([*contexts, qa], dim=0)
        tokens: torch.Tensor = byte_tokenizer.add_special_tokens(
            tokens, bos=True, eos=True
        )

        answer_mask = torch.zeros_like(tokens, dtype=torch.bool)
        a_slice = slice(
            sum(context_lens) + qa_example.question.size(0),
            sum(context_lens)
            + qa_example.question.size(0)
            + qa_example.answer.size(0)
            + 1,
        )  # +1 for EOS
        answer_mask[a_slice] = True

        return QAInput(
            input_ids=tokens,
            answer_masks=answer_mask,
            pad_masks=torch.ones_like(tokens, dtype=torch.bool),
            qa_ids=[qa_example.qa_id],
        )

    def __iter__(self):
        for _ in range(self.samples_per_epoch):
            yield self._sample_triplet()


def qa_input_collate_fn(batch: list[QAInput]):
    seqs = [b.input_ids for b in batch]
    answer_masks = [b.answer_masks for b in batch]
    pad_masks = [b.pad_masks for b in batch]
    qa_ids = [b.qa_ids[0] for b in batch]

    padded_seqs = pad_sequence(seqs, batch_first=True, padding_value=0)
    max_len = padded_seqs.size(1)

    pad_masks = torch.stack(
        [F.pad(m, (0, max_len - m.size(0)), value=False) for m in pad_masks]
    )  # (B,L)

    answer_masks = torch.stack(
        [F.pad(m, (0, max_len - m.size(0)), value=False) for m in answer_masks]
    )  # (B,L)

    return QAInput(
        input_ids=padded_seqs,
        answer_masks=answer_masks,
        pad_masks=pad_masks,
        qa_ids=qa_ids,
    )


class QADataModule(L.LightningDataModule):
    def __init__(self, train_dataset, val_dataset, train_batch_size, val_batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            collate_fn=qa_input_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=qa_input_collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            collate_fn=qa_input_collate_fn,
        )


class HMNetForSQuAD(HMNetForCausalLM, L.LightningModule):
    def __init__(
        self,
        config: HMNetConfig,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        pad_token_id: int = 0,
    ):
        super().__init__(config=config)
        self.lr = lr
        self.weight_decay = weight_decay
        self.pad_token_id = pad_token_id
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.aux_criterion = torch.nn.BCEWithLogitsLoss()

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

    def common_step(self, batch: QAInput, batch_idx: int):
        outputs: CausalLMOutput = self(batch.input_ids, mask=batch.pad_masks)
        return self._compute_loss(
            logits=outputs.logits,
            input_ids=batch.input_ids.long(),
            answer_mask=batch.answer_masks,
            pad_mask=batch.pad_masks,
        )

    def training_step(self, batch: QAInput, batch_idx: int):
        loss = self.common_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: QAInput, batch_idx: int):
        loss = self.common_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return {"val_loss": loss}

    def predict_step(self, batch: QAInput, batch_idx: int):
        outputs = self(batch.input_ids, mask=batch.pad_masks)
        return {"outputs": outputs}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HMNet model")
    parser.add_argument(
        "--model_config",
        type=str,
        required=False,
        help="Path to model config file",
        # default="configs/hmnet_3stage_XL.yaml",
        default="configs/hmnet_1stage_L.yaml",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        help="Path to model file",
        # default="ckpts/hmnet_3stage_XL_from_2stage_XL.pt",
        default=None,
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
    )
    if args.model_path is not None and Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path), strict=False)

    data = load_SQuAD_json()
    train_squad_examples = SQuADExample.get_SQuAD_examples(data["train"])
    train_dataset = QATripletDataset(train_squad_examples, samples_per_epoch=5000)

    val_squad_examples = SQuADExample.get_SQuAD_examples(
        data["dev"], max_question_len=4096, max_answer_len=4096
    )
    val_input_ids_list = QAInput.get_qa_inputs(val_squad_examples)
    val_dataset = QADataset(val_input_ids_list)

    data_module = QADataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_batch_size=args.batch_size,
        val_batch_size=1,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="hmnet-squad-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
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
        # ckpt_path="checkpoints/hmnet-squad-epoch=13-val_loss=0.25.ckpt",
    )
