import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Literal

import lightning as L
import numpy as np
import requests
import torch
import torch.nn.functional as F
import typer
from hmnet.models.config_hnet import HNetConfig
from hnet.models.mixer_seq import HNetForCausalLM
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, IterableDataset

from hmnet.models.config_hmnet import HMNetConfig
from hmnet.models.hmnet import CausalLMOutput, HMNetForCausalLM
from hmnet.models.tokenizer import ByteTokenizer

byte_tokenizer = ByteTokenizer()
app = typer.Typer()


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
        cls, data: dict, max_question_len=128, max_answer_len=64, use_one_answer=False
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
                    answers = qa["answers"]
                    if use_one_answer and len(answers) > 0:
                        answers = answers[:1]
                    for ans in answers:
                        ans_text = ans["text"]
                        if len(ans_text) > max_answer_len:
                            continue
                        if is_impossible:
                            ans_text = "no answer"
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
    SYSTEM_PROMPT: ClassVar[Tensor] = byte_tokenizer.encode(
        ["find a answer in context : "]
    )[0]
    QUESTION_PROMPT: ClassVar[Tensor] = byte_tokenizer.encode([" / question: "])[0]
    ANSWER_PROMPT: ClassVar[Tensor] = byte_tokenizer.encode([" / answer: "])[0]

    @classmethod
    def get_qa_inputs(cls, squad_examples: list[SQuADExample]) -> list["QAInput"]:
        result: list["QAInput"] = []
        for ex in squad_examples:
            answer_ids = ex.answer

            input_ids = torch.cat(
                [
                    cls.SYSTEM_PROMPT,
                    ex.context,
                    cls.QUESTION_PROMPT,
                    ex.question,
                    cls.ANSWER_PROMPT,
                    answer_ids,
                ],
                dim=0,
            )
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
    SYSTEM_PROMPT: Tensor = byte_tokenizer.encode(["find a answer in context : "])[0]
    QUESTION_PROMPT: Tensor = byte_tokenizer.encode([" / question: "])[0]
    ANSWER_PROMPT: Tensor = byte_tokenizer.encode([" / answer: "])[0]

    def __init__(
        self,
        squad_examples: list[SQuADExample],
        max_context_length=512,
        seed: int = 42,
        steps_per_epoch: int = 6000,  # 추가: 한 epoch 당 생성할 샘플 수 (= batches 수 * batch_size)
    ):
        self.data: list[SQuADExample] = squad_examples
        self._len = len(squad_examples)
        self.max_context_length = max_context_length
        self.base_seed = seed
        self.steps_per_epoch = steps_per_epoch
        self.epoch = 0
        self.rank = 0
        self.world_size = 1
        # per-process(또는 싱글) 기본 generator
        self._rng = torch.Generator()
        self._reset_rng()

    def set_epoch(self, epoch: int, rank: int = 0, world_size: int = 1):
        """
        DDP 각 rank에서 매 epoch 호출.
        """
        self.epoch = epoch
        self.rank = rank
        self.world_size = world_size
        self._reset_rng()

    def _reset_rng(self):
        # 서로 다른 rank/worker/epoch 조합마다 고유 seed
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        seed = self.base_seed + self.epoch * 10_000 + self.rank * 1_000 + worker_id
        self._rng.manual_seed(seed)

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

        left = torch.randint(
            self.max_context_length // 4,
            self.max_context_length - self.max_context_length // 4,
            (1,),
            generator=self._rng,
        ).item()
        right = self.max_context_length - left
        window_start = max(0, answer_start - left)
        window_end = min(len(context_ids), answer_start + right)

        return context_ids[window_start:window_end]

    def _sample_triplet(self):
        indices = torch.randint(0, self._len, (3,), generator=self._rng)
        selected_qa_index = torch.randint(0, 3, (1,), generator=self._rng).item()

        selected_data = [self.data[i] for i in indices]
        qa_example = self.data[indices[selected_qa_index]]

        contexts = [
            self._crop_context(d.context, d.answer_start, d.answer.size(0))
            for d in selected_data
        ]
        context_lens = [c.size(0) for c in contexts]

        tokens = torch.cat(
            [
                self.SYSTEM_PROMPT,
                *contexts,
                self.QUESTION_PROMPT,
                qa_example.question,
                self.ANSWER_PROMPT,
                qa_example.answer,
            ],
            dim=0,
        )
        tokens: torch.Tensor = byte_tokenizer.add_special_tokens(
            tokens, bos=True, eos=True
        )

        answer_mask = torch.zeros_like(tokens, dtype=torch.bool)
        answer_mask[-(qa_example.answer.size(0) + 1) :] = True

        return QAInput(
            input_ids=tokens,
            answer_masks=answer_mask,
            pad_masks=torch.ones_like(tokens, dtype=torch.bool),
            qa_ids=[qa_example.qa_id],
        )

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        self._reset_rng()
        for _ in range(self.steps_per_epoch):
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
    def __init__(
        self,
        train_dataset,
        val_dataset,
        pred_dataset,
        train_batch_size=4,
        val_batch_size=1,
        pred_batch_size=1,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.pred_dataset = pred_dataset
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.pred_batch_size = pred_batch_size

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
            self.pred_dataset,
            batch_size=self.pred_batch_size,
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
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {"val_loss": loss}

    def predict_step(self, batch: QAInput, batch_idx: int) -> dict:
        results: CausalLMOutput = self.forward(batch.input_ids, mask=batch.pad_masks)
        outputs = {}
        outputs["logits"] = results.logits
        outputs["qa_ids"] = batch.qa_ids
        outputs["answer_masks"] = batch.answer_masks
        outputs["pad_masks"] = batch.pad_masks
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


class HNetForSQuAD(HNetForCausalLM, L.LightningModule):
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
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return {"val_loss": loss}

    def predict_step(self, batch: QAInput, batch_idx: int) -> dict:
        results: CausalLMOutput = self.forward(batch.input_ids, mask=batch.pad_masks)
        outputs = {}
        outputs["logits"] = results.logits
        outputs["qa_ids"] = batch.qa_ids
        outputs["answer_masks"] = batch.answer_masks
        outputs["pad_masks"] = batch.pad_masks
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return optimizer


class EpochSeedCallback(L.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        ds = trainer.datamodule.train_dataset
        if hasattr(ds, "set_epoch"):
            ds.set_epoch(
                trainer.current_epoch,
                getattr(trainer, "global_rank", 0),
                getattr(trainer, "world_size", 1),
            )


def setup_model_and_data(
    model_config: str,
    model_path: str | None = None,
    train_batch_size: int = 4,
    val_batch_size: int = 1,
    pred_batch_size: int = 1,
    learning_rate: float = 1e-4,
    max_context_length: int = 512,
    train_batches: int = 10_000,
    model_type: str = "HMNet",
):
    if model_type == "HMNet":
        model_cfg = OmegaConf.load(model_config)
        default_model_cfg = OmegaConf.structured(HMNetConfig)
        merged_model_cfg = OmegaConf.merge(default_model_cfg, model_cfg)
        model_config_obj: HMNetConfig = OmegaConf.to_object(merged_model_cfg)
        model = HMNetForSQuAD(
            config=model_config_obj,
            lr=learning_rate,
            weight_decay=0.0,
            pad_token_id=0,
        )
    elif model_type == "HNet":
        model_cfg = OmegaConf.load(model_config)
        default_model_cfg = OmegaConf.structured(HNetConfig)
        merged_model_cfg = OmegaConf.merge(default_model_cfg, model_cfg)
        model_config_obj: HNetConfig = OmegaConf.to_object(merged_model_cfg)
        model = HNetForSQuAD(
            config=model_config_obj,
            lr=learning_rate,
            weight_decay=0.0,
            pad_token_id=0,
        )
    else:
        raise ValueError("model_type must be HMNet or HNet")

    if model_path is not None and Path(model_path).exists():
        model.load_state_dict(torch.load(model_path), strict=False)

    data = load_SQuAD_json()
    train_squad_examples = SQuADExample.get_SQuAD_examples(data["train"])
    train_dataset = QATripletDataset(
        train_squad_examples,
        max_context_length=max_context_length,
        steps_per_epoch=train_batches,
    )

    val_squad_examples = SQuADExample.get_SQuAD_examples(
        data["dev"][:5],
        max_question_len=4096,
        max_answer_len=4096,
        use_one_answer=True,
    )
    val_input_ids_list = QAInput.get_qa_inputs(val_squad_examples)
    val_dataset = QADataset(val_input_ids_list)

    pred_squad_examples = SQuADExample.get_SQuAD_examples(
        data["dev"],
        max_question_len=10240,
        max_answer_len=10240,
        use_one_answer=True,
    )
    pred_input_ids_list = QAInput.get_qa_inputs(pred_squad_examples)
    pred_dataset = QADataset(pred_input_ids_list)

    data_module = QADataModule(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        pred_dataset=pred_dataset,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        pred_batch_size=pred_batch_size,
    )
    return model, data_module


@app.command()
def train(
    model_config: str,
    model_path: str | None = None,
    train_batch_size: int = 4,
    val_batch_size: int = 1,
    pred_batch_size: int = 1,
    learning_rate: float = 1e-4,
    num_epochs: int = 1_000_000,
    max_context_length: int = 512,
    train_batches: int = 10_000,
    ckpt_path: str | None = None,
    model_type: str = "HMNet",
):
    """
    Train HMNet or HNet model on SQuAD dataset.
    """
    model, data_module = setup_model_and_data(
        model_config=model_config,
        model_path=model_path,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        pred_batch_size=pred_batch_size,
        learning_rate=learning_rate,
        max_context_length=max_context_length,
        train_batches=train_batches,
        model_type=model_type,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename=f"{model_type}-squad-{Path(model_config).stem}-{{epoch:02d}}-{{val_loss:.2f}}",
        save_top_k=3,
        mode="min",
    )

    train_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath="checkpoints",
        filename=f"{model_type}-squad-train-{Path(model_config).stem}-{{epoch:02d}}-{{train_loss:.2f}}",
        save_top_k=3,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback, train_checkpoint_callback, EpochSeedCallback()],
        precision="bf16",
    )

    trainer.fit(
        model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


@app.command()
def validate(
    model_config: str,
    model_path: str | None = None,
    ckpt_path: str | None = None,
    val_batch_size: int = 1,
    max_context_length: int = 512,
    model_type: str = "HMNet",
):
    """
    Validate HMNet or HNet model on SQuAD dataset.
    """
    model, data_module = setup_model_and_data(
        model_config=model_config,
        model_path=model_path,
        train_batch_size=0,
        val_batch_size=val_batch_size,
        pred_batch_size=0,
        learning_rate=0.0,
        max_context_length=max_context_length,
        train_batches=0,
        model_type=model_type,
    )
    trainer = L.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        precision="bf16",
    )
    results = trainer.validate(model, datamodule=data_module, ckpt_path=ckpt_path)
    print(results)


class SavePredictionCallback(L.Callback):
    def __init__(self, answer_path: str | None = None):
        super().__init__()
        from pathlib import Path

        # self.output_dir = Path(output_dir)
        # self.output_dir.mkdir(parents=True, exist_ok=True)

        self.answer_path = answer_path
        if self.answer_path:
            self.answer_path = Path(answer_path)
            # 기존 파일이 있으면 삭제
            self.answer_path.parent.mkdir(parents=True, exist_ok=True)
            if self.answer_path.exists():
                self.answer_path.unlink()
            open(self.answer_path, "w").write("{")  # JSON 형식 시작

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        logits = outputs.get("logits", None)
        answer_masks = outputs.get("answer_masks", None)
        qa_ids = outputs.get("qa_ids", None)

        batch_size = len(qa_ids)

        answer_file = open(self.answer_path, "a") if self.answer_path else None
        for i in range(batch_size):
            qa_id = qa_ids[i]

            predict_ids = logits[i].detach().argmax(dim=-1).cpu()
            predict_ids = predict_ids[answer_masks[i]][:-1]  # remove EOS
            answer_text = byte_tokenizer.decode(predict_ids)
            if answer_file:
                answer_file.write(f"{qa_id}:{answer_text},")
        answer_file.close() if answer_file else None

    def on_predict_end(self, trainer, pl_module):
        if self.answer_path:
            # JSON 형식 닫기
            with open(self.answer_path, "a") as f:
                f.write("}\n")


@app.command()
def predict(
    model_config: str,
    model_path: str | None = None,
    ckpt_path: str | None = None,
    predict_dir: str = "predictions",
    pred_batch_size: int = 1,
    max_context_length: int = 512,
    model_type: str = "HMNet",
):
    """
    Predict with HMNet or HNet model on SQuAD dataset.
    """
    model, data_module = setup_model_and_data(
        model_config=model_config,
        model_path=model_path,
        train_batch_size=0,
        val_batch_size=0,
        pred_batch_size=pred_batch_size,
        learning_rate=0.0,
        max_context_length=max_context_length,
        train_batches=0,
        model_type=model_type,
    )
    Path(predict_dir).mkdir(parents=True, exist_ok=True)
    output_file = (
        Path(predict_dir)
        / f"{model_type}-squad-predictions-{Path(model_config).stem}.json"
    )

    save_callback = SavePredictionCallback(answer_path=str(output_file))
    trainer = L.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        precision="32",
        callbacks=[save_callback],
    )
    trainer.predict(model, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    app()
