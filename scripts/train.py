from dataclasses import dataclass
from re import A
import lightning as L
from omegaconf import OmegaConf
import torch
from hmnet.models.hmnet import HMNet
from hmnet.models.config_hmnet import HMNetConfig
from torch.utils.data import DataLoader
import argparse
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import random_split, Dataset
from torchmetrics.classification import AUROC
from lightning.pytorch.callbacks import ModelCheckpoint



class IMDBDataset(Dataset):
    def __init__(self, data_path: str):
        data = torch.load(data_path)
        self.input_ids = data["input_ids"]
        self.labels = data["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 10


def _collate_batch(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    mask = (input_ids != 0).bool()
    return input_ids, torch.concat(labels), mask

class IMDBDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = IMDBDataset(self.data_path)
        self.train_set, self.val_set = random_split(
            self.dataset,
            [int(len(self.dataset) * 0.8), len(self.dataset) - int(len(self.dataset) * 0.8)],
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=_collate_batch,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=_collate_batch,
        )

class HMNetForClassification(L.LightningModule):
    def __init__(
        self,
        model_config: HMNetConfig,
        num_classes: int,
        train_config: TrainConfig | None = None,
    ):
        super().__init__()
        self.model = HMNet(model_config, stage_idx=0)
        self.save_hyperparameters(ignore=["model_config"])
        self.loss = torch.nn.CrossEntropyLoss()
        self.classifier = torch.nn.Linear(model_config.d_model[0], num_classes)
        self.num_classes = num_classes
        self.train_config = train_config or TrainConfig()
        self.embeddings = torch.nn.Embedding(model_config.vocab_size, model_config.d_model[0])

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        x = self.embeddings(x.int())
        x, _, _ = self.model(x, mask=mask)
        x = self.classifier(x[:, 0, :])
        return x

    def training_step(
        self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        inputs, labels, mask = batch
        outputs = self(inputs, mask)
        loss = self.loss(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.model.parameters(), lr=self.train_config.learning_rate
        )

    def validation_step(self, batch: tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        inputs, labels, mask = batch
        outputs = self(inputs, mask)
        loss = self.loss(outputs, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", (outputs.argmax(dim=1) == labels).float().mean(), prog_bar=True)
        self.log(
            "val_auroc",
            AUROC(num_classes=self.num_classes, task="multiclass")(outputs, labels), prog_bar=True
        )

        return loss


def main():
    parser = argparse.ArgumentParser(description="Train HMNet model")
    parser.add_argument(
        "--train_config",
        type=str,
        required=False,
        help="Path to config file",
        default="configs/train_toy.yaml",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=False,
        help="Path to model config file",
        default="configs/hmnet_toy.yaml",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="Path to data file",
        default="data/imdb_dataset.pt",
    )
    args = parser.parse_args()
    torch.set_float32_matmul_precision('medium')

    train_cfg = OmegaConf.load(args.train_config)
    default_train_cfg = OmegaConf.structured(TrainConfig)
    merged_train_cfg = OmegaConf.merge(default_train_cfg, train_cfg)
    train_config: TrainConfig = OmegaConf.to_object(merged_train_cfg)

    model_cfg = OmegaConf.load(args.model_config)
    default_model_cfg = OmegaConf.structured(HMNetConfig)
    merged_model_cfg = OmegaConf.merge(default_model_cfg, model_cfg)
    model_config: HMNetConfig = OmegaConf.to_object(merged_model_cfg)

    model = HMNetForClassification(
        model_config=model_config,
        num_classes=2,  # Assuming binary classification for IMDB
        train_config=train_config,
    )
    data_module = IMDBDataModule(
        data_path=args.data_path,
        batch_size=train_config.batch_size
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="hmnet-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=train_config.num_epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()
