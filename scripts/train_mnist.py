# %% get mnist dataset
from typing import Any
import argparse
import os
import lightning as L
import torch
from einops import rearrange
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor
from torchmetrics import AUROC
from torchvision import datasets, transforms

from hmnet.models.config_hmnet import HMNetConfig, HMNetTrainConfig
from hmnet.models.config_hnet import AttnConfig, SSMConfig
from hmnet.models.hmnet import HMNetForClassification

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="./data", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def prepare_data(self):
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        self.mnist_train = datasets.MNIST(
            self.data_dir, train=True, download=False, transform=self.transform
        )
        self.mnist_val = datasets.MNIST(
            self.data_dir, train=False, download=False, transform=self.transform
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_train, batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)


class HMNetMNISTClassifier(L.LightningModule):
    def __init__(self, train_config: HMNetTrainConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        attn_config = AttnConfig(
            num_heads=[4, 4, 4], rotary_emb_dim=[8, 12, 16], window_size=[-1, -1, -1]
        )
        ssm_config = SSMConfig(d_conv=4, d_state=64, chunk_size=4, expand=2)
        self.num_classes = 10
        config = HMNetConfig(
            arch_layout=["M1", ["T1", ["T1"], "T1"], "M1"],
            d_model=[32, 64, 128],
            d_intermediate=[64, 128, 256],
            encoder_attn_cfg=attn_config,
            decoder_attn_cfg=attn_config,
            encoder_ssm_cfg=ssm_config,
            decoder_ssm_cfg=ssm_config,
        )

        self.backbone = HMNetForClassification(config, num_classes=self.num_classes)
        self.loss = torch.nn.CrossEntropyLoss()
        self.train_config = train_config

    def forward(self, x: Tensor) -> Tensor:
        feats = self.cnn(x)  # (B, 16, 7, 7)
        seq = rearrange(feats, "b c h w -> b (h w) c")
        mask = torch.ones_like(seq[:, :, 0], dtype=torch.bool, device=seq.device)
        return self.backbone(seq, mask)

    def training_step(self, batch, batch_idx: int) -> Tensor:
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs[0], labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.train_config.learning_rate)

    def validation_step(self, batch, batch_idx: int) -> Tensor:
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss(outputs[0], labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log(
            "val_accuracy",
            (outputs[0].argmax(dim=1) == labels).float().mean(),
            prog_bar=True,
        )
        self.log(
            "val_auroc",
            AUROC(num_classes=self.num_classes, task="multiclass")(outputs[0], labels),
            prog_bar=True,
        )
        return loss

    def predict_step(self, batch: Any, batch_idx: int) -> Any:
        inputs, _ = batch
        outputs, chunk_pred, attn_pred = self(inputs)
        return outputs, chunk_pred, attn_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict-out",
        type=str,
        default="predictions.pt",
        help="path to save predictions",
    )
    args = parser.parse_args()

    train_config = HMNetTrainConfig(
        batch_size=2048,
        learning_rate=5e-3,
        num_epochs=20,
    )

    model = HMNetMNISTClassifier(train_config=train_config)
    data_module = MNISTDataModule(batch_size=train_config.batch_size)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints",
        filename="hmnet-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    trainer = L.Trainer(
        max_epochs=model.train_config.num_epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, datamodule=data_module)

    # # After training, run prediction on the validation set and save outputs.
    # # If a best checkpoint was saved, load it for prediction.
    # best_ckpt = (
    #     checkpoint_callback.best_model_path if checkpoint_callback is not None else ""
    # )
    # if best_ckpt and os.path.exists(best_ckpt):
    #     pred_model = HMNetMNISTClassifier.load_from_checkpoint(
    #         best_ckpt, train_config=train_config
    #     )
    # else:
    #     pred_model = model

    # # Run prediction on the validation dataloader from the datamodule
    # preds = trainer.predict(pred_model, datamodule=data_module)

    # # Consolidate predictions: preds is a list (per batch) of either tensors or tuples/lists of tensors
    # def consolidate(pred_list):
    #     if len(pred_list) == 0:
    #         return None
    #     first = pred_list[0]
    #     if isinstance(first, (list, tuple)):
    #         parts = list(zip(*pred_list))
    #         out = [torch.cat([p for p in part], dim=0) for part in parts]
    #         return [t.detach().cpu() for t in out]
    #     else:
    #         out = torch.cat([p for p in pred_list], dim=0)
    #         return out.detach().cpu()

    # consolidated = consolidate(preds)

    # # Prepare a dict to save. If consolidated is a list, map to names
    # save_dict = {}
    # if consolidated is None:
    #     save_dict["predictions"] = None
    # elif isinstance(consolidated, list):
    #     # primary outputs
    #     save_dict["outputs"] = consolidated[0]
    #     if len(consolidated) > 1:
    #         save_dict["chunk_pred"] = consolidated[1]
    #     if len(consolidated) > 2:
    #         save_dict["attn_pred"] = consolidated[2]
    # else:
    #     save_dict["outputs"] = consolidated

    # torch.save(save_dict, args.predict_out)
    # print(f"Saved predictions to {args.predict_out}")


if __name__ == "__main__":
    main()
