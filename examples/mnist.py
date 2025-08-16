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
from hmnet.modules.profiler import RoutingProfiler, ChunkAttnScoreProfiler

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

    def predict_dataloader(self):
        # Use validation set for prediction
        return self.val_dataloader()


class HMNetMNISTClassifier(L.LightningModule):
    def __init__(self, train_config: HMNetTrainConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(
                1, 16, kernel_size=3, stride=2, padding=1
            ),  # B 1 28 28 -> B 16 14 14
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                16, 32, kernel_size=3, stride=2, padding=1
            ),  # B 16 14 14 -> B 32 7 7
            torch.nn.BatchNorm2d(32),
            # torch.nn.Conv2d(
            #     32, 32, kernel_size=3, stride=2, padding=1
            # ),  # B 32 7 7 -> B 32 4 4
            # torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
        )
        encoder_attn_config = AttnConfig(
            num_heads=[4, 4], rotary_emb_dim=[8, 12], window_size=[-1, -1]
        )
        decoder_attn_config = AttnConfig(
            num_heads=[4, 4], rotary_emb_dim=[8, 12], window_size=[10, 4]
        )
        ssm_config = SSMConfig(d_conv=4, d_state=64, chunk_size=4, expand=2)
        self.num_classes = 10
        config = HMNetConfig(
            arch_layout=["T1", ["T1"], "T1"],
            d_model=[32, 64],
            d_intermediate=[64, 128],
            encoder_attn_cfg=encoder_attn_config,
            decoder_attn_cfg=decoder_attn_config,
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
        inputs, labels = batch
        logits, boundary_pred, chunk_attn_pred = self(inputs)
        chunk_attn_profiler = ChunkAttnScoreProfiler(
            chunk_attn_pred, [bp.boundary_mask for bp in boundary_pred]
        )
        dechunk_attn = chunk_attn_profiler.dechunk_all_stage()
        boundary_profiler = RoutingProfiler(boundary_pred)
        dechunked_boundary = boundary_profiler.dechunk_all_stage()
        return {
            "logits": logits,
            "labels": labels,
            "dechunk_attn": torch.stack(list(dechunk_attn), dim=1),
            "dechunked_boundary": torch.stack(list(dechunked_boundary), dim=1),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predict-out",
        type=str,
        default="results/mnist_predictions.pt",
        help="path to save predictions",
    )
    args = parser.parse_args()

    train_config = HMNetTrainConfig(
        batch_size=2048,
        learning_rate=5e-3,
        num_epochs=30,
    )

    model = HMNetMNISTClassifier(train_config=train_config)
    data_module = MNISTDataModule(batch_size=train_config.batch_size)

    # Train if no checkpoint, otherwise load and predict
    if not os.path.exists("ckpts/hmnet_mnist.ckpt"):
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="ckpts",
            filename="hmnet_mnist",
            save_top_k=1,
            mode="min",
        )

        trainer = L.Trainer(
            max_epochs=model.train_config.num_epochs,
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            callbacks=[checkpoint_callback],
            precision="bf16",
        )
        trainer.fit(model, datamodule=data_module)
        # Use best checkpoint for prediction

    torch._dynamo.config.capture_scalar_outputs = True
    model = HMNetMNISTClassifier.load_from_checkpoint("ckpts/hmnet_mnist.ckpt")
    trainer = L.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu", precision="bf16"
    )
    data_module.prepare_data()
    data_module.setup()
    pred_batches = trainer.predict(model, datamodule=data_module)

    # Collate and save predictions
    if pred_batches is not None and len(pred_batches) > 0:
        batch_logits = []
        batch_labels = []
        batch_dechunked_boundary_preds = []
        batch_dechunked_attn_preds = []

        for b in pred_batches:
            if isinstance(b, dict):
                bl = b.get("logits")
                lb = b.get("labels")
                bp = b.get("dechunked_boundary")
                cp = b.get("dechunk_attn")
            elif isinstance(b, (list, tuple)) and len(b) >= 2:
                bl, lb = b[0], b[1]
            else:
                continue
            if bl is not None:
                batch_logits.append(bl.detach().cpu())
            if lb is not None:
                batch_labels.append(lb.detach().cpu())
            if bp is not None:
                batch_dechunked_boundary_preds.append(bp.detach().cpu())
            if cp is not None:
                batch_dechunked_attn_preds.append(cp.detach().cpu())

        if len(batch_logits) == 0:
            return
        logits = torch.cat(batch_logits, dim=0)
        labels = torch.cat(batch_labels, dim=0) if len(batch_labels) else None
        dechunked_boundary_preds = torch.cat(batch_dechunked_boundary_preds)
        dechunked_attn_preds = torch.cat(batch_dechunked_attn_preds)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        out_dir = os.path.dirname(args.predict_out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            "probs": probs,
            "preds": preds,
            "dechunked_boundary_preds": dechunked_boundary_preds,
            "dechunked_attn_preds": dechunked_attn_preds,
        }
        if labels is not None:
            payload["labels"] = labels
        torch.save(payload, args.predict_out)


if __name__ == "__main__":
    main()
