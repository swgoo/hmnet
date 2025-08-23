import argparse
from dataclasses import dataclass
import os
import lightning as L

from omegaconf import OmegaConf
import torch
from hmnet.models.config_hmnet import HMNetConfig
from imdb import IMDBDataModule, HMNetForClassification
from lightning.pytorch.callbacks import ModelCheckpoint


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100_000


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
        default="configs/hmnet_2stage_L.yaml",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="Path to data file",
        default="data/imdb_dataset.pt",
    )
    parser.add_argument(
        "--predict-out",
        type=str,
        default="results/imdb_predictions.pt",
        help="path to save predictions",
    )
    args = parser.parse_args()

    train_cfg = OmegaConf.load(args.train_config)
    default_train_cfg = OmegaConf.structured(TrainConfig)
    merged_train_cfg = OmegaConf.merge(default_train_cfg, train_cfg)
    train_config: TrainConfig = OmegaConf.to_object(merged_train_cfg)

    model_cfg = OmegaConf.load(args.model_config)
    default_model_cfg = OmegaConf.structured(HMNetConfig)
    merged_model_cfg = OmegaConf.merge(default_model_cfg, model_cfg)
    model_config: HMNetConfig = OmegaConf.to_object(merged_model_cfg)

    torch._dynamo.config.capture_scalar_outputs = True
    model = HMNetForClassification(
        model_config=model_config,
        num_classes=2,  # Assuming binary classification for IMDB
        train_config=train_config,
    )

    # model.load_state_dict(torch.load("ckpts/hnet_2stage_L.pt"), strict=False)
    data_module = IMDBDataModule(
        data_path=args.data_path, batch_size=train_config.batch_size
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
        precision="bf16",
    )

    pred_batches = trainer.predict(
        model,
        datamodule=data_module,
        ckpt_path="checkpoints/hmnet-imdb.ckpt",
    )

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
