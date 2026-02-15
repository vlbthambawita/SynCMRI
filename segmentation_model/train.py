# train.py
from __future__ import annotations
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import time
from datetime import datetime
import torch
from torch.amp import autocast,GradScaler
from tqdm import tqdm

from monai.losses.dice import DiceCELoss
from monai.metrics import DiceMetric,MeanIoU,HausdorffDistanceMetric,SurfaceDistanceMetric

# from src.data.mandm_dataloader import MandMDatasetConfig,LoaderConfig,build_mandm_loaders
# from src.data.acdc_dataloader import ACDCDatasetConfig,LoaderConfig,build_acdc_loaders
from src.data.synthetic_dataloader import SyntheticDatasetConfig,LoaderConfig,build_syn_loaders
from src.models.dynunet2d import build_dynunet2d

@dataclass
class TrainConfig:
    # data
    # dataset_name: str = "mandm"
    # cache_root: str = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache"
    # dataset_name: str = "acdc"
    # cache_root: str = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/ACDC Dataset/cache"
    dataset_name: str = "syn_fm_full"
    cache_root: str = "/scratch1/e20-fyp-syn-car-mri-gen/flow-matching/generated/synthetic_imgs/20260214_1240"
    crop_size: int = 128

    # training
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    use_amp: bool = True

    # outputs
    base_out_dir: str = "/scratch1/e20-fyp-syn-car-mri-gen/segmentation-v2/outputs"
    save_images_every: int = 10

def make_run_dir(base_out: str, dataset_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(base_out, f"{ts}_{dataset_name}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "saved_images"), exist_ok=True)
    return run_dir

@torch.no_grad()
def save_debug_figure(
    batch: dict,
    logits: torch.Tensor,
    epoch: int,
    run_dir: str,
    num_classes: int = 4,
):
    """
    Saves a 1x3 subplot: Image | GT | Pred
    batch["image"]: [B,1,H,W]
    batch["mask]:   [B,1,H,W]
    logits:         [B,C,H,W]
    """

    img = batch["image"][0, 0].detach().cpu().numpy()
    gt  = batch["mask"][0, 0].detach().cpu().numpy()

    pred = torch.argmax(logits, dim=1, keepdim=True)  # [B,1,H,W]
    pred = pred[0, 0].detach().cpu().numpy()

    save_path = os.path.join(run_dir, "saved_images", f"epoch_{epoch:03d}.png")

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT Mask")
    plt.imshow(gt, cmap='jet', interpolation="nearest")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Pred Mask")
    plt.imshow(pred, cmap='jet', interpolation="nearest")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def save_loss_curve(run_dir: str, train_losses: list, val_losses: list):
    path = os.path.join(run_dir, "loss_curve.png")
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="train_loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def init_csv_logger(run_dir: str) -> str:
    csv_path = os.path.join(run_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "val_mean_dice", "val_mean_iou", "dice_per_class", "iou_per_class"]
            )
    return csv_path


def append_csv(
    csv_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_mean_dice: float,
    val_mean_iou: float,
    dice_per_class: torch.Tensor,
    iou_per_class: torch.Tensor,
):
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                epoch,
                train_loss,
                val_loss,
                val_mean_dice,
                val_mean_iou,
                dice_per_class.tolist(),
                iou_per_class.tolist(),
            ]
        )

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, best_dice: float):
    torch.save(
        {
            "epoch": epoch,
            "best_dice": best_dice,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )


def build_loss():
    return DiceCELoss(to_onehot_y=True, softmax=True,lambda_dice=0.5,lambda_ce=0.5)

def build_optimizer(model, lr, weight_decay):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def build_dice_metric():
    return DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

def build_iou_metric():
    return MeanIoU(include_background=True, reduction="mean", get_not_nans=False)

def build_hd95_metric():
    return HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean", get_not_nans=False)

def build_surface_dist_metric():
    return SurfaceDistanceMetric(include_background=True,distance_metric="euclidean",get_not_nans=False)

def to_onehot_indices(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    y: [B,1,H,W] int
    returns: [B,C,H,W] float
    """
    return torch.nn.functional.one_hot(y.squeeze(1), num_classes).permute(0, 3, 1, 2).float()

def train_one_epoch(model, loader, loss_fn, optimizer, device, scaler, use_amp: bool) -> float:
    model.train()
    running = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        if use_amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running += loss.item()

    return running / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, loss_fn, device, dice_metric, iou_metric):
    model.eval()
    running = 0.0

    dice_metric.reset()
    iou_metric.reset()

    for batch in tqdm(loader, desc="val", leave=False):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)

        logits = model(x)
        loss = loss_fn(logits, y)
        running += loss.item()

        # Discrete prediction -> one-hot
        pred = torch.argmax(logits, dim=1, keepdim=True)  # [B,1,H,W]
        c = logits.shape[1]
        pred_oh = to_onehot_indices(pred, c)
        y_oh = to_onehot_indices(y, c)

        dice_metric(y_pred=pred_oh, y=y_oh)
        iou_metric(y_pred=pred_oh, y=y_oh)

    val_loss = running / max(1, len(loader))

    dice_per_class = dice_metric.aggregate()  # [C]
    mean_dice = float(dice_per_class.mean().item())

    iou_per_class = iou_metric.aggregate()  # [C]
    mean_iou = float(iou_per_class.mean().item())

    return val_loss, mean_dice, dice_per_class.cpu(), mean_iou, iou_per_class.cpu()


def main():
    cfg = TrainConfig()

    run_dir = make_run_dir(cfg.base_out_dir, cfg.dataset_name)
    csv_path = init_csv_logger(run_dir)
    best_ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    print("RUN DIR:", run_dir)

    # Data
    # dc = MandMDatasetConfig(cache_root=cfg.cache_root, crop_size=cfg.crop_size)
    # lc = LoaderConfig(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # train_loader, val_loader, test_loader, stats = build_mandm_loaders(cfg.cache_root, dc=dc, lc=lc)

    # dc = ACDCDatasetConfig(cache_root=cfg.cache_root, crop_size=cfg.crop_size)
    # lc = LoaderConfig(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    # train_loader, val_loader, test_loader, stats = build_acdc_loaders(cfg.cache_root, dc=dc, lc=lc)

    dc = SyntheticDatasetConfig()
    lc = LoaderConfig(batch_size=cfg.batch_size, num_workers=cfg.num_workers)
    train_loader, val_loader, stats = build_syn_loaders(cfg.cache_root, dc=dc, lc=lc)
    print("DATA STATS:", stats)

    # Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE:", device)

    model = build_dynunet2d(in_channels=1, out_channels=4).to(device)

    # Loss/Optim/Metrics
    loss_fn = build_loss()
    optimizer = build_optimizer(model, cfg.lr, cfg.weight_decay)
    dice_metric = build_dice_metric()
    iou_metric = build_iou_metric()
    scaler = GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    # Sanity check (one batch)
    b = next(iter(train_loader))
    
    print("SANITY:", b["image"].shape, b["mask"].shape, "label_key:")
    print("UNIQUE LABELS:", torch.unique(b["mask"]).tolist())

    with torch.no_grad():
        logits = model(b["image"].to(device))
    print("LOGITS:", logits.shape, "(expected [B,4,H,W])")

    # Training
    train_losses, val_losses = [], []
    best_dice = -1.0
    start = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, scaler, cfg.use_amp)
        va_loss, va_dice, va_dice_pc, va_iou, va_iou_pc = validate(
            model, val_loader, loss_fn, device, dice_metric, iou_metric
        )

        train_losses.append(tr_loss)
        val_losses.append(va_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"val_mean_dice={va_dice:.4f} | val_mean_iou={va_iou:.4f} | "
            f"dice_per_class={va_dice_pc.numpy()} | iou_per_class={va_iou_pc.numpy()}"
        )

        append_csv(csv_path, epoch, tr_loss, va_loss, va_dice, va_iou, va_dice_pc, va_iou_pc)
        save_loss_curve(run_dir, train_losses, val_losses)

        # Save images every N epochs (use val batch for stability)
        if cfg.save_images_every > 0 and epoch % cfg.save_images_every == 0:
            dbg_batch = next(iter(val_loader))
            x_dbg = dbg_batch["image"].to(device)
            logits_dbg = model(x_dbg)
            save_debug_figure(dbg_batch, logits_dbg, epoch, run_dir)
            print(f"Saved image for epoch {epoch}")

        # Save best checkpoint
        if va_dice > best_dice:
            best_dice = va_dice
            save_checkpoint(best_ckpt_path, model, optimizer, epoch, best_dice)
            print(f"Saved BEST checkpoint: {best_ckpt_path} (dice={best_dice:.4f})")

    print(f"Done. Best Dice={best_dice:.4f}. Time={(time.time() - start)/60:.1f} min")


if __name__ == "__main__":
    main()