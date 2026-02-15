# test.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from datetime import datetime
from typing import List

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
)

# -------------------------
# Project imports
# -------------------------
from src.models.dynunet2d import build_dynunet2d
from src.data.mandm_dataloader import build_mandm_loaders, MandMDatasetConfig, LoaderConfig
from src.data.acdc_dataloader import build_acdc_loaders, ACDCDatasetConfig


# -------------------------
# 1) Config
# -------------------------
@dataclass
class TestConfig:
    checkpoint_path: str

    device: str = "cuda"
    num_classes: int = 4

    # output
    base_out_dir: str = "/scratch1/e20-fyp-syn-car-mri-gen/segmentation-v2/test_outputs"
    save_images: int = 20  # number of qualitative samples per dataset

    # datasets
    eval_mandm: bool = True
    eval_acdc: bool = True


# -------------------------
# 2) Helpers
# -------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def one_hot(y: torch.Tensor, c: int) -> torch.Tensor:
    # y: [B,1,H,W] -> [B,C,H,W]
    return torch.nn.functional.one_hot(y.squeeze(1), c).permute(0, 3, 1, 2).float()


def save_triplet(image, gt, pred, save_path: str):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("GT Mask")
    plt.imshow(gt, cmap="jet", interpolation="nearest")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Pred Mask")
    plt.imshow(pred, cmap="jet", interpolation="nearest")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def box_plot(values: List[float], title: str, ylabel: str, save_path: str, higher_is_better=True):
    values = [v for v in values if np.isfinite(v)]  # drop nan/inf
    plt.figure(figsize=(4, 6))
    plt.boxplot(values, showmeans=True)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xticks([])
    plt.xlabel("Higher is Better" if higher_is_better else "Lower is Better")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# -------------------------
# 3) Core evaluation
# -------------------------
@torch.no_grad()
def evaluate_dataset(
    model,
    loader,
    dataset_name: str,
    device,
    cfg: TestConfig,
    run_dir: str,
):
    print(f"\n Evaluating on {dataset_name}")

    out_dir = os.path.join(run_dir, dataset_name)
    img_dir = os.path.join(out_dir, "images")
    plot_dir = os.path.join(out_dir, "plots")

    ensure_dir(out_dir)
    ensure_dir(img_dir)
    ensure_dir(plot_dir)

    dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)
    iou_metric = MeanIoU(include_background=True, reduction="none", get_not_nans=False)
    hd95_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="none", get_not_nans=False)
    asd_metric = SurfaceDistanceMetric(include_background=True, reduction="none", get_not_nans=False)

    dice_metric.reset()
    iou_metric.reset()
    hd95_metric.reset()
    asd_metric.reset()

    saved = 0

    model.eval()
    for batch in tqdm(loader, desc=f"test-{dataset_name}"):
        x = batch["image"].to(device)
        y = batch["mask"].to(device)

        logits = model(x)
        pred = torch.argmax(logits, dim=1, keepdim=True)

        y_oh = one_hot(y, cfg.num_classes)
        pred_oh = one_hot(pred, cfg.num_classes)

        dice_metric(y_pred=pred_oh, y=y_oh)
        iou_metric(y_pred=pred_oh, y=y_oh)
        hd95_metric(y_pred=pred_oh, y=y_oh)
        asd_metric(y_pred=pred_oh, y=y_oh)

        # Save qualitative samples
        if saved < cfg.save_images:
            img = x[0, 0].detach().cpu().numpy()
            gt = y[0, 0].detach().cpu().numpy()
            pd = pred[0, 0].detach().cpu().numpy()
        # # after pred computed
        # diff = (pred != y).float().mean().item()
        # print("Pixel mismatch rate:", diff)
        # print("GT unique:", torch.unique(y).tolist())
        # print("Pred unique:", torch.unique(pred).tolist())
        # per-class pixel counts
        # for k in range(cfg.num_classes):
        #     gt_k = (y == k).sum().item()
        #     pr_k = (pred == k).sum().item()
        #     print(f"class {k}: GT={gt_k}  Pred={pr_k}")
            save_triplet(img,gt,pd,os.path.join(img_dir, f"sample_{saved:03d}.png"),)
            saved += 1

    dice_all = dice_metric.aggregate()
    iou_all = iou_metric.aggregate()
    hd95_all = hd95_metric.aggregate()
    asd_all = asd_metric.aggregate()

    def to_flat_list(t: torch.Tensor) -> List[float]:
        return [float(v) for v in t.detach().cpu().numpy().reshape(-1).tolist()]

    dice_vals = to_flat_list(dice_all)
    iou_vals = to_flat_list(iou_all)
    hd95_vals = to_flat_list(hd95_all)
    asd_vals = to_flat_list(asd_all)

    dice_vals_f = [v for v in dice_vals if np.isfinite(v)]
    iou_vals_f = [v for v in iou_vals if np.isfinite(v)]
    hd95_vals_f = [v for v in hd95_vals if np.isfinite(v)]
    asd_vals_f = [v for v in asd_vals if np.isfinite(v)]

    box_plot(dice_vals, "Dice Distribution", "Dice", os.path.join(plot_dir, "dice_box.png"), True)
    box_plot(iou_vals, "IoU Distribution", "IoU", os.path.join(plot_dir, "iou_box.png"), True)
    box_plot(hd95_vals, "HD95 Distribution", "HD95", os.path.join(plot_dir, "hd95_box.png"), False)
    box_plot(asd_vals, "ASD Distribution", "ASD", os.path.join(plot_dir, "asd_box.png"), False)

    # Save summary
    summary = {
        "dataset": dataset_name,
        "dice": {"mean": float(np.mean(dice_vals_f)), "std": float(np.std(dice_vals_f)), "n": len(dice_vals_f)},
        "iou": {"mean": float(np.mean(iou_vals_f)), "std": float(np.std(iou_vals_f)), "n": len(iou_vals_f)},
        "hd95": {"mean": float(np.mean(hd95_vals_f)), "std": float(np.std(hd95_vals_f)), "n": len(hd95_vals_f)},
        "asd": {"mean": float(np.mean(asd_vals_f)), "std": float(np.std(asd_vals_f)), "n": len(asd_vals_f)},
        "raw_counts": {
            "dice_total": len(dice_vals),
            "iou_total": len(iou_vals),
            "hd95_total": len(hd95_vals),
            "asd_total": len(asd_vals),
            "hd95_inf_nan": int(len(hd95_vals) - len(hd95_vals_f)),
            "asd_inf_nan": int(len(asd_vals) - len(asd_vals_f)),
        },
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f" {dataset_name} done:", summary)


# -------------------------
# 4) Main
# -------------------------
def main():
    cfg = TestConfig(
        checkpoint_path="/scratch1/e20-fyp-syn-car-mri-gen/segmentation-v2/outputs/20260214_1305_syn_fm_full/checkpoints/best.pt"  # <-- UPDATE
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_dynunet2d(in_channels=1, out_channels=cfg.num_classes).to(device)

    ckpt = torch.load(cfg.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state", ckpt))
    print("Checkpoint loaded")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_ds = "syn_fm_syn"
    run_dir = os.path.join(cfg.base_out_dir, f"{ts}_{train_ds}")
    ensure_dir(run_dir)

    # MandM Test
    if cfg.eval_mandm:
        dc = MandMDatasetConfig()
        lc = LoaderConfig()
        mandm_test_loader, _ = build_mandm_loaders(
            dc.cache_root, dc=dc, lc=lc, is_test_set_only=True
        )
        evaluate_dataset(model, mandm_test_loader, "mandm", device, cfg, run_dir)

    # ACDC Test
    if cfg.eval_acdc:
        dc = ACDCDatasetConfig()
        lc = LoaderConfig()
        acdc_test_loader, _ = build_acdc_loaders(
            dc.cache_root, dc=dc, lc=lc, is_test_set_only=True
        )
        evaluate_dataset(model, acdc_test_loader, "acdc", device, cfg, run_dir)

    print(f"\n All tests finished. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()