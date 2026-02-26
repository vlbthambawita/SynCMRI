# src/data/acdc_dataloader.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

Split = Literal["train", "val", "test", "full"]

@dataclass
class ACDCDatasetConfig:
    # raw dirs are ONLY used to list patient IDs (as in your script)
    train_dir: str = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/ACDC Dataset/training"
    test_dir: str = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/ACDC Dataset/testing"

    # cache root
    cache_root: str = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/ACDC Dataset/cache"

    # cache tags (match your cache naming)
    train_tag: str = "train_acdc_v2"
    test_tag: str = "test_acdc_v2"  # used for both val/test because split comes from testing folder

    crop_size: int = 128

    # split: val comes from testing folder
    val_percentage: float = 0.30
    seed: int = 42

    # augmentation params (train only)
    aug_probability: float = 0.5
    aug_rot_range: Tuple[float, float] = (-20, 20)
    aug_scale_range: Tuple[float, float] = (0.9, 1.1)
    aug_trans_range: Tuple[float, float] = (0.1, 0.1)  # fraction of W,H
    aug_flip_prob: float = 0.5


@dataclass
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle_train: bool = True
    drop_last_train: bool = True


def get_patients_from_folder(folder_path: str) -> List[str]:
    """List patient folders from raw ACDC directory."""
    if not os.path.exists(folder_path):
        print(f"Folder path not found: {folder_path}")
        return []
    return sorted([p for p in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, p))])


def build_samples(cache_tag_path: str, patient_ids: List[str]) -> List[Tuple[str, int, int]]:
    """Build (pid, slice_idx, frame_idx) list from cached '*_indices.npy' files."""
    samples: List[Tuple[str, int, int]] = []
    for pid in patient_ids:
        indices_file = os.path.join(cache_tag_path, f"{pid}_indices.npy")
        if not os.path.exists(indices_file):
            continue
        indices = np.load(indices_file)
        if indices.size == 0:
            continue
        for row in indices:
            samples.append((pid, int(row[0]), int(row[1])))

    if len(samples) == 0:
        raise RuntimeError(f"No samples built from {cache_tag_path}. Check indices files.")
    return samples


def split_val_test(patient_ids: List[str], val_percentage: float, seed: int) -> Tuple[List[str], List[str]]:
    """Split testing-folder patients into val and test deterministically."""
    ids = list(patient_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    split_idx = int(len(ids) * val_percentage)
    return ids[:split_idx], ids[split_idx:]


class ACDCDataset(Dataset):
    """
    Loads 2D slice (slice_idx, frame_idx) from cached 4D npy files.
    Applies:
      - ROI crop centered on mask
      - Percentile normalization to [-1, 1]
      - Optional augmentations (flip, affine)
    Returns {"image": img, "mask": mask}
    """
    def __init__(
        self,
        cache_tag_path: str,
        samples: List[Tuple[str, int, int]],
        crop_size: int = 128,
        augment: bool = False,
        aug_probability: float = 0.5,
        aug_rot_range: Tuple[float, float] = (-20, 20),
        aug_scale_range: Tuple[float, float] = (0.9, 1.1),
        aug_trans_range: Tuple[float, float] = (0.1, 0.1),
        aug_flip_prob: float = 0.5,
    ):
        self.cache_tag_path = cache_tag_path
        self.samples = samples
        self.crop_size = crop_size
        self.augment = augment

        self.aug_probability = aug_probability
        self.aug_rot_range = aug_rot_range
        self.aug_scale_range = aug_scale_range
        self.aug_trans_range = aug_trans_range
        self.aug_flip_prob = aug_flip_prob

    def __len__(self) -> int:
        return len(self.samples)

    def _crop_to_mask_center(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        y_idx, x_idx = torch.where(mask[0] > 0)

        if len(y_idx) > 0:
            cy = int(torch.mean(y_idx.float()).item())
            cx = int(torch.mean(x_idx.float()).item())
        else:
            cy = img.shape[1] // 2
            cx = img.shape[2] // 2

        half = self.crop_size // 2
        sy, ey = cy - half, cy + half
        sx, ex = cx - half, cx + half

        pt = max(0, -sy)
        pl = max(0, -sx)
        pb = max(0, ey - img.shape[1])
        pr = max(0, ex - img.shape[2])

        if any([pt, pb, pl, pr]):
            img = TF.pad(img, (pl, pt, pr, pb), fill=float(img.min()))
            mask = TF.pad(mask, (pl, pt, pr, pb), fill=0)
            sy += pt; ey += pt
            sx += pl; ex += pl

        img = img[:, sy:ey, sx:ex]
        mask = mask[:, sy:ey, sx:ex]
        return img, mask

    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply augmentation only with probability
        if random.random() >= self.aug_probability:
            return img, mask

        # Flip
        if random.random() < self.aug_flip_prob:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # Affine: rotation + translate + scale (same params for img & mask)
        angle = random.uniform(self.aug_rot_range[0], self.aug_rot_range[1])
        scale = random.uniform(self.aug_scale_range[0], self.aug_scale_range[1])

        max_dx = float(self.aug_trans_range[0] * img.shape[2])
        max_dy = float(self.aug_trans_range[1] * img.shape[1])
        translate = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))

        img = TF.affine(
            img,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=0.0,
            interpolation=InterpolationMode.BILINEAR,
            fill=float(img.min()),
        )
        mask = TF.affine(
            mask,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=0.0,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        return img, mask

    def _percentile_norm(self, img: torch.Tensor) -> torch.Tensor:
        if torch.isnan(img).any():
            img = torch.nan_to_num(img, nan=0.0)

        img_np = img.numpy()
        p01 = np.percentile(img_np, 1)
        p99 = np.percentile(img_np, 99)

        img_np = np.clip(img_np, p01, p99)
        if (p99 - p01) > 1e-6:
            img_np = (img_np - p01) / (p99 - p01)
            img_np = (img_np * 2.0) - 1.0
        else:
            img_np = np.zeros_like(img_np) - 1.0

        return torch.tensor(img_np, dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pid, slice_idx, frame_idx = self.samples[idx]

        img_path = os.path.join(self.cache_tag_path, f"{pid}_resampled_4d_img.npy")
        mask_path = os.path.join(self.cache_tag_path, f"{pid}_resampled_4d_mask.npy")

        img_vol = np.load(img_path, mmap_mode="r")
        mask_vol = np.load(mask_path, mmap_mode="r")

        img2d = img_vol[:, :, slice_idx, frame_idx]
        mask2d = mask_vol[:, :, slice_idx, frame_idx]

        img = torch.tensor(img2d.copy(), dtype=torch.float32).unsqueeze(0)
        mask2d = mask2d.copy()

        # Remap ACDC labels to match MandM convention
        # ACDC: 1=RV, 2=MYO, 3=LV
        # Target: 1=LV, 2=MYO, 3=RV

        mask_remap = np.zeros_like(mask2d, dtype=np.int64)

        mask_remap[mask2d == 0] = 0  # bg
        mask_remap[mask2d == 3] = 1  # LV
        mask_remap[mask2d == 2] = 2  # MYO
        mask_remap[mask2d == 1] = 3  # RV

        mask = torch.tensor(mask_remap, dtype=torch.long).unsqueeze(0)

        img, mask = self._crop_to_mask_center(img, mask)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = self._percentile_norm(img)

        return {"image": img, "mask": mask}

def build_acdc_loaders(
    cache_root: str,
    dc: Optional[ACDCDatasetConfig] = None,
    lc: Optional[LoaderConfig] = None,
    is_test_set_only: bool = False
):
    dc = dc or ACDCDatasetConfig()
    lc = lc or LoaderConfig()

    train_cache_path = os.path.join(cache_root, dc.train_tag)
    test_cache_path = os.path.join(cache_root, dc.test_tag)

    # patient IDs from raw folders
    train_pids = get_patients_from_folder(dc.train_dir)
    testing_pids = get_patients_from_folder(dc.test_dir)

    if len(train_pids) == 0:
        raise RuntimeError(f"No training patients found in {dc.train_dir}.")
    if len(testing_pids) == 0:
        raise RuntimeError(f"No testing patients found in {dc.test_dir}.")

    # split testing patients into val/test
    val_pids, test_pids = split_val_test(testing_pids, dc.val_percentage, dc.seed)

    # sample lists
    train_samples = build_samples(train_cache_path, train_pids)
    val_samples = build_samples(test_cache_path, val_pids)
    test_samples = build_samples(test_cache_path, test_pids)

    # datasets
    ds_train = ACDCDataset(
        train_cache_path,
        train_samples,
        crop_size=dc.crop_size,
        augment=True,
        aug_probability=dc.aug_probability,
        aug_rot_range=dc.aug_rot_range,
        aug_scale_range=dc.aug_scale_range,
        aug_trans_range=dc.aug_trans_range,
        aug_flip_prob=dc.aug_flip_prob,
    )

    ds_val = ACDCDataset(
        test_cache_path,
        val_samples,
        crop_size=dc.crop_size,
        augment=False,
    )

    ds_test = ACDCDataset(
        test_cache_path,
        test_samples,
        crop_size=dc.crop_size,
        augment=False,
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=lc.batch_size,
        shuffle=lc.shuffle_train,
        num_workers=lc.num_workers,
        pin_memory=lc.pin_memory,
        persistent_workers=lc.persistent_workers and lc.num_workers > 0,
        drop_last=lc.drop_last_train,
    )

    val_loader = DataLoader(
        ds_val,
        batch_size=lc.batch_size,
        shuffle=False,
        num_workers=lc.num_workers,
        pin_memory=lc.pin_memory,
        persistent_workers=lc.persistent_workers and lc.num_workers > 0,
        drop_last=False,
    )

    test_loader = DataLoader(
        ds_test,
        batch_size=lc.batch_size,
        shuffle=False,
        num_workers=lc.num_workers,
        pin_memory=lc.pin_memory,
        persistent_workers=lc.persistent_workers and lc.num_workers > 0,
        drop_last=False,
    )

    stats = {
        "n_train_patients": len(train_pids),
        "n_val_patients": len(val_pids),
        "n_test_patients": len(test_pids),
        "n_train_samples": len(ds_train),
        "n_val_samples": len(ds_val),
        "n_test_samples": len(ds_test),
        "train_cache_path": train_cache_path,
        "test_cache_path": test_cache_path,
        "train_dir": dc.train_dir,
        "test_dir": dc.test_dir,
        "val_percentage": dc.val_percentage,
        "seed": dc.seed,
    }

    if is_test_set_only:
        return test_loader, stats

    return train_loader, val_loader, test_loader, stats


# -------------------------
# 5) Quick check (same style)
# -------------------------
if __name__ == "__main__":
    cache_root = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/ACDC Dataset/cache"

    dc = ACDCDatasetConfig(cache_root=cache_root, crop_size=128)
    lc = LoaderConfig(batch_size=8, num_workers=2)

    train_loader, val_loader, test_loader, stats = build_acdc_loaders(cache_root, dc=dc, lc=lc)
    print("STATS:", stats)

    b = next(iter(train_loader))
    print("BATCH:", b["image"].shape, b["mask"].shape, "unique:", torch.unique(b["mask"]).tolist())
    
    # mask = b["mask"]
    # torch.set_printoptions(threshold=1000000)
    # print(mask)
