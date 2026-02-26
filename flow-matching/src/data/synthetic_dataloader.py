# src/data/syn_dataloader.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Literal

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image  

Split = Literal["train", "val"]

# Fixed global mapping 
MASK_VALUE_TO_CLASS = {
    14: 0,
    38: 1,
    154: 2,
    211: 3,
}
# Configs
@dataclass
class SyntheticDatasetConfig:
    preprocessed_root: str = "/scratch1/e20-fyp-syn-car-mri-gen/flow-matching/generated/synthetic_imgs/20260214_1240"
    crop_size: int = 128
    split_seed: int = 42
    split_cache_dir: Optional[str] = None


@dataclass
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle_train: bool = True
    drop_last_train: bool = True


# Dataset
class SyntheticDataset(Dataset):
    def __init__(self, file_list, root_dir, crop_size=128):
        self.file_list = file_list
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "synth")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.crop_size = crop_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        img_path = os.path.join(self.img_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename)

        # Image
        img = Image.open(img_path).convert("L")
        img_np = np.array(img).astype(np.float32)
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)

        # Mask
        mask = Image.open(mask_path).convert("L")
        mask_u8 = np.array(mask).astype(np.uint8)

        mask_cls = np.zeros_like(mask_u8, dtype=np.uint8)
        for raw_val, cls_id in MASK_VALUE_TO_CLASS.items():
            mask_cls[mask_u8 == raw_val] = cls_id

        # -----------------------------------------
        # REMAP synthetic class-ids -> MandM class-ids
        # syn: 1=RV, 2=LV, 3=MYO
        # want: 1=LV, 2=MYO, 3=RV
        # -----------------------------------------
        # mask_mandm = mask_cls
        mask_mandm = np.zeros_like(mask_cls, dtype=np.uint8)
        mask_mandm[mask_cls == 0] = 0
        mask_mandm[mask_cls == 1] = 3  # LV
        mask_mandm[mask_cls == 3] = 2  # MYO
        mask_mandm[mask_cls == 2] = 1  # RV

        mask_tensor = torch.from_numpy(mask_mandm).long().unsqueeze(0)

        return {"image": img_tensor, "mask": mask_tensor}
# -------------------------
# Builder
# -------------------------
def build_syn_loaders(
    cache_root: str,
    dc: Optional[SyntheticDatasetConfig] = None,
    lc: Optional[LoaderConfig] = None
):
    dc = dc or SyntheticDatasetConfig()
    lc = lc or LoaderConfig()

    # Only load PNG files
    all_files = sorted(
        [f for f in os.listdir(os.path.join(cache_root, "synth")) if f.endswith(".png")]
    )

    total_files = len(all_files)
    print(f"Found {total_files} total PNG images.")
    limited_files = all_files

    train_samples, val_samples = train_test_split(
        limited_files,
        test_size=0.3,
        random_state=dc.split_seed
    )

    ds_train = SyntheticDataset(train_samples, dc.preprocessed_root, dc.crop_size)
    ds_val = SyntheticDataset(val_samples, dc.preprocessed_root, dc.crop_size)

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

    stats = {
        "n_train_samples": len(ds_train),
        "n_val_samples": len(ds_val),
    }

    return train_loader, val_loader, stats



# Test
if __name__ == '__main__':
    cache_root = "/scratch1/e20-fyp-syn-car-mri-gen/flow-matching/generated/synthetic_imgs/20260214_1240"

    dc = SyntheticDatasetConfig()
    lc = LoaderConfig(batch_size=8, num_workers=2)

    train_loader, val_loader, stats = build_syn_loaders(cache_root, dc=dc, lc=lc)
    print("STATS:", stats)

    batch = next(iter(train_loader))
    print("Batch image shape:", batch["image"].shape)
    print("Batch mask shape:", batch["mask"].shape)
    print("unique",torch.unique(batch["mask"]))

    mask = batch["mask"]
    torch.set_printoptions(threshold=1000000)
    print(mask)