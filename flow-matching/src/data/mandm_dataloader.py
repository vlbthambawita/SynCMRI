# src/data/mandm_dataloader.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict,List,Tuple,Optional,Literal

import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

Split = Literal["train", "val", "test","full"]

@dataclass
class MandMDatasetConfig:
    cache_root: str = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache"
    crop_size: int = 128
    train_tag: str = "train_full_processed"
    val_tag: str = "val_processed"
    test_tag: str = "test_processed" 

@dataclass
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    shuffle_train: bool = True
    drop_last_train: bool = True

def get_patients_from_cache(cache_tag_path: str)-> List[str]:
    """Recover patient IDs by scanning for '*_indices.npy'."""
    if not os.path.exists(cache_tag_path):
        print(f"Cache path not found: {cache_tag_path}")
        return []
    files = [ f for f in os.listdir(cache_tag_path) if f.endswith("_indices.npy")]
    return sorted([f.replace("_indices.npy", "") for f in files])

def build_samples(cache_tag_path: str, patient_ids: List[str]) -> List[Tuple[str, int, int]]:
    """
    Builds a flat list of (pid, slice_idx, frame_idx) from cached indices.
    """
    samples: List[Tuple[str, int, int]] = []
    for pid in patient_ids:
        indices_file = os.path.join(cache_tag_path, f"{pid}_indices.npy")
        if not os.path.exists(indices_file):
            continue
        indices = np.load(indices_file)
        for row in indices:
            samples.append((pid, int(row[0]), int(row[1])))
    if len(samples) == 0:
        raise RuntimeError(f"No samples built from {cache_tag_path}. Check indices files.")
    return samples

class MandMDataset(Dataset):
    """
    Loads 2D slice (slice_idx, frame_idx) from cached 4D npy files.
    Applies:
      - ROI crop centered on mask
      - Percentile normalization to [-1, 1]
      - Optional augmentations (flip, rotate)
    """
    def __init__(self,
                 cache_tag_path: str,
                 samples: List[Tuple[str, int, int]],
                 crop_size: int = 128,
                 augment: bool = False,
                 ):
        self.cache_tag_path = cache_tag_path
        self.samples = samples
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self)->int:
        return len(self.samples)

    def _crop_to_mask_center(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mask: (1,H,W)
        y_idx, x_idx = torch.where(mask[0] > 0)

        if len(y_idx) > 0:
            center_y = int(torch.mean(y_idx.float()).item())
            center_x = int(torch.mean(x_idx.float()).item())
        else:
            center_y = img.shape[1] // 2
            center_x = img.shape[2] // 2

        half = self.crop_size // 2
        start_y, end_y = center_y - half, center_y + half
        start_x, end_x = center_x - half, center_x + half

        pad_top = max(0, -start_y)
        pad_left = max(0, -start_x)
        pad_bottom = max(0, end_y - img.shape[1])
        pad_right = max(0, end_x - img.shape[2])

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=float(img.min()))
            mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)

            start_y += pad_top; end_y += pad_top
            start_x += pad_left; end_x += pad_left

        img = img[:, start_y:end_y, start_x:end_x]
        mask = mask[:, start_y:end_y, start_x:end_x]
        return img, mask 

    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            img = TF.hflip(img); mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img); mask = TF.vflip(mask)

        angle = random.uniform(-20, 20)
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=float(img.min()))
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
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

        img = torch.tensor(img2d.copy(), dtype=torch.float32).unsqueeze(0)  # (1,H,W)
        mask = torch.tensor(mask2d.copy(), dtype=torch.long).unsqueeze(0)   # (1,H,W)      
        
        img, mask = self._crop_to_mask_center(img, mask)

        if self.augment:
            img, mask = self._augment(img, mask)

        img = self._percentile_norm(img)
        
        return {"image": img, "mask": mask}
    
def build_mandm_loaders(
    cache_root: str,
    dc: Optional[MandMDatasetConfig] = None,
    lc: Optional[LoaderConfig] = None,
    is_test_set_only: bool = False
):
    dc = dc or MandMDatasetConfig()
    lc = lc or LoaderConfig()

    train_path = os.path.join(cache_root,dc.train_tag)
    val_path = os.path.join(cache_root,dc.val_tag)
    test_path = os.path.join(cache_root,dc.test_tag)

    train_pids = get_patients_from_cache(train_path)
    val_pids = get_patients_from_cache(val_path)
    test_pids = get_patients_from_cache(test_path)

    if len(train_pids) == 0:
        raise RuntimeError(f"No training patients found in {train_path}. Check cache_root.")
    
    train_samples = build_samples(train_path, train_pids)
    val_samples = build_samples(val_path, val_pids)
    test_samples = build_samples(test_path, test_pids)

    ds_train = MandMDataset(train_path,train_samples,augment=True)
    ds_val =  MandMDataset(val_path,val_samples,augment=False)
    ds_test = MandMDataset(test_path,test_samples,augment=False)

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
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
    }
    if is_test_set_only:
        return test_loader,stats

    return train_loader, val_loader, test_loader, stats

if __name__ == '__main__':
    cache_root = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache"

    dc = MandMDatasetConfig(cache_root=cache_root)
    lc = LoaderConfig(batch_size=8, num_workers=2)
    train_loader, val_loader, test_loader, stats = build_mandm_loaders(cache_root,dc=dc,lc=lc)
    print("STATS:", stats)
    