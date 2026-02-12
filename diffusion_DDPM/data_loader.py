import os
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import random

# =========================
# CONFIG & PATHS
# =========================
# ### UPDATE THIS PATH IF NEEDED ###
CACHE_DIR = "/storage/scratch2/e20-syncar-mri-igen/Datasets/MandM Dataset/cache"
CROP_SIZE = 128

# =========================
# HELPER FUNCTIONS
# =========================
def get_patients_from_cache(cache_tag_path):
    """Scans the cache directory to find available Patient IDs."""
    if not os.path.exists(cache_tag_path):
        print(f"⚠ Cache path not found: {cache_tag_path}")
        return []
    files = [f for f in os.listdir(cache_tag_path) if f.endswith("_indices.npy")]
    return sorted([f.replace("_indices.npy", "") for f in files])

def adapter_collate_fn(batch):
    """
    CRITICAL: Converts the dataset tuple output into the dictionary format
    that training.py expects: {'images': ..., 'seg_onehot': ...}
    """
    imgs = [item[0] for item in batch]
    # item[1] is {'image': mask_tensor}
    conds = [item[1]['image'] for item in batch]
    
    imgs_tensor = torch.stack(imgs)   # [B, 1, 128, 128]
    masks_tensor = torch.stack(conds) # [B, 4, 128, 128]

    return {
        "images": imgs_tensor,
        "seg_onehot": masks_tensor # Key must start with 'seg_'
    }

# =========================
# DATASET CLASS
# =========================
class CMRDataset(Dataset):
    def __init__(self, patient_ids, tag, cache_root, crop_size=128, augment=False):
        self.samples = []
        self.tag = tag
        self.cache_path = os.path.join(cache_root, tag)
        self.crop_size = crop_size
        self.augment = augment

        # Load valid indices from the cache
        print(f"Loading indices for {len(patient_ids)} patients...")
        for pid in patient_ids:
            indices_file = os.path.join(self.cache_path, f"{pid}_indices.npy")
            if not os.path.exists(indices_file): 
                continue
            
            try:
                indices = np.load(indices_file)
                for row in indices:
                    self.samples.append((pid, row[0], row[1]))
            except Exception as e:
                print(f"Error loading {pid}: {e}")

    def __len__(self):
        return len(self.samples)
    
    def apply_augmentation(self, img, mask):
        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 2. Random Vertical Flip
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # 3. Random Rotation
        angle = random.uniform(-20, 20)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR, fill=img.min().item())
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST, fill=0)
        return img, mask

    def crop_to_mask_center(self, img, mask):
        # Find mask center
        y_indices, x_indices = np.where(mask[0] > 0)
        
        if len(y_indices) > 0:
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
        else:
            center_y, center_x = img.shape[1] // 2, img.shape[2] // 2

        half_size = self.crop_size // 2
        
        start_y = center_y - half_size
        end_y = center_y + half_size
        start_x = center_x - half_size
        end_x = center_x + half_size

        # Padding Logic
        pad_top = max(0, -start_y)
        pad_bottom = max(0, end_y - img.shape[1])
        pad_left = max(0, -start_x)
        pad_right = max(0, end_x - img.shape[2])

        if any([pad_top, pad_bottom, pad_left, pad_right]):
            img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=img.min().item())
            mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
            
            start_y += pad_top
            end_y += pad_top
            start_x += pad_left
            end_x += pad_left

        return img[:, start_y:end_y, start_x:end_x], mask[:, start_y:end_y, start_x:end_x]

    def __getitem__(self, idx):
        pid, slice_idx, frame_idx = self.samples[idx]
        
        img_path = os.path.join(self.cache_path, f"{pid}_resampled_4d_img.npy")
        mask_path = os.path.join(self.cache_path, f"{pid}_resampled_4d_mask.npy")

        # Memory Map Load
        img_vol = np.load(img_path, mmap_mode='r')
        mask_vol = np.load(mask_path, mmap_mode='r')

        img = img_vol[:, :, slice_idx, frame_idx]
        mask = mask_vol[:, :, slice_idx, frame_idx]

        # To Tensor
        img = torch.tensor(img.copy(), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask.copy(), dtype=torch.long).unsqueeze(0)

        # Augment (Train only)
        if self.augment:
            img, mask = self.apply_augmentation(img, mask)
        
        # Crop
        img, mask = self.crop_to_mask_center(img, mask)

        # Normalize
        if torch.isnan(img).any(): img = torch.nan_to_num(img, nan=0.0)
        img_np = img.numpy() 
        p01, p99 = np.percentile(img_np, 1), np.percentile(img_np, 99.9)
        img_np = np.clip(img_np, p01, p99)
        
        if p99 - p01 > 1e-6:
            img_np = (img_np - p01) / (p99 - p01)
            img_np = (img_np * 2) - 1
        else:
            img_np = np.zeros_like(img_np) - 1

        # Process Mask
        mask = mask.squeeze(0) 
        mask = F.one_hot(mask.long(), num_classes=4).float()
        mask = mask.permute(2, 0, 1) # [4, 128, 128]

        return torch.tensor(img_np, dtype=torch.float32), {'image': mask}

# =========================
# EXPORT FUNCTION
# =========================
def get_data_loaders(batch_size=8):
    """
    Returns (loader_150, val_loader) ready for training.
    """
    train_cache_path = os.path.join(CACHE_DIR, "train_full_processed")
    val_cache_path = os.path.join(CACHE_DIR, "val_processed")

    all_train_patients = get_patients_from_cache(train_cache_path)
    all_val_patients = get_patients_from_cache(val_cache_path)

    if not all_train_patients:
        raise RuntimeError(f"No training data in {train_cache_path}")

    # Select 150 patients
    train_150_ids = all_train_patients[:175] # Adjust indexing as needed

    print(f"Initialize Dataset: Train ({len(train_150_ids)} patients), Val ({len(all_val_patients)} patients)")
    
    ds_train_150 = CMRDataset(train_150_ids, "train_full_processed", CACHE_DIR, crop_size=CROP_SIZE, augment=True)
    ds_val = CMRDataset(all_val_patients, "val_processed", CACHE_DIR, crop_size=CROP_SIZE, augment=False)

    # Note: We pass 'adapter_collate_fn' here!
    loader_150 = DataLoader(ds_train_150, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=adapter_collate_fn)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=adapter_collate_fn)

    return loader_150, val_loader


# =========================
# TEST LOADER
# =========================

def get_test_loader(batch_size=8):
    """
    Returns the dedicated Test Loader (Unseen data).
    """
    # Define Test Path
    test_cache_tag = "test_processed"
    test_cache_path = os.path.join(CACHE_DIR, test_cache_tag)
    
    # Get Patients
    test_ids = get_patients_from_cache(test_cache_path)
    
    if not test_ids:
        print(f"⚠ Warning: No test patients found in {test_cache_path}")
        return None

    print(f"Initialize Dataset: Test ({len(test_ids)} patients)")
    
    # Create Dataset & Loader
    ds_test = CMRDataset(test_ids, test_cache_tag, CACHE_DIR, crop_size=CROP_SIZE, augment=False)
    
    test_loader = DataLoader(
        ds_test, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2, 
        collate_fn=adapter_collate_fn
    )

    return test_loader