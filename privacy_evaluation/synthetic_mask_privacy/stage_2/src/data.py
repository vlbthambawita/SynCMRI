import os
import torch
import numpy as np
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F


def get_patients_from_cache(cache_tag_path):
    """
    Scans the cache directory to find available Patient IDs
    based on the existence of '*_indices.npy' files.
    """
    if not os.path.exists(cache_tag_path):
        print(f"⚠ Cache path not found: {cache_tag_path}")
        return []

    # Find all files ending in '_indices.npy'
    files = [f for f in os.listdir(cache_tag_path) if f.endswith("_indices.npy")]
    
    # Extract Patient ID (e.g., "patient001_indices.npy" -> "patient001")
    pids = sorted([f.replace("_indices.npy", "") for f in files])
    
    return pids

# =========================
# DATASET CLASS (LOADING LOGIC)
# =========================
class CMRDataset(Dataset):
    def __init__(self, patient_ids, tag, cache_root, crop_size=128, augment=False):
        self.samples = []
        self.tag = tag
        self.cache_path = os.path.join(cache_root, tag)
        self.crop_size = crop_size
        self.augment = augment

        # Load valid indices from the cache
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

    def crop_to_mask_center(self, img, mask):
        """
        Smart ROI Crop: Centers the crop on the mask center.
        """
        # Find mask center
        y_indices, x_indices = np.where(mask[0] > 0)
        
        if len(y_indices) > 0:
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
        else:
            # Fallback to image center
            center_y, center_x = img.shape[1] // 2, img.shape[2] // 2

        half_size = self.crop_size // 2
        
        start_y = center_y - half_size
        end_y = center_y + half_size
        start_x = center_x - half_size
        end_x = center_x + half_size

        # Calculate Padding
        pad_top = max(0, -start_y)
        pad_bottom = max(0, end_y - img.shape[1])
        pad_left = max(0, -start_x)
        pad_right = max(0, end_x - img.shape[2])

        # Apply Padding if needed
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=img.min().item())
            mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
            
            start_y += pad_top
            end_y += pad_top
            start_x += pad_left
            end_x += pad_left

        # Crop
        img_crop = img[:, start_y:end_y, start_x:end_x]
        mask_crop = mask[:, start_y:end_y, start_x:end_x]
        
        return img_crop, mask_crop

    def apply_augmentation(self, img, mask):
        """
        Applies Random Rotation (+/- 20) and Random Flips (H/V).
        """
        # 1. Random Horizontal Flip (50%)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        # 2. Random Vertical Flip (50%)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        # 3. Random Rotation (-20 to +20 degrees)
        angle = random.uniform(-20, 20)
        
        # Rotate Image (Bilinear to keep it smooth)
        img = TF.rotate(
            img, 
            angle, 
            interpolation=InterpolationMode.BILINEAR, 
            fill=img.min().item() # Fill corners with dark background
        )
        
        # Rotate Mask (Nearest Neighbor to keep strict labels 0,1,2,3)
        mask = TF.rotate(
            mask, 
            angle, 
            interpolation=InterpolationMode.NEAREST, 
            fill=0 # Fill corners with background class
        )
        
        return img, mask

    def __getitem__(self, idx):
        pid, slice_idx, frame_idx = self.samples[idx]
        
        # Construct paths
        img_path = os.path.join(self.cache_path, f"{pid}_resampled_4d_img.npy")
        mask_path = os.path.join(self.cache_path, f"{pid}_resampled_4d_mask.npy")

        # Load Data (Memory Mapped for speed)
        img_vol = np.load(img_path, mmap_mode='r')
        mask_vol = np.load(mask_path, mmap_mode='r')

        img = img_vol[:, :, slice_idx, frame_idx]
        mask = mask_vol[:, :, slice_idx, frame_idx]

        # Convert to Tensor
        img = torch.tensor(img.copy(), dtype=torch.float32).unsqueeze(0)
        mask = torch.tensor(mask.copy(), dtype=torch.long).unsqueeze(0)

        # Augmentation
        if self.augment:
            img, mask = self.apply_augmentation(img, mask)

        # CROPPING- Cut out the clean center
        img, mask = self.crop_to_mask_center(img, mask)

        # 2. Percentile Normalization
        if torch.isnan(img).any():
            img = torch.nan_to_num(img, nan=0.0)

        img_np = img.numpy() 
        p01 = np.percentile(img_np, 1)
        p99 = np.percentile(img_np, 99.9)
        img_np = np.clip(img_np, p01, p99)
        
        if p99 - p01 > 1e-6:
            img_np = (img_np - p01) / (p99 - p01)
            img_np = (img_np * 2) - 1
        else:
            img_np = np.zeros_like(img_np) - 1
            
        # img = torch.tensor(img_np, dtype=torch.float32)
        # return img, mask

        # Remove channel dim [1, 128, 128] -> [128, 128]
        mask = mask.squeeze(0) 
        
        # One-Hot Encode [128, 128] -> [128, 128, 4]
        mask = F.one_hot(mask.long(), num_classes=4).float()
        
        # Channels First [128, 128, 4] -> [4, 128, 128]
        mask = mask.permute(2, 0, 1)

        # Return as DICTIONARY (This fixes the "too many indices" error everywhere!)
        return torch.tensor(img_np, dtype=torch.float32), {'image': mask}


def get_dataloaders(config):
    # Extract config params
    cache_dir = config.paths['cache_dir']
    train_tag = config.paths.get('train_tag', "train_full_processed")
    test_tag = config.paths.get('test_tag', "test_processed")
    crop_size = config.dataset_params['im_size']
    subset_size = config.dataset_params.get('subset_size', 150)
    batch_size = 8 # Or from config if available

    # Get Patients
    train_cache_path = os.path.join(cache_dir, train_tag)
    test_cache_path = os.path.join(cache_dir, test_tag)

    all_train_patients = get_patients_from_cache(train_cache_path)
    all_test_patients = get_patients_from_cache(test_cache_path)


    # 2. Select Subset for Train
    train_subset = all_train_patients[:subset_size]

    # 3. Create Datasets
    ds_train = CMRDataset(train_subset, train_tag, cache_dir, crop_size=crop_size, augment=False) # Augment is True for train usually, but check privacy eval requirements. The main block said `augment=False # No augmentation for privacy attack`. I should probably check that constraint. 
    ds_test = CMRDataset(all_test_patients, test_tag, cache_dir, crop_size=crop_size, augment=False)

    # 4. Create Loaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader