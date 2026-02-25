import os
import torch
import numpy as np
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader


# CONFIGURATION
CROP_SIZE = 128 
cache_dir = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache" 


#FIND PATIENTS IN CACHE
def get_patients_from_cache(cache_root, tag):
    """
    Scans the specific cache subfolder (e.g., 'train_full_processed') 
    to find valid Patient IDs based on existing *_indices.npy files.
    """
    target_path = os.path.join(cache_root, tag)
    
    if not os.path.exists(target_path):
        print(f"Cache path not found: {target_path}")
        return []

    # We look for indices files because they represent successfully processed patients
    files = [f for f in os.listdir(target_path) if f.endswith("_indices.npy")]
    
    # Extract Patient ID (e.g., "A0S9V9_indices.npy" -> "A0S9V9")
    pids = sorted([f.replace("_indices.npy", "") for f in files])
    
    return pids


#MASK-ONLY DATASET
class MaskOnlyDataset(Dataset):
    def __init__(self, patient_ids, tag, cache_root, crop_size=128):
        self.samples = []
        self.tag = tag
        self.cache_path = os.path.join(cache_root, tag)
        self.crop_size = crop_size

        # 1. Load indices from the cache
        # We assume the indices in the cache are already filtered and valid.
        for pid in patient_ids:
            indices_file = os.path.join(self.cache_path, f"{pid}_indices.npy")
            
            # Safety check
            if not os.path.exists(indices_file): 
                continue
            
            try:
                # Load the list of valid [slice, frame] pairs
                indices = np.load(indices_file)
                for row in indices:
                    self.samples.append((pid, row[0], row[1]))
            except Exception as e:
                print(f"Error loading indices for {pid}: {e}")

    def __len__(self):
        return len(self.samples)

    def crop_to_mask_center(self, mask):
        """
        Calculates center of mass of the mask and crops around it.
        """
        # Find coordinates where mask > 0
        y_indices, x_indices = np.where(mask[0] > 0)
        
        # Calculate Center
        if len(y_indices) > 0:
            center_y = int(np.mean(y_indices))
            center_x = int(np.mean(x_indices))
        else:
            # Fallback to image center if mask is empty (shouldn't happen with filtered indices)
            center_y, center_x = mask.shape[1] // 2, mask.shape[2] // 2

        # Calculate Box Coords
        half_size = self.crop_size // 2
        start_y = center_y - half_size
        end_y = center_y + half_size
        start_x = center_x - half_size
        end_x = center_x + half_size

        # Calculate Padding (if crop goes off edge)
        pad_top = max(0, -start_y)
        pad_bottom = max(0, end_y - mask.shape[1])
        pad_left = max(0, -start_x)
        pad_right = max(0, end_x - mask.shape[2])

        # Apply Pad if needed
        if any([pad_top, pad_bottom, pad_left, pad_right]):
            mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=0)
            # Update coords after padding
            start_y += pad_top
            end_y += pad_top
            start_x += pad_left
            end_x += pad_left

        # Crop
        mask_crop = mask[:, start_y:end_y, start_x:end_x]
        return mask_crop

    def __getitem__(self, idx):
        pid, slice_idx, frame_idx = self.samples[idx]
        
        # Construct path to the 4D Mask file
        mask_path = os.path.join(self.cache_path, f"{pid}_resampled_4d_mask.npy")
        
        # Load using mmap_mode='r' (Very fast, doesn't load whole file to RAM)
        mask_vol = np.load(mask_path, mmap_mode='r')
        
        # Extract specific slice
        mask = mask_vol[:, :, slice_idx, frame_idx]

        # Convert to Tensor (Add channel dimension: [1, H, W])
        mask = torch.tensor(mask.copy(), dtype=torch.long).unsqueeze(0)

        # Apply Smart Crop
        mask = self.crop_to_mask_center(mask)
        
        return mask


#EXPORT FUNCTION
def get_loaders(cache_dir, batch_size=8, num_workers=2):
    """
    Creates and returns the specific DataLoaders required for training.
    """
    # 1. Discover Patients
    train_tag = "train_full_processed"
    val_tag = "val_processed" 

    all_train_patients = get_patients_from_cache(cache_dir, train_tag)
    all_val_patients = get_patients_from_cache(cache_dir, val_tag)

    print(f"Found {len(all_train_patients)} training patients.")
    print(f"Found {len(all_val_patients)} validation patients.")

    if len(all_train_patients) == 0:
        raise RuntimeError(f"❌ No training data found in {cache_dir}")

    # 2. Select the specific subset (e.g., 150 patients)
    train_ids = all_train_patients[:150] 
    
    # 3. Instantiate Datasets
    # Note: We hardcode CROP_SIZE=128 as per your config
    ds_train = MaskOnlyDataset(train_ids, train_tag, cache_dir, crop_size=128)
    ds_val = MaskOnlyDataset(all_val_patients, val_tag, cache_dir, crop_size=128)

    # 4. Create Loaders
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader
