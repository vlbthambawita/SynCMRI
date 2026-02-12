import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from monai.networks.utils import one_hot
import glob
from PIL import Image

# Import your local data loader
current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

#Construct the path to the target folder
target_dir = os.path.join(
    project_root, 
    "mask-generation", 
    "monai-mask-generation", 
    "outputs", 
    "v2"
)
sys.path.append(target_dir)
from data_loading import get_loaders


'''
LPIPS is a perceptual metric trained on natural RGB images (textures, colors).
It is invalid for 4-class integer masks.

Metric Replacement (Dice Score): Instead of LPIPS, Nearest Neighbor Dice Score is implemented.
This effectively measures "perceptual" shape similarity for masks.
A score > 0.99 indicates a highly similar anatomy (potentially privacy-leaking if the anatomy is unique).

L2 Distance (One-Hot): Standard L2 distance on integer masks (0, 1, 2, 3) is mathematically flawed (Ex, Class 3 is not "3 times larger" than Class 1).
Convert all masks to One-Hot Encoding before calculating distances.
This treats every pixel mismatch equally, acting like a Hamming Distance.

NNDR (Nearest Neighbor Distance Ratio): Value ~ 1.0: The synthetic mask is generic.
It is roughly equidistant to its two closest real neighbors. (Good for privacy).
Value < 0.5: The synthetic mask is significantly closer to one specific real patient than any other.
This strongly implies the model has memorized that specific patient (Bad for privacy).
'''

# ==========================================
# CONFIGURATION
# ==========================================
# Paths must match your setup
CACHE_DIR = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache"
OUTPUT_DIR = "privacy_results"

# Model Config (Must match train.py)
IMG_SIZE = (128, 128)
NUM_CLASSES = 4
NUM_SAMPLES = None  # Number of synthetic samples to test
BATCH_SIZE = 16     # Batch size for generation

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. DATA PREPARATION FUNCTIONS
# ==========================================

def load_real_data(loader, max_samples=None):
    """
    Loads real masks from the validation loader and converts them to 
    flattened One-Hot tensors for distance calculation.
    """
    print("🔹 Loading Real Data...")
    real_masks_list = []
    
    for batch_masks in tqdm(loader, desc="Loading Real"):
        # batch_masks shape: [B, 1, 128, 128] (Integer 0-3)
        
        # 1. Convert to One-Hot: [B, 4, 128, 128]
        masks_oh = one_hot(batch_masks.to(DEVICE), num_classes=NUM_CLASSES, dim=1)
        
        # 2. Flatten: [B, 4*128*128] = [B, 65536]
        # We flatten to vectors to compute L2 distance efficiently
        masks_flat = masks_oh.view(masks_oh.shape[0], -1).float()
        
        real_masks_list.append(masks_flat.cpu())
        
        if max_samples and len(real_masks_list) * loader.batch_size >= max_samples:
            break
            
    # Stack into one big tensor: [N_real, Features]
    real_tensor = torch.cat(real_masks_list, dim=0)
    
    if max_samples:
        real_tensor = real_tensor[:max_samples]
        
    print(f" -> Loaded {real_tensor.shape[0]} real samples.")
    return real_tensor

def get_jet_reference_colors(num_classes):
    """
    Calculates the exact RGB values for classes 0, 1, 2, 3 
    based on the 'jet' colormap logic used during saving.
    """
    cmap = plt.get_cmap('jet')
    refs = []
    for i in range(num_classes):
        norm_val = i / (num_classes - 1)
        rgba = cmap(norm_val)
        rgb = (np.array(rgba[:3]) * 255).astype(np.uint8)
        refs.append(rgb)
    return np.array(refs)  # Shape: (4, 3)

def load_rgb_synthetic_data(folder_path, n_samples, device):
    """
    Loads RGB 'jet' images, converts them back to class indices (0-3),
    and returns the flattened One-Hot tensor required for evaluation.
    """
    print(f"🔹 Loading Synthetic RGB Masks from: {folder_path}")
    
    # Get reference colors for mapping
    ref_colors = get_jet_reference_colors(NUM_CLASSES) # Shape (4, 3)
    
    # Find PNG files
    files = sorted(glob.glob(os.path.join(folder_path, "*.png")))[:n_samples]
    
    if len(files) == 0:
        raise ValueError(f"No PNG images found in {folder_path}")

    syn_masks_list = []
    
    for fpath in tqdm(files, desc="Processing RGB Masks"):
        # 1. Load Image and convert to NumPy
        with Image.open(fpath) as img:
            rgb_img = np.array(img.convert('RGB')) # Shape (128, 128, 3)

        # 2. Map RGB pixels back to Class Indices (0, 1, 2, 3)
        # Calculate distance from each pixel to the 4 reference colors
        # Output shape: (128, 128) with integer values 0-3
        diff = rgb_img[:, :, None, :] - ref_colors[None, None, :, :] 
        dist = np.linalg.norm(diff, axis=3)
        mask_labels = np.argmin(dist, axis=2)

        # 3. Convert to Tensor [1, 128, 128]
        mask_t = torch.tensor(mask_labels).long().unsqueeze(0).to(device)

        # 4. Standard Pre-processing (One-Hot -> Flatten)
        mask_oh = one_hot(mask_t.unsqueeze(0), num_classes=NUM_CLASSES, dim=1)
        mask_flat = mask_oh.view(1, -1).float()
        
        syn_masks_list.append(mask_flat.cpu())

    # Stack all samples
    syn_tensor = torch.cat(syn_masks_list, dim=0)
    print(f" -> Loaded and converted {syn_tensor.shape[0]} synthetic samples.")
    return syn_tensor

# ==========================================
# 3. PRIVACY METRICS (MASK ADAPTED)
# ==========================================

def calculate_privacy_metrics(syn_flat, real_flat, device):
    """
    Calculates L2 Distance (One-Hot) and Dice Score NNDR.
    Because inputs are flattened One-Hot vectors:
      - L2 Distance acts like Root-Mean-Squared pixel disagreement.
      - Dot product acts like Intersection (for Dice).
    """
    n_syn = syn_flat.shape[0]
    n_real = real_flat.shape[0]
    
    min_l2_dists = []
    nndr_values = []
    max_dice_scores = []

    # Process in chunks to avoid OOM on GPU if N is large
    chunk_size = 50 
    
    real_flat_gpu = real_flat.to(device) # Move Reference set to GPU
    
    print("🔹 Computing Distances...")
    for i in tqdm(range(0, n_syn, chunk_size)):
        # Batch of synthetic samples
        syn_batch = syn_flat[i : i + chunk_size].to(device)
        
        # --- METRIC 1: Euclidean Distance (on One-Hot) ---
        # dists: [Batch, N_real]
        dists_l2 = torch.cdist(syn_batch, real_flat_gpu, p=2)
        
        # Get Top 2 Nearest Neighbors (smallest distances)
        top2_vals, _ = torch.topk(dists_l2, k=2, dim=1, largest=False)
        
        # Store Nearest Neighbor Distance
        min_l2_dists.extend(top2_vals[:, 0].cpu().numpy())
        
        # Calculate NNDR (Ratio of 1st NN / 2nd NN)
        # Ratio ~ 1.0 means the synth sample is equally far from neighbors (Good)
        # Ratio ~ 0.5 means it is MUCH closer to one specific real sample (Bad/Memorization)
        ratios = top2_vals[:, 0] / (top2_vals[:, 1] + 1e-8)
        nndr_values.extend(ratios.cpu().numpy())

        # --- METRIC 2: Dice Score (Perceptual Proxy) ---
        # Since we have one-hot vectors, we can calculate Dice efficiently via dot product.
        # Dice = 2 * (A . B) / (sum(A) + sum(B))
        # Note: This is a global dice over all classes combined (micro-average)
        
        intersection = torch.matmul(syn_batch, real_flat_gpu.T) # [Batch, N_real]
        sum_syn = torch.sum(syn_batch, dim=1, keepdim=True)
        sum_real = torch.sum(real_flat_gpu, dim=1, keepdim=True).T
        
        dice_matrix = (2. * intersection) / (sum_syn + sum_real + 1e-6)
        
        # Get Max Dice (Best Match)
        best_dice, _ = torch.max(dice_matrix, dim=1)
        max_dice_scores.extend(best_dice.cpu().numpy())
        
    return np.array(min_l2_dists), np.array(nndr_values), np.array(max_dice_scores)

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def run_evaluation():
    
    SYNTH_DATA_PATH = "/scratch1/e20-fyp-syn-car-mri-gen/mask-generation/monai-mask-generation/outputs/v2/mask_inference"

    # Get Data
    train_loader, _ = get_loaders(CACHE_DIR, BATCH_SIZE)
    
    real_data_flat = load_real_data(train_loader, max_samples=NUM_SAMPLES)
    syn_data_flat = load_rgb_synthetic_data(SYNTH_DATA_PATH, NUM_SAMPLES, DEVICE)

    # Calculate Metrics
    l2_dists, nndrs, dice_scores = calculate_privacy_metrics(syn_data_flat, real_data_flat, DEVICE)

    # Analysis
    print("\n--- PRIVACY EVALUATION RESULTS ---")
    
    # Exact Match Check (Dice > 0.99 or L2 < small_epsilon)
    # L2 distance for exact match on One-Hot vectors is 0.0
    n_exact_copies = np.sum(dice_scores > 0.99)
    print(f"WARNING: Found {n_exact_copies} potential exact copies (Dice > 0.99)")

    # Memorization Check (NNDR < 0.6)
    n_memorized = np.sum(nndrs < 0.6)
    print(f"WARNING: Found {n_memorized} potential memorized outliers (NNDR < 0.6)")

    # 5. Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: L2 Distance Distribution
    axes[0].hist(l2_dists, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_title("Distance to Nearest Neighbor (L2 One-Hot)")
    axes[0].set_xlabel("Euclidean Distance (Lower = More Similar)")
    axes[0].set_ylabel("Frequency")

    # Plot 2: Dice Score Distribution
    axes[1].hist(dice_scores, bins=30, color='salmon', edgecolor='black', alpha=0.7)
    axes[1].set_title("Max Dice Similarity (1.0 = Copy)")
    axes[1].set_xlabel("Dice Score (Higher = More Similar)")
    axes[1].axvline(0.99, color='red', linestyle='--', label='Copy Threshold')
    axes[1].legend()

    # Plot 3: NNDR Distribution
    axes[2].hist(nndrs, bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    axes[2].set_title("Nearest Neighbor Distance Ratio (NNDR)")
    axes[2].set_xlabel("Ratio (d1/d2)")
    axes[2].axvline(0.6, color='red', linestyle='--', label='Memorization Risk')
    axes[2].legend()
    
    # Save
    save_path = os.path.join(OUTPUT_DIR, "Mask_Privacy_Eval.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ Results saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    run_evaluation()