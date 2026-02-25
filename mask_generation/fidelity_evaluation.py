import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
import math
from tqdm import tqdm
from PIL import Image
from scipy.stats import wasserstein_distance
from skimage.measure import label, regionprops
from monai.networks.utils import one_hot
from scipy.spatial.distance import directed_hausdorff

# Import your loader for Real Data
from data_loading import get_loaders

# ==========================================
# 1. CONFIGURATION
# ==========================================
CACHE_DIR = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache"
OUTPUT_DIR = "fidelity_results"

# Path to your folder of generated RGB 'Jet' masks
SYNTH_DATA_PATH = "/scratch1/e20-fyp-syn-car-mri-gen/mask-generation/monai-mask-generation/outputs/v2/mask_inference" 

IMG_SIZE = (128, 128)
NUM_CLASSES = 4
NUM_SAMPLES = 1400  # How many samples to compare
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# 2. DATA LOADING (RGB JET -> INTEGER)
# ==========================================

def get_jet_reference_colors(num_classes):
    """Recreates the Jet colormap reference values."""
    cmap = plt.get_cmap('jet')
    refs = []
    for i in range(num_classes):
        norm_val = i / (num_classes - 1)
        rgba = cmap(norm_val)
        rgb = (np.array(rgba[:3]) * 255).astype(np.uint8)
        refs.append(rgb)
    return np.array(refs)

def load_synthetic_rgb_masks(folder_path, n_samples):
    """
    Loads RGB images, maps colors to class integers (0-3).
    Returns Numpy array: [N, 128, 128]
    """
    print(f"🔹 Loading Synthetic RGB Masks from: {folder_path}")
    files = sorted(glob.glob(os.path.join(folder_path, "*.png")))[:n_samples]
    
    if len(files) == 0:
        raise ValueError(f"No PNG images found in {folder_path}")

    ref_colors = get_jet_reference_colors(NUM_CLASSES)
    masks_list = []

    for fpath in tqdm(files, desc="Converting RGB -> Int"):
        with Image.open(fpath) as img:
            rgb_img = np.array(img.convert('RGB')) 
        
        # Map pixels to closest reference color
        diff = rgb_img[:, :, None, :] - ref_colors[None, None, :, :] 
        dist = np.linalg.norm(diff, axis=3)
        mask_labels = np.argmin(dist, axis=2) # [128, 128] integers
        
        masks_list.append(mask_labels)
        
    return np.array(masks_list)

def load_real_masks(loader, n_samples):
    """Loads real masks from DataLoader into Numpy array."""
    print("🔹 Loading Real Masks...")
    masks_list = []
    count = 0
    for batch in loader:
        # batch is [B, 1, 128, 128]
        imgs = batch.squeeze(1).numpy()
        for img in imgs:
            if count >= n_samples: break
            masks_list.append(img)
            count += 1
        if count >= n_samples: break
            
    return np.array(masks_list)

# ==========================================
# 3. METRIC CALCULATIONS
# ==========================================

def get_shape_metrics(mask, class_id):
    """
    Calculates Area, Perimeter, and Roundness for a specific class in a mask.
    Returns None if class is missing.
    """
    binary_mask = (mask == class_id).astype(int)
    
    if np.sum(binary_mask) == 0:
        return None # Class missing
    
    # Label connected components
    lbl_mask = label(binary_mask)
    regions = regionprops(lbl_mask)
    
    # Anatomical Validity Check: Should be exactly 1 component
    n_components = len(regions)
    
    # Get largest component (main anatomy) for shape stats
    main_region = max(regions, key=lambda x: x.area)
    
    area = main_region.area
    perimeter = main_region.perimeter
    # Roundness: 4*pi*Area / Perimeter^2 (1.0 is a perfect circle)
    roundness = (4 * math.pi * area) / (perimeter**2) if perimeter > 0 else 0
    
    return {
        "Area": area,
        "Perimeter": perimeter,
        "Roundness": roundness,
        "Components": n_components
    }

def analyze_dataset(masks_np, group_name):
    """
    Iterates over a dataset and extracts shape statistics for LV, Myo, RV.
    """
    data = []
    classes = {1: "LV", 2: "Myo", 3: "RV"}
    
    print(f"   Analyzing {group_name} ({len(masks_np)} samples)...")
    
    for i, mask in enumerate(masks_np):
        for cid, cname in classes.items():
            stats = get_shape_metrics(mask, cid)
            if stats:
                row = stats.copy()
                row["Structure"] = cname
                row["Group"] = group_name
                row["Sample_ID"] = i
                data.append(row)
            else:
                # Log missing anatomy
                data.append({
                    "Area": 0, "Perimeter": 0, "Roundness": 0, "Components": 0,
                    "Structure": cname, "Group": group_name, "Sample_ID": i
                })
                
    return pd.DataFrame(data)

def compute_best_match_dice(syn_masks, real_masks, num_classes=4):
    """
    Computes the 'Best Match' Dice score for each synthetic mask against the real set.
    """
    print(f"🔹 Computing Best-Match Dice for {num_classes} classes...")
    
    # We will average the scores across the 3 anatomical classes (LV, Myo, RV)
    # Background (class 0) is usually ignored in Dice scores.
    results = []

    # Flatten masks for matrix operations: [N, 128*128]
    # We process one class at a time to keep it simple
    for c in range(1, num_classes): # Skip 0 (Background)
        
        # Create binary vectors for this class
        flat_real = (real_masks == c).reshape(real_masks.shape[0], -1).astype(np.float32)
        flat_syn = (syn_masks == c).reshape(syn_masks.shape[0], -1).astype(np.float32)

        # Intersection: Matmul (N_syn x pixels) @ (pixels x N_real) -> (N_syn x N_real)
        intersection = np.matmul(flat_syn, flat_real.T)
        
        # Sum of pixels (cardinality)
        sum_syn = np.sum(flat_syn, axis=1, keepdims=True)
        sum_real = np.sum(flat_real, axis=1, keepdims=True).T
        
        # Dice Matrix: 2*Int / (Sum1 + Sum2)
        dice_mat = (2.0 * intersection) / (sum_syn + sum_real + 1e-6)
        
        # For each Synthetic mask, find the MAX dice score in the Real set (Nearest Neighbor)
        best_matches = np.max(dice_mat, axis=1)
        results.append(best_matches)

    # Average across the 3 classes (LV, Myo, RV)
    # Shape: [N_syn] (one score per generated mask)
    avg_best_match = np.mean(np.array(results), axis=0)
    
    return avg_best_match

def compute_hausdorff_95(syn_masks, real_masks, num_classes=4, max_samples=100):
    """
    Computes the 95th Percentile Hausdorff Distance for synthetic masks 
    against their nearest neighbors in the real set.
    """
    print(f"🔹 Computing Hausdorff-95 (sampling first {max_samples} for speed)...")
    
    # Limit samples for speed (Hausdorff is very slow)
    syn_subset = syn_masks[:max_samples]
    
    # --- Step 1: Find Nearest Neighbors (Best Match) efficiently ---
    # We use simple overlap (Dot product) to find which real mask is closest
    # Flatten: [N, H*W]
    syn_flat = syn_subset.reshape(syn_subset.shape[0], -1).astype(np.float32)
    # Use a random subset of real masks if too large, or use all
    real_flat = real_masks.reshape(real_masks.shape[0], -1).astype(np.float32)
    
    # Approximate correlation to find best match index
    # [N_syn, N_real]
    scores = np.matmul(syn_flat, real_flat.T)
    best_match_indices = np.argmax(scores, axis=1)
    
    hausdorff_values = []

    # --- Step 2: Calculate Hausdorff for each pair ---
    for i, syn_idx in enumerate(tqdm(range(len(syn_subset)), desc="Hausdorff")):
        real_idx = best_match_indices[i]
        
        mask_s = syn_subset[syn_idx]
        mask_r = real_masks[real_idx]
        
        # Average HD across the 3 anatomical classes (1=LV, 2=Myo, 3=RV)
        class_distances = []
        
        for c in range(1, num_classes):
            # Get coordinates of pixels for this class
            # argwhere returns [y, x] points
            pts_s = np.argwhere(mask_s == c)
            pts_r = np.argwhere(mask_r == c)
            
            if len(pts_s) > 0 and len(pts_r) > 0:
                # Scipy calculates directed HD: A->B and B->A
                d_forward = directed_hausdorff(pts_s, pts_r)[0]
                d_backward = directed_hausdorff(pts_r, pts_s)[0]
                # True HD is the max of the two
                class_distances.append(max(d_forward, d_backward))
            else:
                # Penalty if anatomy is missing in one but present in other
                if len(pts_s) != len(pts_r):
                    class_distances.append(20.0) # Arbitrary high penalty
        
        if class_distances:
            hausdorff_values.append(np.mean(class_distances))
        else:
            hausdorff_values.append(0.0)

    return np.array(hausdorff_values)

def compute_diversity(masks, num_classes=4):
    """
    Computes pairwise Dice scores within a single set (intra-set diversity).
    Lower average = Higher diversity (samples are different from each other).
    """
    print("🔹 Computing Diversity (Pairwise Dice)...")
    results = []
    
    # Sample a random subset (e.g., 100) to keep matrix small
    subset_indices = np.random.choice(len(masks), min(100, len(masks)), replace=False)
    subset = masks[subset_indices]

    for c in range(1, num_classes):
        flat = (subset == c).reshape(subset.shape[0], -1).astype(np.float32)
        intersection = np.matmul(flat, flat.T)
        sums = np.sum(flat, axis=1, keepdims=True)
        
        # Pairwise matrix
        dice_mat = (2.0 * intersection) / (sums + sums.T + 1e-6)
        
        # We only want off-diagonal elements (ignore comparing A with A)
        # Get upper triangle values
        upper_tri = dice_mat[np.triu_indices_from(dice_mat, k=1)]
        results.append(upper_tri)

    return np.mean(np.concatenate(results))

# ==========================================
# 4. MAIN PIPELINE
# ==========================================

def run_fidelity_eval():
    # 1. Load Data
    # IMPORTANT: Use val_loader for evaluation!
    train_loader, val_loader = get_loaders(CACHE_DIR)
    
    real_masks = load_real_masks(train_loader, NUM_SAMPLES)
    syn_masks = load_synthetic_rgb_masks(SYNTH_DATA_PATH, NUM_SAMPLES)
    
    # 2. Extract Features
    print("\n🔹 Computing Shape Statistics...")
    df_real = analyze_dataset(real_masks, "Real")
    df_syn = analyze_dataset(syn_masks, "Synthetic")
    
    df_all = pd.concat([df_real, df_syn], ignore_index=True)

    # --- METRIC 1: Best-Match Dice (Pixel Fidelity) ---
    best_dice_scores = compute_best_match_dice(syn_masks, real_masks)
    print(f"Mean Best-Match Dice: {np.mean(best_dice_scores):.4f}")

    # --- METRIC 2: Hausdorff Distance (Structural Fidelity) ---
    # Note: We limit to 200 samples to prevent the script from freezing for hours
    hausdorff_scores = compute_hausdorff_95(syn_masks, real_masks, max_samples=200)
    print(f"Mean Hausdorff-95: {np.mean(hausdorff_scores):.4f}")

    # --- METRIC 3: Diversity Score ---
    div_real = compute_diversity(real_masks)
    div_syn = compute_diversity(syn_masks)
    print(f"Diversity (Real): {div_real:.4f} | Diversity (Syn): {div_syn:.4f}")
    
    # 3. Compute Distribution Distances (Wasserstein)
    print("\n🔹 Metric 1: Distribution Similarity (Lower is Better)")
    structures = ["LV", "Myo", "RV"]
    metrics = ["Area", "Roundness"]
    
    dist_results = []
    
    print(f"{'Structure':<10} {'Metric':<10} {'Wasserstein Dist':<20} {'Real Mean':<10} {'Syn Mean':<10}")
    print("-" * 65)
    
    for struct in structures:
        for met in metrics:
            # Filter valid data
            r_vals = df_real[df_real["Structure"] == struct][met].values
            s_vals = df_syn[df_syn["Structure"] == struct][met].values
            
            # Earth Mover's Distance
            if len(r_vals) > 0 and len(s_vals) > 0:
                wd = wasserstein_distance(r_vals, s_vals)
                print(f"{struct:<10} {met:<10} {wd:<20.4f} {r_vals.mean():<10.1f} {s_vals.mean():<10.1f}")
                dist_results.append({"Structure": struct, "Metric": met, "WD": wd})

    # 4. Compute Anatomical Validity
    print("\n🔹 Metric 2: Anatomical Validity (Single Connected Component)")
    print(f"{'Structure':<10} {'Real Broken %':<15} {'Syn Broken %':<15}")
    print("-" * 45)
    
    for struct in structures:
        # Check if "Components" column exists and has valid data
        r_sub = df_real[df_real["Structure"] == struct]
        s_sub = df_syn[df_syn["Structure"] == struct]
        
        r_broken = np.mean(r_sub["Components"] != 1) * 100 if len(r_sub) > 0 else 0
        s_broken = np.mean(s_sub["Components"] != 1) * 100 if len(s_sub) > 0 else 0
        
        print(f"{struct:<10} {r_broken:<15.2f} {s_broken:<15.2f}")

    # 5. Visualization
    print("\n🔹 Generating Comprehensive Plots...")
    
    # Create 3x3 Grid
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # ROW 1 & 2: Shape Distributions
    for i, struct in enumerate(structures):
        # Row 1: Area
        sns.kdeplot(data=df_all[df_all["Structure"]==struct], x="Area", hue="Group", fill=True, ax=axes[0, i], common_norm=False)
        axes[0, i].set_title(f"{struct} Area Distribution")
        
        # Row 2: Roundness
        sns.kdeplot(data=df_all[df_all["Structure"]==struct], x="Roundness", hue="Group", fill=True, ax=axes[1, i], common_norm=False)
        axes[1, i].set_title(f"{struct} Roundness Distribution")
        axes[1, i].set_xlim(0, 1.0)
    
    # --- ROW 3: New Fidelity Metrics ---
    
    # Plot 3,0: Best-Match Dice Histogram
    axes[2, 0].hist(best_dice_scores, bins=20, range=(0, 1), color='purple', alpha=0.7, edgecolor='black')
    axes[2, 0].set_title("Best-Match Dice (Pixel Fidelity)")
    axes[2, 0].set_xlabel("Dice Score (Higher is Better)")
    axes[2, 0].set_ylabel("Count")
    axes[2, 0].axvline(np.mean(best_dice_scores), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(best_dice_scores):.2f}')
    axes[2, 0].legend()

    # Plot 3,1: Hausdorff Histogram
    axes[2, 1].hist(hausdorff_scores, bins=20, color='orange', alpha=0.7, edgecolor='black')
    axes[2, 1].set_title("Hausdorff-95 Distance (Edge Fidelity)")
    axes[2, 1].set_xlabel("Pixel Distance (Lower is Better)")
    if len(hausdorff_scores) > 0:
        axes[2, 1].axvline(np.mean(hausdorff_scores), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(hausdorff_scores):.2f}')
        axes[2, 1].legend()

    # Plot 3,2: Diversity Comparison
    div_labels = ['Real Data', 'Synthetic Data']
    div_values = [div_real, div_syn]
    axes[2, 2].bar(div_labels, div_values, color=['skyblue', 'salmon'], edgecolor='black')
    axes[2, 2].set_title("Dataset Diversity (Pairwise Dice)")
    axes[2, 2].set_ylabel("Avg Pairwise Score (Lower = More Diverse)")
    axes[2, 2].set_ylim(0, 1.0)
    
    for j, v in enumerate(div_values):
        axes[2, 2].text(j, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "Mask_Fidelity_Full_Report.png")
    plt.savefig(plot_path)
    print(f"✅ Full Report saved to: {plot_path}")
    
    # Save CSV
    df_all.to_csv(os.path.join(OUTPUT_DIR, "mask_shape_metrics.csv"), index=False)

if __name__ == "__main__":
    run_fidelity_eval()