import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm
from scipy.stats import ks_2samp
from skimage.measure import label, regionprops

# MONAI IMPORTS (Matches your training)
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

# DATA LOADING
from data_loading import get_loaders

#CONFIGURATION
MODEL_PATH = "mask_diffusion_epoch_350.pth"  # Ensure this matches your saved file
CACHE_DIR = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache"
OUTPUT_DIR = "evaluation_results"

# Must match training settings
IMG_SIZE = (128, 128)
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 1000  # How many masks to evaluate (Real vs Gen)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# METRIC FUNCTIONS
def get_class_percentage(mask, class_id):
    total_pixels = mask.size
    class_pixels = np.sum(mask == class_id)
    return (class_pixels / total_pixels) * 100.0

def get_shape_features(mask, class_id):
    # Ensure mask is integer
    mask = mask.astype(int)
    binary_mask = (mask == class_id).astype(int)
    
    if np.sum(binary_mask) == 0: 
        return None

    lbl = label(binary_mask)
    regions = regionprops(lbl)
    
    if not regions: 
        return None

    # Get largest region to ignore noise
    r = max(regions, key=lambda x: x.area)
    
    perimeter = r.perimeter if r.perimeter > 0 else 1
    roundness = (4 * math.pi * r.area) / (perimeter ** 2)
    
    return {
        "Area": r.area,
        "Roundness": roundness,
        "Eccentricity": r.eccentricity,
        "Solidity": r.solidity
    }


# DATA GENERATION & COLLECTION
def collect_metrics(model, val_loader, device, num_samples=100):
    print(f"\nComparing {num_samples} Real vs {num_samples} Generated Masks...")
    
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    records = []
    classes = {1: "LV", 2: "Myo", 3: "RV"}

    # ANALYZE REAL MASKS
    print("🔹 Analyzing Real Masks...")
    count = 0
    
    # We iterate through the loader until we have enough samples
    for batch_masks in val_loader:
        if count >= num_samples: break
        
        # Loader returns (Batch, 1, 128, 128)
        masks_np = batch_masks.squeeze(1).numpy() # Shape: (Batch, 128, 128)
        
        for mask in masks_np:
            if count >= num_samples: break
            
            for cid, cname in classes.items():
                feats = get_shape_features(mask, cid)
                if feats:
                    pct = get_class_percentage(mask, cid)
                    records.append({"Group": "Real", "Structure": cname, "Pixel_Pct": pct, **feats})
            count += 1

    # GENERATE & ANALYZE SYNTHETIC MASKS
    print("🔹 Generating Synthetic Masks...")
    model.eval()
    gen_count = 0
    batch_size = 8 # Generate in batches for speed
    
    with torch.no_grad():
        pbar = tqdm(total=num_samples)
        while gen_count < num_samples:
            # Start with Gaussian Noise
            noise = torch.randn((batch_size, NUM_CLASSES, *IMG_SIZE)).to(device)
            current_img = noise

            # Denoising Loop
            for t in scheduler.timesteps:
                model_output = model(x=current_img, timesteps=torch.Tensor((t,)).to(device), context=None)
                current_img, _ = scheduler.step(model_output, t, current_img)

            # Post-Processing
            # Scale back from [-1, 1] to [0, 1]
            current_img = (current_img + 1) / 2
            # Argmax to get integers (0, 1, 2, 3)
            gen_masks = torch.argmax(current_img, dim=1).cpu().numpy()

            # Extract Metrics
            for mask in gen_masks:
                if gen_count >= num_samples: break
                
                for cid, cname in classes.items():
                    feats = get_shape_features(mask, cid)
                    if feats:
                        pct = get_class_percentage(mask, cid)
                        records.append({"Group": "Generated", "Structure": cname, "Pixel_Pct": pct, **feats})
                
                gen_count += 1
                pbar.update(1)
        pbar.close()

    return pd.DataFrame(records)

def plot_kde_distributions(df, output_dir):
    print("Generating KDE Density Plots...")
    structures = ["LV", "RV", "Myo"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # colors: Blue for Real, Orange for Generated
    palette = {"Real": "tab:blue", "Generated": "tab:orange"} 
    
    for i, struct in enumerate(structures):
        ax = axes[i]
        subset = df[df["Structure"] == struct]
        
        # Draw Density Plot
        sns.kdeplot(
            data=subset, 
            x="Pixel_Pct", 
            hue="Group", 
            fill=True, 
            common_norm=False, 
            palette=palette, 
            alpha=0.3, 
            linewidth=2, 
            ax=ax
        )
        
        ax.set_title(f"{struct} Density Distribution")
        ax.set_xlabel("Pixel Percentage (%)")
        if i != 0: ax.set_ylabel("")
            
    plt.tight_layout()
    save_path = os.path.join(output_dir, "kde_distributions.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved KDE Plots: {save_path}")

# MAIN EXECUTION
def main():
    # Load Data
    _, val_loader = get_loaders(CACHE_DIR, batch_size=8)

    # Load Model
    print("Loading Model...")
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=NUM_CLASSES,
        out_channels=NUM_CLASSES, 
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        num_head_channels=32,
    ).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Ensure MODEL_PATH is correct and the architecture matches train.py exactly.")
        return

    # Run Evaluation
    df_results = collect_metrics(model, val_loader, DEVICE, num_samples=NUM_SAMPLES)
    
    # Save Raw Data
    csv_path = os.path.join(OUTPUT_DIR, "raw_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Metrics saved to {csv_path}")

    # Statistical Summary (KS Test)
    stats_list = []
    print("\n" + "="*80)
    print(f"{'Structure':<10} {'Feature':<15} {'Real Mean':<10} {'Gen Mean':<10} {'p-Value':<10} {'Sig'}")
    print("="*80)

    features = ["Pixel_Pct", "Area", "Roundness", "Eccentricity", "Solidity"]
    
    for struct in ["LV", "RV", "Myo"]:
        for feat in features:
            real_data = df_results[(df_results["Group"]=="Real") & (df_results["Structure"]==struct)][feat]
            gen_data = df_results[(df_results["Group"]=="Generated") & (df_results["Structure"]==struct)][feat]
            
            if len(real_data) > 0 and len(gen_data) > 0:
                _, p = ks_2samp(real_data, gen_data)
                sig = "*" if p < 0.05 else "ns" # p < 0.05 implies distributions are significantly different
                
                print(f"{struct:<10} {feat:<15} {real_data.mean():<10.2f} {gen_data.mean():<10.2f} {p:<10.3f} {sig}")
                
                stats_list.append({
                    "Structure": struct, "Feature": feat, 
                    "Real_Mean": real_data.mean(), "Gen_Mean": gen_data.mean(), 
                    "p_Value": p, "Significance": sig
                })

    pd.DataFrame(stats_list).to_csv(os.path.join(OUTPUT_DIR, "stats_summary.csv"), index=False)

    # Generate Violin Plots
    print("\nGenerating Plots...")
    for feat in features:
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df_results, x="Structure", y=feat, hue="Group", split=True, inner="quartile", palette="muted")
        plt.title(f"Comparison: {feat}")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, f"violin_{feat}.png"))
        plt.close()
    
    plot_kde_distributions(df_results, OUTPUT_DIR)

    print(f"Evaluation Complete! Check output folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()