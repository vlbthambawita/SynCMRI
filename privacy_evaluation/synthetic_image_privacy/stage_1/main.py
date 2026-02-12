import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import lpips
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import glob
from PIL import Image
from torchvision import transforms


# CONFIGURATION
class PrivacyConfig:
    def __init__(self):
        # Path containing 'real' and 'synth' subfolders containing PNG images
        self.base_path = "/scratch1/e20-fyp-syn-car-mri-gen/diffusion/ldm/outputs/mnm_cardiac_generation/inference_10"
        
        # Where to save the plot
        self.output_plot_dir = "/scratch1/e20-fyp-syn-car-mri-gen/privacy-evaluation/outputs"
        self.plot_filename = "Diffusion_Privacy.png"
        
        # Hardware
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Batch size for processing (adjust based on GPU memory)
        self.batch_size = 32

# SETUP & DATA LOADING
def setup_models(device):
    """Initializes the LPIPS perceptual loss model."""
    print(f"Device: {device}")
    #
    loss_fn = lpips.LPIPS(net='vgg').to(device).eval()
    return loss_fn

def load_images_from_folder(folder_path, device):
    """
    Loads all PNG images from the specified folder into a tensor.
    Returns: Tensor [N, 1, H, W] normalized to [0, 1]
    """
    print(f"Loading images from: {folder_path}")
    
    search_path = os.path.join(folder_path, "*.png")
    image_files = sorted(glob.glob(search_path))
    
    if len(image_files) == 0:
        # Fallback: check for jpg/jpeg
        search_path = os.path.join(folder_path, "*.jpg")
        image_files = sorted(glob.glob(search_path))
        if len(image_files) == 0:
            raise ValueError(f"No images found in {folder_path}!")

    print(f" -> Found {len(image_files)} images.")
    
    image_list = []
    to_tensor = transforms.ToTensor() 
    
    for img_file in tqdm(image_files, desc="Reading Images"):
        try:
            with Image.open(img_file) as img:
                img = img.convert('L') # Ensure grayscale for MRI
                tensor_img = to_tensor(img) # [0, 1]
                image_list.append(tensor_img)
        except Exception as e:
            print(f"Skipping {img_file}: {e}")
            
    if not image_list:
        raise ValueError("No valid images loaded!")

    # Stack into a single tensor
    return torch.stack(image_list).to(device)


# 3. METRICS CALCULATION
def calculate_privacy_metrics(synthetic_set, real_set, lpips_model, device, batch_size=32):
    """
    Compares every synthetic image against the ENTIRE real database.
    Calculates L2 Distance, NNDR, and LPIPS.
    """
    #
    print("\n--- Starting Privacy Analysis ---")
    
    n_real = real_set.shape[0]
    n_syn = synthetic_set.shape[0]
    print(f"  -> Comparing {n_syn} Synthetic vs {n_real} Real images...")

    # Flatten Real Data for efficient L2 search [N_real, Pixels]
    real_flat = real_set.view(n_real, -1)
    
    min_l2_dists = []
    min_lpips_dists = []
    nndr_values = []
    
    # Process synthetic images in batches
    synthetic_loader = DataLoader(TensorDataset(synthetic_set), batch_size=batch_size)
    
    with torch.no_grad():
        for (syn_batch,) in tqdm(synthetic_loader, desc="Scanning Database"):
            syn_batch = syn_batch.to(device) 
            
            # --- METRIC 1: L2 Distance ---
            syn_flat = syn_batch.view(syn_batch.shape[0], -1)
            
            # Distance Matrix [Batch, N_real]
            dists_l2 = torch.cdist(syn_flat, real_flat, p=2)
            
            # Find Top 2 Nearest Neighbors for NNDR
            # values: [Batch, 2], indices: [Batch, 2]
            top2_vals, top2_idxs = torch.topk(dists_l2, k=2, dim=1, largest=False)
            
            # Store Min L2 (Nearest Neighbor)
            min_l2_dists.extend(top2_vals[:, 0].cpu().numpy())
            
            # --- METRIC 3: NNDR ---
            # Ratio = Dist_1st_NN / Dist_2nd_NN
            ratios = top2_vals[:, 0] / (top2_vals[:, 1] + 1e-8)
            nndr_values.extend(ratios.cpu().numpy())
            
            # --- METRIC 2: LPIPS ---
            # Calculate LPIPS only against the closest L2 match to save compute
            closest_real_imgs = real_set[top2_idxs[:, 0]]
            
            # LPIPS expects 3 channels [-1, 1]
            syn_3c = syn_batch.repeat(1, 3, 1, 1) * 2 - 1
            real_3c = closest_real_imgs.repeat(1, 3, 1, 1) * 2 - 1
            
            lpips_val = lpips_model(syn_3c, real_3c)
            min_lpips_dists.extend(lpips_val.squeeze().cpu().numpy())

    return np.array(min_l2_dists), np.array(min_lpips_dists), np.array(nndr_values)


# 4. MAIN EXECUTION & VISUALIZATION
def main():
    cfg = PrivacyConfig()
    
    # 1. Setup
    lpips_model = setup_models(cfg.device)
    os.makedirs(cfg.output_plot_dir, exist_ok=True)
    
    # 2. Load Data
    real_path = os.path.join(cfg.base_path, "real")
    syn_path = os.path.join(cfg.base_path, "synth")
    
    if not os.path.exists(real_path) or not os.path.exists(syn_path):
        print(f"❌ Error: Could not find 'real' or 'synth' folders inside {cfg.base_path}")
        return

    real_data = load_images_from_folder(real_path, cfg.device)
    syn_data = load_images_from_folder(syn_path, cfg.device)
    
    # 3. Run Analysis
    l2_dists, lpips_dists, nndrs = calculate_privacy_metrics(
        syn_data, real_data, lpips_model, cfg.device, cfg.batch_size
    )
    
    # 4. Report Findings
    print("\n--- PRIVACY EVALUATION RESULTS ---")
    
    # Thresholds
    n_exact_copies = np.sum(lpips_dists < 0.05)
    print(f"WARNING: Found {n_exact_copies} potential exact copies (LPIPS < 0.05)")
    
    n_memorized = np.sum(nndrs < 0.6)
    print(f"WARNING: Found {n_memorized} potential memorized outliers (NNDR < 0.6)")
    
    # 5. Plot Results
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    
    # Plot L2
    axes[0].hist(l2_dists, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title("Dist. to Nearest Neighbor (L2)")
    axes[0].set_xlabel(f"Euclidean Distance\n(Mean: {np.mean(l2_dists):.2f})")
    
    # Plot LPIPS
    axes[1].hist(lpips_dists, bins=30, color='salmon', edgecolor='black')
    axes[1].set_title("Dist. to Nearest Neighbor (LPIPS)")
    axes[1].axvline(0.05, color='red', linestyle='--', label='Copy Threshold')
    axes[1].legend()
    result_text_lpips = f"WARNING: Found {n_exact_copies} potential\nexact copies (LPIPS < 0.05)"
    axes[1].set_xlabel(f"Perceptual Distance (Lower = More Similar)\n\n{result_text_lpips}", 
                       fontsize=10, color='darkred' if n_exact_copies > 0 else 'black')
    
    # Plot NNDR
    axes[2].hist(nndrs, bins=30, color='lightgreen', edgecolor='black')
    axes[2].set_title("NNDR Distribution")
    axes[2].axvline(0.6, color='red', linestyle='--', label='Memorization Risk')
    axes[2].legend()
    result_text_nndr = f"WARNING: Found {n_memorized} potential\nmemorized outliers (NNDR < 0.6)"
    axes[2].set_xlabel(f"Ratio (1st NN / 2nd NN)\n\n{result_text_nndr}", 
                       fontsize=10, color='darkred' if n_memorized > 0 else 'black')
    
    save_path = os.path.join(cfg.output_plot_dir, cfg.plot_filename)
    plt.savefig(save_path)
    print(f"\nPrivacy Plot saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    main()