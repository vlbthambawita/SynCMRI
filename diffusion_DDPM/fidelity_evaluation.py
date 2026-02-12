import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision.utils as vutils
from diffusers import UNet2DModel, DDPMScheduler

# Metrics
from torch_fidelity import calculate_metrics
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Custom Imports
from eval import SegGuidedDDIMPipeline, SegGuidedDDPMPipeline
from data_loader import get_test_loader, get_data_loaders

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
# Redirect Cache (Important for torch_fidelity)
new_cache_dir = "/storage/scratch2/e20-syncar-mri-igen/.cache"
os.makedirs(new_cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = new_cache_dir
os.environ['XDG_CACHE_HOME'] = new_cache_dir

# Configuration
MODEL_DIR = "ddpm-150-finetuned"  # Your trained model folder
OUTPUT_ROOT = "/storage/scratch2/e20-syncar-mri-igen/diffusion/model_2/fidelity_results"
TEMP_ROOT = "/storage/scratch2/e20-syncar-mri-igen/diffusion/model_2/temporary"

# Clean old temp files
if os.path.exists(TEMP_ROOT):
    shutil.rmtree(TEMP_ROOT)
    
REAL_DIR = os.path.join(TEMP_ROOT, 'real')
SYNTH_DIR = os.path.join(TEMP_ROOT, 'synth')
MASK_DIR = os.path.join(TEMP_ROOT, 'mask')
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(SYNTH_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on {device}")

# ==========================================
# 2. LOAD MODEL & DATA
# ==========================================
print("Loading Model & Scheduler...")

try:
    # Load UNet
    unet = UNet2DModel.from_pretrained(os.path.join(MODEL_DIR, "unet")).to(device)
    
    # Load Scheduler & Convert to DDIM for faster evaluation (50 steps vs 1000)
    scheduler_config = DDPMScheduler.load_config(os.path.join(MODEL_DIR, "scheduler"))
    scheduler = DDPMScheduler.from_config(scheduler_config)
    
except OSError:
    print(f"Error: Could not find model in '{MODEL_DIR}'. Check path.")
    exit()

# Initialize Pipeline
# pipeline = SegGuidedDDIMPipeline(
#     unet=unet, 
#     scheduler=scheduler, 
#     eval_dataloader=None,     
#     external_config=None      
# )

#---- Use for DDPM-----
pipeline = SegGuidedDDPMPipeline(
    unet=unet, 
    scheduler=scheduler, 
    eval_dataloader=None,     
    external_config=None      
)

# Mock Config for Pipeline
class EvalConfig:
    segmentation_channel_mode = "multi"
    class_conditional = False
    use_cfg_for_eval_conditioning = False
    trans_noise_level = None

pipeline.external_config = EvalConfig()

# Load Data
print("⏳ Loading Test Data...")
# Use 'get_test_loader' for official test set, or 'get_data_loaders' for validation
loader = get_test_loader(batch_size=1) 

if loader is None:
    print("Test loader empty. Falling back to Validation loader.")
    _, loader = get_data_loaders(batch_size=1)

print(f"✅ Evaluating on {len(loader)} samples.")

# ==========================================
# 3. INITIALIZE METRICS
# ==========================================
# Note: Data range is 0.0 to 1.0
ssim_metric = SSIM(data_range=1.0, kernel_size=5).to(device)
ms_ssim_metric = MS_SSIM(data_range=1.0, kernel_size=5, betas=(0.0448, 0.2856, 0.3001, 0.2363)).to(device)
psnr_metric = PSNR(data_range=1.0).to(device)
lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

# Define your limit here
MAX_SAMPLES = 1000

# Get the 'jet' colormap object
cmap = plt.get_cmap('jet')

# Generate the 4 colors evenly spaced from 0 to 1
indices = np.linspace(0, 1, 4) 
colors_list = [cmap(i)[:3] for i in indices] # Extract RGB, drop Alpha

# Create the tensor palette
palette = torch.tensor(colors_list, device=device).float()
print("✅ 'Jet' Palette initialized.")

# ==========================================
# 4. GENERATION LOOP
# ==========================================
print("Starting Generation Loop...")

for i, batch in enumerate(tqdm(loader, desc="Sampling")):
    # Stop if we have generated enough
    if i >= MAX_SAMPLES:
        print(f"Reached limit of {MAX_SAMPLES} images.")
        break

    # A. Prepare Inputs
    real_img = batch['images'].to(device)           # [-1, 1]
    masks = batch['seg_onehot'].to(device)          # [B, 4, 128, 128]

    # B. Generate (Using Pipeline)
    with torch.no_grad():
        output = pipeline(
            batch_size=real_img.shape[0], # Batch size 1
            seg_batch={"seg_onehot": masks}, 
            num_inference_steps=1000,       # Fast DDPM/DDIM Sampling
            output_type="np"              # Returns [0, 1] numpy array
        )
    
    # C. Process Synthetic Image (Numpy -> Tensor GPU)
    # Pipeline returns [B, H, W, C] -> Permute to [B, C, H, W]
    syn_tensor = torch.tensor(output.images).permute(0, 3, 1, 2).to(device, dtype=torch.float32)

    # D. Process Real Image (Normalize [-1, 1] -> [0, 1])
    real_img_norm = (real_img + 1) / 2
    real_img_norm = torch.clamp(real_img_norm, 0, 1)

    # E. Update Pixel Metrics
    ssim_metric.update(syn_tensor, real_img_norm)
    ms_ssim_metric.update(syn_tensor, real_img_norm)
    psnr_metric.update(syn_tensor, real_img_norm)

    # LPIPS needs 3 channels (RGB). We repeat the grayscale channel.
    syn_rgb = syn_tensor.repeat(1, 3, 1, 1)
    real_rgb = real_img_norm.repeat(1, 3, 1, 1)
    lpips_metric.update(syn_rgb, real_rgb)

    # F. Save for FID (Torch-Fidelity reads from disk)
    # We save as PNG to ensure no compression artifacts affect FID too much
    vutils.save_image(real_img_norm, os.path.join(REAL_DIR, f'img_{i}.png'))
    vutils.save_image(syn_tensor, os.path.join(SYNTH_DIR, f'img_{i}.png'))

    # Convert One-Hot [B, 4, H, W] -> Indices [B, H, W]
    mask_indices = torch.argmax(masks, dim=1) 
    
    # Apply Palette: [B, H, W] -> [B, H, W, 3]
    mask_rgb = palette[mask_indices.long()]    
    
    # Permute for Saving: [B, H, W, 3] -> [B, 3, H, W]
    mask_rgb = mask_rgb.permute(0, 3, 1, 2) 
    
    # Save to Mask Directory
    vutils.save_image(mask_rgb, os.path.join(MASK_DIR, f'img_{i}.png'))

# ==========================================
# 5. COMPUTE FINAL SCORES
# ==========================================
print("\nComputing Final Metrics...")

# A. Pixel Metrics
final_ssim = ssim_metric.compute().item()
final_ms_ssim = ms_ssim_metric.compute().item()
final_psnr = psnr_metric.compute().item()
final_lpips = lpips_metric.compute().item()

print(f"✅ PSNR:    {final_psnr:.2f} dB")
print(f"✅ SSIM:    {final_ssim:.4f}")
print(f"✅ MS-SSIM: {final_ms_ssim:.4f}")
print(f"✅ LPIPS:   {final_lpips:.4f}")

# B. Distribution Metrics (FID / KID)
print("Calculating FID/KID...")

# If dataset is small (<500), we must reduce KID subset size
kid_subset_size = min(500, len(loader))
# KID requires at least 2 samples
if kid_subset_size < 2: kid_subset_size = 2 

metrics_dict = calculate_metrics(
    input1=REAL_DIR, 
    input2=SYNTH_DIR, 
    cuda=True, 
    fid=True, 
    kid=True, 
    verbose=False,
    kid_subset_size=kid_subset_size
)

final_fid = metrics_dict['frechet_inception_distance']
final_kid_mean = metrics_dict['kernel_inception_distance_mean']
final_kid_std = metrics_dict['kernel_inception_distance_std']

print(f"✅ FID:     {final_fid:.2f}")
print(f"✅ KID:     {final_kid_mean:.4f} ± {final_kid_std:.4f}")

# ==========================================
# 6. PLOT & SAVE RESULTS
# ==========================================
metrics_data = [
    {'name': 'SSIM', 'score': final_ssim, 'std': 0, 'range': [0, 1], 'dir': 'Higher is Better', 'color': 'blue'},
    {'name': 'MS-SSIM', 'score': final_ms_ssim, 'std': 0, 'range': [0, 1], 'dir': 'Higher is Better', 'color': 'cyan'},
    {'name': 'PSNR', 'score': final_psnr, 'std': 0, 'range': [0, 40], 'dir': 'Higher is Better', 'color': 'green'},
    {'name': 'FID', 'score': final_fid, 'std': 0, 'range': [0, max(100, final_fid*1.2)], 'dir': 'Lower is Better', 'color': 'red'},
    {'name': 'KID', 'score': final_kid_mean, 'std': final_kid_std, 'range': [0, max(0.1, final_kid_mean*2)], 'dir': 'Lower is Better', 'color': 'purple'},
    {'name': 'LPIPS', 'score': final_lpips, 'std': 0, 'range': [0, 1], 'dir': 'Lower is Better', 'color': 'magenta'}
]

fig, axes = plt.subplots(1, 6, figsize=(20, 7))
fig.suptitle(f'Fidelity Evaluation: {MODEL_DIR}', fontsize=16)

for i, m in enumerate(metrics_data):
    ax = axes[i]
    bar = ax.bar(m['name'], m['score'], color=m['color'], yerr=m['std'] if m['std']>0 else None, capsize=5)
    ax.set_title(f"{m['name']}\n({m['dir']})")
    ax.set_ylim(m['range'])
    
    # Label
    txt = f"{m['score']:.2f}" + (f"±{m['std']:.2f}" if m['std']>0 else "")
    ax.text(bar[0].get_x() + bar[0].get_width()/2, m['score'], txt, ha='center', va='bottom', fontweight='bold')

# Ensure output directory exists for plot
save_plot_path = os.path.join(OUTPUT_ROOT, "Fidelity_Evaluation.png")

plt.savefig(save_plot_path)
print(f"\nPlot saved to: {save_plot_path}")
