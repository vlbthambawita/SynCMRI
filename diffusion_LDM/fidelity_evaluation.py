import torch
import torch.nn.functional as F
import numpy as np
import os
import shutil
import sys
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Metrics
from torch_fidelity import calculate_metrics
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Custom
from config import Config
from eval import load_models, generate_samples

# Dataloader
sys.path.append("/scratch1/e20-fyp-syn-car-mri-gen/segmentation-v2/src/data")
from mandm_dataloader import build_mandm_loaders, MandMatasetConfig, LoaderConfig

def main():
    # --- SETUP ---
    cfg = Config()
    device = cfg.device
    
    # Output Folders
    output_root = "fidelity_results_ldm"
    temp_root = os.path.join(output_root, "temp")
    real_dir = os.path.join(temp_root, "real")
    synth_dir = os.path.join(temp_root, "synth")
    
    if os.path.exists(temp_root): shutil.rmtree(temp_root)
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(synth_dir, exist_ok=True)
    
    # Redirect Cache for Fidelity
    os.environ['TORCH_HOME'] = output_root
    os.environ['XDG_CACHE_HOME'] = output_root

    # --- LOAD MODELS ---
    ldm, vqvae, scheduler = load_models(cfg, device)

    # --- LOAD DATA ---
    print("⏳ Loading Dataset...")
    # Use ACDC or MandM here
    dc = MandMatasetConfig(cache_root=cfg.train_params['cache_root_mandm'])
    lc = LoaderConfig(batch_size=1, shuffle_train=False) # Batch 1 for safe metric calc
    # Note: Use test_loader or train_loader depending on what you want to eval
    _, _, loader, _ = build_mandm_loaders(dc.cache_root, dc=dc, lc=lc, is_test_set_only=True)
    
    # Limit samples
    MAX_SAMPLES = 1000
    print(f"✅ Evaluating on {min(len(loader), MAX_SAMPLES)} samples.")

    # --- METRICS INIT ---
    ssim_metric = SSIM(data_range=1.0).to(device)
    ms_ssim_metric = MS_SSIM(data_range=1.0, betas=(0.0448, 0.2856, 0.3001, 0.2363)).to(device)
    psnr_metric = PSNR(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    # --- GENERATION LOOP ---
    for i, batch in enumerate(tqdm(loader, desc="Evaluating")):
        if i >= MAX_SAMPLES: break

        real_img = batch['image'].to(device)    # [1, 1, 128, 128]
        real_mask_idx = batch['mask'].to(device)# [1, 1, 128, 128]

        # 1. Prepare Mask (One-Hot)
        masks_flat = real_mask_idx.squeeze(1)
        masks_onehot = F.one_hot(masks_flat, num_classes=cfg.num_classes).permute(0, 3, 1, 2).float()

        # 2. Generate
        syn_img = generate_samples(ldm, vqvae, scheduler, masks_onehot, cfg, device)

        # 3. Normalize Real [-1,1] -> [0,1]
        real_img_norm = (real_img + 1) / 2
        real_img_norm = torch.clamp(real_img_norm, 0, 1)
        
        # Synth is usually decoded to [-1, 1] or similar by VQVAE, ensure [0, 1]
        syn_img_norm = (syn_img + 1) / 2
        syn_img_norm = torch.clamp(syn_img_norm, 0, 1)

        # 4. Pixel Metrics
        ssim_metric.update(syn_img_norm, real_img_norm)
        ms_ssim_metric.update(syn_img_norm, real_img_norm)
        psnr_metric.update(syn_img_norm, real_img_norm)
        
        # LPIPS needs 3 channels
        lpips_metric.update(syn_img_norm.repeat(1,3,1,1), real_img_norm.repeat(1,3,1,1))

        # 5. Save for FID
        vutils.save_image(real_img_norm, os.path.join(real_dir, f"{i}.png"))
        vutils.save_image(syn_img_norm, os.path.join(synth_dir, f"{i}.png"))

    # --- FINAL COMPUTATION ---
    print("\nComputing Final Metrics...")
    
    # Distribution Metrics
    kid_subset = min(500, i)
    metrics_dict = calculate_metrics(
        input1=real_dir, input2=synth_dir, 
        cuda=True, fid=True, kid=True, verbose=False,
        kid_subset_size=kid_subset
    )
    
    results = {
        "PSNR": psnr_metric.compute().item(),
        "SSIM": ssim_metric.compute().item(),
        "MS-SSIM": ms_ssim_metric.compute().item(),
        "LPIPS": lpips_metric.compute().item(),
        "FID": metrics_dict['frechet_inception_distance'],
        "KID_Mean": metrics_dict['kernel_inception_distance_mean'],
        "KID_Std": metrics_dict['kernel_inception_distance_std']
    }

    print("-" * 30)
    for k, v in results.items():
        print(f"{k}: {v:.4f}")
    print("-" * 30)

    # Cleanup
    # shutil.rmtree(temp_root)

if __name__ == "__main__":
    main()