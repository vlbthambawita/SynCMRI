import torch
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import sys

# Import custom modules
from config import Config
from eval import load_models, generate_samples

# Import Dataloader (ACDC or MandM)
sys.path.append("/scratch1/e20-fyp-syn-car-mri-gen/segmentation-v2/src/data")
from mandm_dataloader import build_mandm_loaders, MandMDatasetConfig, LoaderConfig

def run_test():
    # 1. Setup
    cfg = Config()
    device = cfg.device
    os.makedirs("test_results_ldm", exist_ok=True)

    # 2. Load Models
    ldm, vqvae, scheduler = load_models(cfg, device)

    # 3. Load Data
    print("⏳ Loading Test Data...")
    
    dc = MandMDatasetConfig(cache_root=cfg.train_params['cache_root_mandm'])
    lc = LoaderConfig(batch_size=8, shuffle_train=True) # Shuffle to get random samples
    _, _, test_loader, _ = build_mandm_loaders(dc.cache_root, dc=dc, lc=lc, is_test_set_only=True)
    
    batch = next(iter(test_loader))
    real_imgs = batch['image'].to(device)
    real_masks_indices = batch['mask'].to(device) # [B, 1, H, W]

    # 4. Convert Mask to One-Hot for LDM
    # [B, 1, H, W] -> [B, 4, H, W]
    masks_flat = real_masks_indices.squeeze(1)
    masks_onehot = F.one_hot(masks_flat, num_classes=cfg.num_classes).permute(0, 3, 1, 2).float()

    # 5. Generate
    print("🚀 Generating samples...")
    syn_imgs = generate_samples(ldm, vqvae, scheduler, masks_onehot, cfg, device)

    # 6. Save Visual Comparison
    # Normalize for saving (-1,1 -> 0,1)
    real_viz = (real_imgs + 1) / 2
    syn_viz = (syn_imgs + 1) / 2
    
    # Visualize Mask (collapse channels)
    mask_viz = masks_flat.unsqueeze(1).float() / (cfg.num_classes - 1)

    # Stack: Mask | Real | Synthetic
    grid = torch.cat([mask_viz, real_viz, syn_viz], dim=0)
    save_path = "test_results_ldm.png"
    save_image(grid, save_path, nrow=8)
    
    print(f"✅ Test saved to {save_path}")

if __name__ == "__main__":
    run_test()