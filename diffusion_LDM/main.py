import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import os
import sys

# Import Custom Modules
from config import Config
from diffusion import Unet, VQVAE, LinearNoiseScheduler

# Import Data Loaders
sys.path.append("/scratch1/e20-fyp-syn-car-mri-gen/segmentation-v2/src/data")
from mandm_dataloader import build_mandm_loaders, MandMDatasetConfig, LoaderConfig

def train():
    # 1. Setup
    cfg = Config()
    device = cfg.device
    
    # Create Output Directory
    run_name = "ldm_run_v1"
    output_dir = os.path.join("training_outputs", run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Training Output Directory: {output_dir}")

    # 2. Load Data
    print("Building Data Loaders...")
    # Change to ACDC if needed
    dc = MandMDatasetConfig(cache_root=cfg.train_params['cache_root_mandm']) 
    lc = LoaderConfig(batch_size=cfg.ldm_params['ldm_batch_size'], num_workers=4, shuffle_train=True)
    train_loader, val_loader, _, _ = build_mandm_loaders(dc.cache_root, dc=dc, lc=lc)
    print(f"✅ Data Loaded. Train Batches: {len(train_loader)}")

    # 3. Load Models
    print("⏳ Initializing Models...")
    
    # A. VQ-VAE (Frozen - Encoder)
    vqvae = VQVAE(im_channels=1, model_config=cfg.autoencoder_params).to(device)
    if os.path.exists(cfg.train_params['vqvae_ckpt_path']):
        vqvae.load_state_dict(torch.load(cfg.train_params['vqvae_ckpt_path'], map_location=device))
        print("✅ VQ-VAE Weights Loaded (Frozen).")
    else:
        raise FileNotFoundError(f"❌ VQ-VAE Checkpoint not found at {cfg.train_params['vqvae_ckpt_path']}")
    
    # Freeze VQ-VAE
    for param in vqvae.parameters():
        param.requires_grad = False
    vqvae.eval()

    # B. LDM (U-Net - Trainable)
    ldm = Unet(im_channels=cfg.autoencoder_params['z_channels'], model_config=cfg.ldm_params).to(device)
    ldm.train()

    # C. Scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=cfg.diffusion_params['num_timesteps'],
        beta_start=cfg.diffusion_params['beta_start'],
        beta_end=cfg.diffusion_params['beta_end']
    )

    # 4. Optimizer
    optimizer = AdamW(ldm.parameters(), lr=cfg.ldm_params['ldm_lr'])
    criterion = nn.MSELoss()

    # 5. Training Loop
    print("🔥 Starting Training...")
    NUM_EPOCHS = 100 # Adjust as needed
    
    for epoch in range(NUM_EPOCHS):
        ldm.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for batch in pbar:
            # Prepare Inputs
            images = batch['image'].to(device) # [B, 1, 128, 128]
            masks_idx = batch['mask'].to(device) # [B, 1, 128, 128]

            # 1. Encode Images to Latent Space (using VQ-VAE)
            with torch.no_grad():
                # vqvae.encode returns (z, quant_losses)
                # z shape: [B, 4, 16, 16] (if downsample=3)
                latents, _ = vqvae.encode(images)
                
                # IMPORTANT: Detach latents so we don't backprop into VQVAE
                latents = latents.detach()

            # 2. Prepare Condition (Masks)
            # Convert Indices to One-Hot: [B, 4, 128, 128]
            masks_flat = masks_idx.squeeze(1)
            masks_onehot = F.one_hot(masks_flat, num_classes=cfg.num_classes).permute(0, 3, 1, 2).float()
            
            # 3. Add Noise (Forward Diffusion)
            t = torch.randint(0, cfg.diffusion_params['num_timesteps'], (images.shape[0],), device=device).long()
            noise = torch.randn_like(latents).to(device)
            noisy_latents = scheduler.add_noise(latents, noise, t)

            # 4. Predict Noise (U-Net)
            # We pass the mask as 'cond_input'
            cond_input = {'image': masks_onehot}
            noise_pred = ldm(noisy_latents, t, cond_input=cond_input)

            # 5. Optimization
            loss = criterion(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(MSE=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")

        # 6. Save Checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(output_dir, f"ldm_epoch_{epoch+1}.pth")
            torch.save(ldm.state_dict(), save_path)
            print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train()