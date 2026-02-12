import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import sys

from diffusion import Unet, VQVAE, LinearNoiseScheduler
from config import Config

def load_models(config, device):
    """Loads VQVAE and LDM from checkpoints defined in config."""
    print("⏳ Loading Models...")
    
    # 1. Load VQ-VAE
    vqvae = VQVAE(im_channels=1, model_config=config.autoencoder_params).to(device)
    if os.path.exists(config.train_params['vqvae_ckpt_path']):
        vqvae.load_state_dict(torch.load(config.train_params['vqvae_ckpt_path'], map_location=device))
        print("✅ VQ-VAE Loaded.")
    else:
        print(f"❌ VQ-VAE checkpoint not found at {config.train_params['vqvae_ckpt_path']}")

    vqvae.eval()

    # 2. Load LDM
    ldm = Unet(im_channels=config.autoencoder_params['z_channels'], model_config=config.ldm_params).to(device)
    if os.path.exists(config.train_params['ldm_ckpt_path']):
        ldm.load_state_dict(torch.load(config.train_params['ldm_ckpt_path'], map_location=device))
        print("✅ LDM Loaded.")
    else:
         print(f"❌ LDM checkpoint not found at {config.train_params['ldm_ckpt_path']}")
    
    ldm.eval()

    # 3. Scheduler
    scheduler = LinearNoiseScheduler(
        num_timesteps=config.diffusion_params['num_timesteps'],
        beta_start=config.diffusion_params['beta_start'],
        beta_end=config.diffusion_params['beta_end']
    )

    return ldm, vqvae, scheduler

def generate_samples(ldm, vqvae, scheduler, masks, config, device):
    """
    Generates synthetic images from input masks.
    masks: Tensor [B, 4, H, W] (One-Hot Encoded)
    Returns: Tensor [B, 1, H, W] (Synthetic Images)
    """
    ldm.eval()
    vqvae.eval()
    
    batch_size = masks.shape[0]
    
    # 1. Calculate Latent Dim
    # Count how many 'True' in down_sample list [True, True, True] -> 3
    # 128 / 2^3 = 16
    num_downsamples = sum(config.autoencoder_params['down_sample'])
    latent_dim = config.im_size // (2 ** num_downsamples)
    z_channels = config.autoencoder_params['z_channels']

    # 2. Prepare Condition
    cond_input = {'image': masks.to(device)}

    # 3. Start from Noise
    z = torch.randn(batch_size, z_channels, latent_dim, latent_dim).to(device)

    # 4. Reverse Diffusion
    # print("Sampling...")
    for t in reversed(range(config.diffusion_params['num_timesteps'])):
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        with torch.no_grad():
            noise_pred = ldm(z, t_tensor, cond_input=cond_input)
            z, _ = scheduler.sample_prev_timestep(z, noise_pred, t_tensor)

    # 5. Decode to Pixel Space
    with torch.no_grad():
        generated_imgs = vqvae.decode(z)
    
    # Clamp to valid range if needed
    # generated_imgs = torch.clamp(generated_imgs, -1, 1)

    return generated_imgs