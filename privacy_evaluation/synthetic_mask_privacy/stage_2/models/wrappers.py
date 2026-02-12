import torch
import abc
import os
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from monai.networks.utils import one_hot

class GenerativeModelWrapper(abc.ABC):
    """
    Abstract Interface for Privacy Evaluation.
    Any model (Diffusion, Flow, GAN) must implement `get_training_pair`.
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # # VQVAE is NOT needed for DDPM, so we make it optional
        # if config.model_type == 'diffusion-ldm': 
        #     self.vae = self._load_vae()
        # else:
        #     self.vae = None

    @abc.abstractmethod
    def get_training_pair(self, images, condition):
        pass

class MaskDiffusionWrapper(GenerativeModelWrapper):
    """
    Wrapper for the MONAI Mask Diffusion Model.
    """
    def __init__(self, config, device):
        # We don't call super() init because we don't need VQVAE
        self.config = config
        self.device = device
        self.model = self._load_model()
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        
    def _load_model(self):
        print("Loading Mask Diffusion Model...")
        # Initialize MONAI UNet
        model = DiffusionModelUNet(**self.config.mask_model_params).to(self.device)
        
        # Load Weights
        ckpt_path = self.config.paths['mask_ckpt_path']
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
            print(f"✅ Mask Model loaded from {ckpt_path}")
        else:
            raise FileNotFoundError(f"❌ Checkpoint not found: {ckpt_path}")
            
        model.eval()
        return model

    def get_training_pair(self, images, condition):
        """
        For masks, the 'image' input IS the mask.
        We ignore the 'images' arg (which is the MRI) and use 'condition['image']'
        """
        with torch.no_grad():
            # 1. Get Mask Data
            # Loader returns mask in condition['image'] as [B, 4, 128, 128] (One-Hot)
            if 'image' in condition:
                masks = condition['image'].to(self.device)
            else:
                # Fallback if loader implementation varies
                masks = images.to(self.device)

            # Ensure Float and Range [-1, 1] (Standard for Diffusion)
            # Masks are 0 or 1. (x * 2) - 1 makes them -1 or 1.
            clean_input = masks.float() * 2 - 1
            
            # 2. Sample Timestep
            # We attack the "middle" steps where the model learns structure
            batch_size = clean_input.shape[0]
            t = torch.randint(
                100, 900,  # Focus on middle range (structure formation)
                (batch_size,), 
                device=self.device
            ).long()

            # 3. Add Noise
            noise = torch.randn_like(clean_input).to(self.device)
            noisy_input = self.scheduler.add_noise(
                original_samples=clean_input, 
                noise=noise, 
                timesteps=t
            )

            # 4. Predict Noise
            # Unconditional mask generation -> context=None
            noise_pred = self.model(x=noisy_input, timesteps=t, context=None)
            
            return noise_pred, noise