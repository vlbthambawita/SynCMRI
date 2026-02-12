import torch
import abc
import os
from .diffusion import Unet, LinearNoiseScheduler, VQVAE

# Import Diffusers logic
from diffusers import UNet2DModel, DDPMScheduler
# Import flow match
from .flow_match import build_flow_model

class GenerativeModelWrapper(abc.ABC):
    """
    Abstract Interface for Privacy Evaluation.
    Any model (Diffusion, Flow, GAN) must implement `get_training_pair`.
    """
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # VQVAE is NOT needed for DDPM, so we make it optional
        if config.model_type == 'diffusion-ldm': 
            self.vae = self._load_vae()
        else:
            self.vae = None

    def _load_vae(self):
        # Common VQVAE loading logic
        vae = VQVAE(im_channels=self.config.dataset_params['im_channels'],
                    model_config=self.config.autoencoder_params).to(self.device)
        
        ckpt_path = os.path.join(
            self.config.paths['ckpt_dir'],
            self.config.train_params['task_name'],
            self.config.train_params['vqvae_ckpt_name']
        )
        # Load weights
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            # Handle if checkpoint is wrapped in 'state_dict' key or is direct dictionary
            if 'state_dict' in checkpoint:
                vae.load_state_dict(checkpoint['state_dict'])
            else:
                vae.load_state_dict(checkpoint)
            print(f"✅ VQVAE loaded from {ckpt_path}")
        else:
            raise FileNotFoundError(f"❌ VQVAE checkpoint not found at: {ckpt_path}")
            
        vae.eval()
        return vae

    @abc.abstractmethod
    def get_training_pair(self, images, condition):
        pass

class DiffusionWrapper(GenerativeModelWrapper):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.model = self._load_diffusion_model()
        self.scheduler = LinearNoiseScheduler(
            config.diffusion_params['num_timesteps'],
            config.diffusion_params['beta_start'],
            config.diffusion_params['beta_end']
        )

    def _load_diffusion_model(self):
        print("Loading Diffusion Model...")
        model = Unet(im_channels=self.config.autoencoder_params['z_channels'],
                     model_config=self.config.ldm_params).to(self.device)
        
        # Construct Path
        ckpt_path = os.path.join(
            self.config.paths['ckpt_dir'],
            self.config.train_params['task_name'],
            self.config.train_params['ldm_ckpt_name']
        )

        # Load Weights
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            # Handle if checkpoint is wrapped in 'state_dict' or 'model_state_dict'
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"✅ Diffusion Model loaded from {ckpt_path}")
        else:
            raise FileNotFoundError(f"❌ Diffusion checkpoint not found at: {ckpt_path}")
        
        model.eval()
        return model

    def get_training_pair(self, images, condition):
        with torch.no_grad():
            # 1. Encode to Latent
            latent, _ = self.vae.encode(images)
            
            # 2. Sample Timestep
            batch_size = latent.shape[0]
            # Focusing attack on middle timesteps where structure is formed
            t_start = int(self.config.diffusion_params['num_timesteps'] * 0.3)
            t_end = int(self.config.diffusion_params['num_timesteps'] * 0.6)
            t = torch.randint(t_start, t_end, (batch_size,), device=self.device)

            # 3. Add Noise
            noise_true = torch.randn_like(latent).to(self.device)
            noisy_latent = self.scheduler.add_noise(latent, noise_true, t)

            # 4. Predict Noise
            noise_pred = self.model(noisy_latent, t, cond_input=condition)
            
            return noise_pred, noise_true

class DDPMWrapper(GenerativeModelWrapper):
    """
    Wrapper for the Pixel-Space DDPM trained using HuggingFace Diffusers.
    """
    def __init__(self, config, device):
        super().__init__(config, device)
        self.model = self._load_ddpm_model()
        self.scheduler = DDPMScheduler(num_train_timesteps=config.ddpm_params['num_train_timesteps'])

    def _load_ddpm_model(self):
        print("Loading DDPM Model from Diffusers...")
        model_path = self.config.paths['ddpm_path']
        
        try:
            # 1. Try loading as a Pipeline subdirectory (common in save_pretrained)
            # Usually saved in model_path/unet
            unet_path = os.path.join(model_path, "unet")
            if os.path.exists(unet_path):
                model = UNet2DModel.from_pretrained(unet_path).to(self.device)
            else:
                # 2. Try loading directly from the folder
                model = UNet2DModel.from_pretrained(model_path).to(self.device)
                
            print(f"✅ DDPM UNet loaded successfully from {model_path}")
            model.eval()
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load DDPM model from {model_path}. Error: {e}")

    def get_training_pair(self, images, condition):
        """
        DDPM Logic:
        1. Input: Real Image (1 ch) + Mask (4 ch)
        2. Process: Add Noise to Image -> Concatenate Mask -> Predict Noise
        """
        with torch.no_grad():
            # images: [B, 1, 128, 128]
            # condition['image']: [B, 4, 128, 128] (The segmentation mask)
            
            masks = condition['image']
            
            # 1. Sample Timestep
            batch_size = images.shape[0]
            # Focus attack on middle steps (structure formation)
            t = torch.randint(
                int(self.config.ddpm_params['num_train_timesteps']*0.3), 
                int(self.config.ddpm_params['num_train_timesteps']*0.6), 
                (batch_size,), 
                device=self.device
            ).long()

            # 2. Add Noise (Only to the image, not the mask)
            noise = torch.randn_like(images).to(self.device)
            noisy_images = self.scheduler.add_noise(images, noise, t)

            # 3. Concatenate (This matches your training logic: 1 channel image + 4 channel mask = 5 inputs)
            # [B, 1, 128, 128] cat [B, 4, 128, 128] -> [B, 5, 128, 128]
            model_input = torch.cat([noisy_images, masks], dim=1)

            # 4. Predict
            # Diffusers returns a tuple/object, we need .sample
            noise_pred = self.model(model_input, t).sample
            
            return noise_pred, noise

class FlowMatchingWrapper(GenerativeModelWrapper):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.model = self._load_flow_model()
        
    def _load_flow_model(self):
        print("Loading Flow Matching Model...")
        ckpt_path = self.config.paths['flow_match_path']
        
        # Build structure
        model = build_flow_model(self.config.flow_model_params, self.device)
        
        # Load weights
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            # Handle state dict wrapping
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load
            model.load_state_dict(state_dict)
            print(f"✅ Flow Model loaded from {ckpt_path}")
        else:
            raise FileNotFoundError(f"❌ Flow checkpoint not found at: {ckpt_path}")
        
        model.eval()
        return model

    def get_training_pair(self, images, condition):
        """
        Flow Matching Training Pair Logic
        """
        with torch.no_grad():
            x1 = images  # Real Data [B, 1, H, W]
            x0 = torch.randn_like(x1).to(self.device) # Noise
            
            batch_size = x1.shape[0]
            
            # 1. Sample t in [0, 1]
            t = torch.rand((batch_size,), device=self.device)
            
            # 2. Interpolate (Forward Process)
            t_expand = t.view(batch_size, 1, 1, 1)
            x_t = t_expand * x1 + (1 - t_expand) * x0
            
            # 3. Calculate Target Velocity
            target_velocity = x1 - x0
            
            # 4. Prepare Conditions
            masks = condition.get('image', None)
            
            # === FIX STARTS HERE ===
            # The ControlNet expects 1 channel, but masks are 4 channels (one-hot).
            # Convert [B, 4, H, W] -> [B, 1, H, W] using argmax
            if masks is not None and masks.shape[1] == 4:
                # Argmax converts one-hot back to integer class labels (0, 1, 2, 3)
                # Keepdim=True ensures we get [B, 1, H, W]
                masks = torch.argmax(masks, dim=1, keepdim=True).float()
            # === FIX ENDS HERE ===

            # 5. Predict Velocity
            pred_velocity = self.model(
                x=x_t, 
                t=t, 
                cond=None, 
                masks=masks
            )
            
            return pred_velocity, target_velocity