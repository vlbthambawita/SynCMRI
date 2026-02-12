import torch

class Config:
    def __init__(self):
        # 1. Dataset & Paths
        self.im_size = 128
        self.num_classes = 4  # (BG, LV, Myo, RV)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Paths (Update these to your actual weights)
        self.train_params = {
            'ldm_ckpt_path': 'ldm_cardiac_cond128_150_10.pth',
            'vqvae_ckpt_path': 'vqvae_cardiac_autoencoder128_150_10.pth',
            'cache_root_mandm': "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache",
        }

        # 2. Diffusion Parameters
        self.diffusion_params = {
            'num_timesteps': 1000,
            'beta_start': 0.00085,
            'beta_end': 0.012
        }

        # 3. LDM Architecture (U-Net)
        self.ldm_params = {
            'ldm_batch_size': 16,
            'ldm_lr': 5e-5,
            'num_timesteps': 1000,
            'down_channels': [128, 256, 512],
            'mid_channels': [512, 256],
            'down_sample': [True, True],
            'attn_down': [False, True],
            'time_emb_dim': 256,
            'norm_channels': 32,
            'num_heads': 8,
            'conv_out_channels': 128,
            'num_down_layers': 2,
            'num_mid_layers': 2,
            'num_up_layers': 2,
            'condition_config': {
                'condition_types': ['image'],
                'image_condition_config': {
                    'image_condition_input_channels': 4, # Mask Channels
                    'image_condition_output_channels': 1,
                    'image_condition_h': 128,
                    'image_condition_w': 128,
                    'cond_drop_prob': 0.0
                }
            }
        }

        # 4. Autoencoder Architecture (VQ-VAE)
        self.autoencoder_params = {
            'z_channels': 4,
            'codebook_size': 8192,
            'down_channels': [64, 128, 256],
            'mid_channels': [256, 256],
            'down_sample': [True, True, True], # 3 downsamples -> 128/8 = 16x16 latent
            'attn_down': [False, False, False],
            'norm_channels': 32,
            'num_heads': 4,
            'num_down_layers': 2,
            'num_mid_layers': 2,
            'num_up_layers': 2
        }