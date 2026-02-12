# config.py
class Config:
    def __init__(self):
        # Switch this to 'flow_matching' when using your friend's model
        self.model_type = 'flow_matching'  # Options: 'diffusion-ldm', 'diffusion-ddpm', 'flow_matching'
        
        self.dataset_params = {
            'im_channels': 1, 
            'im_size': 128, 
            'subset_size': 175
        }
        
        self.train_params = {
            'task_name': 'mnm_cardiac_generation',
            'ldm_ckpt_name': 'ldm_cardiac_cond128_150_12.pth',
            'vqvae_ckpt_name': 'vqvae_cardiac_autoencoder128_150_12.pth'
        }
        
        self.paths = {
            'cache_dir': "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache",
            'train_tag': "train_full_processed",
            'test_tag': "test_processed",
            'ckpt_dir': "/scratch1/e20-fyp-syn-car-mri-gen/diffusion/ldm/outputs",
            'pri_eval2_img_path': "/scratch1/e20-fyp-syn-car-mri-gen/privacy-evaluation/outputs/privacy_roc_curve.png",

            'ddpm_path': "/scratch1/e20-fyp-syn-car-mri-gen/diffusion/ddpm/ddpm-150-model-v3",

            'flow_match_path': "/scratch1/e20-fyp-syn-car-mri-gen/flow-matching/saved_model/20251226_1247/model.pt"
        }

        # LDM Model specific configs (keep your existing dicts here)
        self.diffusion_params = {'num_timesteps': 1000, 'beta_start': 0.00085, 'beta_end': 0.012}
        self.ldm_params = {'down_channels': [128, 256, 512], 'mid_channels': [512, 256], 'down_sample': [True, True], 'attn_down': [False, True], 'time_emb_dim': 256, 'norm_channels': 32, 'num_heads': 8, 'conv_out_channels': 128, 'num_down_layers': 2, 'num_mid_layers': 2, 'num_up_layers': 2, 'condition_config': {'condition_types': ['image'], 'image_condition_config': {'image_condition_input_channels': 4, 'image_condition_output_channels': 1, 'image_condition_h': 128, 'image_condition_w': 128, 'cond_drop_prob': 0.1}}}
        self.autoencoder_params = {'z_channels': 4, 'codebook_size': 8192, 'down_channels': [64, 128, 256], 'mid_channels': [256, 256], 'down_sample': [True, True], 'attn_down': [False, False], 'norm_channels': 32, 'num_heads': 4, 'num_down_layers': 2, 'num_mid_layers': 2, 'num_up_layers': 2}

        # DDPM Params (matches your training setup)
        self.ddpm_params = {
            'num_train_timesteps': 1000,
            'image_size': 128,
            'prediction_type': 'epsilon' # predicting noise
        }

        # Flow Matching Config (From your code snippet)
        self.flow_model_params = {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "num_res_blocks": [2, 2, 2, 2],
            "num_channels": [32, 64, 128, 256],
            "attention_levels": [False, False, False, True],
            "norm_num_groups": 32,
            "resblock_updown": True,
            "num_head_channels": [32, 64, 128, 256],
            "transformer_num_layers": 6,
            "use_flash_attention": False,
            "with_conditioning": True,
            "cross_attention_dim": 256,
            # ControlNet-specific
            "mask_conditioning": True,
            "conditioning_embedding_num_channels": (16,),
            "max_timestep": 1000
        }