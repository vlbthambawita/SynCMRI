class Config:
    def __init__(self):
        # Change this to enable mask mode
        self.eval_mode = 'mask'

        self.dataset_params = {
            'im_channels': 1, 
            'im_size': 128, 
            'subset_size': 175
        }

        # Add Mask Model Paths
        self.paths = {
            'cache_dir': "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache",
            'train_tag': "train_full_processed",
            'test_tag': "test_processed",
            'mask_ckpt_path': "/scratch1/e20-fyp-syn-car-mri-gen/mask-generation/monai-mask-generation/outputs/v2/mask_diffusion_epoch_350.pth", # Path to your saved mask model
            'mask_privacy_output': "/scratch1/e20-fyp-syn-car-mri-gen/privacy-evaluation/mask-privacy/stage2/privacy_roc_mask.png"
        }

        # Mask Model Params (Must match the MONAI UNet you used for masks)
        self.mask_model_params = {
            "spatial_dims": 2,
            "in_channels": 4,  # One-hot encoded masks
            "out_channels": 4, 
            "num_channels": (64, 128, 256, 512),
            "attention_levels": (False, False, True, True),
            "num_res_blocks": 2,
            "num_head_channels": 32,
        }