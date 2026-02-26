# src/models/dynunet2d.py
from __future__ import annotations

import os
import time
import torch
import matplotlib.pyplot as plt
from torch import nn
from generative.networks.nets import DiffusionModelUNet, ControlNet
from flow_matching.solver import ODESolver

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
import math

class MergedModel(nn.Module):
    
    def __init__(self, unet: DiffusionModelUNet, controlnet: ControlNet = None, max_timestep=1000):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.max_timestep = max_timestep

        self.has_controlnet = controlnet is not None
        self.has_conditioning = unet.with_conditioning

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor = None,
        masks: torch.Tensor = None,
    ):
        # Scale continuous t -> discrete timesteps
        t = t * (self.max_timestep - 1)
        t = t.floor().long()

        # If t is scalar, expand to batch size
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        if cond is not None and cond.dim() == 2:
            cond = cond.unsqueeze(1)

        if self.has_controlnet:
            # cond is expected to be a ControlNet conditioning, e.g. mask
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                x=x, timesteps=t, controlnet_cond=masks, context=cond
            )
            output = self.unet(
                x=x,
                timesteps=t,
                context=cond,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )
        else:
            # If no ControlNet, cond might be cross-attention or None
            output = self.unet(x=x, timesteps=t, context=cond)

        return output

def build_model(model_config: dict, device: torch.device = None) -> MergedModel:
    """
    Builds a UNet+ControlNet model

    Args:
        model_config: Dictionary containing model configuration.
        device: Device to move the model to.

    Returns:
        A MergedModel instance.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Make a copy so the original config remains unaltered.
    mc = model_config.copy()

    # Pop out keys that are not needed by the model constructors.
    mask_conditioning = mc.pop("mask_conditioning", False)
    max_timestep = mc.pop("max_timestep", 1000)
    # Pop out ControlNet specific key, if present.
    cond_embed_channels = mc.pop("conditioning_embedding_num_channels", None)

    # Build the base UNet by passing all remaining items as kwargs.
    unet = DiffusionModelUNet(**mc)

    controlnet = None
    if mask_conditioning:
        mc.pop("out_channels", None)
        # Ensure the controlnet has its specific key.
        if cond_embed_channels is None:
            cond_embed_channels = (16,)
        # Pass the same config kwargs to ControlNet plus the controlnet-specific key.
        controlnet = ControlNet(**mc, conditioning_embedding_num_channels=cond_embed_channels)
        controlnet.load_state_dict(unet.state_dict(), strict=False)

    model = MergedModel(unet=unet, controlnet=controlnet, max_timestep=max_timestep)

    # Print number of trainable parameters.
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters.")
    model_size_mb = num_params * 4 / (1024**2)
    print(f"Model size: {model_size_mb:.2f} MB")

    return model.to(device)

if __name__ == '__main__':
    model_config = {
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
}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    model = build_model(model_config, device)

    
