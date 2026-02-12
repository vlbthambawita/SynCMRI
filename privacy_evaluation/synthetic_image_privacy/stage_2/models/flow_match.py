# src/models/flow_match.py
import torch
import torch.nn as nn
from generative.networks.nets import DiffusionModelUNet, ControlNet 
# Ensure 'generative' and 'flow_matching' packages/folders are available


class ODESolver:
    """
    A simple ODE Solver for Flow Matching.
    Integrates the velocity field v(x, t) to move from noise (t=0) to data (t=1).
    """
    def __init__(self, velocity_model):
        self.velocity_model = velocity_model

    def sample(self, x_init, time_grid, method="euler", step_size=0.1, return_intermediates=False, cond=None, masks=None):
        """
        Args:
            x_init: Initial noise [B, C, H, W]
            time_grid: Tensor of time points from 0 to 1 (e.g., torch.linspace(0, 1, steps))
            cond: Conditioning (e.g., Class labels or embeddings)
            masks: Segmentation masks (for ControlNet/SPADE)
        """
        x = x_init
        intermediates = [x]
        
        device = x.device
        
        # We iterate through the time steps
        for i in range(len(time_grid) - 1):
            t0 = time_grid[i]
            t1 = time_grid[i+1]
            dt = t1 - t0

            # 1. Get Velocity Estimate at current x and t
            # Expand t0 to batch size
            t_batch = torch.ones(x.shape[0], device=device) * t0
            
            # Predict velocity v
            v = self.velocity_model(x, t_batch, cond=cond, masks=masks)

            # 2. Update x (Euler Step: x_new = x + v * dt)
            if method == "euler":
                x = x + v * dt
            elif method == "midpoint":
                # Midpoint method (more accurate, slower)
                x_mid = x + v * (dt / 2)
                t_mid = t_batch + (dt / 2)
                v_mid = self.velocity_model(x_mid, t_mid, cond=cond, masks=masks)
                x = x + v_mid * dt
            
            if return_intermediates:
                intermediates.append(x)

        if return_intermediates:
            return torch.stack(intermediates)
        else:
            return x

class MergedModel(nn.Module):
    """
    Merged model that wraps a UNet and an optional ControlNet.
    Takes in x, time in [0,1], and (optionally) a ControlNet condition.
    """
    def __init__(self, unet: DiffusionModelUNet, controlnet: ControlNet = None, max_timestep=1000):
        super().__init__()
        self.unet = unet
        self.controlnet = controlnet
        self.max_timestep = max_timestep
        self.has_controlnet = controlnet is not None

    def forward(self, x, t, cond=None, masks=None):
        # Scale continuous t -> discrete timesteps
        t = t * (self.max_timestep - 1)
        t = t.floor().long()

        # If t is scalar, expand to batch size
        if t.dim() == 0:
            t = t.expand(x.shape[0])

        if cond is not None and cond.dim() == 2:
            cond = cond.unsqueeze(1)

        if self.has_controlnet:
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
            output = self.unet(x=x, timesteps=t, context=cond)

        return output

def build_flow_model(model_config: dict, device: torch.device) -> MergedModel:
    """
    Builds the MergedModel based on config.
    """
    mc = model_config.copy()

    # Pop out keys specific to wrapper/builder
    mask_conditioning = mc.pop("mask_conditioning", False)
    max_timestep = mc.pop("max_timestep", 1000)
    cond_embed_channels = mc.pop("conditioning_embedding_num_channels", None)

    # Build Base UNet
    unet = DiffusionModelUNet(**mc)

    controlnet = None
    if mask_conditioning:
        # ControlNet needs slightly modified config (remove out_channels usually)
        if "out_channels" in mc:
            mc.pop("out_channels")
        
        if cond_embed_channels is None:
            cond_embed_channels = (16,)
            
        controlnet = ControlNet(**mc, conditioning_embedding_num_channels=cond_embed_channels)
        
        # Load shared weights usually happens here, but we assume
        # loading the final MergedModel state_dict handles this.

    model = MergedModel(unet=unet, controlnet=controlnet, max_timestep=max_timestep)
    return model.to(device)