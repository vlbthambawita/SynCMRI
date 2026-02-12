import torch
import torch.nn as nn

class MaskPrivacyMetric(nn.Module):
    """
    Simple MSE Loss for Mask Privacy.
    For Diffusion models, the training loss (MSE between predicted and true noise)
    is the strongest signal for membership inference.
    """
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, pred_noise, true_noise):
        # pred_noise: [B, 4, 128, 128]
        # true_noise: [B, 4, 128, 128]
        
        # Calculate MSE per sample (keep batch dimension)
        # Result shape: [B]
        mse = torch.mean((pred_noise - true_noise) ** 2, dim=[1, 2, 3])
        
        # We assume 'error' is the signal. 
        # For consistency with the pipeline, we can return just the MSE.
        # Reshape to [B, 1] so it looks like a feature vector
        return mse.unsqueeze(1)