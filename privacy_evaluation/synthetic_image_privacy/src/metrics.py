import torch
import torch.nn as nn
import torch.fft as fft
from torchmetrics.image import StructuralSimilarityIndexMeasure

# ==========================================
# FCRE (Frequency-Calibrated Error)
# ==========================================
class FCRECalculator(nn.Module):
    def __init__(self, device, low_cutoff=0.15, high_cutoff=0.85):
        """
        Calculates error based on mid-frequencies and SSIM.
        low_cutoff: fraction of radius to ignore (low freq)
        high_cutoff: fraction of radius to ignore (high freq)
        """
        super().__init__()
        self.device = device
        
        # --- CRITICAL FIX HERE ---
        # Added reduction='none'. 
        # This ensures we get a score for EVERY image in the batch [N], 
        # instead of one average score for the whole batch [].
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0, reduction='none').to(device) 
        
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def get_mid_freq_loss(self, error_map):
        """
        Applies a Band-Pass filter to the error map to isolate mid-frequencies.
        """
        # 1. FFT
        fft_map = fft.fft2(error_map)
        fft_shift = fft.fftshift(fft_map)
        
        # 2. Create Mask
        h, w = error_map.shape[-2:]
        cy, cx = h // 2, w // 2
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y, x = y.to(self.device), x.to(self.device)
        
        # Calculate radius from center
        radius = torch.sqrt((x - cx)**2 + (y - cy)**2)
        max_radius = min(cy, cx)
        
        # Bandpass Mask (1 for mid-freq, 0 for others)
        mask = ((radius >= (max_radius * self.low_cutoff)) & 
                (radius <= (max_radius * self.high_cutoff))).float()
        
        # Expand mask for batch/channel dims
        mask = mask.unsqueeze(0).unsqueeze(0) 
        
        # 3. Apply Mask & Inverse FFT
        filtered_fft = fft_shift * mask
        filtered_fft_ishift = fft.ifftshift(filtered_fft)
        filtered_error = fft.ifft2(filtered_fft_ishift).real
        
        # 4. MSE of filtered error
        return torch.mean(filtered_error ** 2, dim=[1, 2, 3])

    def forward(self, pred_noise, true_noise):
        # 1. Filtered Mid-Frequency Error
        error_map = pred_noise - true_noise
        freq_error = self.get_mid_freq_loss(error_map)
        
        # 2. SSIM (Structural Similarity)
        # Now returns a tensor of shape [Batch_Size] because reduction='none'
        ssim_score = self.ssim(pred_noise, true_noise)
        
        # Return vector: [Mid_Freq_MSE, 1 - SSIM]
        # We use (1-SSIM) because we want "Error" (higher is bad)
        return torch.stack([freq_error, 1 - ssim_score], dim=1)