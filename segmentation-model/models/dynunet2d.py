# src/models/dynunet2d.py
from __future__ import annotations

from typing import Sequence, Optional
import torch

from monai.networks.nets import DynUNet

def build_dynunet2d(
    in_channels: int = 1,
    out_channels: int = 4,
    dropout: float = 0.0,
) -> torch.nn.Module:
    # 5 stages (>=3 required)
    # Must satisfy: len(kernel_size) == len(strides)
    kernel_size = [[3, 3]] * 5
    strides = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2]]
    upsample_kernel_size = [[2, 2]] * 4  # one per upsampling step (stages-1)

    model = DynUNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        upsample_kernel_size=upsample_kernel_size,
        filters=(32, 64, 128, 256, 512),
        dropout=dropout,
        norm_name="INSTANCE",
        act_name=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
        deep_supervision=False,
    )
    return model

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'CPU'
    model = build_dynunet2d().to(device)
    print(model)