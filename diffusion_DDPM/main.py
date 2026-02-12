import torch
from torch import nn
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

# Import logic from the files you already have
from training import TrainingConfig, train_loop
from training import TrainingConfig, train_loop
from data_loader import get_data_loaders


# MAIN EXECUTION
if __name__ == "__main__":
    
    # 1. Get Data Loaders (The new pipeline)
    # This automatically loads the cache, applies 150 split, and returns loaders
    print("Setting up data loaders...")
    loader_150, val_loader = get_data_loaders(batch_size=8)
    
    print(f"✅ Data Ready. Train Batches: {len(loader_150)} | Val Batches: {len(val_loader)}")

    # 2. Setup Model Config
    train_config = TrainingConfig(
        image_size=128,
        train_batch_size=8,      # Must match loader batch_size
        eval_batch_size=8,       # Must match loader batch_size
        # gradient_accumulation_steps=8,
        num_epochs=400,          # Adjust as needed
        save_image_epochs=10,
        segmentation_guided=True,
        segmentation_channel_mode="multi", 
        output_dir="ddpm-150-model-v3"
    )

    # 3. Setup UNet (5 input channels: 1 Image + 4 Mask)
    model = UNet2DModel(
        sample_size=128,
        in_channels=5, 
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D", "DownBlock2D", "DownBlock2D",
            "DownBlock2D", "AttnDownBlock2D", "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
            "UpBlock2D", "UpBlock2D", "UpBlock2D"
        ),
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = nn.DataParallel(model)
    model.to(device)

    # 4. Setup Training Details
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=(len(loader_150) * train_config.num_epochs),
        # num_training_steps=(len(loader_150) * train_config.num_epochs) // train_config.gradient_accumulation_steps,
    )

    # 5. Run Training
    print("\nStarting Training Loop with New Data Pipeline...")
    train_loop(
        config=train_config,
        model=model,
        noise_scheduler=noise_scheduler,
        optimizer=optimizer,
        train_dataloader=loader_150,
        eval_dataloader=val_loader,
        lr_scheduler=lr_scheduler,
        device=device
    )