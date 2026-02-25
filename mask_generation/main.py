import torch
import torch.nn.functional as F
from monai.utils import set_determinism
from monai.networks.utils import one_hot
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
import matplotlib.pyplot as plt

from data_loading import get_loaders

#CONFIGURATION
CACHE_DIR = "/scratch1/e20-fyp-syn-car-mri-gen/datasets/MandM Dataset/cache"
IMG_SIZE = (128, 128)
BATCH_SIZE = 8            # Higher batch size is better for diffusion
NUM_EPOCHS = 400          # Diffusion needs long training
LR = 1e-4
NUM_CLASSES = 4           # 0:Back, 1:LV, 2:MYO, 3:RV
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    #Reproducibility
    set_determinism(42)

    print("Initializing Data Loaders...")
    train_loader, val_loader = get_loaders(CACHE_DIR, BATCH_SIZE)

    #Define Network
    # We use in_channels=NUM_CLASSES because input is One-Hot
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=NUM_CLASSES,
        out_channels=NUM_CLASSES, 
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        num_head_channels=32,
    ).to(DEVICE)

    # Standard DDPM Scheduler
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    val_losses = []

    #Training Loop
    print(f"Starting training on {DEVICE}...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for step, batch_masks in enumerate(train_loader):
            
            # Input Shape: (Batch, 1, 128, 128) | Values: Integers 0,1,2,3
            clean_images = batch_masks.to(DEVICE)
            
            #Convert to One-Hot: (B, 1, H, W) -> (B, 4, H, W)
            clean_images = one_hot(clean_images, num_classes=NUM_CLASSES, dim=1)
            #Ensure Float32 for the network
            clean_images = clean_images.float()
            clean_images = clean_images * 2 -1

            #Sample Noise
            noise = torch.randn_like(clean_images).to(DEVICE)
            
            #Sample Timesteps
            timesteps = torch.randint(
                low=0, 
                high=scheduler.num_train_timesteps, 
                size=(clean_images.shape[0],), 
                device=DEVICE
            ).long()
            
            #Add Noise (Forward Diffusion)
            noisy_images = scheduler.add_noise(
                original_samples=clean_images, 
                noise=noise, 
                timesteps=timesteps
            )
            
            #Predict Noise
            optimizer.zero_grad()
            noise_pred = model(x=noisy_images, timesteps=timesteps, context=None)
            
            #Loss
            loss = F.mse_loss(noise_pred, noise)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()  # Switch to evaluation mode
        val_epoch_loss = 0
        
        with torch.no_grad():
            for step, batch_masks in enumerate(val_loader):
                clean_images = batch_masks.to(DEVICE)
                
                # Apply EXACT same preprocessing as training
                clean_images = one_hot(clean_images, num_classes=NUM_CLASSES, dim=1)
                clean_images = clean_images.float()
                clean_images = clean_images * 2 - 1 

                noise = torch.randn_like(clean_images).to(DEVICE)
                
                timesteps = torch.randint(
                    low=0, 
                    high=scheduler.num_train_timesteps, 
                    size=(clean_images.shape[0],), 
                    device=DEVICE
                ).long()
                
                noisy_images = scheduler.add_noise(
                    original_samples=clean_images, 
                    noise=noise, 
                    timesteps=timesteps
                )
                
                # Forward pass only
                noise_pred = model(x=noisy_images, timesteps=timesteps, context=None)
                loss = F.mse_loss(noise_pred, noise)
                val_epoch_loss += loss.item()
        
        # Calculate averages
        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_epoch_loss / len(val_loader)
        
        # Store for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        #Save Checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save(model.state_dict(), f"mask_diffusion_epoch_{epoch+1}.pth")

    torch.save(model.state_dict(), "mask_diffusion_final.pth")
    print("Training Complete. Model saved.")

    #PLOTTING CODE STARTS HERE
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label="Train Loss")
    plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label="Validation Loss", linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Diffusion Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("Saved loss curve to 'loss_curve.png'")

if __name__ == "__main__":
    main()