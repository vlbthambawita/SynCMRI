import torch
import matplotlib.pyplot as plt
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from tqdm import tqdm
from scipy import ndimage

#CONFIGURATION
MODEL_PATH = "mask_diffusion_epoch_400.pth"
IMG_SIZE = (128, 128)
NUM_CLASSES = 4
NUM_SAMPLES = 8  # How many masks to generate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    #Load Model structure (Must match training)
    model = DiffusionModelUNet(
        spatial_dims=2,
        in_channels=NUM_CLASSES,
        out_channels=NUM_CLASSES,
        num_channels=(64, 128, 256, 512),
        attention_levels=(False, False, True, True),
        num_res_blocks=2,
        num_head_channels=32,
    ).to(DEVICE)
    
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    scheduler = DDPMScheduler(num_train_timesteps=1000)

    #Generation Loop
    print(f"Generating {NUM_SAMPLES} unconditional masks...")
    
    # Start from pure Gaussian noise: Shape (Batch, Classes, H, W)
    # Note we generate 4 channels of noise, representing the 4 class probabilities
    noise = torch.randn((NUM_SAMPLES, NUM_CLASSES, *IMG_SIZE)).to(DEVICE)
    current_img = noise

    # Iterative Denoising
    for t in tqdm(scheduler.timesteps):
        with torch.no_grad():
            model_output = model(x=current_img, timesteps=torch.Tensor((t,)).to(DEVICE), context=None)
            current_img, _ = scheduler.step(model_output, t, current_img)
            step_output = scheduler.step(model_output, t, current_img)
            current_img = step_output[0]
    
    current_img = (current_img + 1) / 2
    # The output 'current_img' contains continuous values (logits) for each class. We use argmax to select the most likely class for each pixel.
    generated_masks = torch.argmax(current_img, dim=1).cpu().numpy()

    # 4. Visualization and Saving
    fig, axs = plt.subplots(1, NUM_SAMPLES, figsize=(15, 3))
    
    # Define a color map for 0,1,2,3 (Background, LV, MYO, RV)
    # Adjust vmin/vmax to ensure colors stay consistent
    for i in range(NUM_SAMPLES):
        mask = generated_masks[i]
        
        # Simple cleanup (optional): Remove small disconnected noise if needed
        # mask = ndimage.median_filter(mask, size=3) 

        axs[i].imshow(mask, cmap='jet', vmin=0, vmax=NUM_CLASSES-1, interpolation='nearest')
        axs[i].axis('off')
        axs[i].set_title(f"Sample {i+1}")

    plt.tight_layout()
    plt.savefig("generated_unconditional_masks400.png")
    print("Saved 'generated_unconditional_masks.png'.")
    plt.show()

if __name__ == "__main__":
    main()