import torch
import os
from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from torchvision.utils import save_image

# Import custom classes from your files
from eval import SegGuidedDDPMPipeline, SegGuidedDDIMPipeline
from data_loader import get_test_loader, get_data_loaders

def test_model():
    # 1. Setup Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = "ddpm-150-finetuned" # Ensure this matches your training output folder name
    
    print(f"Loading model from {output_dir}...")

    # 2. Load the Trained Model & Scheduler
    try:
        unet = UNet2DModel.from_pretrained(os.path.join(output_dir, "unet")).to(device)
        #---- Use for DDPM-----
        scheduler = DDPMScheduler.from_pretrained(os.path.join(output_dir, "scheduler"))
        #---- Use for DDIM-----
        # scheduler = DDIMScheduler.from_pretrained(os.path.join(output_dir, "scheduler"))
    except OSError:
        print("Error: Could not find the model.")
        print(f"Did training finish? Check if '{output_dir}/unet' exists.")
        return

    # 3. Create the Pipeline

    #---- Use for DDPM-----
    pipeline = SegGuidedDDPMPipeline(
        unet=unet, 
        scheduler=scheduler, 
        eval_dataloader=None,     
        external_config=None      
    )

    # ---- Use for DDIM-----
    # pipeline = SegGuidedDDIMPipeline(
    #     unet=unet, 
    #     scheduler=scheduler, 
    #     eval_dataloader=None,     
    #     external_config=None      
    # )

    # Option A: Use the official Test Set
    print("Loading Test Data...")
    loader = get_test_loader(batch_size=8)
    
    # Option B: If you want to use Validation data instead, uncomment this:
    # _, loader = get_data_loaders(batch_size=8)

    if loader is None or len(loader) == 0:
        print("Error: Loader is empty.")
        return

    # Get one batch of data
    batch = next(iter(loader))
    real_images = batch['images'].to(device)
    masks = batch['seg_onehot'].to(device) # Shape: [8, 4, 128, 128]

    # 5. Mock a Config Object for the Pipeline
    class EvalConfig:
        segmentation_channel_mode = "multi"
        class_conditional = False
        use_cfg_for_eval_conditioning = False
        trans_noise_level = None 
    
    pipeline.external_config = EvalConfig()

    # 6. Generate Images
    print("Generating samples... (This might take a minute)")
    with torch.no_grad():
        output = pipeline(
            batch_size=8,
            seg_batch={"seg_onehot": masks}, 
            num_inference_steps=1000, #Change accroding to choice of DDPM or DDIM
            output_type="np" # Returns numpy array
        )
        fake_images = torch.tensor(output.images).permute(0, 3, 1, 2) 

    # 7. Visualize Results
    # Collapse 4-channel mask to 1-channel for visualization
    mask_viz = torch.argmax(masks, dim=1, keepdim=True).float() / 3.0 

    # Normalize Real Images (-1..1 -> 0..1)
    real_images = (real_images + 1) / 2
    
    # Concatenate lists (Mask | Real | Fake)
    final_grid = torch.cat([mask_viz.cpu(), real_images.cpu(), fake_images.cpu()], dim=0)

    os.makedirs("test_results", exist_ok=True)
    save_path = "test_results/test_sample_DDPM4.png"
    save_image(final_grid, save_path, nrow=8)
    
    print(f"✅ Test Complete! Check the image at: {save_path}")
    print("Top Row: Input Masks")
    print("Middle Row: Real Patient MRI")
    print("Bottom Row: Generated Artificial MRI")

if __name__ == "__main__":
    test_model()