import torch
import numpy as np
import random
from config import Config
from src.data import get_dataloaders
from models.wrappers import DiffusionWrapper, DDPMWrapper, FlowMatchingWrapper
from src.pipeline import run_attack

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    set_seed(42)
    cfg = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Setup Data
    train_loader, test_loader = get_dataloaders(cfg)
    
    # 2. Select Model Wrapper based on Config
    if cfg.model_type == 'diffusion-ldm':
        print("Initializing Diffusion-ldm Pipeline...")
        wrapper = DiffusionWrapper(cfg, device)
    elif cfg.model_type == 'diffusion-ddpm':
        print("Initializing Pixel-Space DDPM Pipeline...")
        wrapper = DDPMWrapper(cfg, device)
    elif cfg.model_type == 'flow_matching':
        print("Initializing Flow Matching Pipeline...")
        wrapper = FlowMatchingWrapper(cfg, device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model_type}")

    # 3. Run Evaluation
    auc = run_attack(cfg, wrapper, train_loader, test_loader)
    print(f"\nFinal Privacy AUC: {auc:.4f}")

if __name__ == "__main__":
    main()