import torch
import numpy as np
import random
from config import Config
from src.data import get_dataloaders
from src.metrics import MaskPrivacyMetric
from models.wrappers import MaskDiffusionWrapper
from src.pipeline import run_attack_flexible

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
    
    # 2. Select Pipeline
    if cfg.eval_mode == 'mask':
        print("Initializing MASK Privacy Pipeline...")
        wrapper = MaskDiffusionWrapper(cfg, device)
        # Use Simple MSE for masks
        metric_calc = MaskPrivacyMetric(device)

    # 3. Run Evaluation (Modified to accept metric calculator)    
    auc = run_attack_flexible(cfg, wrapper, train_loader, test_loader, metric_calc)
    print(f"\nFinal Privacy AUC: {auc:.4f}")

if __name__ == "__main__":
    main()