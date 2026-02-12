import torch
from tqdm import tqdm
import numpy as np
from config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def collect_signals(wrapper, loader, label, metric_calc, device):
    signals = []
    labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader):
            # Handle Data Loading format
            if isinstance(batch, (list, tuple)):
                im, cond = batch
            else:
                im = batch
                cond = {}
                
            im = im.float().to(device)
            if isinstance(cond, dict) and 'image' in cond:
                cond['image'] = cond['image'].to(device)

            # Get Pair
            pred, target = wrapper.get_training_pair(im, cond)

            # Calculate Error (Using the passed metric_calc)
            # For masks: returns MSE [B, 1]
            # For images: returns [B, 2] (Freq + SSIM)
            error_metrics = metric_calc(pred, target)
            
            signals.append(error_metrics.cpu().numpy())
            labels.extend([label] * im.shape[0])
            
        return np.concatenate(signals, axis=0), np.array(labels)

config = Config()

def plot_roc_curve(y_test, preds, auc_score, save_path= config.paths['mask_privacy_output']):
    """
    Helper function to plot and save the ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_test, preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--') # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Privacy Attack ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(save_path)
    print(f"📉 ROC Curve saved to {save_path}")
    plt.close()

def run_attack_flexible(config, wrapper, train_loader, test_loader, metric_calc):
    """
    Unified attack function for both Images and Masks.
    """
    device = wrapper.device
    
    # 1. Collect Signals
    print("Collecting Training Member Signals...")
    X_mem, y_mem = collect_signals(wrapper, train_loader, 1, metric_calc, device)
    
    print("Collecting Non-Member Signals...")
    X_non, y_non = collect_signals(wrapper, test_loader, 0, metric_calc, device)
    
    X = np.concatenate([X_mem, X_non], axis=0)
    y = np.concatenate([y_mem, y_non], axis=0)
    
    # 2. Train Attack Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    attack_model = LogisticRegression(solver='liblinear')
    attack_model.fit(X_train, y_train)
    
    # 3. Score
    preds = attack_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    
    save_path = config.paths.get('mask_privacy_output', 'roc.png')
    plot_roc_curve(y_test, preds, auc, save_path=save_path)
    
    return auc