import numpy as np
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from .metrics import FCRECalculator
from config import Config

def collect_signals(wrapper, loader, label, fcre_calc, device):
    signals = []
    labels = []
    
    print(f"Collecting signals for Label {label}...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            if isinstance(batch, (list, tuple)):
                im, cond = batch
            else:
                im = batch
                cond = {}
                
            im = im.float().to(device)
            if isinstance(cond, dict) and 'image' in cond:
                cond['image'] = cond['image'].to(device)
                
            # === MAGIC HAPPENS HERE ===
            # The wrapper handles the specific logic (Noise vs Flow)
            # We just get Prediction and Target back.
            pred, target = wrapper.get_training_pair(im, cond)
            # ==========================

            # Calculate Error
            error_metrics = fcre_calc(pred, target)
            
            signals.append(error_metrics.cpu().numpy())
            labels.extend([label] * im.shape[0])
            
        return np.concatenate(signals, axis=0), np.array(labels)

config = Config()

def plot_roc_curve(y_test, preds, auc_score, save_path= config.paths['pri_eval2_img_path']):
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

def run_attack(config, wrapper, train_loader, test_loader):
    device = wrapper.device
    fcre_calc = FCRECalculator(device)
    
    # 1. Collect Signals
    X_mem, y_mem = collect_signals(wrapper, train_loader, 1, fcre_calc, device)
    X_non, y_non = collect_signals(wrapper, test_loader, 0, fcre_calc, device)
    
    X = np.concatenate([X_mem, X_non], axis=0)
    y = np.concatenate([y_mem, y_non], axis=0)
    
    # 2. Train Attack Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    attack_model = LogisticRegression(solver='liblinear')
    attack_model.fit(X_train, y_train)
    
    # 3. Score
    preds = attack_model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    
    print(f"PRIVACY AUC: {auc:.4f}")
    plot_roc_curve(y_test, preds, auc)
    return auc