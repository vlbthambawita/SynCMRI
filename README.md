# SynCMRI: Synthetic Cardiac MRI Generation

> **Balancing Fidelity, Utility, and Privacy in Synthetic Cardiac MRI Generation: A Comparative Study**

## Abstract

Deep learning in cardiac MRI (CMR) is fundamentally constrained by data scarcity and privacy regulations. This study systematically benchmarks three generative architectures—**Denoising Diffusion Probabilistic Models (DDPM)**, **Latent Diffusion Models (LDM)**, and **Flow Matching (FM)**—for synthetic CMR generation. Utilizing a two-stage pipeline where anatomical masks condition image synthesis, we evaluate generated data across three critical axes: **fidelity**, **utility**, and **privacy**. This work quantifies the trade-offs between computational efficiency, cross-domain generalization, and patient confidentiality, establishing a framework for safe and effective synthetic data augmentation in medical imaging.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Evaluation Framework](#evaluation-framework)
- [Configuration](#configuration)
- [Citation](#citation)

---

## Overview

SynCMRI implements a two-stage pipeline for segmentation-guided synthetic cardiac MRI generation:

1. **Stage 1 — Mask Generation**: Anatomical masks (LV, Myo, RV, background) are produced using external mask-generation models (e.g., MONAI-based segmentation).
2. **Stage 2 — Image Generation**: Masks condition three generative models to produce synthetic cardiac MRI images.

### Supported Generative Architectures

| Model | Description |
|-------|-------------|
| **DDPM** | Pixel-space diffusion with UNet2D, 5-channel input (1 image + 4 mask channels). Supports DDPM and DDIM sampling. |
| **LDM** | VQ-VAE encoder + latent diffusion U-Net with mask conditioning. More efficient than pixel-space diffusion. |
| **Flow Matching** | Flow-based generative model with ControlNet-style mask conditioning (uses external flow-matching repo). |

### Evaluation Axes

- **Fidelity**: PSNR, SSIM, MS-SSIM, LPIPS, FID, KID
- **Utility**: Train DynUNet2D segmentation on synthetic data; evaluate on MandM and ACDC (Dice, IoU, HD95, ASD)
- **Privacy**: Copy/memorization risk (L2, LPIPS/Dice, NNDR); membership inference attack via FCRE (frequency-calibrated reconstruction error)

---

## Project Structure

```
SynCMRI/
├── README.md                           # This file
├── diffusion_DDPM/                     # DDPM (pixel-space diffusion)
│   ├── main.py                         # Training entry point
│   ├── training.py                     # Training loop, SegGuided pipeline
│   ├── test.py                         # Inference and qualitative comparison
│   ├── eval.py                         # SegGuided pipelines, add_segmentations_to_noise
│   ├── fidelity_evaluation.py          # FID, KID, SSIM, PSNR, LPIPS
│   ├── data_loader.py                  # MandM CMRDataset, get_data_loaders
│   └── utils.py                        # Image visualization utilities
├── diffusion_LDM/                      # Latent Diffusion Model
│   ├── main.py                         # LDM training
│   ├── diffusion.py                    # Unet, VQVAE, LinearNoiseScheduler
│   ├── config.py                       # Paths, LDM/VQ-VAE architecture
│   ├── test.py                         # LDM inference
│   ├── eval.py                         # load_models, generate_samples
│   └── fidelity_evaluation.py          # Fidelity metrics for LDM
├── flow-matching/                      # Flow Matching generative model
│   └── src/
│       ├── data/                       # Data loading
│       ├── models/                     # Flow matching model architectures
│       └── utils/                      # Helper functions
├── mask_generation/                    # Synthetic mask generation pipeline
│   ├── main.py                         # Entry point for mask generation
│   ├── generate.py                     # Mask synthesis logic
│   ├── evaluate.py                     # Mask evaluation metrics
│   ├── fidelity_evaluation.py          # Pair-wise Dice, mask-level metrics
│   └── data_loading.py                 # Mask dataset loading
├── segmentation_model/                 # Downstream utility evaluation
│   ├── train.py                        # Train DynUNet2D on real/synthetic data
│   ├── test.py                         # Evaluate on MandM & ACDC
│   ├── data/
│   │   ├── mandm_dataloader.py         # MandM cardiac MRI dataloader
│   │   ├── acdc_dataloader.py          # ACDC dataloader
│   │   └── synthetic_dataloader.py     # Synthetic image/mask dataloader
│   └── models/
│       └── dynunet2d.py                # DynUNet2D (MONAI)
└── privacy_evaluation/                 # Privacy analysis
    ├── synthetic_image_privacy/
    │   ├── stage_1/main.py             # L2, LPIPS, NNDR (copy risk)
    │   └── stage_2/
    │       ├── main.py                 # Membership inference (FCRE-based)
    │       ├── config.py               # Model type, paths, hyperparams
    │       ├── src/
    │       │   ├── pipeline.py         # ROC AUC attack, collect_signals, plot
    │       │   ├── data.py             # Data loading
    │       │   └── metrics.py          # FCRE calculator
    │       └── models/                 # Diffusion/DDPM/Flow Matching wrappers
    └── synthetic_mask_privacy/
        ├── stage_1/main.py             # L2, Dice, NNDR on synthetic masks
        └── stage_2/                    # Mask privacy pipeline
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch with CUDA (recommended for training)
- See below for pip installable packages

### Install Dependencies

```bash
pip install torch torchvision
pip install diffusers einops
pip install monai
pip install torch_fidelity torchmetrics lpips
pip install scikit-learn matplotlib tqdm pandas numpy pillow
```

For Flow Matching and some privacy evaluation components:

```bash
pip install generative-monai  # MONAI generative package
```

### Environment Notes

- **DDPM** uses `diffusers` (UNet2DModel, DDPMScheduler, DDIMScheduler).
- **LDM** uses custom components in `diffusion_LDM/diffusion.py`.
- **Segmentation** uses MONAI `DynUNet`, `DiceCELoss`, and metrics.
- **Privacy** uses `lpips`, `scikit-learn`, and custom FCRE metrics.

---

## Data Preparation

### MandM Dataset

The project expects a preprocessed MandM cache with the following structure:

```
<CACHE_ROOT>/
├── train_full_processed/     # Training split
│   ├── <patient_id>_resampled_4d_img.npy
│   ├── <patient_id>_resampled_4d_mask.npy
│   └── <patient_id>_indices.npy
├── val_processed/            # Validation split
└── test_processed/           # Test split
```

- **Images**: 4D arrays resampled to a consistent size; typically 128×128 per slice.
- **Masks**: 4 classes—Background, LV, Myocardium (Myo), RV—one-hot encoded [B, 4, H, W] or index [B, 1, H, W].

### Update Data Paths

Set the cache root in the relevant config files:

- **DDPM**: `diffusion_DDPM/data_loader.py` → `CACHE_DIR`
- **LDM**: `diffusion_LDM/config.py` → `train_params['cache_root_mandm']`
- **FM**: `flow-matching/src/data/mandm_dataloader.py` → `cache_root`
- **Mask Generation**: `mask_generation/data_loading.py` → `cache_dir`
- **Segmentation**: `segmentation_model/train.py` → `TrainConfig.cache_root`
- **Privacy**: `privacy_evaluation/synthetic_image_privacy/stage_2/config.py` → `paths['cache_dir']`

---

## Usage

### 1. Train DDPM

```bash
cd diffusion_DDPM
python main.py
```

- Adjust `batch_size`, `num_epochs`, `output_dir` in `main.py` and `TrainingConfig`.
- Outputs checkpoints and sample images to `output_dir` (e.g., `ddpm-150-model-v3`).

### 2. Train LDM

```bash
cd diffusion_LDM
# Ensure VQ-VAE checkpoint path is set in config.py
python main.py
```

- Configure `config.py`: `ldm_ckpt_path`, `vqvae_ckpt_path`, `cache_root_mandm`.

### 3. Generate Samples (DDPM)

```bash
cd diffusion_DDPM
python test.py
```

- Uses test loader and a trained model (e.g., `ddpm-150-finetuned`).
- Compare real vs. generated images qualitatively.

### 4. Generate Samples (LDM)

```bash
cd diffusion_LDM
python test.py
```

### 5. Fidelity Evaluation

```bash
# DDPM
cd diffusion_DDPM
python fidelity_evaluation.py

# LDM
cd diffusion_LDM
python fidelity_evaluation.py
```

- Computes FID, KID, SSIM, MS-SSIM, PSNR, LPIPS between real and generated images.
- Outputs metrics and optionally saved image grids.

### 6. Train Segmentation (Utility Evaluation)

```bash
cd segmentation_model
# Edit train.py: set dataset_name and cache_root for MandM, ACDC, or synthetic
python train.py
```

- `dataset_name`: `"mandm"`, `"acdc"`, or synthetic (e.g., `"syn_fm_full"`).
- `cache_root`: path to the corresponding data directory.

### 7. Test Segmentation

```bash
cd segmentation_model
# Set checkpoint_path, eval_mandm, eval_acdc in test.py
python test.py
```

- Evaluates on MandM and/or ACDC; reports Dice, IoU, HD95, ASD.

### 8. Privacy Evaluation

**Stage 1 (Image) — Copy/Memorization Risk**

```bash
cd privacy_evaluation/synthetic_image_privacy/stage_1
# Expects base_path with 'real' and 'synth' image subdirectories
python main.py
```

- Computes L2, LPIPS, NNDR histograms between real and synthetic images.

**Stage 2 (Image) — Membership Inference**

```bash
cd privacy_evaluation/synthetic_image_privacy/stage_2
# Set model_type in config.py: 'flow_matching', 'diffusion-ldm', or 'diffusion-ddpm'
python main.py
```

- Uses FCRE (frequency-calibrated error + SSIM) and logistic regression.
- Outputs ROC AUC and ROC curve plot.

**Stage 1 (Mask)**

```bash
cd privacy_evaluation/synthetic_mask_privacy/stage_1
python main.py
```

- L2, Dice, NNDR on synthetic masks vs. real masks.

---

## Evaluation Framework

### Fidelity Metrics

| Metric | Description |
|--------|-------------|
| **FID** | Fréchet Inception Distance |
| **KID** | Kernel Inception Distance |
| **SSIM / MS-SSIM** | Structural similarity |
| **PSNR** | Peak signal-to-noise ratio |
| **LPIPS** | Learned perceptual image patch similarity |

### Utility Metrics (Segmentation)

| Metric | Description |
|--------|-------------|
| **Dice** | Dice coefficient per class |
| **IoU** | Intersection over Union |
| **HD95** | 95th percentile Hausdorff distance |
| **ASD** | Average surface distance |

### Privacy Metrics

| Stage | Metrics |
|-------|---------|
| **Stage 1** | L2, LPIPS (images) or L2, Dice (masks); NNDR (nearest-neighbor distance ratio) |
| **Stage 2** | FCRE + logistic regression → ROC AUC for membership inference |

---

## Configuration

### Key Config Files

| Component | Config File | Main Parameters |
|-----------|-------------|-----------------|
| LDM | `diffusion_LDM/config.py` | `cache_root_mandm`, checkpoint paths, diffusion/autoencoder params |
| Privacy (Image) | `privacy_evaluation/synthetic_image_privacy/stage_2/config.py` | `model_type`, `paths`, LDM/DDPM/Flow Matching params |
| Segmentation | `segmentation_model/train.py` | `TrainConfig`: `dataset_name`, `cache_root`, `epochs`, `batch_size` |

### Data Formats

- **Images**: 128×128, 1 channel, normalized to [-1, 1] or [0, 1].
- **Masks**: 4 classes (background, LV, Myo, RV); one-hot [B, 4, 128, 128] or index [B, 1, 128, 128].
- **Synthetic data**: PNG images and masks; class remapping to MandM scheme in `synthetic_dataloader.py`.

---
## Hugging Face Resources

### Interactive Visualization 

```bash
https://huggingface.co/spaces/ishanthathsara/SynCMRIApp
```

### Pretrained Models 



---

## Citation

If you use SynCMRI in your research, please cite:

```bibtex
@article{TBA,
  title={Balancing Fidelity, Utility, and Privacy in Synthetic Cardiac MRI Generation: A Comparative Study},
  author={...},
  year={2025}
}
```

---

## License

See the repository for license information.
