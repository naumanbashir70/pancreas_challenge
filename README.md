# wift-nnU-Net: A Unified and Accelerated Framework for Pancreatic Lesion Segmentation and Classification

> 3D CT pipeline that segments the pancreas and lesions, and predicts lesion subtype with a shared encoder and task-specific heads. Built on **nnU-Net v2** with a 3D ResEncUNet-M backbone and a lightweight classification branch.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-brightgreen.svg)]()
[![nnU--Net v2](https://img.shields.io/badge/nnU--Net-v2-informational.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

---

## Highlights
- Multi-task: shared 3D encoder, dual heads for segmentation and classification.
- Fast inference: AMP, channels_last_3d, cuDNN benchmark, TF32 allowed.
- Reproducible: fixed preprocessing, plans, and evaluation scripts.
- No external pretraining or data.

## Dataset layout
```
data/
├─ train/
│  ├─ subtype0/ quiz_0_###_0000.nii.gz  image
│  │             quiz_0_###.nii.gz      mask (0 bg, 1 pancreas, 2 lesion)
│  ├─ subtype1/
│  └─ subtype2/
├─ validation/   (same structure as train)
└─ test/         quiz_###_0000.nii.gz   images only
```
Labels: 0 background, 1 normal pancreas, 2 pancreas lesion.

## Methods
- Backbone: nnU-Net v2 3D ResEncUNet-M, 6 stages, feature sizes [32, 64, 128, 256, 320, 320].
- Branching:
  - Segmentation head: standard nnU-Net decoder with deep supervision.
  - Classification head: global feature pooling on encoder stage 6 followed by MLP. Supports attention pooling.
- Preprocessing: CT intensity normalization, resampling to spacing `[2.0, 0.732421875, 0.732421875]` mm, patch size `[64, 128, 192]`.
- Augmentation: nnU-Net default 3D spatial and intensity transforms.
- Optimization: AdamW, cosine learning rate schedule, batch size 2, mixed precision, joint loss for seg + cls.
- Logging: Weights and Biases for losses and validation curves.

## Inference speed
The predictor enables three safe accelerations that keep accuracy stable:
1. Automatic mixed precision
2. 3D channels_last to pick faster cuDNN kernels
3. cuDNN benchmarking and TF32 fast paths

Observed on validation: ~1.3 s per case.

## Results (validation, 36 cases)
Whole pancreas (labels 1 | 2)
- Dice: **0.939 ± 0.030**
- MSD: **0.362 ± 0.320 mm**
- HD95: **2.098 ± 2.348 mm**
- NSD: **96.1 ± 3.6 %**

Lesion only (label 2)
- Dice: **0.667 ± 0.305**
- MSD: **3.914 ± 12.136 mm**
- HD95: **9.426 ± 16.540 mm**
- NSD: **70.6 ± 30.7 %**

Combined (macro over whole and lesion)
- Dice: **0.803 ± 0.255**
- MSD: **2.087 ± 8.584 mm**
- HD95: **5.657 ± 12.135 mm**
- NSD: **83.3 ± 25.2 %**



## Quick start
```bash
# 1) Install
conda create -y -n nnunetv2 python=3.11
conda activate nnunetv2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install nnunetv2 monai nibabel simpleitk tqdm wandb

# 2) Prepare data
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_results=/path/to/nnUNet_results
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed

# Place the dataset under $nnUNet_raw/Dataset701_MultiTaskSegClass

# 3) Train
cmd = "nnUNetv2_train 701 3d_fullres 0 -tr nnUNetTrainerWandBMTFix -p nnUNetResEncUNetMPlans"
subprocess.check_call(shlex.split(cmd), env=os.environ.copy())

 Validation and Inferencing code are highlighted in the notebook
```



## Citation
- Isensee et al. nnU-Net: a self configuring method for biomedical image segmentation. Nat Methods, 2021.
- Maier-Hein et al. Metrics Reloaded: recommendations for image analysis validation. Nat Methods, 2024.
- Cao et al. Large-scale pancreatic cancer detection via non-contrast CT and deep learning. Nat Med, 2023.

## License
MIT

---

_Last updated: 2025-11-06_
