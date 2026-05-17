# FairVision 🎯

> Detecting and Mitigating Bias in a CNN-Based Age Group Classification System Using FairFace

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Dataset](https://img.shields.io/badge/Dataset-FairFace-purple.svg)](https://github.com/joojs/fairface)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

---

## Overview

FairVision is a fairness-aware age group classification system built from scratch using PyTorch. The model predicts one of **nine age brackets** from a single face image and is audited for bias across **race** (7 groups) and **gender** (2 groups) using the FairFace dataset.

The project investigates the standard fairness–accuracy trade-off and implements two bias mitigation strategies to reduce demographic performance gaps without pre-trained backbone models.

---

## Age Classes

| Label | Range |
|-------|-------|
| 0 | 0–2 |
| 1 | 3–9 |
| 2 | 10–19 |
| 3 | 20–29 |
| 4 | 30–39 |
| 5 | 40–49 |
| 6 | 50–59 |
| 7 | 60–69 |
| 8 | 70+ |

---

## Dataset — FairFace (0.25 Configuration)

| Split | Samples | Purpose |
|-------|---------|---------|
| Internal Train | 78,070 | Parameter updates |
| Internal Val | 8,674 | Checkpoint selection |
| Test (held-out) | 10,954 | Final evaluation only |

- **Age imbalance ratio:** ~30× (rarest: 70+, 842 samples vs. most common: 20–29, 25,598 samples)
- **Gender:** Near-balanced — 53% Male / 47% Female
- **Race groups:** East Asian, Indian, Black, White, Middle Eastern, Latino/Hispanic, Southeast Asian

---

## Model Architecture

A custom VGG-inspired CNN trained **from scratch** (no pre-trained weights).

```
Input (160×160×3)
  → Block 1: Conv2d 3→32,  BN, ReLU ×2 + MaxPool  →  32×80×80
  → Block 2: Conv2d 32→64, BN, ReLU ×2 + MaxPool  →  64×40×40
  → Block 3: Conv2d 64→128, BN, ReLU ×2 + MaxPool → 128×20×20
  → Block 4: Conv2d 128→256, BN, ReLU ×2 + MaxPool → 256×10×10
  → Block 5: Conv2d 256→512, BN, ReLU ×2 + MaxPool → 512×5×5
  → GlobalAvgPool → FC(512→256) → Dropout(0.4) → FC(256→9)
Output: 9 logits
```

**Peak VRAM:** 251 MB — deliberately lightweight and GPU-efficient.

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimiser | AdamW |
| Learning Rate | 3×10⁻⁴ |
| Weight Decay | 1×10⁻⁴ |
| LR Scheduler | CosineAnnealingLR (T_max=30) |
| Epochs | 30 |
| Batch Size | 64 |
| Image Size | 160×160 |
| Loss (Baseline) | CrossEntropyLoss |

**Augmentation (train only):** random horizontal flip, rotation ±15°, colour jitter, random erasing (p=0.1).

**Inference:** Test-Time Augmentation (TTA) — 5 crops with horizontal flip averaged.

---

## Results Summary

### Overall Performance (Baseline)

| Metric | Value |
|--------|-------|
| Accuracy | 58.07% |
| F1 Weighted | 0.5705 |
| F1 Macro | 0.5490 |
| Precision (Weighted) | 0.5736 |
| Recall (Weighted) | 0.5807 |

### Model Comparison

| Model | Accuracy | Race Gap | Gender Gap | Worst Race Acc |
|-------|----------|----------|------------|----------------|
| Baseline | 58.07% | 0.0602 | 0.0035 | 0.5572 (Black) |
| **Mit1: Focal + Soft Weights** | **56.97%** | **0.0383** | 0.0046 | **0.5559 (White)** |
| Mit2: Sampler + Focal + SW | 53.60% | 0.0443 | 0.0006 | 0.5058 (Black) |

---

## Bias Mitigation Strategies

### Mitigation 1 — Focal Loss + Soft Class Weights ✅ **(Recommended)**
- Focal Loss (γ=2) down-weights easy examples, focusing learning on minority/hard classes
- Inverse-frequency soft weights (w_c = 1/√freq_c) amplify gradient signals from rare age classes
- **Race gap reduced by 36%** (0.0602 → 0.0383) at only −1.1 pp overall accuracy cost

### Mitigation 2 — Weighted Random Sampler + Focal Loss + Soft Weights
- Adds batch-level over-sampling via `WeightedRandomSampler`
- Three-layer intervention: data sampling + loss shaping + loss re-weighting
- Near-zero gender gap (0.0006) but higher accuracy cost (−4.47 pp) and worse worst-group absolute accuracy

---

## Deployment

The deployed model is **Mitigation 1**, served as a **Streamlit web application**.

**Features:**
- Upload any face image (JPG/PNG)
- Preprocessing identical to training (160×160, ImageNet normalisation)
- Top-3 predicted age classes with softmax confidence bars
- Responsible-use disclaimer and model limitations displayed

---

## Environment

```
GPU:     Tesla P100-PCIE-16GB (Kaggle)
PyTorch: 2.5.1+cu121
CUDA:    12.1
```

---

## Ethical Considerations & Limitations

- **58% accuracy** on a 9-class task is insufficient for high-stakes individual decisions
- Binary gender encoding excludes non-binary identities
- Race labels are self-reported approximations, not objective categories
- Model trained exclusively on FairFace — out-of-distribution faces may degrade accuracy
- No pre-trained backbone used (per assignment constraints); transfer learning could add 10–20+ pp
- Any real-world deployment must comply with applicable privacy regulations (GDPR, etc.) and obtain informed consent

> ⚠️ **This system should never serve as the sole basis for consequential decisions affecting individuals.**

---

## References

1. Karkkainen & Joo (2021). *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age.* WACV 2021.
2. Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017. arXiv:1708.02002
3. Paszke et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019.
4. Mehrabi et al. (2021). *A Survey on Bias and Fairness in Machine Learning.* ACM Computing Surveys.
5. Barocas, Hardt & Narayanan (2023). *Fairness and Machine Learning.* MIT Press.

---

## Author

**K.T. Imasha Dewmi**  
Institute of Software Engineering (IJSE) — Certified AI & ML Engineer  
Academic Year: 2025/2026
