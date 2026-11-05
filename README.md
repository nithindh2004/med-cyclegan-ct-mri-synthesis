# CycleGAN — MRI ↔ CT Medical Image Translation


## Project Overview

Medical imaging modalities such as MRI and CT provide complementary information about anatomical structures. This project explores **unpaired MRI-to-CT and CT-to-MRI image translation** using a CycleGAN-based deep learning framework.

The implementation investigates a frequency-aware, CBAM-augmented CycleGAN architecture with spectral-normalized discriminators to improve image translation between the two domains without requiring strictly paired training images.

The framework is developed using PyTorch and includes model training, dataset preparation, image generation, and quantitative evaluation.

## Key Technical Highlights

- **Unpaired Image-to-Image Translation:** Uses CycleGAN to learn mappings between MRI and CT image domains without requiring paired training samples.
- **Attention-Augmented Architecture:** Incorporates CBAM-based attention mechanisms into the CycleGAN framework.
- **Spectral Normalization:** Uses spectral-normalized discriminators to support more stable adversarial training.
- **Cycle Consistency:** Employs cycle-consistency and identity losses to preserve important image characteristics.
- **Model Evaluation:** Evaluates generated images using MAE, PSNR, SSIM, and FID.
- **Training Optimization:** Incorporates computational optimization techniques, including PyTorch Automatic Mixed Precision (AMP).

## Experimental Results

The following metrics are reported in the project report, comparing the baseline CycleGAN with the proposed Med-CycleGAN implementation.

| Evaluation Metric | Baseline CycleGAN | Med-CycleGAN |
|:---|---:|---:|
| PSNR (↑) | 27.4 | **29.5** |
| SSIM (↑) | 0.731 | **0.781** |
| FID (↓) | 85.2 | **62.4** |
| MAE (↓) | 0.058 | **0.041** |

*Note: These values are taken from the project report and have not been independently reproduced.*

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **Computer Vision:** Torchvision, Pillow
- **Numerical Computing:** NumPy
- **Medical Image Handling:** pydicom (optional DICOM support)
- **Development Environment:** Google Colab / Jupyter Notebook

## Project Documentation

- [Detailed Project Report](docs/Project_Report.pdf)
- [Project Presentation](docs/Project_Presentation.pdf)

---

### Model Architecture

### Generator: ResNet-based
```
c7s1-64 → d128 → d256 → [R256 × 9] → u128 → u64 → c7s1-1
  ↑ Encoder ↑           ↑ Transformer ↑    ↑ Decoder ↑
```
- **c7s1-k**: 7×7 Conv-InstanceNorm-ReLU, k filters, stride 1
- **dk**: 3×3 Conv-InstanceNorm-ReLU, k filters, stride 2 (downsample)
- **Rk**: Residual block, two 3×3 convolutions
- **uk**: 3×3 fractional-stride Conv-InstanceNorm-ReLU (upsample)

### Discriminator: PatchGAN (70×70)
```
C64 → C128 → C256 → C512 → Conv(1)
```
- **Ck**: 4×4 Conv-InstanceNorm-LeakyReLU, stride 2
- No InstanceNorm on C64 (first layer)
- Output: probability map (each patch = real or fake)

### Training Objectives and Loss Functions
```
L_total = L_GAN(G, Dy) + L_GAN(F, Dx) + λ·L_cyc(G,F) + 0.5λ·L_identity(G,F)
```
- **Adversarial loss** (LSGAN variant for stability)
- **Cycle consistency loss**: F(G(x)) ≈ x  and  G(F(y)) ≈ y
- **Identity loss**: G(y) ≈ y  and  F(x) ≈ x

---

## Project Structure

```text
cyclegan_mri_ct/
│
├── train.py          ← Main training script
├── test.py           ← Run inference on new images
├── evaluate.py       ← Compute MAE / PSNR / SSIM / FID
├── prepare_data.py   ← Download & organize datasets
├── models.py         ← Generator + Discriminator architectures
├── dataset.py        ← PyTorch Dataset (MRI/CT/DICOM support)
├── utils.py          ← ReplayBuffer, LR scheduler, checkpoints
├── requirements.txt  ← Python dependencies
│
├── data/
│   ├── trainA/       ← MRI training images
│   ├── trainB/       ← CT training images
│   ├── testA/        ← MRI test images
│   └── testB/        ← CT test images
│
├── checkpoints/      ← Saved model checkpoints
└── outputs/          ← Generated images & test results
    └── samples/      ← Training samples saved every N epochs
```

## Kaggle Datasets

| Dataset | Link | Domain |
|---------|------|--------|
| Brain MRI | [Brain MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | MRI (trainA) |
| RSNA CT Scans | [SIIM Medical Images](https://www.kaggle.com/datasets/kmader/siim-medical-images) | CT (trainB) |
| MRI+CT Combined | [Medical Imaging Datasets](https://www.kaggle.com/datasets/arjunbasandrai/medical-imaging-datasets) | Both |

## Individual Translation Examples

**Input MRI**

![Input MRI](assets/paired_input_mri.png)

**Ground Truth CT**

![Ground Truth CT](assets/paired_ground_truth_ct.png)

**Generated CT**

![Generated CT](assets/paired_generated_ct.png)

---
*Original paper: Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks", ICCV 2017*
