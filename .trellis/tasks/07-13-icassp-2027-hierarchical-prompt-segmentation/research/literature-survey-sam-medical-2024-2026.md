# Literature Survey: SAM-Based Medical Image Segmentation (2024–2026)

- Scope: ICASSP 2027 submission related work
- Date: 2026-07-13

---

## 1. Foundation & Predecessor

### SAM-Med2D (2024)
- **Authors**: Junlong Cheng, Jin Ye, et al. (Sichuan University / Shanghai AI Lab)
- **Key**: Base model used in this project. 4.6M images, 19.7M masks, comprehensive fine-tuning with box/point/mask prompts.
- **Role**: Method §2.1 — core backbone citation.
- **Local PDF**: `report/论文/medsam/sam-med2d.pdf`

### MedSAM (2024, Nature Communications)
- **Authors**: Jun Ma, Yuting He, Bo Wang, et al.
- **DOI**: `10.1038/s41467-024-44824-z`
- **Key**: Original SAM→medical adaptation. 1.57M image-mask pairs, 10 modalities, 30+ cancer types.
- **Role**: Introduction — SAM medicalization milestone.
- **Local PDF**: `report/论文/medsam/Segment anything in medical images.pdf`

### Self-Prompt / IJCNN 2025 (published as "YOLO-Driven Prompt Generation for SAM-Based Brain Tumor Segmentation")
- **Authors**: Bolin Lv, Ming Yuan
- **DOI**: `10.1109/IJCNN64981.2025.11228325`
- **Venue**: IJCNN 2025 (CCF-C), Rome, June 30 – July 5, 2025
- **Key**: YOLO + 3-point prompts + Adapter + label erosion/dilation. 180-slice 2D study, TC-only binary segmentation. No oracle/automatic prompt decomposition, no hierarchy supervision, no patient-level 3D evaluation.
- **Role**: Related Work — direct predecessor; this work extends it with multi-class 3D evaluation, PJT, and hierarchy loss.
- **Local PDF**: `report/论文/medsam/Self-Prompt Segmentation Model for Brain Tumors.pdf`
- **Gap audit**: `research/ijcnn2025-predecessor-gap.md`

---

## 2. Prompt Robustness (Justification for PJT)

### PP-SAM: Perturbed Prompts for Robust Adaptation (CVPR 2024 Workshop)
- **Authors**: Md Mostafijur Rahman, Mustafa Munir, Debesh Jha, Ulas Bagci, Radu Marculescu (UT Austin / Northwestern)
- **Venue**: CVPRW 2024 (DEF-AI-MIA)
- **Key Findings**:
  - Zero-shot SAM is highly vulnerable to bounding box perturbations.
  - Variable perturbation (0–50px random) during fine-tuning superior to fixed perturbation.
  - 1-shot PP-SAM boosts Dice by 20% and 37% with 50px/100px perturbations.
  - Best strategy: freeze mask decoder, fine-tune image + prompt encoders.
- **Role**: Primary citation for PJT design rationale.
- **Code**: `github.com/SLDGroup/PP-SAM`

### Optimization of MedSAM Based on Bounding Box Adaptive Perturbation (ICIC 2025)
- **Authors**: Boyi Li, Ye Yuan, Wenjun Tan (Northeastern University, China)
- **Key**: Bounding box adaptive perturbation algorithm for MedSAM. Addresses segmentation errors when boxes are shrunk or misaligned.
- **Role**: Supporting citation — MedSAM-specific perturbation work.

### MedSAM Guider: Lightweight Module to Mitigate Prompt Sensitivity (2025)
- **Authors**: Muhammad Nouman, Ghada Khoriba, Essam A. Rashed
- **Key**: LoRA + Squeeze-and-Excitation plug-in module. Tunes <0.5% of MedSAM's 93.7M parameters. Tested with standard, rotated, and contour prompt types.
- **Role**: Related Work — alternative approach to prompt robustness.

---

## 3. Automatic Prompt Generation (Justification for YOLO + Class-Specific Prompts)

### ConfMamba-SAM: Memory-Augmented Prompting for Brain Lesion Segmentation (2026)
- **Venue**: IEEE
- **Key**: Memory-driven prompt generator with learnable prototype banks across adjacent slices. State space model encoder.
- **Role**: Related Work — latest automatic prompt approach for brain lesions.

### Sub-Region-Aware Modality Fusion and Adaptive Prompting (Jan 2026)
- **arXiv**: `2601.15734`
- **Key**: Per-sub-region modality weighting + adaptive sub-region-specific prompts. Dice 0.901 on BraTS 2020, ET improvement 9.8% over nnU-Net.
- **Role**: Strong support for class-specific (per-sub-region) prompting approach.

### SAM-PEFT: Parameter-Efficient Fine-Tuning with Prompt-Free Segmentation (2026)
- **Venue**: IEEE
- **Key**: Lightweight Prompt Encoder (LPE) with anatomical positional encoding for automatic internal prompt generation. LoRA adapters (~5.8% params). Dice 0.915 on BraTS 2025.
- **Role**: Related Work — prompt-free direction, PEFT comparison.

---

## 4. Hierarchy & Multi-Task Learning (Justification for Hierarchy Loss)

### PCUM: Prompt Collaboration with Uncertainty Modeling (2026)
- **Key**: Dual-prompting from confident + uncertain regions. Dice 73.12% with only 10 labeled cases on BraTS 2025. Semi-supervised.
- **Role**: Related Work — uncertainty-aware multi-region segmentation.

### MTL-SAM3D: Multi-Task Learning 3D SAM for Glioma Segmentation and IDH Genotyping (2025)
- **Venue**: IEEE
- **Key**: LoRA fine-tuning on query/value projections. Zero-shot cross-domain transfer (Dice 0.91→0.82). Joint segmentation + genotype classification.
- **Role**: Related Work — multi-task learning with SAM for glioma.

---

## 5. SAM2 / MedSAM2 Evolution (for Discussion section)

### Medical SAM 2 (MedSAM-2) (2024)
- **Authors**: Jiayuan Zhu, et al. (University of Oxford)
- **arXiv**: `2408.00874`
- **Key**: Treats 2D/3D medical segmentation as video object tracking via SAM2 pipeline. Self-sorting memory bank. Evaluated on 5 2D + 9 3D tasks.
- **Role**: Discussion — future 3D extension direction.
- **Local PDF**: `report/论文/medsam/2408.00874v2.pdf`

### Comparative Evaluation of SAM/SAM2/MedSAM/MedSAM2 (2025)
- **Venue**: J. Imaging Informatics in Medicine
- **Key**: SAM (96.08% Dice) outperformed SAM2 (84.98%) on lung CT with box prompts. SAM2 not always better than v1 for medical tasks.
- **Role**: Discussion — justifies staying with SAM-Med2D v1 for this work.

---

## 6. Foundation Model vs. Specialized Model Debate

### Is Segmentation Solved? Vision Foundation Models for Head and Neck Tumor Segmentation (2026)
- **Venue**: Physics in Medicine & Biology
- **Key**: Larger bounding boxes → notable performance degradation. MedSAM better for primary tumors; SAM better for lymph nodes. Conventional CNNs remain more stable and clinically reliable.
- **Role**: Discussion — supports "parameter-efficient prompt robustness" positioning over SOTA claims.

### Necessity and Impact of Specialization of Large Foundation Models (2025)
- **Venue**: Medical Physics
- **Key**: Out-of-box MedSAM inferior to nnU-Net by 10-20% Dice. After specialized fine-tuning, LiteMedSAM matches nnU-Net. "Specialized fine tuning is necessary to make large foundation models clinically relevant."
- **Role**: Introduction/Related Work — justifies fine-tuning approach.

---

## 7. Additional Resources

### MedSAM-U: Uncertainty-Guided Auto Multi-Prompt Adaptation (2024)
- **Authors**: Nan Zhou, Ke Zou, et al.
- **Key**: Uncertainty-guided framework for auto-refining multi-prompt inputs. 1.7-20.5% improvement across modalities.
- **Local PDF**: `report/论文/medsam/MedSAM-U Uncertainty-Guided Auto.pdf`

### U-MedSAM (CVPR 2024 Challenge)
- **Key**: Uncertainty-aware loss + Sharpness-Aware Minimization. CVPR 2024 MedSAM on Laptop challenge entry.
- **Local PDF**: `report/论文/medsam/Challenge Summary U-MedSAM.pdf`

### Brain Tumor Segmentation Survey (fbioe-12-1392807, 2024)
- **Key**: Recent DL-based brain tumor segmentation survey using multi-modal MRI. Covers CNN, ViT, and hybrid architectures.
- **Local PDF**: `report/论文/medsam/fbioe-12-1392807.pdf`

---

## Paper Citation Map

| Paper Section | Primary Citations |
|---|---|
| **Introduction** | MedSAM (2024), Medical Physics specialization study (2025), IJCNN 2025 predecessor |
| **Related Work — SAM medicalization** | MedSAM → SAM-Med2D → MedSAM-2 |
| **Related Work — Prompt robustness** | PP-SAM (CVPRW 2024), MedSAM adaptive perturbation (ICIC 2025) |
| **Related Work — Automatic prompting** | ConfMamba-SAM (2026), Sub-Region-Aware Prompting (2026), SAM-PEFT (2026) |
| **Method — PJT justification** | PP-SAM (CVPRW 2024) |
| **Method — Hierarchy loss** | PCUM (2026), MTL-SAM3D (2025) |
| **Discussion — SAM2 comparison** | MedSAM-2 (2024), SAM vs SAM2 comparative study (2025) |
| **Discussion — Positioning** | "Is Segmentation Solved?" (2026), Medical Physics specialization study (2025) |
