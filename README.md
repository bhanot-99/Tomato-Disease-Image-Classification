# 🍅 Tomato Disease Image Classification
### A Comparative Study of Color, Segmented, and Class-Aware Selective Mixed Training Strategies using MobileNetV2
---

## 📌 Overview

This project investigates how **image preprocessing domain** affects deep learning performance for tomato leaf disease classification. Using **MobileNetV2** transfer learning on the **PlantVillage dataset** (10 classes, ~9,300+ images), five distinct training strategies were systematically compared. This culminates in a novel **class-aware selective mixing** approach (Strategy E) that routes each disease class to its optimal image domain based on per-class accuracy analysis. 

> **Research Hypothesis:** Different tomato diseases rely on different visual cues (color contrast vs. texture/shape). A smart, class-aware mixing strategy that trains each class on its optimal image domain should outperform any single uniform strategy.

Additionally, the project explores robust **fine-tuning** on the New Plant Diseases Dataset (NPDD), and validates against field-captured imagery — where performance collapses to near chance. See `FINDINGS.md` section 3 and `outputs/field_validation/FIELD_VALIDATION_REPORT.md`.

---

## 🎯 Key Results (Internal Dataset)

All five training strategies were evaluated on the internal PlantVillage dataset. Strategy E emerged as the definitive winner, achieving **93.88% accuracy** and outperforming all uniform baselines.

| Strategy | Description | Test Accuracy | Status |
|----------|-------------|:-------------:|:------:|
| **A** | Color images (baseline) | **89.06%** | ✅ |
| **B** | Segmented images | **86.13%** | ✅ |
| **C** | Random 50/50 mix | **87.48%** | ✅ |
| **D** | Fine-tune A on segmented (lr=1e-5) | **88.42%** | ✅ |
| **E** | Class-aware selective mixing | **93.88%** | ✅ 🏆 |

### ★ Strategy E — Class-Aware Selective Mixing (Original Contribution)
The core research contribution. Each disease class was explicitly routed to the image domain (color or segmented) where Strategy A vs. B analysis showed it performed best.
- **Color domain** → Bacterial Spot, Early Blight, Leaf Mold, Healthy
- **Segmented domain** → Septoria Leaf Spot, Spider Mites, Yellow Leaf Curl Virus
- **Mixed domain** → Late Blight, Target Spot, Mosaic Virus

---

## 🌍 Real-World Generalization & Fine-Tuning

### Cross-Dataset Validation
Models trained on controlled lab backgrounds degrade in complex natural environments. Four candidate field datasets were audited; three were rejected on data-quality grounds (PlantDoc, Tomato-Village Variant-a and Variant-c). **PlantWild** (1,966 tomato images, 8 classes) was evaluated:

| Strategy | Internal | PlantWild (field) | Gap |
|---|---|---|---|
| A | 89.21% | 17.19% | −72.0 |
| B | 88.42% | 12.11% | −76.3 |
| C | 87.42% | 14.45% | −73.0 |
| D | 90.92% | 16.63% | −74.3 |
| **E** | **93.71%** | **12.46%** | **−81.2** |

Chance is 12.5%, and top-3 accuracy is also at chance. **Strategy E's internal advantage does not transfer** — it has the largest gap of the five and finishes second-worst. The absolute figures are a lower bound rather than a measurement, since 20–40% of three classes show lesions on fruit rather than foliage; the comparison *between* strategies is unaffected, as all five were scored on identical images. Full detail and the dataset audits: `outputs/field_validation/FIELD_VALIDATION_REPORT.md`.

### Fine-Tuning on New Plant Diseases Dataset (NPDD)
To address the domain shift, **Strategy E was fine-tuned** on the New Plant Diseases Dataset (18,345 training images, offering better class balance and lighting variations).
- **Configuration:** Unfroze the last 30 MobileNetV2 layers, learning rate `1e-5`, label smoothing `0.1`.
- **Validation Accuracy:** **94.11%** (+0.23pp over original Strategy E), best of 10 epochs.
- **NPDD test folder:** 16/16 correct — but this is 16 images covering only 3 of 10 classes, from NPDD's own test split, **not** field imagery. It is not a reportable result.
- **Improved Confidence Calibration:** Label smoothing substantially lowers confidence on out-of-domain images (~86% → ~57%) without improving accuracy — the model becomes appropriately uncertain rather than confidently wrong.

> **Correction (2026-08-22).** This section previously claimed 97.21% validation at `lr=1e-4` and "93.75% on real-world field test images". Those figures came from a run that does not exist in this repository, and the 16-image set is not field imagery. See `PAPER_REVIEW_NOTES.md` Issue 3 and `DOC_AUDIT.md`.

---

## 🛠️ Critical Preprocessing Fixes
During development, significant improvements were implemented to properly align with MobileNetV2 requirements:
- **Input Scaling:** Migrated from basic `rescale=1./255` `[0, 1]` to MobileNetV2's native `preprocess_input()` mapping pixels to `[-1, 1]`.
- **Label Smoothing:** Applied `0.1` label smoothing to the loss function to prevent logit saturation and overconfident mispredictions.

---

## 🗂️ Repository Structure
```
TomatoClassification/
├── dataset/                    # Original PlantVillage split
├── validation/                 # NPDD dataset
├── models/                     # Saved .h5 models (Strategy A-E & Finetuned)
├── notebooks/                  
│   ├── 01-05_*                 # Training for Strategies A-E
│   ├── 06_gradcam.ipynb        # GradCAM visualization
│   ├── 07_severity_estimator.ipynb # Disease severity estimation
│   └── 09_finetune_and_test_new_plant_diseases.ipynb
├── outputs/                    # Visualizations, charts, and CSV results
├── src/                        # Core Python modules (model, train, evaluate, gradcam)
├── project_technical_report.txt # Full narrative research report
└── README.md
```

---

## 🌿 Dataset — PlantVillage (10 Tomato Classes)

| # | Class | # | Class |
|---|---|---|---|
| 0 | Bacterial Spot | 5 | Spider Mites (Two-spotted) |
| 1 | Early Blight | 6 | Target Spot |
| 2 | Late Blight | 7 | Yellow Leaf Curl Virus |
| 3 | Leaf Mold | 8 | Mosaic Virus |
| 4 | Septoria Leaf Spot | 9 | Healthy |

---

## 🔬 Analysis & Experiments Tracker

- [x] Strategy A — Color baseline training & evaluation
- [x] Strategy B — Segmented training & evaluation
- [x] Strategy C — Random mixed training & evaluation
- [x] Strategy D — Fine-tuning experiment
- [x] Strategy E — Class-aware selective mixing
- [x] Implementation of proper MobileNetV2 Preprocessing & Label Smoothing
- [x] **Grad-CAM visualizations** (explainability module)
- [x] **Severity estimation** regression module
- [x] **Cross-dataset validation** (4 datasets audited, PlantWild evaluated)
- [x] **Fine-tuning on diverse data** (NPDD)

---

## ⚙️ Setup & Requirements

### Environment
- Python 3.11
- TensorFlow 2.16.2
- VS Code + Jupyter

### Installation

```bash
git clone https://github.com/bhanot-99/Tomato-Disease-Image-Classification.git
cd Tomato-Disease-Image-Classification
pip install -r requirements.txt
```

---

## 📄 Research Publication (Target)

> **Title:** *"Class-Aware Domain Mixing for Robust Tomato Disease Classification: A Comparative Study of Color, Segmented and Selective Mixed Training Strategies"*

**Target Venues:**
- Computers and Electronics in Agriculture *(Elsevier)*
- IEEE Access
- Applied Artificial Intelligence *(Taylor & Francis)*

---

## 👤 Author

**Jatin Bhanot**  
B.E. Computer Science & Engineering  
Chitkara University, Himachal Pradesh  
Student ID: 2211981181  

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://plantvillage.psu.edu/) — Penn State University
- MobileNetV2 — Google Research (Howard et al., 2018)
- TensorFlow / Keras team
