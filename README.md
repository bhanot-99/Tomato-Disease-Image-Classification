# 🍅 Tomato Disease Image Classification
### A Comparative Study of Color, Segmented, and Class-Aware Selective Mixed Training Strategies using MobileNetV2
---

## 📌 Overview

This project investigates how **image preprocessing domain** affects deep learning performance for tomato leaf disease classification. Using **MobileNetV2** transfer learning on the **PlantVillage dataset** (10 classes, ~9,300+ images), five distinct training strategies were systematically compared. This culminates in a novel **class-aware selective mixing** approach (Strategy E) that routes each disease class to its optimal image domain based on per-class accuracy analysis. 

> **Research Hypothesis:** Different tomato diseases rely on different visual cues (color contrast vs. texture/shape). A smart, class-aware mixing strategy that trains each class on its optimal image domain should outperform any single uniform strategy.

Additionally, the project explores **cross-dataset generalization** using the PlantDoc dataset, and robust **fine-tuning** on the New Plant Diseases Dataset (NPDD) to bridge the domain gap between lab-controlled and real-field images.

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

### Cross-Dataset Validation (PlantDoc)
When testing our PlantVillage-trained Strategy E model on real-world field images from the **PlantDoc** dataset, accuracy dropped to **~24%**. This highlighted a severe **domain shift**: models trained purely on controlled lab backgrounds struggle in complex, natural environments.

### Fine-Tuning on New Plant Diseases Dataset (NPDD)
To address the domain shift, **Strategy E was fine-tuned** on the New Plant Diseases Dataset (18,345 training images, offering better class balance and lighting variations).
- **Configuration:** Unfroze the last 30 MobileNetV2 layers, learning rate `1e-4`, label smoothing `0.1`.
- **Validation Accuracy:** Jumped to **97.21%** (+3.33% over original Strategy E).
- **Unseen Field Test:** Achieved **93.75% accuracy** on entirely flat, real-world field test images.
- **Improved Confidence Calibration:** The model now correctly expresses lower confidence on ambiguous class pairs (like Early Blight vs. Septoria Leaf Spot) instead of being overconfident, thanks to Categorical Cross-Entropy with label smoothing.

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
├── validation/                 # PlantDoc and NPDD datasets
├── models/                     # Saved .h5 models (Strategy A-E & Finetuned)
├── notebooks/                  
│   ├── 01-05_*                 # Training for Strategies A-E
│   ├── 06_gradcam.ipynb        # GradCAM visualization
│   ├── 07_severity_estimator.ipynb # Disease severity estimation
│   ├── 08_cross_dataset_validation_plantdoc.ipynb
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
- [x] **Cross-dataset validation** (PlantDoc)
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
