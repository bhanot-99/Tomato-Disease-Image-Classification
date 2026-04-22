# 🍅 Tomato Disease Image Classification
### A Comparative Study of Color, Segmented, and Class-Aware Selective Mixed Training Strategies using MobileNetV2
---

## 📌 Overview

This project investigates how **image preprocessing domain** affects deep learning performance for tomato leaf disease classification. Using **MobileNetV2** transfer learning on the **PlantVillage dataset** (10 classes, ~9,300+ images), five distinct training strategies were systematically compared. This culminates in a novel **class-aware selective mixing** approach (Strategy E) that routes each disease class to its optimal image domain based on per-class accuracy analysis. 

> **Research Hypothesis:** Different tomato diseases rely on different visual cues (color contrast vs. texture/shape). A smart, class-aware mixing strategy that trains each class on its optimal image domain should outperform any single uniform strategy.

---

## 🎯 Key Results 

All five training strategies have been completed. Strategy E emerged as the definitive winner, achieving a deterministic **93.88% accuracy** and outperforming all uniform baselines.

| Strategy | Description | Test Accuracy | Status |
|----------|-------------|:-------------:|:------:|
| **A** | Color images (baseline) | **89.06%** | ✅ |
| **B** | Segmented images | **86.13%** | ✅ |
| **C** | Random 50/50 mix | **87.48%** | ✅ |
| **D** | Fine-tune A on segmented (lr=1e-5) | **88.42%** | ✅ |
| **E** | Class-aware selective mixing | **93.88%** | ✅ 🏆 |

### Per-Class Accuracy Head-to-Head

| Disease Class | A (Color) | B (Segmented) | C (Mixed) | D (Fine-Tuned) | ★ E (Selective) | 🏆 Winner |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Bacterial Spot | **99.3%** | 78.7% | 90.0% | 88.7% | 98.0% | **A** 🔵 |
| Early Blight | 82.0% | 74.7% | 62.0% | 74.7% | **82.7%** | **E** ⭐ |
| Late Blight | 89.3% | 92.0% | **94.7%** | 91.3% | 88.0% | **C** 🟢 |
| Leaf Mold | 92.3% | 79.0% | 92.0% | 92.3% | **95.1%** | **E** ⭐ |
| Septoria Leaf Spot | 85.3% | 90.7% | 89.3% | 89.3% | **97.3%** | **E** ⭐ |
| Spider Mites | 77.3% | 89.3% | 88.7% | 85.3% | **96.7%** | **E** ⭐ |
| Target Spot | 86.0% | 85.3% | 87.3% | 76.7% | **91.3%** | **E** ⭐ |
| Yellow Leaf Curl Virus | 95.3% | **96.7%** | 90.7% | 96.0% | **96.7%** | **E** ⭐ |
| Mosaic Virus | 91.1% | 82.1% | 93.8% | 96.4% | **97.3%** | **E** ⭐ |
| Healthy | 94.0% | 90.0% | 88.0% | **98.7%** | 96.7% | **D** 🟣 |
| **OVERALL** | 89.06% | 86.13% | 87.48% | 88.42% | **93.88%** | **E** ⭐ |

**Class-level wins:** Strategy E dominates with **7 out of 10** class wins. Strategies A, C, and D each won a single class, while Strategy B won none outright.

---

## 🗂️ Repository Structure
```
TomatoClassification/
│
├── dataset/
│   ├── processed/              # Color images — 9,325 images
│   ├── processed_segmented/    # Segmented images — 9,325 images
│   └── processed_mixed/        # Mixed images — 9,746 images
│
├── models/
│   ├── best_model.h5           # Strategy A — best model
│   └── best_model_B_segmented.h5  # Strategy B — best model
│
├── notebooks/
│   ├── 01_preprocessing.ipynb              # Color image preprocessing
│   ├── 01b_preprocessing_segmented.ipynb   # Segmented image preprocessing
│   ├── 01c_preprocessing_mixed.ipynb       # Mixed dataset preprocessing
│   ├── 02_model_training.ipynb             # Strategy A training
│   ├── 03_evaluation.ipynb                 # Strategy A evaluation
│   ├── 03b_evaluation_strategy_B.ipynb     # A vs. B full comparison
│   ├── 04_training_strategy_B.ipynb        # Strategy B training
│   ├── 05_training_strategy_C.ipynb        # Strategy C training
│   └── 03_evaluation_mixed.ipynb           # A vs. B vs. C full comparison
│
├── src/
│   └── model.py                # Model architecture scaffold
│
└── README.md
```

---

## 🌿 Dataset — PlantVillage (10 Tomato Classes)

| # | Class |
|---|-------|
| 0 | Tomato Bacterial Spot |
| 1 | Tomato Early Blight |
| 2 | Tomato Late Blight |
| 3 | Tomato Leaf Mold |
| 4 | Tomato Septoria Leaf Spot |
| 5 | Tomato Spider Mites (Two-spotted) |
| 6 | Tomato Target Spot |
| 7 | Tomato Yellow Leaf Curl Virus |
| 8 | Tomato Mosaic Virus |
| 9 | Tomato Healthy |

**Dataset split:** 75% train / 15% validation / 15% test  
**Image size:** 224×224 (MobileNetV2 input)  
**Augmentation:** Horizontal/vertical flip, rotation, zoom, brightness shift

---

## 🏗️ Model Architecture — MobileNetV2 Transfer Learning

```
Input (224×224×3)
     ↓
MobileNetV2 (ImageNet pretrained, frozen base)
     ↓
Global Average Pooling
     ↓
Dense(256, ReLU) + Dropout(0.3)
     ↓
Dense(10, Softmax)
```

**Training config:**
- Optimizer: Adam (lr=1e-4)
- Loss: Categorical Cross-Entropy
- Callback: EarlyStopping + ModelCheckpoint
- Fine-tune phase (Strategy D): lr=1e-5, top layers unfrozen, 10 epochs

---

**Training config:**
- Optimizer: Adam (lr=1e-4)
- Loss: Categorical Cross-Entropy
- Callback: EarlyStopping + ModelCheckpoint

---

## 🧪 Training Strategies

### Strategy A — Color Images (Baseline)
Full-color PlantVillage images. Preserves all chromatic disease signatures.  
📌 **Result: 89.06% test accuracy** (Loss: 0.3366, F1: 0.893). Color context proved to be the strongest global signal.

### Strategy B — Segmented Images
Leaf region isolated via background segmentation. Removes soil/background noise; preserves shape and texture.  
📌 **Result: 86.13% test accuracy** (Loss: 0.4064, F1: 0.864). Double-edged sword: improved Spider Mites texture recognition but caused a 20.6% collapse in Bacterial Spot by destroying color-contrast signals.

### Strategy C — Random 50/50 Mix
Training set composed of 50% color + 50% segmented images, randomly mixed.  
📌 **Result: 87.48% test accuracy** (Loss: 0.3631, F1: 0.876). Failed to achieve the "best of both worlds." Conflicting representations confused the model, causing Early Blight accuracy to collapse from 82% to 62%.

### Strategy D — Fine-tune Strategy A on Segmented
Takes the trained Strategy A model and continues training on segmented images at a very low learning rate (lr=1e-5, 10 epochs).  
📌 **Result: 88.42% test accuracy** (Loss: 0.3688, F1: 0.890). Sequential domain adaptation showed minimal catastrophic forgetting (only -0.64% overall vs. A) and produced the best result for the 'Healthy' class (98.7%), but ultimately failed to outperform the baseline average.

### ★ Strategy E — Class-Aware Selective Mixing (Original Contribution)
The core research contribution. Each disease class was explicitly routed to the image domain (color or segmented) where Strategy A vs. B analysis showed it performed best:

- **Color domain** → Bacterial Spot, Early Blight, Leaf Mold, Healthy
- **Segmented domain** → Septoria Leaf Spot, Spider Mites, Yellow Leaf Curl Virus
- **Mixed domain** → Late Blight, Target Spot, Mosaic Virus

📌 **Result: 93.88% test accuracy** (Loss: 0.1880, F1: 0.940). 
This curated, class-optimized dataset provided the definitive best result, yielding a **+4.82% improvement** over the standard baseline. The approach proved 100% reproducible across 3 independent runs with a fixed seed ($\sigma = 0$).

---

## 🔬 Analysis & Experiments Tracker

- [x] Strategy A — Color baseline training & evaluation
- [x] Strategy B — Segmented training & evaluation
- [x] A vs. B per-class comparison
- [x] Strategy C — Random mixed training & evaluation
- [x] Strategy D — Fine-tuning experiment
- [x] Strategy E — Class-aware selective mixing
- [x] Final comparative evaluation of all 5 strategies
- [ ] Grad-CAM visualizations (explainability module)
- [ ] Severity estimation regression module
- [ ] Robustness testing (noise, blur, occlusion)

---

## ⚙️ Setup & Requirements

### Environment
- Python 3.11
- TensorFlow 2.10 (GPU)
- VS Code + Jupyter

### Installation

```bash
git clone [https://github.com/bhanot-99/Tomato-Disease-Image-Classification.git](https://github.com/bhanot-99/Tomato-Disease-Image-Classification.git)
cd Tomato-Disease-Image-Classification
pip install -r requirements.txt
```

### Requirements

```
tensorflow==2.10
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python
Pillow
jupyter
```

### GPU (Windows)
Ensure CUDA 11.2 + cuDNN 8.1 are installed to match TensorFlow 2.10.

---

## 🚀 Running the Notebooks

Run in order:

```
01_preprocessing.ipynb              → Prepare color dataset
01b_preprocessing_segmented.ipynb   → Prepare segmented dataset
01c_preprocessing_mixed.ipynb       → Prepare mixed dataset
02_model_training.ipynb             → Train Strategy A
03_evaluation.ipynb                 → Evaluate Strategy A
04_training_strategy_B.ipynb        → Train Strategy B
03b_evaluation_strategy_B.ipynb     → Compare A vs B
05_training_strategy_C.ipynb        → Train Strategy C
03_evaluation_mixed.ipynb           → Evaluate Strategy C (A vs B vs C)
```

---

## 📊 Visualizations

Training curves, confusion matrices, and per-class accuracy comparisons are generated within the evaluation notebooks. Grad-CAM saliency maps will be added in a future update to provide model explainability.

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

## 📜 License

This project is for research purpose

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://plantvillage.psu.edu/) — Penn State University
- MobileNetV2 — Google Research (Howard et al., 2018)
- TensorFlow / Keras team
