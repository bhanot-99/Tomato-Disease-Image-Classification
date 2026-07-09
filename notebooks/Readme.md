# 🍅 Tomato Disease Image Classification
### A Comparative Study of Color, Segmented, and Class-Aware Selective Mixed Training Strategies using MobileNetV2

> **Lab Oriented Project — CS183**  
> Chitkara University, Himachal Pradesh  
> Student ID: 2211981181

---

## 📌 Overview

This project investigates how **image preprocessing domain** affects deep learning performance for tomato leaf disease classification. Using **MobileNetV2** transfer learning on the **PlantVillage dataset** (10 classes, ~9,300+ images), five distinct training strategies are systematically compared — culminating in a novel **class-aware selective mixing** approach (Strategy E) that routes each disease class to its optimal image domain based on per-class accuracy analysis.

> **Research Hypothesis:** Different tomato diseases rely on different visual cues (color contrast vs. texture/shape). A smart, class-aware mixing strategy that trains each class on its optimal image domain should outperform any single uniform strategy.

---

## 🎯 Key Results (Completed Strategies)

| Strategy | Description | Test Accuracy |
|----------|-------------|:-------------:|
| **A** | Color images (baseline) | **89.06%** ✅ |
| **B** | Segmented images | **86.13%** ✅ |
| **C** | Random 50/50 mix | **87.54%** ✅ |
| **D** | Fine-tune A on segmented (lr=1e-5) | ⏳ Planned |
| **E** | Class-aware selective mixing (original) | ⏳ Planned |

> Strategy A (Color) leads overall, but no single strategy dominates every class — which is exactly what motivates the class-aware approach in Strategy E.

### Per-Class Accuracy — A vs. B vs. C

| Disease Class | A (Color) | B (Segmented) | C (Mixed) | C−A | C−B | 🏆 Best |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Bacterial Spot | 99.3% | 78.7% | 90.0% | −9.3% | +11.3% | **A** 🔵 |
| Early Blight | 82.0% | 74.7% | 62.0% | −20.0% | −12.7% | **A** 🔵 |
| Late Blight | 89.3% | 92.0% | **94.7%** | +5.4% | +2.7% | **C** 🟢 |
| Leaf Mold | **92.3%** | 79.0% | 92.0% | −0.3% | +13.0% | **A** 🔵 |
| Septoria Leaf Spot | 85.3% | **90.7%** | 89.3% | +4.0% | −1.4% | **B** 🟠 |
| Spider Mites | 77.3% | **89.3%** | 88.7% | +11.4% | −0.6% | **B** 🟠 |
| Target Spot | 86.0% | 85.3% | **87.3%** | +1.3% | +2.0% | **C** 🟢 |
| Yellow Leaf Curl Virus | 95.3% | **96.7%** | 90.7% | −4.6% | −6.0% | **B** 🟠 |
| Mosaic Virus | 91.1% | 82.1% | **93.8%** | +2.6% | +11.7% | **C** 🟢 |
| Healthy | **94.0%** | 90.0% | 88.0% | −6.0% | −2.0% | **A** 🔵 |
| **OVERALL** | **89.1%** | 86.1% | 87.5% | — | — | **A** 🔵 |

**Class-level wins:** A wins 4/10 🔵 · B wins 3/10 🟠 · C wins 3/10 🟢

> No single strategy dominates — each wins different classes. This directly motivates **Strategy E**: class-aware routing where each disease is trained on its optimal domain.

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

**Dataset split:** 80% train / 10% validation / 10% test  
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

## 🧪 Training Strategies

### Strategy A — Color Images (Baseline)
Full-color PlantVillage images. Preserves all chromatic disease signatures.  
📌 **Result: 89.06% test accuracy**

### Strategy B — Segmented Images
Leaf region isolated via background segmentation. Removes soil/background noise; preserves shape and texture.  
📌 **Result: 86.13% test accuracy**

### Strategy C — Random 50/50 Mix
Training set composed of 50% color + 50% segmented images, randomly mixed. Tests whether exposure to both domains improves generalization without class-specific routing.  
📌 **Result: 87.54% test accuracy** — sits between A and B overall, wins 3 classes (Late Blight, Target Spot, Mosaic Virus). Confirms that naive mixing alone isn't sufficient; smart routing is needed.

### Strategy D — Fine-tune Strategy A on Segmented *(Planned)*
Takes the trained Strategy A model and continues training on segmented images at a very low learning rate (lr=1e-5, 10 epochs). Tests whether domain adaptation via fine-tuning is effective.

### Strategy E — Class-Aware Selective Mixing *(Planned — Original Contribution)*
The core research contribution. Each disease class is routed to the image domain (color or segmented) where Strategy A vs. B analysis shows it performs best:

- **Color domain** → Bacterial Spot, Early Blight, Leaf Mold, Healthy (A wins all four)
- **Segmented domain** → Septoria Leaf Spot, Spider Mites, Yellow Leaf Curl Virus (B wins all three)
- **Mixed domain** → Late Blight, Target Spot, Mosaic Virus (C wins all three)

This creates a curated, class-optimized training dataset that is neither uniformly color nor uniformly segmented.

---

## 🔬 Planned Analysis & Experiments

- [x] Strategy A — Color baseline training & evaluation
- [x] Strategy B — Segmented training & evaluation
- [x] A vs. B per-class comparison
- [x] Strategy C — Random mixed training & evaluation
- [x] A vs. B vs. C full comparison
- [ ] Strategy D — Fine-tuning experiment
- [ ] Strategy E — Class-aware selective mixing
- [ ] Grad-CAM visualizations (explainability)
- [ ] Severity estimation module
- [ ] Robustness testing (noise, blur, occlusion)

---

## ⚙️ Setup & Requirements

### Environment
- Python 3.11
- TensorFlow 2.10 (GPU)
- VS Code + Jupyter

### Installation

```bash
git clone https://github.com/bhanot-99/Tomato-Disease-Image-Classification.git
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

This project is for academic and research purposes under CS183 — Lab Oriented Project.

---

## 🙏 Acknowledgements

- [PlantVillage Dataset](https://plantvillage.psu.edu/) — Penn State University
- MobileNetV2 — Google Research (Howard et al., 2018)
- TensorFlow / Keras team
