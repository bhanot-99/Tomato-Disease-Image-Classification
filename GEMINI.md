# TOMATO DISEASE CLASSIFICATION — PROJECT MEMORY
# Read this file at the start of any new session on this project.
# Path: /media/bhanot/Main_Storage/Development/8th Sem Project/TomatoClassification/GEMINI.md
# Last updated: 2026-07-09

---

## Project Identity

- **What it is**: 8th semester research project on tomato leaf disease classification
- **Architecture**: MobileNetV2 (ImageNet pretrained) + custom head (GAP → Dense(128,relu) → Dropout(0.3) → Dense(10,softmax))
- **Framework**: TensorFlow/Keras 2.16.2, Python 3.11.15
- **Environment**: `myenv/` virtualenv in project root (activate with `source myenv/bin/activate`)
- **10 classes** (alphabetical = index order used by ImageDataGenerator):
  - `[0]` Bacterial Spot · `[1]` Early Blight · `[2]` Late Blight · `[3]` Leaf Mold
  - `[4]` Septoria Leaf Spot · `[5]` Spider Mites · `[6]` Target Spot
  - `[7]` Yellow Leaf Curl Virus · `[8]` Mosaic Virus · `[9]` Healthy

---

## Repository Layout

```
TomatoClassification/
├── dataset/processed/          ← Original PlantVillage split (train/val/test)
│   └── train/                  6,527 images, 10 classes (~700/class, Mosaic=261)
├── validation/
│   ├── PlantDoc-Dataset/       731 real-field images, 8 classes (no Spider Mites, no Target Spot)
│   └── New_Plant_Diseases_Dataset/
│       ├── train/              18,345 images, 10 classes (~1,700–1,960/class)
│       ├── valid/              4,585 images, 10 classes
│       └── test/test/          33 flat images (no subfolders), 16 are tomato
├── models/
│   ├── model_A_color_final.h5              11 MB  Strategy A (Color Only)
│   ├── model_B_segmented_final.h5          11 MB  Strategy B (Segmented Only)
│   ├── model_C_Mixed_final.h5              11 MB  Strategy C (50/50 Mix)
│   ├── model_D_finetuned_final.h5          23 MB  Strategy D (A→Seg fine-tune)
│   ├── model_E_selective_V3_final.h5       11 MB  Strategy E (Class-Aware Routing) ← MAIN MODEL
│   ├── model_E_finetuned_NPDD_best.h5      23 MB  Strategy E fine-tuned on NPDD (best ckpt)
│   └── model_E_finetuned_NPDD_final.h5     23 MB  Strategy E fine-tuned on NPDD (final)
├── notebooks/
│   ├── 01_model_training_color.ipynb       Strategy A training
│   ├── 02_model_training_segmented.ipynb   Strategy B training
│   ├── 03_model_training_mixed.ipynb       Strategy C training
│   ├── 04_model_training_finetuned.ipynb   Strategy D training
│   ├── 05_model_training_selective.ipynb   Strategy E training
│   ├── 06_gradcam.ipynb                    GradCAM visualization
│   ├── 07_severity_estimator.ipynb         Disease severity estimation
│   ├── 08_cross_dataset_validation_plantdoc.ipynb  PlantDoc cross-dataset test
│   └── 09_finetune_and_test_new_plant_diseases.ipynb  NPDD fine-tune + test
├── scripts/
│   ├── patch_notebooks.py      Applied preprocess_input + label_smoothing to nb 01-08
│   └── generate_nb09.py        Generated notebook 09 programmatically
├── outputs/
│   ├── cross_dataset_validation/   PlantDoc charts + CSV
│   └── npdd_validation/            NPDD fine-tune charts + CSV
├── src/                        dataset.py, model.py, train.py, evaluate.py, predict.py, gradcam.py, severity.py
├── project_technical_report.txt   556-line full narrative report (read this for full context)
└── myenv/                      Python virtualenv
```

---

## The 5 Strategies

| Strategy | Description | Internal Acc | PlantDoc Acc |
|---|---|---|---|
| A | Color-only training images | 89.06% | 22.98% |
| B | Segmented-only (background removed) | 86.13% | 24.21% |
| C | Random 50/50 color+segmented mix | 87.48% | 24.49% |
| D | Fine-tuned: Strategy A weights → segmented data | 88.42% | 21.61% |
| **E** | **Class-Aware Routing** (each class uses its best-performing domain) | **93.88%** | **24.35%** |

Strategy E is the **main research contribution** — each of the 10 disease classes is routed to whichever image type (color or segmented) it historically performed best on.

---

## Critical Issues Found & Fixed (July 2026)

### Issue 1 — Wrong MobileNetV2 Preprocessing (FIXED in notebooks 01-08)
**What was wrong**: All training notebooks used `rescale=1./255` → pixels in `[0, 1]`  
**What MobileNetV2 needs**: `preprocess_input()` → pixels in `[-1, 1]` via `(pixel/127.5) - 1.0`  
**Fix applied** (via `scripts/patch_notebooks.py`):
```python
# BEFORE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, ...)
val_datagen = ImageDataGenerator(rescale=1./255)

# AFTER
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, rotation_range=15, ...)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
```
**Also fixed in notebook 08 inference** (`arr / 255.0` → `preprocess_input(arr)`)  
⚠️ **Models A-E must be RETRAINED** for this fix to take effect. The `.h5` files still have old normalization.

### Issue 2 — Overconfident Predictions / Sharp Decision Boundaries (FIXED)
**What was wrong**: `loss='categorical_crossentropy'` → models were 84% confident on wrong answers  
**Fix applied** to all training notebooks:
```python
# BEFORE
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# AFTER
model.compile(
    optimizer=Adam(lr=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
```
Label smoothing (0.1) distributes 10% of probability mass across all classes, preventing logit saturation and producing calibrated uncertainty.

### Issue 3 — PlantDoc Cross-Dataset Results Were ~22-25% (Root Cause: Domain Shift)
The models were trained only on controlled lab images (white background, flat leaf) and completely failed on real field images. Not a code bug — a fundamental data distribution mismatch. Full analysis in `project_technical_report.txt` Section 3.

---

## NPDD Fine-Tuning Results (Notebook 09) — July 2026

**What**: Fine-tuned `model_E_selective_V3_final.h5` on `New_Plant_Diseases_Dataset`  
**Why NPDD**: 18,345 training images (vs 6,527 original), ~2.8x larger, better class balance, mix of lab+real images

### Fine-tuning configuration:
```python
# Unfreezing: last 30 MobileNetV2 layers unfrozen (BatchNorm stays frozen)
# Optimizer:  Adam(learning_rate=1e-4)   # lower LR than original (1e-3)
# Loss:       CategoricalCrossentropy(label_smoothing=0.1)
# Epochs:     20 max (EarlyStopping patience=5, ReduceLROnPlateau patience=3 factor=0.5)

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.8, 1.2],  # field lighting variation
    shear_range=0.1,              # perspective distortion
)
```

### Training outcome:
- Ran **14 epochs**, stopped by EarlyStopping at epoch 14, best weights from **epoch 9**
- Key event: `ReduceLROnPlateau` halved LR from `1e-4 → 5e-5` after epoch 8 → epoch 9 jumped from 95.27% → **97.21%** val accuracy
- **Best validation accuracy: 97.21%** (+3.33% over original Strategy E's 93.88%)
- Saved as `model_E_finetuned_NPDD_final.h5` (23 MB, larger because unfrozen layers saved)

### Validation results (4,585 images, all 10 classes):
| Class | F1 | Notes |
|---|---|---|
| Bacterial Spot | 98.5% | |
| Early Blight | 96.7% | |
| Late Blight | 97.8% | |
| Leaf Mold | 98.0% | |
| Septoria Leaf Spot | 95.9% | |
| Spider Mites | 95.7% | Lowest recall (93.6%) |
| Target Spot | 93.4% | Lowest precision (90.9%) — confused with other blotch diseases |
| Yellow Leaf Curl | 99.7% | Near-perfect |
| Mosaic Virus | 99.4% | Was 0% on PlantDoc — fixed by 6.8x more training data |
| Healthy | 97.0% | |
| **Weighted avg** | **97.2%** | |

### Test results (16 flat images, 3 classes only):
- **93.75% accuracy (15/16 correct)**
- Only miss: `TomatoEarlyBlight2.JPG` → predicted Septoria (52% confidence — model expressed uncertainty correctly)
- Early Blight avg confidence: 67.7% (lower, reflecting class ambiguity) — label smoothing working

---

## Known Confusable Class Pairs
- **Early Blight ↔ Septoria Leaf Spot**: Both have small dark lesions with yellow halos. Hardest pair in tomato disease literature.
- **Target Spot ↔ Early Blight / Bacterial Spot**: Blotch patterns look similar.
- **Spider Mites ↔ Septoria**: Spider mite stippling can resemble early Septoria spots.

---

## Preprocessing — The Correct Way (Use This Always)
```python
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np

# For generators (training/eval):
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# For single image inference:
img = load_img(path, target_size=(224, 224))
arr = img_to_array(img)          # [0, 255] float32
arr = preprocess_input(arr)      # [-1, 1] float32
arr = np.expand_dims(arr, 0)     # (1, 224, 224, 3)
pred = model.predict(arr, verbose=0)[0]  # (10,) softmax probs
```

---

## What Still Needs Doing
1. **Retrain A-E** using patched notebooks (preprocess_input + label_smoothing) — patches are already applied to notebooks, just run them
2. **Re-run notebook 08** (PlantDoc) after retraining to see actual improvement from the fixes
3. Consider **stronger augmentation**: CutMix, MixUp, RandAugment
4. NPDD test folder only tests 3/10 classes — need broader real-world test set
5. Early Blight / Septoria pair — consider specialized binary classifier for this decision

---

## Quick Reference — Key Numbers
| Metric | Value |
|---|---|
| Original PlantVillage training set | 6,527 images |
| NPDD training set | 18,345 images |
| Strategy E internal accuracy | 93.88% |
| Strategy E PlantDoc accuracy | 24.35% |
| Strategy E NPDD validation accuracy | **97.21%** |
| NPDD test accuracy (16 images) | **93.75%** |
| Label smoothing value used | 0.1 |
| Fine-tune LR | 1e-4 (→ 5e-5 via ReduceLR) |
| Layers unfrozen in MobileNetV2 | Last 30 (BatchNorm kept frozen) |
| Image input size | 224 × 224 × 3 |
| Full report | `project_technical_report.txt` (556 lines) |
