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
│   └── 09_finetune_and_test_new_plant_diseases.ipynb  NPDD fine-tune + test
├── scripts/
│   ├── patch_notebooks.py      Applied preprocess_input + label_smoothing to nb 01-08
│   └── generate_nb09.py        Generated notebook 09 programmatically
├── outputs/
│   └── npdd_validation/            NPDD fine-tune charts + CSV
├── src/                        dataset.py, model.py, train.py, evaluate.py, predict.py, gradcam.py, severity.py
├── project_technical_report.txt   556-line full narrative report (read this for full context)
└── myenv/                      Python virtualenv
```

---

## The 5 Strategies

| Strategy | Description | Internal Acc |
|---|---|---|
| A | Color-only training images | 89.06% |
| B | Segmented-only (background removed) | 86.13% |
| C | Random 50/50 color+segmented mix | 87.48% |
| D | Fine-tuned: Strategy A weights → segmented data | 88.42% |
| **E** | **Class-Aware Routing** (each class uses its best-performing domain) | **93.88%** |

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

### Issue 3 — Field generalization is unmeasured (Domain Shift)
The models were trained only on controlled lab images (white background, flat leaf), so degradation on real field images is expected but currently **unquantified**: the dataset previously used for this was withdrawn on data-quality grounds. See `FINDINGS.md` section 3.

---

## NPDD Fine-Tuning Results (Notebook 09) — July 2026

**What**: Fine-tuned `model_E_selective_V3_final.h5` on `New_Plant_Diseases_Dataset`  
**Why NPDD**: 18,345 training images (vs 6,527 original), ~2.8x larger, better class balance, mix of lab+real images

### Fine-tuning configuration:
```python
# Unfreezing: last 30 MobileNetV2 layers unfrozen (BatchNorm stays frozen)
# Optimizer:  Adam(learning_rate=1e-5)   # two orders below original (1e-3)
# Loss:       CategoricalCrossentropy(label_smoothing=0.1)
# Epochs:     10 max (EarlyStopping patience=5, ReduceLROnPlateau patience=3 factor=0.5)

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
- Ran **10 epochs** — hit `max_epochs`, EarlyStopping never fired; best weights from **epoch 9** restored
- LR held at `1e-5` for the whole run; `ReduceLROnPlateau` never triggered
- Val accuracy per epoch: 81.16 → 87.00 → 87.63 → 87.72 → 90.64 → 91.60 → 92.11 → 91.78 → **94.11** → 92.67
- **Best validation accuracy: 94.11%** (+0.23pp over original Strategy E's 93.88%)
- Saved as `model_E_finetuned_NPDD_final.h5` (23 MB, larger because unfrozen layers saved)

> **CORRECTION (2026-08-22).** This section previously described a 14-epoch run at
> LR 1e-4 halved to 5e-5 reaching 97.21%. No such run exists. The figures above are
> transcribed from the saved cell outputs of notebook 09. See Issue 3 in
> `PAPER_REVIEW_NOTES.md`.

### Validation results (4,585 images, all 10 classes):
| Class | Precision | Recall | F1 |
|---|---|---|---|
| Bacterial Spot | 94.6% | 95.3% | 95.0% |
| Early Blight | 95.3% | 88.7% | 91.9% |
| Late Blight | 95.6% | 94.2% | 94.9% |
| Leaf Mold | 95.2% | 97.4% | 96.3% |
| Septoria Leaf Spot | 92.7% | 87.6% | 90.1% |
| Spider Mites | 94.7% | 89.9% | 92.2% |
| Target Spot | 82.3% | 94.3% | 87.9% | 
| Yellow Leaf Curl | 99.6% | 98.0% | 98.8% |
| Mosaic Virus | 97.7% | 96.2% | 97.0% |
| Healthy | 94.8% | 98.8% | 96.7% |
| **Weighted avg** | **94.3%** | **94.1%** | **94.1%** |

Weakest: Target Spot (82.3% precision — other blotch diseases over-assigned to it)
and Septoria (87.6% recall). Early Blight recall 88.7%, consistent with the
Early Blight / Septoria confusion pair.

### Test results (16 flat images, 3 classes only):
- **100.00% accuracy (16/16 correct)**, avg confidence 75.54%
- Early Blight avg confidence 55.8% vs Healthy 85.4% / Yellow Leaf Curl 89.1% — lower
  confidence on the harder class, consistent with its 88.7% validation recall
- **Not reportable as a result**: 16 images, 3 of 10 classes

> **CORRECTION (2026-08-22).** Previously recorded as 93.75% (15/16) with
> `TomatoEarlyBlight2.JPG` missed as Septoria at 52% confidence. The CSV on disk
> (`outputs/npdd_validation/test_predictions_NPDD.csv`) records that image as Early
> Blight at 89.22% confidence, Septoria probability 2.01%, and all 16 as correct.

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
2. **Evaluate a field-captured dataset** (see `FINDINGS.md` section 3) to measure real-world generalization
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
| Strategy E NPDD validation accuracy | **94.11%** |
| NPDD test accuracy (16 images, 3 classes — not reportable) | **100.00%** |
| Label smoothing value used | 0.1 |
| Fine-tune LR | 1e-5 (constant; ReduceLR never fired) |
| Layers unfrozen in MobileNetV2 | Last 30 (BatchNorm kept frozen) |
| Image input size | 224 × 224 × 3 |
| Full report | `project_technical_report.txt` (556 lines) |
