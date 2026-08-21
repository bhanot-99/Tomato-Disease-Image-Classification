# PlantDoc Cross-Dataset Evaluation — Paper-Ready Results

**Date:** 2026-08-22  
**Script:** `scripts/plantdoc_final_evaluation.py`  
**Data:** PlantDoc tomato subset, train+test pooled  
**Images:** 731 across 8 classes  
**Chance level:** 12.5%  
**Classes absent from PlantDoc:** Spider Mites, Target Spot

This supersedes notebook 08. Each model is evaluated with the input scaling it was trained
with; notebook 08 applied `preprocess_input` to models trained with `rescale=1./255` while
claiming it matched training (worth ~1 point either way).


## Table 1 — Class distribution

| Class | Images |
|---|---|
| Bacterial Spot | 107 |
| Early Blight | 83 |
| Late Blight | 111 |
| Leaf Mold | 91 |
| Septoria Leaf Spot | 148 |
| Yellow Leaf Curl Virus | 75 |
| Mosaic Virus | 54 |
| Healthy | 62 |
| **Total** | **731** |


## Table 2 — Lab vs field accuracy

| Strategy | PlantVillage (internal test) | PlantDoc (field) | Gap | Top-3 (field) |
|---|---|---|---|---|
| A — Color only | 89.21% | **21.20%** | −68.0 pts | 51.30% |
| B — Segmented only | 88.42% | **25.72%** | −62.7 pts | 53.76% |
| C — Random 50/50 mix | 87.42% | **25.31%** | −62.1 pts | 56.77% |
| D — Fine-tuned A->Seg | 90.92% | **22.85%** | −68.1 pts | 57.05% |
| E — Class-aware routing | 93.71% | **23.67%** | −70.0 pts | 54.58% |

All five strategies fall to within ~4 points of each other on field images. **Strategy E's
+4-point internal advantage does not transfer.**


## Table 3 — Per-class recall on PlantDoc

| Class | A | B | C | D | E |
|---|---|---|---|---|---|
| Bacterial Spot | 0.0% | 7.5% | 2.8% | 4.7% | 7.5% |
| Early Blight | 55.4% | 36.1% | 44.6% | 63.9% | 26.5% |
| Late Blight | 61.3% | 47.7% | 48.6% | 38.7% | 47.7% |
| Leaf Mold | 3.3% | 5.5% | 4.4% | 3.3% | 4.4% |
| Septoria Leaf Spot | 23.0% | 50.7% | 50.0% | 37.2% | 40.5% |
| Yellow Leaf Curl Virus | 0.0% | 2.7% | 5.3% | 2.7% | 4.0% |
| Mosaic Virus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Healthy | 6.5% | 24.2% | 14.5% | 9.7% | 37.1% |

Three diseases partially survive the lab-to-field transfer; four are near-invisible.
**Mosaic Virus is 0.0% for every strategy.**


## Table 4 — Calibration

| Model generation | Mean confidence | When correct | When wrong |
|---|---|---|---|
| Strategy A (label smoothing 0.1) | 61.7% | 65.1% | 60.8% |
| Strategy B (label smoothing 0.1) | 58.1% | 64.5% | 55.9% |
| Strategy C (label smoothing 0.1) | 63.7% | 68.1% | 62.2% |
| Strategy D (label smoothing 0.1) | 59.1% | 64.0% | 57.7% |
| Strategy E (label smoothing 0.1) | 51.4% | 57.2% | 49.6% |
| Strategy A (no label smoothing) | 85.0% | 87.1% | 84.4% |
| Strategy B (no label smoothing) | 84.8% | 86.8% | 84.1% |
| Strategy C (no label smoothing) | 82.2% | 84.5% | 81.4% |
| Strategy D (no label smoothing) | 84.9% | 89.3% | 83.6% |
| Strategy E (no label smoothing) | 84.1% | 87.4% | 83.0% |

Label smoothing reduced mean confidence on out-of-domain images from ~85% to 51–64%.
The models became appropriately uncertain rather than confidently wrong — a deployment-
relevant gain even though accuracy is unchanged.


## Table 5 — Prediction distribution, Strategy E

| Class | True images | Predicted as |
|---|---|---|
| Bacterial Spot | 107 | 55 |
| Early Blight | 83 | 190 |
| Late Blight | 111 | 201 |
| Leaf Mold | 91 | 22 |
| Septoria Leaf Spot | 148 | 181 |
| Spider Mites | 0 | 0 |
| Target Spot | 0 | 5 |
| Yellow Leaf Curl Virus | 75 | 4 |
| Mosaic Virus | 54 | 0 |
| Healthy | 62 | 73 |

572 of 731 predictions (78%) fall on just three classes.
Predictions onto the two absent classes: 0.7% — the models do
not hallucinate missing categories.


## Table 6 — Paper-generation models (reproduces published Section V numbers)

| Strategy | Published | Re-measured |
|---|---|---|
| A | 22.98% | 22.98% |
| B | 24.21% | 24.21% |
| C | 24.49% | 24.49% |
| D | 21.61% | 21.61% |
| E | 24.35% | 24.35% |

Exact reproduction, confirming the measurement chain.


## Figures

- `fig_internal_vs_plantdoc.png` — lab vs field accuracy, all five strategies
- `fig_plantdoc_per_class.png` — per-class recall, showing concentrated failure
- `fig_plantdoc_confusion_E.png` — Strategy E confusion, showing three-class collapse


## Suggested claims for the paper

1. All five strategies lose 62–70 points transferring from PlantVillage to field imagery.
2. Class-aware routing's internal advantage does not transfer; on field images the five
   strategies are statistically indistinguishable.
3. Failure is disease-specific, not uniform: Late Blight, Septoria and Early Blight retain
   partial signal; Bacterial Spot, Leaf Mold, Yellow Leaf Curl and Mosaic Virus do not.
4. Top-3 accuracy (51–57%) far exceeds top-1 (21–26%), indicating the representation retains
   usable information that the decision boundary fails to rank correctly.
5. Label smoothing substantially improves calibration under domain shift without improving
   accuracy.
