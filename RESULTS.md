# Experiment Results Log

Running log of actual results (metrics, files produced, dates) from work done on this project going forward. Each entry records what was run, when, and what came out of it, so the paper and `PAPER_REVIEW_NOTES.md` can be updated from real numbers rather than memory.

---

## Log format

```
## [Date] — [What was run]
- Command/notebook: ...
- Environment: ...
- Result: ...
- Files produced: ...
- Notes: ...
```

---

## Entries

## 2026-08-22 — Issue 2: PlantDoc cross-dataset validation, all three model generations

- Script: `scripts/eval_plantdoc.py` (new; portable, and fixes a bug in notebook 08 — see below)
- Data: `validation/PlantDoc-Dataset`, train+test pooled = **731 tomato images**, 8 of the 10
  PlantVillage classes (Spider Mites and Target Spot have no PlantDoc equivalent). Chance = 12.5%.
- Every model evaluated under BOTH input scalings, to separate a suspected measurement artifact
  from real domain shift.

### Notebook 08 has a preprocessing bug — but it is NOT the cause

Notebook 08 cell 5 applies `mobilenet_v2.preprocess_input` (pixels -> [-1,1]) at inference to
models trained with `rescale=1./255` (pixels -> [0,1]); its comment claims this "matches
training". It does not. **However, the ablation shows this costs ~1 point, not 60.** Strategy A
scores 22.98% under correct scaling vs 23.80% under the mismatched scaling. The bug is worth
fixing for correctness but explains none of the collapse. (Hypothesis raised and rejected in the
same run — recorded here so it is not re-investigated later.)

### The domain shift is real and robust to every fix so far

| Generation | A | B | C | D | E |
|---|---|---|---|---|---|
| Paper (leaky routing, 1./255) | 23.0% | 24.2% | 24.5% | 21.6% | 24.4% |
| Honest routing, 1./255 | 22.6% | 22.6% | 22.0% | 21.3% | 24.2% |
| Honest routing + preprocess_input | 21.2% | 25.7% | 25.3% | 22.8% | 23.7% |

All 15 models sit in a 21-26% band. Neither the Issue 1 leak fix nor the Issue 4 preprocessing
fix moves real-world performance. **The narrative proposed in PAPER_REVIEW_NOTES.md Issue 2 —
"show the fix recovering performance" — is not supported. There is no recovery.** What the data
supports is a robust negative result: a model can reach 93.7% on PlantVillage and stay at ~24%
on field photographs, and the standard remedies do not close it.

Note also that Strategy E's internal advantage (+4 points over A) does **not** transfer: on
PlantDoc all five strategies are within ~4 points of each other, i.e. indistinguishable.
Class-aware routing is a PlantVillage-specific gain.

### Label smoothing worked — on calibration, not accuracy

| Generation | mean confidence |
|---|---|
| Paper | 82-85% |
| Honest routing, 1./255 | 82-90% |
| Honest routing + preprocess_input + label smoothing 0.1 | **51-64%** |

The models stopped being confidently wrong and became appropriately uncertain on out-of-domain
images. This is a genuine, reportable deployment improvement (a 51%-confident prediction can be
escalated to a human; an 85%-confident wrong one cannot) and is worth a paragraph even though
accuracy is flat.

### Failure is concentrated, not uniform — Strategy E per class

| Class | Paper model | Honest + preprocess_input |
|---|---|---|
| Late Blight | 68.5% | 47.7% |
| Septoria Leaf Spot | 41.2% | 40.5% |
| Early Blight | 41.0% | 26.5% |
| Healthy | 4.8% | **37.1%** |
| Bacterial Spot | 1.9% | 7.5% |
| Leaf Mold | 1.1% | 4.4% |
| Yellow Leaf Curl Virus | 1.3% | 4.0% |
| Mosaic Virus | 0.0% | 0.0% |

Three diseases are partially recognised in the field; four are essentially invisible, and Mosaic
Virus is 0.0% in every configuration tested. This is a far more specific finding than a single
aggregate 24% and should replace it in the paper.

Hallucination onto the two absent classes is negligible (0.1-3.0% of predictions), ruling out
"the model invents missing classes" as an explanation.

- Files produced: `outputs/cross_dataset_validation/plantdoc_preprocessing_ablation.json`
  (per model: overall, per class, mean confidence, hallucination rate, under both scalings)

### Issue 2 status: MEASURED, narrative needs rewriting

The failure is confirmed, quantified across three model generations, and traced to specific
classes. What is NOT available is a fix that recovers performance — so the paper should report
the negative result honestly (with the calibration improvement and the per-class breakdown as
the substantive contributions) rather than promising a remedy. Untested remaining lever:
aggressive augmentation for lighting/background/angle, which requires retraining; and NPDD
fine-tuning (notebook 09), which is a different claim — adapting to a second lab dataset, not
generalising to field photos.

---

## 2026-08-22 — Issue 1 fix: leak-free Strategy E, corrected preprocessing (pass 2 of 2)

- Script: `scripts/pipeline.py --preproc mobilenet`
- Same protocol as pass 1 (val-derived routing, single test evaluation), but with the Issue 4
  fix applied: `mobilenet_v2.preprocess_input` instead of `rescale=1./255`, plus
  `CategoricalCrossentropy(label_smoothing=0.1)`. Runtime ~75 min. All runs used the full
  epoch budget; no EarlyStopping.

### Final test accuracy — all three protocols

| Strategy | Paper (leaky routing, 1./255) | Leak-free, 1./255 | Leak-free + preprocess_input |
|---|---|---|---|
| A — color        | 89.06% | 89.35% | **89.21%** |
| B — segmented    | 86.13% | 87.71% | **88.42%** |
| C — mixed        | 87.48% | 86.56% | **87.42%** |
| D — fine-tuned   | 88.42% | 87.35% | **90.92%** |
| E — selective    | 93.88% | 93.28% | **93.71%** |

**Strategy E is robust.** 93.28% and 93.71% under two independent honest protocols, against a
published 93.88% that was leak-inflated. The contribution stands. Strategy D is the one that
clearly benefits from correct preprocessing (+3.57 over pass 1); A and C are unchanged within
noise, which is expected — the internal test set shares whatever scaling training used, so
`preprocess_input` cannot show its value here. Its payoff should appear on cross-dataset
PlantDoc (notebook 08), which has not yet been re-run.

### IMPORTANT CAVEAT — the routing table is not stable

The two passes differ only in input scaling, yet the val-derived routing table agrees on just
**6 of 10 classes**:

| Class | Pass 1 route | Pass 2 route | Agree |
|---|---|---|---|
| Bacterial_spot | color | segmented | no |
| Early_blight | color | color | yes |
| Late_blight | color | segmented | no |
| Leaf_Mold | color | segmented | no |
| Septoria_leaf_spot | segmented | segmented | yes |
| Spider_mites | color | mixed | no |
| Target_Spot | color | color | yes |
| Yellow_Leaf_Curl_Virus | segmented | segmented | yes |
| Mosaic_virus | segmented | segmented | yes |
| healthy | mixed | mixed | yes |

Four classes flip domain under a benign change, because their A/B/C margins are 1-5 points on a
143-150 image val split — inside sampling noise. Only Target_Spot (color) and Septoria /
Yellow_Leaf_Curl / Mosaic (segmented) hold across both passes with margins worth trusting.

Strategy E's *accuracy* is stable (93.28 / 93.71) even though the table underneath it is not.
That is the honest reading: the benefit comes from routing per class at all, not from the
specific assignments the paper presents as findings. The paper currently narrates each routing
choice as a discovered biological property ("background color is the signal for Bacterial
Spot"); pass 2 routes Bacterial Spot the opposite way and still gains 6 points on it. Those
per-class explanations are not supported and should be cut or heavily hedged.

**Recommended for the paper:** report routing determined by validation data, state that
assignments with margins below ~5 points are noise-sensitive, and — Tier 3 in
PAPER_REVIEW_NOTES.md, now clearly worth doing — run multiple seeds so the routing table can be
reported as a majority vote with stability counts rather than a single draw.

### Pass 2 per-class, E vs A (test)

| Class | E | A | Gain | Routed to |
|---|---|---|---|---|
| Septoria_leaf_spot | 91.3% | 79.3% | **+12.0** | segmented |
| Early_blight | 94.7% | 87.3% | +7.3 | color |
| Bacterial_spot | 92.0% | 86.0% | +6.0 | segmented |
| Late_blight | 97.3% | 92.0% | +5.3 | segmented |
| Target_Spot | 90.0% | 86.0% | +4.0 | color |
| Yellow_Leaf_Curl_Virus | 96.7% | 92.7% | +4.0 | segmented |
| healthy | 99.3% | 95.3% | +4.0 | mixed |
| Mosaic_virus | 98.2% | 94.6% | +3.6 | segmented |
| Spider_mites | 89.3% | 90.0% | -0.7 | mixed |
| Leaf_Mold | 90.9% | 92.3% | -1.4 | segmented |

Septoria leaf spot is the largest gain in both passes (+16.0 and +12.0) and is routed to
segmented in both — the single most defensible finding in the whole routing story.

- Files produced:
  - `outputs/results/final_test_mobilenet.json`
  - `outputs/routing_analysis/strategy_{A,B,C}_val_per_class_mobilenet.json`
    (the unsuffixed `strategy_{A,B,C}_val_per_class.json` files that notebook 05 loads now hold
    these pass-2 values)
  - `outputs/routing_analysis/class_routing_val_derived_mobilenet.json`
  - `models/strategy_{A,B,C,D,E}_mobilenet_{best,final}.keras`
  - `dataset/processed_selective_mobilenet/`

### Issue 1 status: RESOLVED

Routing is derived from validation data only; the test split is evaluated once, for all
strategies, at the end. Still open from PAPER_REVIEW_NOTES.md: notebook 05's in-file
`CLASS_ROUTING` constant is still the legacy test-derived table and the notebooks still carry
Windows paths, so the notebooks do not yet reproduce this result — `scripts/pipeline.py` is the
authoritative path. Issue 2 (PlantDoc) needs notebook 08 re-run against these new models.

---

## 2026-08-21 — Issue 1 fix: leak-free Strategy E, original scaling (pass 1 of 2)

- Script: `scripts/pipeline.py --preproc rescale` (new)
- Environment: Linux, `.venv/`, TensorFlow 2.16.2 + CUDA, RTX 3050 4GB. Runtime ~76 min for
  all 5 strategies. Seed 42, 20 epochs (10 for D), batch 32, MobileNetV2 frozen base,
  `rescale=1./255` and plain categorical crossentropy — i.e. identical training setup to the
  models reported in the paper, so these numbers are directly comparable.
- **Method change (the fix):** the per-class accuracies that decide `CLASS_ROUTING` are now
  measured on the **val** split. The test split is evaluated exactly once, at the end, for all
  five strategies. Previously the routing was chosen from test-set per-class accuracy and then
  Strategy E was reported on that same test set.

### Final test accuracy — leak-free vs. published

| Strategy | Paper (test-derived routing) | Leak-free | Delta |
|---|---|---|---|
| A — color        | 89.06% | **89.35%** | +0.29 |
| B — segmented    | 86.13% | **87.71%** | +1.58 |
| C — mixed        | 87.48% | **86.56%** | -0.92 |
| D — fine-tuned   | 88.42% | **87.35%** | -1.07 |
| E — selective    | 93.88% | **93.28%** | -0.60 |

**Strategy E holds up.** With routing derived only from validation data it still beats the best
single-domain baseline by **+3.93 points** (93.28 vs 89.35). The leak was worth ~0.6 points —
within run-to-run noise. The paper's central claim survives an honest protocol.

### Val-derived routing table (`outputs/routing_analysis/class_routing_val_derived_rescale.json`)

| Class | A (color) | B (segmented) | C (mixed) | Routed to |
|---|---|---|---|---|
| Bacterial_spot        | 96.7% | 91.3% | 92.7% | color |
| Early_blight          | 76.7% | 68.0% | 70.0% | color |
| Late_blight           | 92.7% | 88.0% | 86.0% | color |
| Leaf_Mold             | 94.4% | 90.2% | 93.0% | color |
| Septoria_leaf_spot    | 84.0% | 92.0% | 84.7% | segmented |
| Spider_mites          | 94.0% | 90.0% | 91.3% | color |
| Target_Spot           | 93.3% | 78.0% | 78.7% | color |
| Yellow_Leaf_Curl_Virus| 96.0% | 97.3% | 94.7% | segmented |
| Mosaic_virus          | 91.1% | 96.4% | 89.3% | segmented |
| healthy               | 94.0% | 94.0% | 95.3% | mixed |

Val overall: A 91.28%, B 87.99%, C 87.42%.

**Table I in the paper needs correcting.** Its showcase figure — Bacterial Spot "99.3% color vs
78.7% segmented, +20.6%" — is honestly **96.7% vs 91.3%, a +5.4 point gap**. The domain effect
is real but roughly a quarter of the published magnitude. Three classes (Yellow Leaf Curl,
Mosaic, healthy) are decided by margins of <=1.3 points, i.e. within noise — those routing
choices are not evidence-backed either way and should be described as such rather than as
findings. The genuinely large, defensible gaps are Target_Spot (+15.3 color) and
Septoria_leaf_spot (+8.0 segmented).

### Where Strategy E's gain actually comes from (test, per class, E vs A)

| Class | E | A | Routed to | Gain |
|---|---|---|---|---|
| Septoria_leaf_spot    | 98.0% | 82.0% | segmented | **+16.0** |
| Yellow_Leaf_Curl_Virus| 99.3% | 94.0% | segmented | +5.3 |
| Early_blight          | 83.3% | 76.7% | color     | +6.6 |
| Leaf_Mold             | 96.5% | 91.6% | color     | +4.9 |
| healthy               | 96.0% | 93.3% | mixed     | +2.7 |
| Target_Spot           | 88.0% | 86.0% | color     | +2.0 |
| Mosaic_virus          | 96.4% | 94.6% | segmented | +1.8 |
| Bacterial_spot        | 96.7% | 96.7% | color     |  0.0 |
| Spider_mites          | 86.7% | 86.7% | color     |  0.0 |
| Late_blight           | 94.0% | 95.3% | color     | -1.3 |

Nearly half of Strategy E's headline advantage is one class — Septoria leaf spot, +16 points from
segmentation. Worth stating plainly in the paper instead of implying a uniform benefit.

- Files produced:
  - `outputs/results/final_test_rescale.json` (full histories, val per-class, test per-class)
  - `outputs/routing_analysis/strategy_{A,B,C}_val_per_class_rescale.json` — the JSON files
    `05_model_training_selective.ipynb` expects and that had never been generated
  - `outputs/routing_analysis/class_routing_val_derived_rescale.json`
  - `models/strategy_{A,B,C,D,E}_rescale_{best,final}.keras`
  - `dataset/processed_selective_rescale/` (symlinked from the routed domains)
- Notes: `.keras` format rather than `.h5` — Keras 3 rejects `.h5` for full-model checkpointing.
  No EarlyStopping triggered; all runs used their full epoch budget.

---

## 2026-08-21 — Dataset processed (Issue 1, step 0)

- Script: `scripts/preprocess_dataset.py` (new; portable replacement for notebooks
  `01_preprocessing_color`, `01b_preprocessing_segmented`, `01c_preprocessing_mixed`,
  which hardcode Windows paths and cannot run on this machine)
- Environment: Linux, Python 3.12, `.venv/` at project root
- Source: `dataset/color` + `dataset/segmented` (PlantVillage, extracted from `archive.zip`)
  — 10 classes, 18,160 raw images per domain
- Result: three processed roots, each 9,325 images (6,527 train / 1,399 val / 1,399 test)
  - `dataset/processed` (color, Strategy A)
  - `dataset/processed_segmented` (Strategy B)
  - `dataset/processed_mixed` (Strategy C)
  - Total of 9,325 matches the cohort size reported in the paper.
- Protocol (unchanged from the paper): cap 1,000 images/class, 70/15/15 split, seed 42,
  resized 224x224 LANCZOS.
- **Deliberate change vs. the original notebooks:** color and segmented images are paired by
  base ID (`<id>.JPG` <-> `<id>_final_masked.jpg`) and a *single* split is derived per class
  and applied identically to color, segmented and mixed. The original notebooks split each
  domain independently, so the same leaf could sit in `color/train` and `segmented/test`.
  Comparing per-class accuracy across domains to choose Strategy E's routing is only valid
  if both domains use the same split — so this is a prerequisite for the Issue 1 fix, not a
  cosmetic change. It also removes a second, previously unflagged cross-domain leak in
  Strategy C, whose mixed set was rebuilt by re-splitting pooled color+segmented images.
- Per-class split sizes: 700/150/150 for the eight classes at the 1,000 cap; Leaf Mold
  666/143/143 (951 paired images); Mosaic Virus 261/56/56 (373 images).
- Notes: 1 Spider-mites color image has no segmented counterpart and was dropped (18,160 ->
  9,325 after capping is unaffected). The exact image *selection* of the original Windows run
  cannot be reproduced (it depended on `os.listdir` order), so all strategies must be
  retrained on this split regardless — which the Issue 1 fix required anyway.

---

