# Consolidated Findings — Strategy E Re-validation

**Date:** 2026-08-22
**Scope:** Issues 1 and 2 from `PAPER_REVIEW_NOTES.md`, resolved against re-processed data and
retrained models.
**Companion documents:** `PAPER_REVIEW_NOTES.md` (the review that motivated this work),
`RESULTS.md` (chronological log with full run detail and file inventory).

This document consolidates what the experiments established, organised around what needs to
change in the paper rather than around when it was run.

---

## 1. Headline: Strategy E survives an honest protocol

The paper's routing table was built from per-class accuracy measured on the **test** split, and
Strategy E was then reported on that same split. Rebuilt so that routing only ever sees the
**validation** split, with the test split evaluated exactly once at the end:

| Strategy | Paper (leaky) | Leak-free, `1./255` | Leak-free + `preprocess_input` |
|---|---|---|---|
| A — color | 89.06% | 89.35% | **89.21%** |
| B — segmented | 86.13% | 87.71% | **88.42%** |
| C — mixed | 87.48% | 86.56% | **87.42%** |
| D — fine-tuned | 88.42% | 87.35% | **90.92%** |
| **E — selective routing** | **93.88%** | **93.28%** | **93.71%** |

Two independent honest protocols land within 0.6 points of the published figure, and Strategy E
still beats the best single-domain baseline by ~4 points. **The central contribution of the paper
holds.** The leak inflated the headline by ~0.6 points — within run-to-run noise.

Reproduce with `scripts/pipeline.py --preproc {rescale,mobilenet}` (~2.5 h each on an RTX 3050).

---

## 2. What must change in the paper

### 2.1 Table I is inflated (must fix)

The showcase figure — Bacterial Spot, "99.3% color vs 78.7% segmented, +20.6%" — is honestly
**96.7% vs 91.3%, a +5.4 point gap**. The domain effect is real but roughly a quarter of the
published magnitude.

Bacterial Spot is also a poor choice of showcase: it contributes **zero** to Strategy E's final
gain. Septoria leaf spot (+16.0, the largest gain in both passes) or Target Spot (+15.3 in favour
of color) make the argument far better.

### 2.2 Per-class routing rationales are unsupported (must cut or hedge)

The two passes differ only in input scaling, yet the val-derived routing table agrees on just
**6 of 10 classes**. Four classes flip domain, because their A/B/C margins are 1-5 points on a
~150-image validation split — inside sampling noise.

Strategy E's *accuracy* is stable (93.28 / 93.71) even though the table underneath it is not.
The honest reading: **the gain comes from routing per class at all, not from the specific
assignments.** The paper currently narrates each choice as a discovered biological property
("background color is the signal for Bacterial Spot") — pass 2 routes Bacterial Spot the opposite
way and still gains 6 points on it. Those explanations should be removed or heavily hedged.

Assignments stable across both passes, and therefore reportable: Septoria -> segmented,
Target Spot -> color, Yellow Leaf Curl -> segmented, Mosaic -> segmented.

### 2.3 Strategy E's advantage does not transfer (must state)

On PlantDoc field images all five strategies are within ~4 points of each other, i.e.
indistinguishable. Class-aware routing is a PlantVillage-specific gain. Stating this limit
directly is far safer than leaving a reviewer to find it.

### 2.4 Section V-D omission (must replace)

See section 3 below. The current text shows only the flattering NPDD result and omits the
PlantDoc failure entirely, while the public repo contains the failing notebook — the gap reads
as selective reporting.

Note that NPDD fine-tuning is a **different claim**: adapting to a second lab dataset, not
generalising to field photographs. It cannot substitute for the PlantDoc result.

### 2.5 Smaller corrections (from `PAPER_REVIEW_NOTES.md` Issue 5, unchanged)

Missing reference [4]; Section V-A cites Table II for numbers that are in Table I; "balanced"
should be "capped" (minority classes were never upsampled); no Limitations section; no comparison
table against prior published results; single seed, no confidence intervals.

---

## 3. The PlantDoc result: a robust negative

731 pooled PlantDoc tomato images, 8 of 10 classes present (Spider Mites and Target Spot have no
equivalent), chance = 12.5%. All 15 models evaluated under both input scalings:

| Generation | A | B | C | D | E |
|---|---|---|---|---|---|
| Paper (leaky routing, `1./255`) | 23.0% | 24.2% | 24.5% | 21.6% | 24.4% |
| Honest routing, `1./255` | 22.6% | 22.6% | 22.0% | 21.3% | 24.2% |
| Honest routing + `preprocess_input` | 21.2% | 25.7% | 25.3% | 22.8% | 23.7% |

**Neither the leak fix nor the preprocessing fix moves real-world performance.** The remedy
narrative proposed in `PAPER_REVIEW_NOTES.md` Issue 2 — "show the fix recovering performance" —
is *not supported by the data*. There is no recovery to show.

A hypothesis raised and **rejected** in the same run, recorded so it is not re-investigated:
notebook 08 cell 5 applies `preprocess_input` to models trained with `rescale=1./255` while
claiming it "matches training". The bug is real and worth fixing, but the ablation shows it costs
~1 point (22.98% vs 23.80%), not 60.

### 3.1 What does hold up

**Calibration improved substantially.** Label smoothing worked exactly as intended — mean
confidence on out-of-domain images fell from ~85% to **51-64%**. The models stopped being
confidently wrong and became appropriately uncertain. This is a genuine deployment gain: a
51%-confident prediction can be escalated to a human, an 85%-confident wrong one cannot. Worth a
paragraph even though accuracy is flat.

**Failure is concentrated, not uniform** (Strategy E, per class):

| Class | Paper model | Honest + `preprocess_input` |
|---|---|---|
| Late Blight | 68.5% | 47.7% |
| Septoria Leaf Spot | 41.2% | 40.5% |
| Early Blight | 41.0% | 26.5% |
| Healthy | 4.8% | **37.1%** |
| Bacterial Spot | 1.9% | 7.5% |
| Leaf Mold | 1.1% | 4.4% |
| Yellow Leaf Curl Virus | 1.3% | 4.0% |
| Mosaic Virus | 0.0% | 0.0% |

Three diseases partially survive the lab-to-field gap; four are essentially invisible, and Mosaic
Virus is 0.0% in **every** configuration tested. This is a far more specific and more publishable
finding than a single aggregate 24% — "models fail on field images" is known, "the gap is
disease-specific, and here is which visual signals do not survive background clutter" is a
contribution.

Hallucination onto the two absent classes is negligible (0.1-3.0%), ruling out "the model invents
missing classes" as an explanation.

### 3.2 Suggested framing

Report the negative result honestly, with the per-class breakdown and the calibration improvement
as the substantive contributions, and an explicit statement that standard remedies (correct
preprocessing, leak-free training, label smoothing) do not close the gap. A paper that
demonstrates 93.7% internally and ~24% in the field, and says so plainly, is more credible than
one that omits the second number.

---

## 4. Methodological changes made

**Split protocol.** Color and segmented images are the same leaves (`<id>.JPG` <->
`<id>_final_masked.jpg`). The original notebooks split each domain independently, so one leaf
could sit in `color/train` and `segmented/test`. Cross-domain per-class comparison is only valid
on a shared split, so images are now paired by base ID and **one split per class** is applied to
all three domains. This also removed a second, previously unflagged leak in Strategy C, whose
mixed set was built by re-splitting pooled color+segmented images.

Cohort unchanged: 9,325 images (6,527 / 1,399 / 1,399), cap 1,000/class, 70/15/15, seed 42.

**Evaluation protocol.** Routing derived from validation only; test split evaluated once, at the
end, for all strategies.

---

## 5. Open work

| Item | Effort |
|---|---|
| Multi-seed routing stability — report the table as a majority vote with stability counts | ~4 h compute |
| Aggressive augmentation (lighting/background/angle) — the one untested lever for the field gap | ~3 h compute, retrain |
| Reconcile NPDD numbers: paper says 94.11% @ lr 1e-5; `project_technical_report.txt` says 97.21% @ 1e-4 -> 5e-5 (Issue 3) | ~30 min |
| Port notebooks to Linux paths and rebuild notebook 05's `CLASS_ROUTING` from the val data | ~30 min |
| Fix notebook 08 cell 5 preprocessing bug | ~10 min |
| Paper edits per section 2 | writing |

**Reproducibility caveat:** the notebooks do not currently reproduce these numbers. Notebook 05
still holds the legacy test-derived `CLASS_ROUTING`, and the training/preprocessing notebooks
still carry `D:\Development\...` paths. `scripts/pipeline.py`, `scripts/preprocess_dataset.py`
and `scripts/eval_plantdoc.py` are the authoritative path until that is fixed — which matters,
because a reviewer opening the public repo today finds the old leaky table.

Retrained models (263 MB) are gitignored and regenerable from the scripts.
