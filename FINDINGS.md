# Consolidated Findings — Strategy E Re-validation

**Date:** 2026-08-22
**Scope:** Issue 1 from `PAPER_REVIEW_NOTES.md`, resolved against re-processed data and
retrained models. Issue 2 (field generalisation) is open pending evaluation on a
field-captured dataset — see section 3.
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

### 2.3 Strategy E's advantage is demonstrated in-domain only (must state)

Every number supporting class-aware routing comes from PlantVillage. Whether the advantage
survives on field imagery is untested pending the replacement dataset (section 3), so the paper
must scope the claim to the internal benchmark rather than implying general superiority.

### 2.4 Section V-D rests on the wrong evidence (must replace)

The current text supports its generalisation claim with the NPDD fine-tuning result alone. NPDD
is a **different claim**: adapting to a second lab dataset, not generalising to field
photographs. It cannot substitute for a field evaluation. Section V-D should be rewritten once
the replacement field dataset (section 3) has been evaluated.

### 2.5 Smaller corrections (from `PAPER_REVIEW_NOTES.md` Issue 5, unchanged)

Missing reference [4]; Section V-A cites Table II for numbers that are in Table I; "balanced"
should be "capped" (minority classes were never upsampled); no Limitations section; no comparison
table against prior published results; single seed, no confidence intervals.

---

## 3. Field validation: PlantDoc excluded on data-quality grounds

Cross-dataset validation was originally run against the tomato subset of PlantDoc (Singh et al.,
2020) — 731 images, 8 of the 10 classes. That evaluation has been **withdrawn and removed from
this repository** after a visual audit of the subset found the label and content quality
inadequate to support any claim, positive or negative.

Findings from the audit, by inspection of every image in two classes:

- **Images that are not leaves.** The healthy class (`Tomato leaf`, 62 images) contains a stock
  *illustration* of two red tomato fruits with flowers, a photograph of chopped herbs on a
  cutting board, and a leaf photographed together with a ripe fruit.
- **Composite figures and lecture slides.** A nine-panel scientific figure grid with per-specimen
  captions, and a herbarium-style plate, both filed as single training images. The mosaic virus
  class contains a lecture slide, title text included, whose sub-panels show a bowl of fruit.
- **Label errors.** Leaves with unmistakable yellowing and black lesions are filed as healthy.
- **Wrong species.** Several images in both audited classes are not tomato foliage.
- **Watermarks.** Roughly a third of the audited images carry Shutterstock, Alamy or
  Depositphotos watermark bars overlaid on the leaf.

These follow from the dataset's construction: PlantDoc was assembled by web scraping, and the
tomato subset was not filtered to leaf photographs of the labelled species. A model scored
against it is partly being scored on its ability to classify stock illustrations and slide
screenshots, so neither a low score nor a high one is interpretable.

**Replacement in progress.** Field-captured imagery is being sourced instead, with
`Tomato-Village` (Gehlot et al., *Multimedia Systems*, 2023 — original photography from fields in
Jodhpur and Jaipur, Rajasthan) the leading candidate, and `PlantWild` (Wei et al., 2024) a
secondary one. PlantWild is itself web-scraped and must pass the same audit before use. Until a
replacement is evaluated, **the paper has no field-generalisation result** and must not claim
one.

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
| Evaluate the replacement field dataset (section 3) and rewrite Section V-D from it | ~1 h compute |
| Aggressive augmentation (lighting/background/angle) — untested lever for the field gap; `scripts/pipeline.py --aug heavy` | ~4 h compute, retrain |
| ~~Reconcile NPDD numbers (Issue 3)~~ — **done**: the paper (94.11% @ 1e-5) matches notebook 09 exactly; `project_technical_report.txt` and `GEMINI.md` described a run that never happened and have been corrected | done |
| **Audit `project_technical_report.txt` and `GEMINI.md` against notebook outputs** — Issue 3 showed both contain fabricated run detail, so every number sourced only from them is unverified | ~2-3 h |
| Port notebooks to Linux paths and rebuild notebook 05's `CLASS_ROUTING` from the val data | ~30 min |
| Paper edits per section 2 | writing |

**Reproducibility caveat:** the notebooks do not currently reproduce these numbers. Notebook 05
still holds the legacy test-derived `CLASS_ROUTING`, and the training/preprocessing notebooks
still carry `D:\Development\...` paths. `scripts/pipeline.py` and `scripts/preprocess_dataset.py`
are the authoritative path until that is fixed — which matters,
because a reviewer opening the public repo today finds the old leaky table.

Retrained models (263 MB) are gitignored and regenerable from the scripts.
