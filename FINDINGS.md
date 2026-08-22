# Consolidated Findings — Strategy E Re-validation

**Date:** 2026-08-22
**Scope:** Issues 1 and 2 from `PAPER_REVIEW_NOTES.md`, resolved against re-processed data and
retrained models. Issue 2 is answered — see section 3 — but the paper has not yet been rewritten
to match.
**Companion documents:** `PROJECT_OVERVIEW.md` (start here — the map to everything else),
`PAPER_REVIEW_NOTES.md` (the review that motivated this work),
`outputs/field_validation/FIELD_VALIDATION_REPORT.md` (authority for all field results and
dataset audits), `DOC_AUDIT.md`, `RESULTS.md` (chronological run log).

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

Every number supporting class-aware routing comes from PlantVillage. It has now been tested on
field imagery and **it does not survive**: on PlantWild, Strategy E has the largest
lab-to-field gap of the five (−81.2 points) and finishes second-worst, inverting its +4.5-point
internal advantage over Strategy A (section 3). The paper must scope the claim to the internal
benchmark explicitly, and state the inversion rather than leaving it implied.

### 2.4 Section V-D rests on the wrong evidence (must replace)

The current text supports its generalisation claim with the NPDD fine-tuning result alone. NPDD
is a **different claim**: adapting to a second lab dataset, not generalising to field
photographs. It cannot substitute for a field evaluation.

The field evaluation now exists (section 3), so V-D can and must be rewritten from it. It should
be built around the *relative* finding — that routing's advantage does not transfer — plus the
calibration result, both of which are robust to the evaluation set's contamination because all
five strategies were scored on identical images. It should **not** report a point estimate of
field accuracy. This is the highest-value remaining edit and needs no further compute.

### 2.5 Smaller corrections (from `PAPER_REVIEW_NOTES.md` Issue 5, unchanged)

Missing reference [4]; Section V-A cites Table II for numbers that are in Table I; "balanced"
should be "capped" (minority classes were never upsampled); no Limitations section; no comparison
table against prior published results; single seed, no confidence intervals.

---

## 3. Field validation: PlantDoc withdrawn, PlantWild evaluated

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

**Replacement: found, audited, evaluated.** Two further candidates were rejected —
`Tomato-Village` Variant-a (detached leaves on white sheets, i.e. the same domain as
PlantVillage; 1,616 files derived from 391 source photographs; its own splits leak) and
Variant-c (correct field domain, but only one class overlaps the ten PlantVillage labels; it
retains value as an open-set test). `PlantWild` (Wei et al., 2024) was accepted with reservations
and evaluated: 1,966 tomato images, 8 of 10 classes.

| Strategy | Internal | PlantWild | Gap |
|---|---|---|---|
| A — color | 89.21% | 17.19% | −72.0 |
| B — segmented | 88.42% | 12.11% | −76.3 |
| C — mixed | 87.42% | 14.45% | −73.0 |
| D — fine-tuned | 90.92% | 16.63% | −74.3 |
| **E — routing** | **93.71%** | **12.46%** | **−81.2** |

Chance is 12.5%, and top-3 accuracy is also at chance (33–38.5% against a 37.5% baseline), so
the representation retains almost no usable signal. **Strategy E has the largest gap and
finishes second-worst.**

The absolute figures are a **lower bound, not a measurement**: PlantWild is web-scraped, and
20–40% of the three fruit-affected classes show lesions on fruit rather than foliage — images no
leaf classifier could be expected to get right. The *paired comparison* between strategies is
unaffected, because all five were scored on identical images, so a contaminated image is equally
unclassifiable for every model.

What is safe to claim, with no further work:

1. All five strategies lose 72–81 points transferring from PlantVillage to in-the-wild imagery.
2. Class-aware routing's internal advantage does not transfer; Strategy E has the largest gap.
3. Label smoothing cuts out-of-domain confidence from ~86% to ~57% without changing accuracy —
   the models become appropriately uncertain rather than confidently wrong. Replicated across two
   independent field datasets, and the cleanest positive result in this work.

Not claimable without a leaf-only filtered re-evaluation: any precise field accuracy figure, and
any explanation of *why* particular classes fail.

Full detail, including the four dataset audits and their evidence, is in
`outputs/field_validation/FIELD_VALIDATION_REPORT.md`.

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
| ~~Evaluate a replacement field dataset~~ — **done**: PlantWild evaluated, three other candidates rejected on audit (section 3) | done |
| **Rewrite Section V-D** from the field result — relative finding plus calibration, no point estimate | writing, no compute |
| Aggressive augmentation (lighting/background/angle) — untested lever for the field gap; `scripts/pipeline.py --aug heavy` | ~4 h compute, retrain |
| ~~Reconcile NPDD numbers (Issue 3)~~ — **done**: the paper (94.11% @ 1e-5) matches notebook 09 exactly; `project_technical_report.txt` and `GEMINI.md` described a run that never happened and have been corrected | done |
| ~~Audit `project_technical_report.txt` and `GEMINI.md`~~ — **done**, see `DOC_AUDIT.md`: the NPDD section was the only fabricated region; 162 of 168 percentage claims corroborate exactly | done |
| Optional field strengtheners: cite the withdrawn PlantDoc result for the *relative* finding only (free, doubles replication); paired McNemar test on A vs E; leaf-only filtered PlantWild re-evaluation; Variant-c open-set test | varies |
| Port notebooks to Linux paths and rebuild notebook 05's `CLASS_ROUTING` from the val data | ~30 min |
| Paper edits per section 2 | writing |

**Reproducibility caveat:** the notebooks do not currently reproduce these numbers. Notebook 05
still holds the legacy test-derived `CLASS_ROUTING`, and the training/preprocessing notebooks
still carry `D:\Development\...` paths. `scripts/pipeline.py` and `scripts/preprocess_dataset.py`
are the authoritative path until that is fixed — which matters,
because a reviewer opening the public repo today finds the old leaky table.

Retrained models (263 MB) are gitignored and regenerable from the scripts.
