# Field Validation — Dataset Audits and PlantWild Evaluation

**Date:** 2026-08-22
**Script:** `scripts/field_evaluation.py`
**Supersedes:** the withdrawn PlantDoc evaluation (see `FINDINGS.md` section 3)

This report covers four candidate field datasets, three of which were rejected on data-quality
grounds, and the evaluation of Strategies A–E against the one that was usable.

---

## 1. Summary

| | Result |
|---|---|
| Datasets audited | 4 (PlantDoc, Tomato-Village Variant-a, Tomato-Village Variant-c, PlantWild) |
| Datasets rejected | 3 (a 4th, Variant-c, usable only as an open-set test) |
| Dataset evaluated | PlantWild tomato subset, 1,966 images, 8 of 10 classes |
| **Best field accuracy** | **17.19% (Strategy A)** against a 12.5% chance baseline |
| Worst field accuracy | 12.11% (Strategy B) — indistinguishable from chance |
| Generalization gap | −72 to −81 points |

The headline finding is harsher than the withdrawn PlantDoc result. On genuine in-the-wild
imagery the models perform at or barely above chance, and **top-3 accuracy is also at chance**,
which means the representation retains almost no usable signal — a materially different
conclusion from the PlantDoc data, where top-3 (51–57%) sat well above its 37.5% baseline.

An important caveat on interpretation is in section 5: part of this gap is measurement error
introduced by the evaluation set itself, not model failure.

---

## 2. Dataset audits

### 2.1 PlantDoc — REJECTED (content and label quality)

731 tomato images, 8 classes, web-scraped. Image-level audit of two complete classes found
fruit-only stock illustrations, a lecture slide with title text whose panels show a bowl of
fruit, a nine-panel scientific figure grid, a herbarium plate, chopped herbs on a cutting board,
several non-tomato species, diseased leaves labelled healthy, and Shutterstock/Alamy/Depositphotos
watermarks on roughly a third of images. Data, code and results removed from the repository.

### 2.2 Tomato-Village Variant-a — REJECTED (wrong domain, augmented duplicates)

Field-captured in Jodhpur and Jaipur, Rajasthan (Gehlot et al., *Multimedia Systems*, 2023), so
the provenance is sound — but three disqualifying problems:

1. **Not field imagery.** All three usable classes are single detached leaves on a plain white or
   lilac sheet. This is the same domain PlantVillage occupies, so it cannot measure lab-to-field
   transfer. It would score well and mean nothing.
2. **4× over-counted.** 1,616 files derive from **391 unique source photographs** — the rest are
   rotation-augmented copies (`IMG20220323100545.jpg`, `_5`, `_6`, `_7`). Perceptual hashing puts
   637 of 1,616 in near-duplicate groups. The 391 originals carry EXIF; augmented copies had it
   stripped, which is why Healthy (never augmented) is 100% EXIF and the blights are 12%.
3. **Its own splits leak.** 167 source photos have augmented copies in more than one of
   train/val/test.

Class overlap with the 10 PlantVillage labels was only 3 (Early blight, Late blight, Healthy);
the other 5 classes (Leaf Miner, Magnesium/Nitrogen/Potassium Deficiency, Spotted Wilt Virus)
have no PlantVillage equivalent.

### 2.3 Tomato-Village Variant-c — REJECTED (label space does not overlap)

This is the in-situ field photography the paper's title implies, and **the domain is right**:
whole plants in soil, potted plants outdoors, foliage against brick walls, hands holding leaves
in situ, natural lighting, cluttered real backgrounds. Bounding boxes are leaf-level, so crops
would yield exactly the field leaf images a classifier needs.

It fails on label space instead. Annotations carry bare numeric ids (`2.0`–`7.0`) with no mapping
file in the repository. The ids are alphabetical indices into Variant-a's eight class folders,
confirmed by cropping boxes per id and inspecting them:

| id | Class | Boxes (all) | Boxes (non-augmented) | Visual confirmation |
|---|---|---|---|---|
| 3.0 | Leaf Miner | 107,199 | 7,589 | serpentine white mines — unmistakable |
| 7.0 | Spotted Wilt Virus | 31,397 | 2,826 | bronzed necrotic patches |
| 4.0 | Magnesium Deficiency | 15,870 | 1,569 | interveinal yellowing |
| **2.0** | **Late Blight** | **3,938** | **336** | dark necrotic lesions |
| 5.0 | Nitrogen Deficiency | 1,693 | 147 | pale, uniformly yellowing leaves |
| 6.0 | Potassium Deficiency | 1,126 | 118 | marginal chlorosis |

Ids 0.0 (Early_blight) and 1.0 (Healthy) are absent entirely. **Exactly one class — Late Blight —
overlaps the 10 PlantVillage labels, with 336 non-augmented boxes.** A ten-class model cannot be
evaluated on a single class.

Augmentation inflation is the same as Variant-a: 14,368 images derive from **1,796 unique source
photographs** (`Jaipur_Pots (384)_aug7.jpg`), an 8× multiplier.

**Residual value:** five of the six classes are conditions the model has never been trained on
(leaf miner, three deficiencies, spotted wilt), photographed in the field with expert labels.
That makes Variant-c a ready-made **open-set test**: feed the leaf crops and measure whether the
model confidently asserts a known disease on a condition outside its label space. Given that the
`_rescale` models sit at ~86% mean confidence out of domain, this would likely produce a strong
deployment-safety result. It is the only remaining use for this dataset.

### 2.4 PlantWild — ACCEPTED WITH RESERVATIONS

18,542 images / 89 classes overall (Wei et al., 2024, CC BY-NC-ND 4.0); the tomato subset is
1,966 images across the same 8 classes PlantDoc covered. Web-scraped, but with quality control
PlantDoc lacked: five annotators filtered against extension-service exemplars, labels
cross-validated by at least two experts.

**Provenance** (from the dataset's own `url_record.json`, all 1,966 tomato images):

| Metric | Value |
|---|---|
| Distinct source domains | 480 |
| Stock-photo domains | 65 (3.3%) |
| Slide / paper domains | 59 (3.0%) |

Top sources are Reddit (127), Chinese forums, ResearchGate, plantix.net, Wikipedia, Flickr,
INRA's ephytia, invasive.org — predominantly user and extension-service photography. This is a
far healthier profile than PlantDoc.

**Visual audit** — 96-image random sample from each of the 8 classes:

| Class | n | Principal contamination | Est. rate |
|---|---|---|---|
| Late blight | 295 | **lesions on fruit, not leaves** | ~40% |
| Early blight | 346 | fruit lesions; blog graphics with text overlays | ~20–25% |
| Bacterial leaf spot | 280 | fruit lesions; one line drawing | ~20% |
| Tomato leaf (healthy) | 226 | **red-fruit photographs**; stock cutouts; watermarks | ~20% |
| Yellow leaf curl | 171 | a presentation title slide; multi-panel figures; ad watermarks | ~10% |
| Septoria leaf spot | 220 | detached-leaf studio shots; a few other species | ~10% |
| Leaf mold | 239 | annotation overlays (drawn circles); text branding | ~10% |
| Mosaic virus | 189 | Shutterstock bars; cropped figure panels | ~5–10% |

The dominant failure mode is **systematic, not random**: for pathogens that visibly attack fruit
(late blight, early blight, bacterial spot), image searches return fruit. A leaf classifier
cannot be expected to classify fruit rot, so these images measure nothing about the model.

Note on ordering: this audit and its contamination estimates were recorded **before** any model
was run against the dataset.

---

## 3. Evaluation results

1,966 images, 8 of 10 classes present (Spider Mites and Target Spot absent), chance = 12.5%.
Models are the leak-free `_mobilenet` generation (`preprocess_input` + label smoothing 0.1).

### Table 1 — Lab vs field

| Strategy | PlantVillage (internal test) | PlantWild (field) | Gap | Top-3 |
|---|---|---|---|---|
| A — Color only | 89.21% | **17.19%** | −72.0 | 38.5% |
| B — Segmented only | 88.42% | **12.11%** | −76.3 | 33.0% |
| C — Random 50/50 mix | 87.42% | **14.45%** | −73.0 | 36.1% |
| D — Fine-tuned A→Seg | 90.92% | **16.63%** | −74.3 | 37.3% |
| E — Class-aware routing | 93.71% | **12.46%** | **−81.2** | 37.3% |

Chance = 12.5% top-1, 37.5% top-3.

**Two observations that matter more than the headline number:**

- **Top-3 is at chance.** 33–38.5% against a 37.5% baseline. On the withdrawn PlantDoc data top-3
  reached 51–57%, indicating the representation held signal the decision boundary misranked. Here
  there is no such reserve.
- **Strategy E has the largest gap of the five (−81.2) and finishes second-worst in the field.**
  Its +4.5-point internal advantage over Strategy A inverts to a 4.7-point deficit. Class-aware
  routing is a PlantVillage-specific gain, and the effect is stronger here than PlantDoc suggested.

### Table 2 — Per-class recall

| Class | n | A | B | C | D | E |
|---|---|---|---|---|---|---|
| Early Blight | 346 | 55.5% | 29.8% | 50.3% | 63.9% | 35.5% |
| Healthy | 226 | 54.0% | 23.5% | 25.7% | 27.9% | 13.3% |
| Yellow Leaf Curl Virus | 171 | 9.9% | 5.8% | 5.3% | 2.3% | 21.1% |
| Bacterial Spot | 280 | 0.7% | 13.6% | 6.1% | 7.5% | 7.9% |
| Leaf Mold | 239 | 1.3% | 10.5% | 8.4% | 7.5% | 7.5% |
| Late Blight | 295 | 0.0% | 2.7% | 1.0% | 0.0% | 4.7% |
| Mosaic Virus | 189 | 1.1% | 0.0% | 1.6% | 0.0% | 0.5% |
| Septoria Leaf Spot | 220 | 0.0% | 0.5% | 0.0% | 0.0% | 0.5% |

Only Early Blight and Healthy retain usable recall. Septoria, Late Blight and Mosaic Virus are
effectively zero across every strategy.

### Table 3 — Prediction collapse (Strategy A)

| Class | True | Predicted |
|---|---|---|
| Healthy | 226 | **848** |
| Early Blight | 346 | **811** |
| Leaf Mold | 239 | 182 |
| Yellow Leaf Curl Virus | 171 | 62 |
| Late Blight | 295 | 23 |
| Bacterial Spot | 280 | 11 |
| Mosaic Virus | 189 | 3 |
| Septoria Leaf Spot | 220 | **0** |
| Spider Mites (absent) | 0 | 11 |
| Target Spot (absent) | 0 | 15 |

1,659 of 1,966 predictions (84%) fall on two classes. Septoria receives none at all.
Hallucination onto the two absent classes is negligible (26 predictions, 1.3%).

### Table 4 — Calibration

| Generation | Mean confidence | When wrong |
|---|---|---|
| `_mobilenet` (label smoothing 0.1) | 52.0–62.8% | 52.4–62.6% |
| `_rescale` (no label smoothing) | 82.2–89.8% | — |

Label smoothing cuts out-of-domain confidence from ~86% to ~57% without improving accuracy. This
replicates on an independent dataset the one positive finding from the withdrawn evaluation, and
is the most defensible claim in this report: the models became appropriately uncertain rather
than confidently wrong.

---

## 4. Preprocessing generation comparison

| Strategy | `_rescale` field | `_mobilenet` field |
|---|---|---|
| A | 14.50% | 17.19% |
| B | 14.14% | 12.11% |
| C | 16.68% | 14.45% |
| D | 16.63% | 16.63% |
| E | 16.94% | 12.46% |

Both generations sit in the same 12–17% band. **Correct MobileNetV2 preprocessing does not
recover field performance** — consistent with the withdrawn evaluation's finding, now replicated
on different data. Its benefit is calibration, not accuracy.

---

## 5. Interpretation, and what cannot yet be claimed

The measured 12–17% is a **lower bound on model capability**, because the evaluation set contains
images the model was never designed to classify. Roughly 20–40% of the three fruit-affected
classes show lesions on fruit rather than foliage. Late Blight, at ~40% fruit contamination,
scores 0.0–4.7% — those numbers are not separable into "model failure" and "wrong task" without a
leaf-only filter.

**The single most valuable next step is a leaf-only filtered re-evaluation**: apply the exclusion
criteria fixed in section 2.4 (not a leaf photograph of the labelled species / composite figure or
slide / annotation or text overlay / wrong species) image by image, publish the excluded list, and
report raw and filtered accuracy side by side. Until that exists, the honest statement is a range,
not a point estimate.

What can be claimed now, safely:

1. All five strategies lose 72–81 points transferring from PlantVillage to in-the-wild imagery.
2. Class-aware routing's internal advantage does not transfer; Strategy E has the largest gap and
   is second-worst in the field despite being best internally.
3. Failure is disease-specific: Early Blight and Healthy retain partial signal; Septoria, Late
   Blight and Mosaic Virus retain essentially none.
4. Label smoothing substantially improves out-of-domain calibration without improving accuracy —
   replicated across two independent field datasets.
5. Correct preprocessing does not close the gap.

What cannot be claimed without the filtered re-evaluation: any precise field accuracy figure, and
any conclusion about *why* Late Blight and Bacterial Spot fail.

---

## 6. Recommendation on retraining with heavy augmentation

**Do not run it yet.** `scripts/pipeline.py --aug heavy` exists and works (rotation 40°, both
flips, shear, zoom 0.6–1.4, shift 0.25, brightness 0.5–1.5, channel shift 40), and a run was
started and stopped earlier. The evidence now argues against spending 4–5 hours on it:

- Top-3 accuracy is **at chance**. Augmentation redistributes decision boundaries within an
  existing representation; it cannot manufacture signal that is not there. Where PlantDoc showed a
  51–57% top-3 reserve worth attacking, this shows none.
- Augmentation cannot add what the training data lacks. PlantVillage is single leaves on plain
  backgrounds; brightness and rotation jitter do not synthesise soil, stems, occlusion or
  whole-plant framing. Strategies B and E draw on segmented images whose backgrounds are *black*,
  so for those the background gap is structurally unreachable.
- The evaluation set is contaminated. Retraining against a target that is 20–40% mis-specified in
  three classes risks optimising for the wrong thing and mis-reading the result either way.

Correct order: filter the evaluation set → re-measure → then decide on augmentation against a
trustworthy target. If a filtered re-evaluation shows a top-3 reserve, augmentation becomes worth
testing; if it does not, the finding is that the gap is dataset composition, and the paper should
say so.

---

## 7. Files

| Path | Contents |
|---|---|
| `scripts/field_evaluation.py` | Dataset-agnostic field evaluation; class mapping declared in-file |
| `outputs/field_validation/plantwild_mobilenet.json` | Full results, leak-free `_mobilenet` models |
| `outputs/field_validation/plantwild_rescale.json` | Full results, `_rescale` models |
| `validation/candidates/plantwild_tomato/` | PlantWild tomato subset, 1,966 images, 8 classes |
| `validation/candidates/Tomato-Village/` | Rejected for accuracy evaluation; Variant-c retains open-set value |
