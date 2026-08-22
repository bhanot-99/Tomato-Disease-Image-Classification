# Documentation Audit — `project_technical_report.txt` and `GEMINI.md`

**Date:** 2026-08-22
**Trigger:** Issue 3 (`PAPER_REVIEW_NOTES.md`) found both documents describing an NPDD
fine-tuning run that never happened. Since neither is a machine-written log, every number
sourced solely from them had to be treated as unverified.
**Question:** is the NPDD section an isolated failure, or are these documents unreliable
throughout?

**Answer: isolated.** Apart from the NPDD section, both documents are faithful to the
notebooks. One structural gap remains, and it concerns the paper's headline number.

---

## 1. Method

Saved cell outputs were extracted from all 22 notebooks in `notebooks/`, plus
`08_cross_dataset_validation_plantdoc.ipynb` recovered from commit `f741b5b` (deleted from
the working tree with the withdrawn PlantDoc evaluation). Every percentage and every
dataset count asserted in the two documents was then checked against that corpus.

Notebook outputs are authoritative for *what was run*. They are not authoritative for what
the paper should report — Issues 1 and 4 established that the notebook generation carries
the routing leak and the `rescale=1./255` preprocessing error. This audit asks only whether
the documents faithfully record their own runs.

## 2. Percentage claims

| Document | Claims checked | Corroborated | Orphaned |
|---|---|---|---|
| `project_technical_report.txt` | 113 | 109 | 4 |
| `GEMINI.md` | 55 | 53 | 2 |

All six orphans are benign:

- `6.6%` (Leaf Mold recall) and `-69.5%` (Strategy E generalization gap) trace to notebook
  08's saved output. Sourced, but **superseded twice** — first by the final PlantDoc
  evaluation, then by the PlantWild evaluation. They should be removed or re-labelled as
  historical, since the run they describe was withdrawn.
- The remaining four are figures newly derived from
  `outputs/npdd_validation/test_predictions_NPDD.csv` while correcting the NPDD section
  (per-class mean confidences), plus one quotation inside a CORRECTION note.

**No fabricated numbers were found outside the NPDD section.**

## 3. Dataset counts

Every count the documents cite is verifiable in notebook output — 6,527 / 1,399 / 1,462 /
18,345 / 4,585 / 261 — with one exception:

- **`1,790`** — asserted as the NPDD Mosaic Virus training count in the claim that a "6.8x
  increase in Mosaic Virus training data" (261 → 1,790) fixed that class. NPDD per-class
  counts are never printed in any notebook. The figure is consistent with the stated 6.8x
  multiplier but is unverifiable, and the causal claim built on it was never tested.
  Hedged in the corrected §7; `GEMINI.md` still carries the "6.8x" phrasing.

## 4. Results tables — verified exactly

| Claim | Source | Status |
|---|---|---|
| Strategy A 89.06% | `01_evaluation_color` | exact |
| Strategy B 86.13% | `02_evaluation_segmented` | exact |
| Strategy C 87.48% | `03_evaluation_mixed` | exact |
| Strategy D 88.42% | `04_evaluation_finetuned` | exact |
| Table I showcase: Bacterial Spot 99.3% color / 78.7% segmented | `01`/`02` per-class recall (0.993 / 0.787) | exact |

Table I is faithfully transcribed. Its problem is the routing leak (Issue 1), not
transcription — a distinction worth keeping straight when revising the paper.

Note: Strategy C is evaluated on 1,462 test images against 1,399 for A/B/D, the visible
signature of the Strategy C re-split leak recorded in `FINDINGS.md` §4.

## 5. The one structural gap: 93.88% has no notebook provenance

**Strategy E's 93.88% — the paper's headline result — cannot be traced to any notebook.**

- `05_model_training_selective.ipynb` is the Strategy E training notebook. Its saved output
  shows the dataset assembly failing on **all 30** source paths (10 classes × 3 splits) with
  `FAILED: Source not found: D:\Development\...`, then printing
  `✅ Strategy E Dataset Assembly Complete!` regardless. No training ran; no accuracy was
  produced. This is a silent-failure bug — the success banner is unconditional — and is
  worth fixing independently of the audit.
- There is no Strategy E evaluation notebook. Strategies A–D each have one; E does not.
- `93.88` appears only in notebooks 08 and 09, both of which *consume* it as a comparison
  baseline. Neither produces it.

**This does not mean the number is wrong.** It is independently corroborated: the leak-free
re-runs through `scripts/pipeline.py` produced 93.28% and 93.71% under two preprocessing
regimes, and the trained model files (`model_E_selective_V3_final.h5` and the V1/V2
variants) exist on disk. The number is *substantiated*; its *original run record* is
missing.

The practical consequence is reproducibility, not correctness — and it is sharper than the
caveat currently in `FINDINGS.md` §5, which says notebook 05 "still holds the legacy
test-derived `CLASS_ROUTING`." That understates it: notebook 05's recorded run produced
nothing at all. `scripts/pipeline.py` is the only executable path to Strategy E.

## 6. Actions

| Item | Status |
|---|---|
| Correct NPDD sections in both documents | done (Issue 3) |
| Re-label or remove the superseded notebook-08 figures (`6.6%`, `-69.5%`) in §3 of the report | open |
| Hedge or drop the unverifiable `1,790` / "6.8x" claim in `GEMINI.md` | open |
| Sharpen the `FINDINGS.md` §5 reproducibility caveat to state that notebook 05 produced no run | open |
| Fix notebook 05's unconditional success banner | open |
| Add a Strategy E evaluation path so the headline number has a reproducible source | open (largely covered by `scripts/pipeline.py`) |

## 7. Scope not covered

This pass checked numeric claims. It did not verify prose claims that assert *mechanism*
rather than measurement — the Grad-CAM interpretations and the severity-estimator grading,
both of which `PAPER_REVIEW_NOTES.md` Issue 5 already flags as qualitative-only and stated
with unearned confidence. Those notebooks do run and do produce output; what is unverified
is the interpretation laid over it.
