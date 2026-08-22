# Research Paper Review Notes — "Beyond Uniform Augmentation" (Strategy E / Tomato Disease Classification)

Date: 2026-08-21
Scope: Review of `Research Paper.pdf` against the actual project code, notebooks, `project_technical_report.txt`, and `GEMINI.md`, with a focus on what is likely capping the paper's review score and how to raise it.

---

## 1. What the project is

Five training strategies for a MobileNetV2-based tomato leaf disease classifier (10 classes, PlantVillage dataset), differing only in image preprocessing domain:

- **Strategy A** — Color images only (baseline) — 89.06%
- **Strategy B** — Segmented images only (background removed) — 86.13%
- **Strategy C** — Random 50/50 color+segmented mix — 87.48%
- **Strategy D** — Fine-tuned: Strategy A → segmented — 88.42%
- **Strategy E** — Class-aware selective routing (each class routed to its best-performing domain) — **93.88%** (proposed contribution)

The paper additionally covers Grad-CAM explainability, a post-processing severity estimator, and a "generalization validation" fine-tuning experiment on the New Plant Diseases Dataset (NPDD).

---

## 2. Bottom line

The paper can likely score meaningfully higher than its current level — and much of the fix is *finishing work already started in the repo*, not new work. The biggest issues are a data leak in the headline result and a mismatch between what the paper reports and what the project's own records/notebooks show.

---

## 3. Issues found, ranked by impact

### Issue 1 — Data leakage in the core result (highest impact)
The per-class accuracy numbers used to *decide* the Strategy E routing (Table I: e.g. "Bacterial Spot: 99.3% color vs 78.7% segmented") were computed on the **test set** — the same test set later used to report Strategy E's final headline accuracy (93.88%). This means the routing decision was chosen using the same data later used to "prove" it works — a classic data leak that inflates the reported number.

**This is already flagged inside the repo's own notebooks.** A comment block in `01_evaluation_color.ipynb`, `02_evaluation_segmented.ipynb`, and `05_model_training_selective.ipynb` explicitly says:

> "The per-class numbers used to build CLASS_ROUTING in 05_model_training_selective.ipynb were taken from the TEST set evaluation above... a data leak that inflates Strategy E's reported performance, and a likely reason Strategy E generalised the WORST to field imagery despite having the BEST internal test score."

A partial fix already exists (cells that compute per-class accuracy on the **validation** split instead, exporting to JSON at `outputs/routing_analysis/strategy_{A,B,C}_val_per_class.json`), but:
- Those JSON files do not currently exist anywhere in the repo (never generated — the val-set cells were never run).
- `05_model_training_selective.ipynb`'s `CLASS_ROUTING` table is still explicitly marked in-code as `⚠️ LEGACY / TEST-SET DERIVED`.
- `model_E_selective_V3_final.h5` (the model behind the paper's 93.88%) was trained using this leaky, test-set-derived routing table.

**Fix:** Run the validation-set evaluation cells in notebooks 01/02/03, generate the JSON files, rebuild `CLASS_ROUTING` from validation data in notebook 05, retrain Strategy E, and report the honest (leak-free) test accuracy.

### Issue 2 — The paper claims generalization it has not measured
Section V-D ("Generalization Validation") supports its real-world claim with the NPDD fine-tuning
result alone. NPDD is a second *lab* dataset, so it evidences domain adaptation, not field
generalization. As it stands the section implies a capability the experiments do not establish.

Cross-dataset validation was previously run against PlantDoc, but that evaluation has been
**withdrawn and removed from this repository**: an image-level audit found the tomato subset
contains fruit-only stock illustrations, composite figures and lecture slides, watermarked stock
photography, wrong-species images, and diseased leaves labelled healthy. Scores against it are not
interpretable in either direction. The audit findings are recorded in `FINDINGS.md` section 3.

**Fix:** evaluate on a field-captured dataset with reliable labels and rewrite Section V-D from
that result, whatever it says. Until then the paper must scope its generalization claim to
lab-to-lab adaptation. Candidate replacement datasets and their own quality caveats are tracked in
`FINDINGS.md` section 3.

### Issue 3 — RESOLVED (2026-08-22): the paper was right, the internal docs were fabricated

This issue was originally filed as "the paper's numbers don't match the project's own
records," on the assumption that `project_technical_report.txt` and `GEMINI.md` were
run logs. They are not. Adjudicated against the saved cell outputs of
`notebooks/09_finetune_and_test_new_plant_diseases.ipynb` and the artifacts in
`outputs/npdd_validation/`:

| | Notebook 09 (authoritative) | Paper §V-D | Internal docs (as filed) |
|---|---|---|---|
| Learning rate | `1.0000e-05`, all 10 epochs | α=1×10⁻⁵ ✓ | 1e-4 → 5e-5 ✗ |
| Best val accuracy | 0.9411 | 94.11% ✓ | 97.21% ✗ |
| Delta vs 93.88% | +0.23% (printed) | +0.23% ✓ | +3.33% ✗ |
| Epochs | max 10, ran 10, EarlyStopping never fired | — | 14, stopped at 14 ✗ |
| Val trajectory | 81.16 → 87.00 → 87.63 → … → 94.11 | — | 90.95 → 93.13 → 91.84 → … ✗ |
| NPDD test (16 img) | 16/16, avg conf 75.54% | not cited | 15/16 = 93.75% ✗ |

**The paper matches the notebook on every checkable detail and needs no correction.**
§V-D never cites the 16-image test result, so nothing there needs fixing either.

The internal documents instead contain a detailed but unsupported account of a run
that does not exist: a fabricated 14-epoch table, a "system pause ~7hrs" at epoch 13,
and a causal explanation ("the jump from 95.27% to 97.21% happened exactly when
ReduceLROnPlateau halved the LR") for an event with no basis in the log. The test-set
claim is refuted directly by `test_predictions_NPDD.csv`, which records
`TomatoEarlyBlight2.JPG` as Early Blight at 89.22% confidence (Septoria probability
2.01%) — not a miss at 52%. Every per-image confidence in that table differed from the
CSV. Per-class validation F1s differed too (98.5/96.7/97.8… vs the notebook's
95.0/91.9/94.9…).

Given the filename, the most likely origin is that `GEMINI.md` is an LLM-written
summary that hallucinated plausible training details, which `project_technical_report.txt`
then inherited.

**Done:** §6/7/8 of `project_technical_report.txt` and the NPDD section of `GEMINI.md`
corrected to the notebook's figures, each with an inline CORRECTION note.

**Consequence — audited, see `DOC_AUDIT.md`:** both documents were checked in full against
the saved outputs of all 22 notebooks (plus notebook 08 recovered from `f741b5b`). The NPDD
section turned out to be the **only** fabricated region: 162 of 168 percentage claims are
corroborated exactly, all six orphans are benign, and every dataset count checks out except
`1,790` (NPDD Mosaic Virus, never printed anywhere). Strategies A-D and Table I transcribe
exactly.

One structural gap did surface: **93.88% — the paper's headline — has no notebook
provenance.** Notebook 05's recorded run failed on all 30 source paths and printed success
anyway; there is no Strategy E evaluation notebook; and `93.88` appears only in notebooks 08
and 09, which consume it as a baseline. The number is independently substantiated by the
leak-free re-runs (93.28% / 93.71%) and the model files on disk — so this is a
reproducibility problem, not a correctness one.

Caveat: a second, later NPDD run whose notebook was never saved cannot be formally
excluded. Against it — notebook 09 is the only one, and the CSV artifact on disk matches
it exactly while contradicting the docs; a later run would have overwritten that CSV.
NPDD is not on disk (~2.7 GB), so a re-run is not currently possible.

### Issue 4 — All reported models used the wrong preprocessing
Root-cause analysis (already done, in `project_technical_report.txt` Section 3) found every training notebook used `rescale=1./255` (maps pixels to [0,1]), but MobileNetV2 was pretrained expecting `preprocess_input()` (maps pixels to [-1,1]). This is invisible on the internal benchmark (test set uses the same wrong scaling) but degrades feature quality, and is part of why cross-dataset performance collapsed. The fix (`preprocess_input` + label smoothing 0.1) is already written into `scripts/patch_notebooks.py` and applied to the notebooks, but the `.h5` models the paper reports on were **not retrained** with the fix — meaning 93.88% may not even be the true ceiling of the architecture.

**Fix:** Retrain Strategies A–E with the patched notebooks and report the corrected numbers (this is item 1 on the repo's own "what still needs to be done" list in both `project_technical_report.txt` and `GEMINI.md`).

### Issue 5 — Smaller, easy-to-fix issues
- **Missing reference:** [4] (EfficientNet) is cited in the Introduction/Related Work text but absent from the reference list (jumps from [3] to [5]).
- **Mislabeled table reference:** Section V-A says "Per-class results in TABLE II reveal..." but the cited numbers (99.3%, 78.7%, etc.) are actually in Table I.
- **"Balanced" dataset is imprecise:** the standardization protocol *downsamples* classes above 1,000 images to exactly 1,000, but never *upsamples* classes below 1,000 (e.g., Mosaic Virus started at 373). Calling the resulting 9,325-image cohort "balanced" overstates it — "capped" is more accurate, or minority classes should actually be balanced via oversampling/augmentation.
- **No Limitations section** — journals typically expect one explicitly; currently limitations are scattered/implicit rather than stated.
- **No quantitative comparison table** against other published tomato-disease-classification accuracies (e.g., Mohanty et al., Brahimi et al.) — reviewers commonly expect this in Results or Related Work.
- **Single train/test split, no repeated runs** — no confidence intervals, no statistical significance testing (e.g., paired test) between strategies; all results come from one SEED=42 run each.
- **Grad-CAM and severity claims are qualitative only** — described with strong confidence language ("confirms," "validates," "surgical precision") but backed only by visual inspection, not a quantitative metric (e.g., activation-overlap with lesion regions).
- **Promotional tone** — words like "devastating," "catastrophic," "definitive," "mandatory next baseline" read as informal/hype for a formal journal (target venues: Computers and Electronics in Agriculture, IEEE Access, Applied AI) even though the underlying data supports the claims. Toning down the rhetoric while keeping the numbers would read as more rigorous.

---

## 4. Recommended priority order

**Tier 1 — highest score impact, mostly finishing work already started in the repo:**
1. Close the routing data leak: run the validation-set evaluation cells (01/02/03), generate the missing JSON files, rebuild `CLASS_ROUTING` in notebook 05 from validation data, retrain Strategy E, report the honest number.
2. Retrain Strategies A–E using the already-patched notebooks (`preprocess_input` + label smoothing) so all headline numbers reflect correct MobileNetV2 preprocessing.
3. Evaluate a field-captured dataset after retraining to measure real-world generalization.
4. ~~Reconcile the NPDD fine-tuning numbers~~ — **done 2026-08-22**; paper was correct, internal docs corrected. Replaced by: audit `project_technical_report.txt` and `GEMINI.md` in full against notebook outputs (see Issue 3).
5. Rewrite Section V-D from the field result once measured, and state the limitation plainly if the gap persists.

**Tier 2 — writing/polish fixes, low effort:**
6. Add an explicit Limitations section.
7. Add a comparison table vs. prior published results.
8. Fix the missing [4] reference and the Table I/II mislabeling.
9. Correct "balanced" → more precise language about class capping.
10. Tone down promotional language while keeping the data-driven claims.

**Tier 3 — stronger but higher effort:**
11. Multiple seeds / k-fold evaluation for confidence intervals.
12. Quantitative Grad-CAM validation metric (not just visual inspection).

---

## 5. Source material used for this review
- `Research Paper.pdf` (the paper itself)
- `project_technical_report.txt` (556-line full narrative, including root-cause analysis and fixes)
- `GEMINI.md` (project memory / status file)
- `README.md`
- Notebooks: `01_evaluation_color.ipynb`, `02_evaluation_segmented.ipynb`, `05_model_training_selective.ipynb` (inline code/comments confirming the data-leak issue and its partial, unfinished fix)
