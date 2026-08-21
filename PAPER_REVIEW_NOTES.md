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

> "The per-class numbers used to build CLASS_ROUTING in 05_model_training_selective.ipynb were taken from the TEST set evaluation above... a data leak that inflates Strategy E's reported performance, and a likely reason Strategy E generalised the WORST to PlantDoc despite having the BEST internal test score."

A partial fix already exists (cells that compute per-class accuracy on the **validation** split instead, exporting to JSON at `outputs/routing_analysis/strategy_{A,B,C}_val_per_class.json`), but:
- Those JSON files do not currently exist anywhere in the repo (never generated — the val-set cells were never run).
- `05_model_training_selective.ipynb`'s `CLASS_ROUTING` table is still explicitly marked in-code as `⚠️ LEGACY / TEST-SET DERIVED`.
- `model_E_selective_V3_final.h5` (the model behind the paper's 93.88%) was trained using this leaky, test-set-derived routing table.

**Fix:** Run the validation-set evaluation cells in notebooks 01/02/03, generate the JSON files, rebuild `CLASS_ROUTING` from validation data in notebook 05, retrain Strategy E, and report the honest (leak-free) test accuracy.

### Issue 2 — The paper hides its most interesting finding (PlantDoc domain-shift failure)
Cross-dataset validation (`08_cross_dataset_validation_plantdoc.ipynb`) shows **all 5 strategies score only ~22–25% on real-world PlantDoc field images** — barely above the 12.5% random-chance baseline for 8 classes:

| Strategy | Internal (PlantVillage) | PlantDoc (real-world) | Gap |
|---|---|---|---|
| A | 89.06% | 22.98% | −66.1% |
| B | 86.13% | 24.21% | −61.9% |
| C | 87.48% | 24.49% | −63.0% |
| D | 88.42% | 21.61% | −66.8% |
| E | 93.88% | 24.35% | −69.5% (**largest gap of all 5**) |

The paper's Section V-D ("Generalization Validation") omits this entirely and shows only the flattering NPDD fine-tuning result. If a reviewer checks the public repo (which contains the PlantDoc notebook, root-cause analysis, and fix), the gap between what the paper implies and what the code shows reads as selective reporting / cherry-picking — more damaging to credibility than the low number itself.

**Fix — reframe as a strength, not a hidden weakness:** report the PlantDoc failure honestly, walk through the root-cause analysis already done in `project_technical_report.txt` (wrong MobileNetV2 preprocessing, insufficient augmentation for lighting/background variation, overconfident loss function), then show the fix (proper `preprocess_input`, label smoothing, NPDD fine-tuning) recovering performance. "Found a real failure → diagnosed it → fixed it" is a stronger, more publishable narrative than "everything worked."

### Issue 3 — Paper's numbers don't match the project's own records
- **Paper (Section V-D):** NPDD fine-tuning achieved 94.11% validation accuracy, learning rate 1e-5, unfroze top 30 layers.
- **`project_technical_report.txt` / `GEMINI.md` (actual run logs):** NPDD fine-tuning achieved **97.21%** validation accuracy, learning rate started at 1e-4 and was halved to 5e-5 via `ReduceLROnPlateau`, same 30-layer unfreezing.

These cannot both be correct. Any reviewer cross-checking code against paper will find this, and a single confirmed mismatch undermines trust in every other number in the paper.

**Fix:** Reconcile — determine which run is authoritative (re-run notebook 09 if needed) and correct the paper to match the actual notebook output.

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
3. Re-run notebook 08 (PlantDoc) after retraining to measure real improvement from the fixes.
4. Reconcile the NPDD fine-tuning numbers in the paper (94.11%/1e-5) against the actual notebook/report output (97.21%/1e-4→5e-5).
5. Fold the honest PlantDoc failure → root-cause → fix narrative into the paper instead of omitting it.

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
