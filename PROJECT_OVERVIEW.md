# Project Overview — Start to Now

**Last updated:** 2026-08-22
**Purpose:** one place to understand what this project is, what has actually been done,
what is wrong with it, and what to do about it. Written to be read start-to-finish by
someone who has lost track of the details.

Every other document in this repo is a *fragment* — a review, a findings log, an audit.
This one is the map. Where they disagree with each other, section 7 says which to trust.

---

## 1. What the project is, in one paragraph

You are classifying tomato leaf diseases from photographs — 10 categories (9 diseases plus
healthy) — using MobileNetV2, a small neural network that runs on a CPU. The training data
is PlantVillage, a public dataset of single leaves photographed against plain backgrounds
in a lab. The research question is not "can we classify leaves" (long solved) but something
narrower and more interesting: **does it help to feed different diseases different
*versions* of the same photograph?**

PlantVillage ships each leaf twice — once as a normal colour photo, once "segmented" with
the background cut away to black. The hypothesis is that some diseases are easier to spot
in colour (where the disease changes leaf colour) and others in segmented images (where
shape and texture matter and the background is a distraction). If that is true, routing each
disease class to whichever version suits it should beat using one version for everything.

That routing idea is **Strategy E**, and it is the paper's original contribution.

---

## 2. The five strategies

| | What it does |
|---|---|
| **A** | Train on colour images only. The baseline. |
| **B** | Train on segmented images only. |
| **C** | Train on a random 50/50 mix of both. The "naive mixing" control. |
| **D** | Train on colour, then fine-tune on segmented. The "sequential" control. |
| **E** | **Route each class to its best domain.** The contribution. |

C and D matter because they are the obvious alternatives. If E only beat A, a reviewer would
ask "did you just benefit from seeing more data?" C and D rule that out — they see the same
data, mixed without class-awareness, and do worse. That is what makes E a real finding.

---

## 3. What has actually been done — the timeline

### Phase 1 — The original work (through July 2026)
Five strategies trained and evaluated. Strategy E won at **93.88%**. Grad-CAM visualisations
produced (heatmaps showing where the model looks), a severity estimator built (how much of
the leaf is diseased), and Strategy E fine-tuned on a second lab dataset, NPDD. A paper was
written and a technical report and `GEMINI.md` summary produced.

### Phase 2 — The review (2026-08-21)
The paper was reviewed against the actual code. Five issues were found and written into
`PAPER_REVIEW_NOTES.md`. Two were serious enough to threaten the result.

### Phase 3 — Fixing them (2026-08-22, this work)
Issues 1, 3 and 4 resolved; Issue 2 resolved with an unwelcome answer; Issue 5 still open.
A documentation audit was added after Issue 3 turned up fabricated content. Details below.

---

## 4. Where every number stands today

### The internal benchmark (PlantVillage) — trustworthy

These are the leak-free numbers, produced by `scripts/pipeline.py`, with correct MobileNetV2
preprocessing. **These supersede everything in the paper, the README, and the older docs.**

| Strategy | Paper (leaky) | Leak-free, `1./255` | **Leak-free + `preprocess_input`** |
|---|---|---|---|
| A — colour | 89.06% | 89.35% | **89.21%** |
| B — segmented | 86.13% | 87.71% | **88.42%** |
| C — random mix | 87.48% | 86.56% | **87.42%** |
| D — fine-tuned | 88.42% | 87.35% | **90.92%** |
| **E — routing** | **93.88%** | **93.28%** | **93.71%** |

**The contribution survives.** Two independent honest protocols land within 0.6 points of
the published figure, and E still beats the best single-domain baseline by ~3-4 points.

### The field benchmark (PlantWild) — trustworthy as a comparison, not as a score

| Strategy | Internal | Field | Gap |
|---|---|---|---|
| A | 89.21% | 17.19% | −72.0 |
| B | 88.42% | 12.11% | −76.3 |
| C | 87.42% | 14.45% | −73.0 |
| D | 90.92% | 16.63% | −74.3 |
| **E** | **93.71%** | **12.46%** | **−81.2 (worst)** |

Chance is 12.5%. Top-3 accuracy is also at chance. **Strategy E's advantage does not
transfer** — it has the largest gap of the five and finishes second-worst in the field.

### NPDD fine-tuning — trustworthy, and smaller than previously claimed
**94.11%** validation accuracy, learning rate 1e-5, best of 10 epochs. This is adaptation
to a second *lab* dataset — it is not evidence of field performance.

---

## 5. The issues — what is wrong and how to fix it

### Issue 1 — Data leakage in the headline result ✅ RESOLVED

**What it was.** To decide the routing table ("Bacterial Spot goes to colour"), per-class
accuracies were measured **on the test set**. Strategy E was then scored on that *same* test
set. The routing was tuned using the data later used to prove it works — so the 93.88% was
partly measuring its own answer key.

**Why it mattered.** This is the single most common reason a machine-learning paper gets
rejected. A reviewer who spots it discards the headline number.

**How it was fixed.** Routing is now derived only from the validation split; the test split
is touched exactly once, at the end. A second, previously unnoticed leak in Strategy C was
also removed (its mixed set had been re-split from pooled images, so the same leaf could
appear in both train and test).

**Outcome:** 93.28% / 93.71% versus the published 93.88%. The leak was worth about 0.6
points — inside normal run-to-run noise. **The finding was real.**

**Still to do in the paper:** Table I's showcase number is inflated. "Bacterial Spot: 99.3%
colour vs 78.7% segmented, +20.6%" is honestly **96.7% vs 91.3%, +5.4** — a quarter of the
published gap. Bacterial Spot is also a bad example: it contributes *nothing* to E's gain.
Use Septoria (+16.0) or Target Spot (+15.3) instead.

---

### Issue 2 — The paper claims real-world generalization it never measured ✅ RESOLVED, badly

**What it was.** Section V-D claims the model generalizes, citing the NPDD fine-tuning
result. But NPDD is another *lab* dataset. Adapting from one lab dataset to another says
nothing about photographs taken in an actual field, which is what the paper implies.

**What was done.** Four candidate field datasets were audited. Three were rejected:

- **PlantDoc** — web-scraped and badly contaminated: stock illustrations of tomato *fruit*,
  a lecture slide with its title text, a nine-panel figure grid, chopped herbs on a cutting
  board, wrong species, diseased leaves labelled healthy, and Shutterstock/Alamy watermarks
  on roughly a third of images. An earlier evaluation against it was withdrawn.
- **Tomato-Village Variant-a** — sound provenance, but the images are detached leaves on
  white sheets (the same domain as PlantVillage, so it measures nothing), and 1,616 files
  turned out to be 391 real photographs plus rotated copies, with its own splits leaking.
- **Tomato-Village Variant-c** — genuinely field-captured, but only **one** of its classes
  overlaps your 10 labels. Cannot score a ten-class model on one class.

**PlantWild** was accepted with reservations and evaluated. Result: **12-17% accuracy
against a 12.5% chance baseline**, with top-3 also at chance.

**What this means.** The models do not work on in-the-wild photographs. More pointedly,
**Strategy E's advantage inverts** — best internally, near-worst in the field.

**The honest caveat.** PlantWild is also web-scraped: for diseases that attack fruit (late
blight, early blight, bacterial spot), image search returns fruit, so 20-40% of those
classes are photographs a leaf classifier could never get right. So 12-17% is a *lower
bound*, not a measurement.

**What is safe to claim right now, with no further work:**
1. All five strategies lose 72-81 points going from lab to field.
2. Class-aware routing's advantage does not transfer — E has the largest gap.
3. Label smoothing cuts out-of-domain overconfidence from ~86% to ~57% without hurting
   accuracy. The model becomes appropriately uncertain instead of confidently wrong. This is
   the one clean *positive* result, and it replicates across two independent datasets.

Claims 1-3 are robust to the contamination, because **all five strategies were scored on the
identical images** — a bad image is equally unclassifiable for every model, so the
*comparison between strategies* is fair even though the *absolute number* is not.

**How to finish it:**
- **Reframe Section V-D around the relative finding** ("the routing advantage does not
  transfer"), not around a field accuracy figure. Zero compute. Do this first.
- Add a paired McNemar test between A and E on the shared images — makes the inversion
  statistical rather than descriptive.
- *Optional, recovers an absolute number:* audit PlantWild image-by-image, exclude non-leaf
  images, publish the exclusion list, report raw and filtered side by side. The compute is
  minutes; the cost is the human audit (~1,150 images carry most of the contamination).
- *Optional, cheap and valuable:* the withdrawn PlantDoc result can be cited for the
  *relative* finding only — it shows the same inversion (E's gap −70.0, largest of five).
  That turns one observation into two.

> **Note.** This is not a failure of the project. A controlled demonstration that a
> benchmark-winning mechanism is benchmark-*specific* is a genuine contribution — the
> PlantVillage literature is full of 99% accuracy claims that have never been tested this
> way. Reported honestly, this strengthens the paper. Suppressing it would not.

---

### Issue 3 — Paper numbers didn't match the project's records ✅ RESOLVED (backwards)

**What it was.** The paper said NPDD fine-tuning reached 94.11% at learning rate 1e-5. The
technical report and `GEMINI.md` said 97.21% at 1e-4 halved to 5e-5. Both cannot be right,
and it was assumed the paper was wrong.

**What was found.** The opposite. Notebook 09's saved output — the actual execution record —
matches the **paper** exactly on every checkable detail: 1e-5, 94.11%, +0.23%, 10 epochs.

The internal documents describe a run **that never happened**: a fabricated 14-epoch table,
a "system pause ~7hrs" at epoch 13, and an invented explanation for a jump to 97.21%. Their
test-set claim (15/16, one image missed as Septoria at 52% confidence) is contradicted by
your own CSV on disk, which records that image as correctly predicted at 89.22%, with all 16
correct.

The likely cause is benign: `GEMINI.md` appears to be an LLM-written summary that
hallucinated plausible-sounding training details, which the technical report then inherited.

**Fixed.** Both documents corrected from the notebook, each with an inline CORRECTION note.
The README carried the same false numbers and has been corrected too. **The paper needs no
change.**

---

### Issue 4 — Wrong image preprocessing ✅ RESOLVED

**What it was.** Every training notebook scaled pixels to [0,1] using `rescale=1./255`, but
MobileNetV2 was pretrained expecting [-1,1] via `preprocess_input()`. Invisible on the
internal benchmark, because the test set was scaled the same wrong way — but it degrades the
features the pretrained network provides.

**Fixed.** All strategies retrained with `preprocess_input` plus label smoothing 0.1. Both
generations of models exist on disk (`strategy_*_rescale_*.keras` and
`strategy_*_mobilenet_*.keras`), so results can be compared across the fix.

**Worth knowing:** correcting preprocessing did *not* recover field performance (12-17%
either way). Its real benefit is calibration — see Issue 2, claim 3.

---

### Issue 5 — Smaller paper problems ⬜ OPEN

None are hard; all are things reviewers routinely catch.

| Problem | Fix |
|---|---|
| Reference [4] cited but missing from the reference list | Add it |
| Section V-A cites Table II for numbers that are in Table I | Fix the cross-reference |
| Dataset called "balanced" | It is **capped** — classes over 1,000 were cut down, but minority classes were never boosted. Change the word |
| No Limitations section | Add one. You now have real material for it |
| No comparison against prior published results | Add a table (Mohanty et al., Brahimi et al.) |
| Single seed, no confidence intervals | Re-run with 3-5 seeds, report mean ± spread |
| Grad-CAM and severity claims stated as proven ("confirms", "validates") but only ever eyeballed | Either add a quantitative metric or soften to "consistent with" |
| Promotional tone — "devastating", "catastrophic", "definitive", "mandatory next baseline" | Tone down. The numbers are strong enough without it |

---

### Issue 6 (new, from the documentation audit) ⬜ OPEN

Issue 3 raised the question of whether the two internal documents were unreliable
*throughout*. They were audited in full against the saved outputs of all 22 notebooks.
Result in `DOC_AUDIT.md`: **the NPDD section was the only fabricated region.** 162 of 168
percentage claims corroborate exactly; Strategies A-D and Table I transcribe perfectly.

But the audit surfaced one structural gap:

**The paper's headline number, 93.88%, has no notebook provenance.**

- Notebook 05 (Strategy E training) has a saved run that **failed on all 30 source paths**
  — Windows `D:\` paths that did not exist — and then printed
  `✅ Strategy E Dataset Assembly Complete!` anyway. Nothing was trained.
- There is no Strategy E evaluation notebook. A, B, C and D each have one.
- `93.88` appears only in notebooks 08 and 09, which *use* it as a comparison baseline.
  Neither produces it.

**This does not mean the number is wrong.** The leak-free re-runs independently produced
93.28% and 93.71%, and the model files exist. The number is substantiated; its original
*run record* is missing. This is a reproducibility problem, not a correctness one — but a
reviewer who opens the public repo today cannot reproduce the headline result.

**Fix:** the notebooks need to be brought in line with `scripts/pipeline.py` — Linux paths,
routing rebuilt from validation data, and the unconditional success banner in notebook 05
removed so a failed run cannot report success again.

---

## 6. What is trustworthy right now

| | Status |
|---|---|
| Strategy E beats A-D on PlantVillage by ~3-4 points | ✅ Solid — survives an honest protocol, verified twice |
| Strategies A/B/C/D internal accuracies | ✅ Verified exactly against notebooks |
| Table I per-class numbers | ⚠️ Correctly transcribed, but leak-inflated. Use the leak-free versions |
| The specific routing table (which class → which domain) | ⚠️ **Unstable.** Two runs differing only in pixel scaling agree on just 6 of 10 classes. The gain comes from routing per class *at all*, not from the specific assignments — cut the per-class biological explanations |
| NPDD 94.11% | ✅ Verified against notebook 09 |
| Field performance is catastrophic and E's advantage inverts | ✅ Solid as a comparison; ⚠️ the absolute 12-17% is a lower bound |
| Label smoothing improves out-of-domain calibration | ✅ Solid — replicated on two datasets |
| Grad-CAM and severity interpretations | ⚠️ Never quantitatively verified |
| Anything in `project_technical_report.txt` / `GEMINI.md` about NPDD | ✅ Now corrected |
| Notebooks reproducing the headline number | ❌ They do not. `scripts/pipeline.py` is the only working path |

---

## 7. Which file to trust for what

The docs have accumulated and partly contradict each other. Order of authority:

1. **`scripts/pipeline.py` + `scripts/field_evaluation.py`** — the code that actually runs.
   Ultimate authority.
2. **`FINDINGS.md`** — the consolidated result of the re-validation work. Authority for the
   internal benchmark, and its section 3 now carries the field result as well.
3. **`outputs/field_validation/FIELD_VALIDATION_REPORT.md`** — authority for everything
   field-related, including the dataset audits.
4. **`DOC_AUDIT.md`** — what in the older documents can and cannot be trusted.
5. **`PAPER_REVIEW_NOTES.md`** — the issue list and its status.
6. **`project_technical_report.txt`, `GEMINI.md`, `README.md`** — narrative history. NPDD
   sections corrected; treat other unverified specifics with care.
7. **`RESULTS.md`** — chronological run log.
8. **`Research Paper.pdf`** — the current draft. Correct on NPDD; still carries the leaky
   Table I and the unsupported Section V-D.

---

## 8. What to do next, in order

**Housekeeping — done.** All the work described here is committed and merged to `main`
(PR #2), and `FINDINGS.md` section 3 and the README now carry the field result rather than
describing it as pending.

**Then — the paper, in this order:**

1. **Rewrite Section V-D** around the relative field finding plus the calibration result.
   No compute needed, and it is what makes the paper honest.
2. **Fix Table I** with the leak-free numbers and swap the showcase class to Septoria.
3. **Cut the per-class routing rationales.** The routing table is not stable enough to
   support "background colour is the signal for Bacterial Spot"-type explanations. Keep the
   four assignments that were stable across both runs; report the rest as a majority vote.
4. **Add a Limitations section.** You have unusually good material: the leak and its
   correction, the routing instability, the field collapse, and the dataset-quality audits.
5. **Work through Issue 5's list.** Mechanical, and each one is a point a reviewer will
   otherwise raise.

**Optional, in value order:**
- Cite the withdrawn PlantDoc result for the relative finding only (cheap, doubles your
  replication).
- Multi-seed runs for confidence intervals (~4h compute) — directly answers Issue 5.
- Leaf-only filtered PlantWild re-evaluation (recovers an absolute field number).
- Tomato-Village Variant-c as an open-set safety test: five field-photographed conditions
  the model was never trained on. Would likely yield a strong *positive* deployment result
  to pair with the negative transfer finding.
- Port the notebooks to Linux paths so the public repo reproduces (Issue 6).

**Explicitly not recommended:** retraining with heavy augmentation. `--aug heavy` exists and
works, but top-3 accuracy is already at chance, meaning there is no latent signal for
augmentation to recover — and augmentation cannot synthesise soil, stems and occlusion that
the training data never contained. Filter the evaluation set first, then decide.

---

## 9. The honest summary

The core contribution is real and survived an honest re-test. What the work does not support
— and what the paper currently implies — is that this helps in a real field. Reported as
"class-aware routing gives a reliable in-domain gain that does **not** survive domain shift,
and here is the controlled evidence," this is a more valuable paper than the one that claims
field readiness without having measured it. Most of the remaining work is writing, not
computing.
