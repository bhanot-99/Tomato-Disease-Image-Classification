#!/usr/bin/env python3
"""
Definitive PlantDoc cross-dataset evaluation for the paper (Issue 2).

Supersedes notebook 08 and scripts/eval_plantdoc.py. Evaluates the five
Issue-1 generation models (val-derived routing + mobilenet_v2.preprocess_input)
alongside the five paper-generation models, on the pooled PlantDoc tomato set,
and emits every number and figure the paper needs.

Each model is evaluated with the input scaling it was TRAINED with. Notebook 08
applied preprocess_input to models trained with rescale=1./255 while claiming it
matched training; that is corrected here (it is worth ~1 point either way).

Outputs per model: overall accuracy, top-3 accuracy, per-class recall, mean and
median confidence, confidence when correct vs wrong, rate of prediction onto the
two classes absent from PlantDoc, and full 10x10 confusion matrices.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLANTDOC = ROOT / "validation" / "PlantDoc-Dataset"
MODEL_DIR = ROOT / "models"
OUT = ROOT / "outputs" / "cross_dataset_validation"

IMAGE_SIZE = (224, 224)
BATCH = 32

PLANTVIL_CLASSES = [
    "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold",
    "Septoria Leaf Spot", "Spider Mites", "Target Spot",
    "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy",
]
ABSENT = [5, 6]  # no PlantDoc equivalent

PLANTDOC_TO_PLANTVIL = {
    "Tomato leaf bacterial spot": 0,
    "Tomato Early blight leaf": 1,
    "Tomato leaf late blight": 2,
    "Tomato mold leaf": 3,
    "Tomato Septoria leaf spot": 4,
    "Tomato leaf yellow virus": 7,
    "Tomato leaf mosaic virus": 8,
    "Tomato leaf": 9,
}

# label, file, description, training scaling, internal PlantVillage test accuracy
MODELS = [
    ("A", "strategy_A_mobilenet_final.keras", "Color only", "mobilenet", 0.8921),
    ("B", "strategy_B_mobilenet_final.keras", "Segmented only", "mobilenet", 0.8842),
    ("C", "strategy_C_mobilenet_final.keras", "Random 50/50 mix", "mobilenet", 0.8742),
    ("D", "strategy_D_mobilenet_final.keras", "Fine-tuned A->Seg", "mobilenet", 0.9092),
    ("E", "strategy_E_mobilenet_final.keras", "Class-aware routing", "mobilenet", 0.9371),
]
LEGACY = [
    ("A_paper", "model_A_color_final.h5", "Color only (paper)", "rescale", 0.8906),
    ("B_paper", "model_B_segmented_final.h5", "Segmented only (paper)", "rescale", 0.8613),
    ("C_paper", "model_C_Mixed_final.h5", "Random 50/50 mix (paper)", "rescale", 0.8748),
    ("D_paper", "model_D_finetuned_final.h5", "Fine-tuned A->Seg (paper)", "rescale", 0.8842),
    ("E_paper", "model_E_selective_V3_final.h5", "Class-aware routing (paper)", "rescale", 0.9388),
]


def load_images():
    from tensorflow.keras.preprocessing import image as ki
    arrs, recs = [], []
    for folder, idx in PLANTDOC_TO_PLANTVIL.items():
        n = 0
        for split in ("train", "test"):
            d = PLANTDOC / split / folder
            if not d.exists():
                continue
            for p in sorted(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))):
                try:
                    arrs.append(ki.img_to_array(ki.load_img(p, target_size=IMAGE_SIZE)))
                except Exception as e:
                    print(f"  ! skip {p.name}: {e}")
                    continue
                recs.append({"true": idx, "folder": folder, "split": split})
                n += 1
        print(f"  {folder:<30} -> {PLANTVIL_CLASSES[idx]:<24} {n:>4} images")
    return np.stack(arrs), recs


def scale(arr, mode):
    if mode == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input(arr.copy())
    return arr / 255.0


def evaluate(probs, y):
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == y
    top3 = np.argsort(-probs, 1)[:, :3]
    per_class = {}
    for folder, i in PLANTDOC_TO_PLANTVIL.items():
        m = y == i
        per_class[PLANTVIL_CLASSES[i]] = {
            "n": int(m.sum()),
            "recall": float(correct[m].mean()),
            "mean_confidence": float(conf[m].mean()),
        }
    cm = np.zeros((10, 10), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    return {
        "overall": float(correct.mean()),
        "top3": float((top3 == y[:, None]).any(1).mean()),
        "mean_confidence": float(conf.mean()),
        "median_confidence": float(np.median(conf)),
        "confidence_when_correct": float(conf[correct].mean()),
        "confidence_when_wrong": float(conf[~correct].mean()),
        "predicted_absent_class_rate": float(np.isin(pred, ABSENT).mean()),
        "prediction_distribution": {PLANTVIL_CLASSES[i]: int((pred == i).sum()) for i in range(10)},
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def figures(results, y):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [m[0] for m in MODELS]
    internal = [m[4] for m in MODELS]
    field = [results["models"][m[0]]["overall"] for m in MODELS]
    chance = 1 / len(PLANTDOC_TO_PLANTVIL)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    ax.bar(x - 0.2, internal, 0.4, label="PlantVillage (internal test)", color="#2E7D32")
    ax.bar(x + 0.2, field, 0.4, label="PlantDoc (field)", color="#C62828")
    ax.axhline(chance, ls="--", c="gray", lw=1, label=f"chance ({chance:.1%})")
    for i, (a, b) in enumerate(zip(internal, field)):
        ax.text(i, max(a, b) + 0.03, f"-{(a-b)*100:.0f} pts", ha="center", fontsize=8, color="#C62828")
    ax.set_xticks(x); ax.set_xticklabels([f"Strategy {l}" for l in labels])
    ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1.1)
    ax.set_title("Lab vs field accuracy, all five strategies")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_internal_vs_plantdoc.png", dpi=200); plt.close(fig)

    names = [PLANTVIL_CLASSES[i] for i in sorted(PLANTDOC_TO_PLANTVIL.values())]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.15
    for j, (lab, *_rest) in enumerate(MODELS):
        vals = [results["models"][lab]["per_class"][n]["recall"] for n in names]
        ax.bar(np.arange(len(names)) + (j - 2) * w, vals, w, label=f"Strategy {lab}")
    ax.axhline(chance, ls="--", c="gray", lw=1)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Per-class recall on PlantDoc")
    ax.set_title("Field failure is concentrated in specific diseases")
    ax.legend(fontsize=8, ncol=5); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_plantdoc_per_class.png", dpi=200); plt.close(fig)

    cm = np.array(results["models"]["E"]["confusion_matrix"], float)
    keep = sorted(PLANTDOC_TO_PLANTVIL.values())
    sub = cm[np.ix_(keep, range(10))]
    sub = sub / np.maximum(sub.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(sub, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_xticklabels(PLANTVIL_CLASSES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(keep))); ax.set_yticklabels([PLANTVIL_CLASSES[i] for i in keep], fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Strategy E on PlantDoc — predictions collapse onto three classes")
    for a in range(sub.shape[0]):
        for b in range(sub.shape[1]):
            if sub[a, b] > 0.02:
                ax.text(b, a, f"{sub[a,b]:.0%}", ha="center", va="center", fontsize=6,
                        color="white" if sub[a, b] > 0.5 else "black")
    fig.colorbar(im, shrink=0.8); fig.tight_layout()
    fig.savefig(OUT / "fig_plantdoc_confusion_E.png", dpi=200); plt.close(fig)
    print(f"  figures -> {OUT}")


def main():
    if not PLANTDOC.exists():
        print(f"PlantDoc not found at {PLANTDOC}", file=sys.stderr)
        return 1
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading PlantDoc (train+test pooled):")
    arr, recs = load_images()
    y = np.array([r["true"] for r in recs])
    print(f"\nTotal {len(y)} images, {len(PLANTDOC_TO_PLANTVIL)} classes, chance "
          f"{1/len(PLANTDOC_TO_PLANTVIL):.1%}\n")

    cache = {}
    results = {
        "dataset": "PlantDoc (tomato subset, train+test pooled)",
        "n_images": int(len(y)),
        "n_classes_present": len(PLANTDOC_TO_PLANTVIL),
        "classes_absent": [PLANTVIL_CLASSES[i] for i in ABSENT],
        "chance_level": 1 / len(PLANTDOC_TO_PLANTVIL),
        "class_counts": {PLANTVIL_CLASSES[i]: int((y == i).sum())
                         for i in sorted(PLANTDOC_TO_PLANTVIL.values())},
        "models": {}, "legacy_models": {},
    }

    for target, spec in (("models", MODELS), ("legacy_models", LEGACY)):
        print(f"\n{'='*74}\n{'Issue-1 generation' if target=='models' else 'Paper generation'}\n{'='*74}")
        for lab, fname, desc, mode, internal in spec:
            path = MODEL_DIR / fname
            if not path.exists():
                print(f"  ! {fname} missing -- skipped")
                continue
            if mode not in cache:
                cache[mode] = scale(arr, mode)
            model = tf.keras.models.load_model(path, compile=False)
            probs = model.predict(cache[mode], batch_size=BATCH, verbose=0)
            r = evaluate(probs, y)
            r.update({"file": fname, "description": desc, "scaling": mode,
                      "internal_test_accuracy": internal,
                      "generalization_gap": internal - r["overall"]})
            results[target][lab] = r
            print(f"  {lab:<8} {desc:<28} field={r['overall']:6.2%}  top3={r['top3']:6.2%}  "
                  f"conf={r['mean_confidence']:5.1%}  gap=-{r['generalization_gap']*100:.1f}pts")
            del model
            tf.keras.backend.clear_session()

    (OUT / "plantdoc_final_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT/'plantdoc_final_results.json'}")
    figures(results, y)
    return 0


if __name__ == "__main__":
    sys.exit(main())
