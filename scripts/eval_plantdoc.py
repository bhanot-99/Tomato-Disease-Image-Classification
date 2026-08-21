#!/usr/bin/env python3
"""
Cross-dataset validation on PlantDoc (PAPER_REVIEW_NOTES.md Issue 2).

Portable replacement for notebook 08, which additionally fixes a bug in it:
notebook 08 applies `mobilenet_v2.preprocess_input` (pixels -> [-1, 1]) at
inference to models that were TRAINED with `rescale=1./255` (pixels -> [0, 1]).
Its comment claims this "matches training"; it does not. The published 22-25%
PlantDoc accuracies were therefore measured with mismatched input scaling, so
part of the reported "domain shift catastrophe" may be a measurement artifact
rather than a property of the models.

This script evaluates every model under BOTH scalings, which separates the two
effects:

  old .h5 + preprocess_input  -> reproduces the published number
  old .h5 + rescale           -> the true domain-shift number
  new *_rescale + rescale     -> honest-routing models, original scaling
  new *_mobilenet + preprocess_input -> does the Issue 4 fix aid generalization

Only 8 of the 10 PlantVillage classes exist in PlantDoc (Spider Mites and
Target Spot have no equivalent), so chance is 12.5%. Predictions of the two
unmapped classes are still recorded, to detect hallucination onto them.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PLANTDOC = ROOT / "validation" / "PlantDoc-Dataset"
MODEL_DIR = ROOT / "models"
OUT_DIR = ROOT / "outputs" / "cross_dataset_validation"

IMAGE_SIZE = (224, 224)
BATCH = 32

PLANTVIL_CLASSES = [
    "Bacterial Spot", "Early Blight", "Late Blight", "Leaf Mold",
    "Septoria Leaf Spot", "Spider Mites", "Target Spot",
    "Yellow Leaf Curl Virus", "Mosaic Virus", "Healthy",
]
UNMAPPED = {5, 6}  # no PlantDoc equivalent

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

# (label, filename, description, scaling the model was TRAINED with)
MODELS = [
    # Paper generation: leaky routing, trained with rescale=1./255
    ("A_paper", "model_A_color_final.h5", "Color Only (paper)", "rescale"),
    ("B_paper", "model_B_segmented_final.h5", "Segmented Only (paper)", "rescale"),
    ("C_paper", "model_C_Mixed_final.h5", "Random 50/50 Mix (paper)", "rescale"),
    ("D_paper", "model_D_finetuned_final.h5", "Fine-tuned A->Seg (paper)", "rescale"),
    ("E_paper", "model_E_selective_V3_final.h5", "Class-Aware Routing (paper)", "rescale"),
    # Issue 1 generation: val-derived routing
    ("A_rescale", "strategy_A_rescale_final.keras", "Color, honest routing", "rescale"),
    ("B_rescale", "strategy_B_rescale_final.keras", "Segmented, honest routing", "rescale"),
    ("C_rescale", "strategy_C_rescale_final.keras", "Mixed, honest routing", "rescale"),
    ("D_rescale", "strategy_D_rescale_final.keras", "Fine-tuned, honest routing", "rescale"),
    ("E_rescale", "strategy_E_rescale_final.keras", "Selective, honest routing", "rescale"),
    # Issue 1 + Issue 4: val-derived routing and correct preprocessing
    ("A_mobilenet", "strategy_A_mobilenet_final.keras", "Color, preprocess_input", "mobilenet"),
    ("B_mobilenet", "strategy_B_mobilenet_final.keras", "Segmented, preprocess_input", "mobilenet"),
    ("C_mobilenet", "strategy_C_mobilenet_final.keras", "Mixed, preprocess_input", "mobilenet"),
    ("D_mobilenet", "strategy_D_mobilenet_final.keras", "Fine-tuned, preprocess_input", "mobilenet"),
    ("E_mobilenet", "strategy_E_mobilenet_final.keras", "Selective, preprocess_input", "mobilenet"),
]


def load_images():
    """Pool PlantDoc train+test; ~80-140 images/class instead of ~9 from test alone."""
    from tensorflow.keras.preprocessing import image as keras_image
    records, raw = [], []
    for folder, true_idx in PLANTDOC_TO_PLANTVIL.items():
        for split in ("train", "test"):
            d = PLANTDOC / split / folder
            if not d.exists():
                print(f"  ! missing {d}")
                continue
            for p in sorted(list(d.glob("*.jpg")) + list(d.glob("*.jpeg")) + list(d.glob("*.png"))):
                try:
                    img = keras_image.load_img(p, target_size=IMAGE_SIZE)
                except Exception as e:
                    print(f"  ! skip {p.name}: {e}")
                    continue
                raw.append(keras_image.img_to_array(img))
                records.append({"path": str(p), "true": true_idx, "folder": folder, "split": split})
    arr = np.stack(raw)  # uint8-valued float array, 0..255
    print(f"Loaded {len(arr)} PlantDoc images across {len(PLANTDOC_TO_PLANTVIL)} classes")
    return arr, records


def scale(arr, mode):
    if mode == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return preprocess_input(arr.copy())
    return arr / 255.0


def score(preds, y_true):
    y_pred = preds.argmax(axis=1)
    conf = preds.max(axis=1)
    overall = float((y_pred == y_true).mean())
    per_class = {}
    for folder, idx in PLANTDOC_TO_PLANTVIL.items():
        m = y_true == idx
        per_class[PLANTVIL_CLASSES[idx]] = float((y_pred[m] == idx).mean()) if m.any() else None
    halluc = float(np.isin(y_pred, list(UNMAPPED)).mean())
    return {"overall": overall, "per_class": per_class,
            "mean_confidence": float(conf.mean()),
            "hallucinated_absent_classes": halluc}


def main():
    if not PLANTDOC.exists():
        print(f"PlantDoc not found at {PLANTDOC}", file=sys.stderr)
        return 1
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    arr, records = load_images()
    y_true = np.array([r["true"] for r in records])
    scaled = {m: scale(arr, m) for m in ("rescale", "mobilenet")}

    results = {"n_images": len(records), "chance_level": 1 / len(PLANTDOC_TO_PLANTVIL),
               "models": {}}

    for label, fname, desc, trained_with in MODELS:
        path = MODEL_DIR / fname
        if not path.exists():
            print(f"  ! {fname} not found -- skipped")
            continue
        model = tf.keras.models.load_model(path, compile=False)
        entry = {"file": fname, "description": desc, "trained_with": trained_with, "eval": {}}
        for mode in ("rescale", "mobilenet"):
            preds = model.predict(scaled[mode], batch_size=BATCH, verbose=0)
            s = score(preds, y_true)
            s["matches_training"] = (mode == trained_with)
            entry["eval"][mode] = s
        results["models"][label] = entry
        c = entry["eval"][trained_with]["overall"]
        w = entry["eval"]["mobilenet" if trained_with == "rescale" else "rescale"]["overall"]
        print(f"  {label:<14} {desc:<32} correct={c:6.2%}   mismatched={w:6.2%}")
        del model
        tf.keras.backend.clear_session()

    out = OUT_DIR / "plantdoc_preprocessing_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")

    print(f"\n{'='*78}\nPlantDoc accuracy, correct scaling per model (chance = 12.5%)\n{'='*78}")
    for gen, suffix in (("paper (leaky routing)", "_paper"),
                        ("honest routing, 1./255", "_rescale"),
                        ("honest routing + preprocess_input", "_mobilenet")):
        row = []
        for s in "ABCDE":
            k = f"{s}{suffix}"
            if k in results["models"]:
                e = results["models"][k]
                row.append(f"{s}={e['eval'][e['trained_with']]['overall']:.1%}")
        print(f"  {gen:<36} {'  '.join(row)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
