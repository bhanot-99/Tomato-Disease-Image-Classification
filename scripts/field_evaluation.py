#!/usr/bin/env python3
"""
Cross-dataset evaluation of Strategies A-E against a field-image dataset.

Replaces the withdrawn PlantDoc evaluation (see FINDINGS.md section 3). Written
to be dataset-agnostic: point --data at a directory of class folders and supply
a mapping from those folder names onto the 10 PlantVillage class indices.

The mapping is declared up front, in this file, and the exclusion of classes
with no PlantVillage equivalent happens before any model runs. Nothing here
looks at model output to decide what to evaluate on.

Usage:
    python scripts/field_evaluation.py --dataset plantwild
    python scripts/field_evaluation.py --dataset plantwild --tag rescale
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = ROOT / "validation" / "candidates"
OUT_DIR = ROOT / "outputs" / "field_validation"
MODEL_DIR = ROOT / "models"
IMAGE_SIZE = (224, 224)
BATCH = 32

# PlantVillage training class order (alphabetical, as ImageDataGenerator assigns it)
PV_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]
PV_SHORT = ["Bacterial Spot", "Early Blight", "Healthy", "Late Blight", "Leaf Mold",
            "Septoria Leaf Spot", "Spider Mites", "Target Spot", "Mosaic Virus",
            "Yellow Leaf Curl Virus"]

DATASETS = {
    "plantwild": {
        "root": CANDIDATES / "plantwild_tomato" / "plantwild" / "images",
        # folder name -> PlantVillage class index
        "mapping": {
            "tomato bacterial leaf spot": 0,
            "tomato early blight": 1,
            "tomato leaf": 2,               # healthy
            "tomato late blight": 3,
            "tomato leaf mold": 4,
            "tomato septoria leaf spot": 5,
            "tomato mosaic virus": 8,
            "tomato yellow leaf curl virus": 9,
        },
    },
}

MODELS = [
    ("A", "Color only"),
    ("B", "Segmented only"),
    ("C", "Random 50/50 mix"),
    ("D", "Fine-tuned A->Seg"),
    ("E", "Class-aware routing"),
]


def load_images(paths, preproc):
    from tensorflow.keras.preprocessing import image as ki
    if preproc == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    else:
        preprocess_input = None
    arrs = []
    for p in paths:
        im = ki.load_img(p, target_size=IMAGE_SIZE)
        a = ki.img_to_array(im)
        arrs.append(preprocess_input(a) if preprocess_input else a / 255.0)
    return np.asarray(arrs, dtype=np.float32)


def evaluate(model, X, y_true, n_classes=10):
    probs = model.predict(X, batch_size=BATCH, verbose=0)
    pred = probs.argmax(1)
    conf = probs.max(1)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    hit1 = pred == y_true
    hit3 = np.array([t in row for t, row in zip(y_true, top3)])
    per_class = {}
    for c in sorted(set(y_true.tolist())):
        m = y_true == c
        per_class[PV_SHORT[c]] = {
            "n": int(m.sum()),
            "recall": float(hit1[m].mean()),
            "top3": float(hit3[m].mean()),
        }
    return {
        "overall": float(hit1.mean()),
        "top3": float(hit3.mean()),
        "mean_confidence": float(conf.mean()),
        "confidence_when_correct": float(conf[hit1].mean()) if hit1.any() else None,
        "confidence_when_wrong": float(conf[~hit1].mean()) if (~hit1).any() else None,
        "per_class": per_class,
        "prediction_counts": {PV_SHORT[i]: int((pred == i).sum()) for i in range(n_classes)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="plantwild")
    ap.add_argument("--tag", default="mobilenet", help="model generation tag")
    ap.add_argument("--preproc", choices=["mobilenet", "rescale"], default=None,
                    help="defaults to matching --tag")
    args = ap.parse_args()
    preproc = args.preproc or ("mobilenet" if args.tag == "mobilenet" else "rescale")

    spec = DATASETS[args.dataset]
    root, mapping = spec["root"], spec["mapping"]
    if not root.exists():
        sys.exit(f"dataset root not found: {root}")

    paths, labels = [], []
    for folder, idx in sorted(mapping.items(), key=lambda kv: kv[1]):
        d = root / folder
        if not d.exists():
            sys.exit(f"missing class folder: {d}")
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                paths.append(f)
                labels.append(idx)
    y = np.asarray(labels)
    present = sorted(set(labels))
    print(f"dataset      : {args.dataset}")
    print(f"images       : {len(paths)}")
    print(f"classes      : {len(present)} of 10 present -> chance = {100/len(present):.1f}%")
    absent = [PV_SHORT[i] for i in range(10) if i not in present]
    print(f"absent       : {', '.join(absent) if absent else '(none)'}")
    for c, n in sorted(Counter(labels).items()):
        print(f"    {PV_SHORT[c]:24} {n:5}")

    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    print(f"\nloading {len(paths)} images with preproc={preproc} ...")
    X = load_images(paths, preproc)
    print(f"tensor {X.shape}  range [{X.min():.2f}, {X.max():.2f}]")

    run = {"dataset": args.dataset, "tag": args.tag, "preproc": preproc,
           "n_images": len(paths), "classes_present": [PV_SHORT[i] for i in present],
           "classes_absent": absent, "chance": 1.0 / len(present),
           "class_counts": {PV_SHORT[c]: n for c, n in sorted(Counter(labels).items())},
           "models": {}}

    print(f"\n{'strategy':<10} {'top-1':>8} {'top-3':>8} {'mean conf':>10}")
    for letter, desc in MODELS:
        path = MODEL_DIR / f"strategy_{letter}_{args.tag}_final.keras"
        if not path.exists():
            print(f"  strategy {letter}: MISSING {path.name}")
            continue
        model = tf.keras.models.load_model(path, compile=False)
        res = evaluate(model, X, y)
        res["description"] = desc
        res["model_file"] = path.name
        run["models"][letter] = res
        print(f"{letter} {desc:<20} {res['overall']:>7.2%} {res['top3']:>7.2%} "
              f"{res['mean_confidence']:>9.1%}")
        tf.keras.backend.clear_session()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.dataset}_{args.tag}.json"
    out.write_text(json.dumps(run, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
