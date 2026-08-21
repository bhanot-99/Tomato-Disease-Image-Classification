#!/usr/bin/env python3
"""
Leak-free Strategy E pipeline (fixes PAPER_REVIEW_NOTES.md Issue 1).

The paper's Strategy E routing table was built from per-class accuracy measured
on the TEST split, and Strategy E was then reported on that same test split.
This script rebuilds the whole chain so the routing decision only ever sees the
VAL split, and the test split is touched exactly once, at the end, for all
strategies:

  1. train A (color), B (segmented), C (mixed)
  2. per-class accuracy of A/B/C on VAL  -> outputs/routing_analysis/*.json
  3. routing table = argmax over A/B/C per class (val-derived)
  4. assemble dataset/processed_selective_<tag> from that table (symlinks)
  5. train D (A fine-tuned on segmented) and E (selective)
  6. evaluate A-E on TEST once -> outputs/results/final_test_<tag>.json

--preproc selects the input scaling, so Issue 1 and Issue 4 can be separated:
  rescale   = 1./255, what every reported model in the paper used
  mobilenet = mobilenet_v2.preprocess_input + label smoothing 0.1 (the repo's
              documented fix, see scripts/patch_notebooks.py)
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dataset"
DOMAIN_PATHS = {
    "color": DATA / "processed",
    "segmented": DATA / "processed_segmented",
    "mixed": DATA / "processed_mixed",
}
ROUTING_DIR = ROOT / "outputs" / "routing_analysis"
RESULTS_DIR = ROOT / "outputs" / "results"
MODEL_DIR = ROOT / "models"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 20
FINETUNE_EPOCHS = 10
SEED = 42
TAG_DOMAIN = {"A": "color", "B": "segmented", "C": "mixed"}


def setup_tf():
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(SEED)
    print(f"TensorFlow {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")
    return tf


def datagen_kwargs(preproc):
    if preproc == "mobilenet":
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        return {"preprocessing_function": preprocess_input}
    return {"rescale": 1.0 / 255}


def make_loss(tf, preproc):
    if preproc == "mobilenet":
        return tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    return "categorical_crossentropy"


def generators(tf, root, preproc, splits=("train", "val", "test")):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    base = datagen_kwargs(preproc)
    train_gen = ImageDataGenerator(
        rotation_range=15, horizontal_flip=True, zoom_range=0.1,
        width_shift_range=0.1, height_shift_range=0.1, **base
    )
    eval_gen = ImageDataGenerator(**base)
    out = {}
    for split in splits:
        g = train_gen if split == "train" else eval_gen
        out[split] = g.flow_from_directory(
            str(Path(root) / split), target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
            class_mode="categorical", shuffle=(split == "train"), seed=SEED,
        )
    return out


def build_model(tf, preproc):
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.3)(x)
    out = Dense(NUM_CLASSES, activation="softmax")(x)
    model = Model(base.input, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=make_loss(tf, preproc), metrics=["accuracy"])
    return model


def callbacks_for(tf, ckpt):
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    return [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(str(ckpt), monitor="val_accuracy", save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]


def train_one(tf, name, root, preproc, tag):
    print(f"\n{'='*70}\nTraining {name}  <- {root}\n{'='*70}")
    gens = generators(tf, root, preproc, ("train", "val"))
    model = build_model(tf, preproc)
    ckpt = MODEL_DIR / f"{name}_{tag}_best.keras"
    hist = model.fit(gens["train"], epochs=EPOCHS, validation_data=gens["val"],
                     callbacks=callbacks_for(tf, ckpt), verbose=2)
    final = MODEL_DIR / f"{name}_{tag}_final.keras"
    model.save(final)
    print(f"saved {final}")
    return model, hist.history


def finetune_D(tf, model_a_path, preproc, tag):
    """Strategy D: Strategy A fine-tuned on segmented, top 30 layers, BN frozen."""
    print(f"\n{'='*70}\nTraining strategy_D (fine-tune A on segmented)\n{'='*70}")
    model = tf.keras.models.load_model(model_a_path, compile=False)
    trainable = model.layers[-30:]
    for layer in model.layers:
        layer.trainable = False
    for layer in trainable:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=make_loss(tf, preproc), metrics=["accuracy"])
    gens = generators(tf, DOMAIN_PATHS["segmented"], preproc, ("train", "val"))
    ckpt = MODEL_DIR / f"strategy_D_{tag}_best.keras"
    hist = model.fit(gens["train"], epochs=FINETUNE_EPOCHS, validation_data=gens["val"],
                     callbacks=callbacks_for(tf, ckpt), verbose=2)
    final = MODEL_DIR / f"strategy_D_{tag}_final.keras"
    model.save(final)
    return model, hist.history


def per_class_accuracy(model, gen):
    from sklearn.metrics import confusion_matrix
    preds = np.argmax(model.predict(gen, verbose=0), axis=1)
    true = gen.classes
    cm = confusion_matrix(true, preds, labels=list(range(len(gen.class_indices))))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    names = list(gen.class_indices.keys())
    overall = float((preds == true).mean())
    return overall, {n: float(a) for n, a in zip(names, per_class)}


def build_selective(routing, out_root):
    """Symlink each class folder from its routed domain into one dataset root."""
    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val", "test"):
        for cls, domain in routing.items():
            src = DOMAIN_PATHS[domain] / split / cls
            dst = out_root / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in os.listdir(src):
                link = dst / f
                if not link.exists():
                    link.symlink_to((src / f).resolve())
    print(f"Strategy E dataset assembled at {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preproc", choices=["rescale", "mobilenet"], default="mobilenet")
    ap.add_argument("--tag", default=None, help="run tag (default: same as --preproc)")
    args = ap.parse_args()
    tag = args.tag or args.preproc

    tf = setup_tf()
    for d in (ROUTING_DIR, RESULTS_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    run = {"tag": tag, "preproc": args.preproc, "epochs": EPOCHS, "seed": SEED,
           "history": {}, "val_per_class": {}, "test": {}}
    models = {}

    # --- 1/2. Train A,B,C and measure per-class accuracy on VAL only ---
    for t, domain in TAG_DOMAIN.items():
        m, h = train_one(tf, f"strategy_{t}", DOMAIN_PATHS[domain], args.preproc, tag)
        models[t] = m
        run["history"][t] = {k: [float(v) for v in vs] for k, vs in h.items()}

        val_gen = generators(tf, DOMAIN_PATHS[domain], args.preproc, ("val",))["val"]
        overall, per_class = per_class_accuracy(m, val_gen)
        run["val_per_class"][t] = {"overall": overall, "per_class": per_class}
        payload = {"strategy": t, "source_split": "val", "domain": domain,
                   "preproc": args.preproc, "overall_accuracy": overall,
                   "per_class_accuracy": per_class}
        p = ROUTING_DIR / f"strategy_{t}_val_per_class_{tag}.json"
        p.write_text(json.dumps(payload, indent=2))
        # also write the plain name the notebooks look for
        (ROUTING_DIR / f"strategy_{t}_val_per_class.json").write_text(json.dumps(payload, indent=2))
        print(f"VAL overall {overall:.4f} -> {p.name}")

    # --- 3. Val-derived routing table ---
    classes = sorted(run["val_per_class"]["A"]["per_class"])
    routing, table = {}, []
    for cls in classes:
        scores = {t: run["val_per_class"][t]["per_class"][cls] for t in "ABC"}
        best = max(scores, key=scores.get)
        routing[cls] = TAG_DOMAIN[best]
        table.append({"class": cls, **{f"acc_{t}": scores[t] for t in "ABC"},
                      "winner": best, "domain": TAG_DOMAIN[best]})
    routing_path = ROUTING_DIR / f"class_routing_val_derived_{tag}.json"
    routing_path.write_text(json.dumps(
        {"source_split": "val", "preproc": args.preproc, "routing": routing, "table": table}, indent=2))
    print(f"\nVal-derived routing ({routing_path.name}):")
    print(f"  {'class':<45} {'A':>7} {'B':>7} {'C':>7}  ->")
    for r in table:
        print(f"  {r['class']:<45} {r['acc_A']:>7.1%} {r['acc_B']:>7.1%} {r['acc_C']:>7.1%}  -> {r['domain']}")
    run["routing"] = routing

    # --- 4/5. Assemble and train E, plus D ---
    sel_root = DATA / f"processed_selective_{tag}"
    build_selective(routing, sel_root)
    models["E"], hE = train_one(tf, "strategy_E", sel_root, args.preproc, tag)
    run["history"]["E"] = {k: [float(v) for v in vs] for k, vs in hE.items()}

    models["D"], hD = finetune_D(tf, MODEL_DIR / f"strategy_A_{tag}_final.keras", args.preproc, tag)
    run["history"]["D"] = {k: [float(v) for v in vs] for k, vs in hD.items()}

    # --- 6. Single test-set evaluation of all five ---
    test_roots = {"A": DOMAIN_PATHS["color"], "B": DOMAIN_PATHS["segmented"],
                  "C": DOMAIN_PATHS["mixed"], "D": DOMAIN_PATHS["segmented"], "E": sel_root}
    print(f"\n{'='*70}\nFINAL TEST EVALUATION ({args.preproc})\n{'='*70}")
    for t in "ABCDE":
        gen = generators(tf, test_roots[t], args.preproc, ("test",))["test"]
        overall, per_class = per_class_accuracy(models[t], gen)
        run["test"][t] = {"overall": overall, "per_class": per_class,
                          "data_root": str(test_roots[t])}
        print(f"  Strategy {t}: {overall:.4f}")

    out = RESULTS_DIR / f"final_test_{tag}.json"
    out.write_text(json.dumps(run, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
