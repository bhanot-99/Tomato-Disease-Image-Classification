"""
patch_notebooks.py
==================
Patches all 5 training notebooks with two improvements:

  1. preprocess_input  — Replace `rescale=1./255` with
     `tf.keras.applications.mobilenet_v2.preprocess_input`
     (maps pixels to [-1, 1] as MobileNetV2 expects).

  2. Label Smoothing   — Replace `loss='categorical_crossentropy'`
     with `CategoricalCrossentropy(label_smoothing=0.1)` to
     prevent overconfident, brittle decision boundaries.

Run from the project root:
    python scripts/patch_notebooks.py
"""

import json
import shutil
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
NOTEBOOKS_DIR   = Path(__file__).parent.parent / "notebooks"
LABEL_SMOOTHING = 0.1

TARGETS = [
    "01_model_training_color.ipynb",
    "02_model_training_segmented.ipynb",
    "03_model_training_mixed.ipynb",
    "04_model_training_finetuned.ipynb",
    "05_model_training_selective.ipynb",
]

# ── Patch helpers ─────────────────────────────────────────────────────────────

def patch_imports(src: str) -> tuple[str, list[str]]:
    """
    Inject the preprocess_input import if not already present.
    Returns (new_src, list_of_changes).
    """
    changes = []

    # Already has the import — skip
    if "preprocess_input" in src:
        return src, changes

    # Inject after the MobileNetV2 import line
    mobilenet_import = "from tensorflow.keras.applications import MobileNetV2"
    new_import = (
        "from tensorflow.keras.applications import MobileNetV2\n"
        "from tensorflow.keras.applications.mobilenet_v2 import preprocess_input"
    )
    if mobilenet_import in src:
        src = src.replace(mobilenet_import, new_import)
        changes.append("  ✅ Added `preprocess_input` import")
    else:
        # Fallback: inject at top after `import tensorflow as tf`
        tf_line = "import tensorflow as tf"
        src = src.replace(
            tf_line,
            tf_line + "\nfrom tensorflow.keras.applications.mobilenet_v2 import preprocess_input",
        )
        changes.append("  ✅ Added `preprocess_input` import (fallback position)")

    return src, changes


def patch_datagenerators(src: str) -> tuple[str, list[str]]:
    """
    Replace rescale=1./255 with preprocessing_function=preprocess_input
    in ImageDataGenerator calls.
    Returns (new_src, list_of_changes).
    """
    changes = []

    if "rescale=1./255" not in src:
        return src, changes

    count = src.count("rescale=1./255")

    # ── Training generator ────────────────────────────────────────────────────
    # Pattern A: rescale on its own line with trailing comma (in augmentation block)
    old_train = "    rescale=1./255,\n    rotation_range="
    new_train = "    preprocessing_function=preprocess_input,\n    rotation_range="
    if old_train in src:
        src = src.replace(old_train, new_train)
        changes.append("  ✅ Patched train ImageDataGenerator (rescale → preprocess_input)")

    # ── Val/Test generator ────────────────────────────────────────────────────
    # Pattern B: standalone  ImageDataGenerator(rescale=1./255)
    old_valtest = "ImageDataGenerator(rescale=1./255)"
    new_valtest = "ImageDataGenerator(preprocessing_function=preprocess_input)"
    if old_valtest in src:
        replaced = src.count(old_valtest)
        src = src.replace(old_valtest, new_valtest)
        changes.append(
            f"  ✅ Patched val/test ImageDataGenerator(rescale) × {replaced}"
        )

    # ── Fine-tune notebook: lighter augmentation block ────────────────────────
    # (rotation_range=10 instead of 15)
    old_ft = "    rescale=1./255,\n    rotation_range=10"
    new_ft = "    preprocessing_function=preprocess_input,\n    rotation_range=10"
    if old_ft in src:
        src = src.replace(old_ft, new_ft)
        changes.append("  ✅ Patched fine-tune train ImageDataGenerator (rescale → preprocess_input)")

    # Sanity check — any remaining rescale=1./255 we missed?
    remaining = src.count("rescale=1./255")
    if remaining:
        changes.append(f"  ⚠️  {remaining} occurrence(s) of rescale=1./255 still present — check manually")

    return src, changes


def patch_loss(src: str) -> tuple[str, list[str]]:
    """
    Replace string shorthand 'categorical_crossentropy' in compile()
    with CategoricalCrossentropy(label_smoothing=0.1).
    Returns (new_src, list_of_changes).
    """
    changes = []

    # Already patched
    if "label_smoothing" in src:
        return src, changes

    replacements = [
        # Quoted string inside compile(loss=...)
        (
            "loss='categorical_crossentropy'",
            f"loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing={LABEL_SMOOTHING})",
        ),
        (
            'loss="categorical_crossentropy"',
            f"loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing={LABEL_SMOOTHING})",
        ),
    ]

    for old, new in replacements:
        if old in src:
            count = src.count(old)
            src = src.replace(old, new)
            changes.append(
                f"  ✅ Replaced {old!r} with CategoricalCrossentropy"
                f"(label_smoothing={LABEL_SMOOTHING}) × {count}"
            )

    return src, changes


# ── Per-cell dispatcher ───────────────────────────────────────────────────────

def patch_cell(cell: dict) -> tuple[dict, list[str]]:
    """Apply all patches to a single notebook cell. Returns patched cell + log."""
    if cell.get("cell_type") != "code":
        return cell, []

    src = "".join(cell["source"])
    all_changes = []

    src, c = patch_imports(src)
    all_changes += c

    src, c = patch_datagenerators(src)
    all_changes += c

    src, c = patch_loss(src)
    all_changes += c

    if all_changes:
        # Rebuild source as list of lines (Jupyter stores it that way)
        cell = dict(cell)
        lines = src.splitlines(keepends=True)
        # Ensure last line has no trailing newline only if original didn't
        cell["source"] = lines

    return cell, all_changes


# ── Main ──────────────────────────────────────────────────────────────────────

def patch_notebook(path: Path) -> None:
    print(f"\n📓 Patching: {path.name}")
    print("─" * 60)

    # Back up original
    backup = path.with_suffix(".ipynb.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  💾 Backup saved → {backup.name}")
    else:
        print(f"  ℹ️  Backup already exists ({backup.name}), skipping backup")

    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    total_changes = []
    new_cells = []
    for i, cell in enumerate(nb["cells"]):
        patched_cell, changes = patch_cell(cell)
        if changes:
            print(f"\n  Cell {i}:")
            for c in changes:
                print(f"    {c}")
        total_changes += changes
        new_cells.append(patched_cell)

    nb["cells"] = new_cells

    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    if total_changes:
        print(f"\n  ✅ {len(total_changes)} change(s) applied — notebook saved.")
    else:
        print("  ℹ️  No changes needed (already patched).")


def main():
    print("=" * 60)
    print("  🔧 Notebook Patcher — preprocess_input + Label Smoothing")
    print("=" * 60)

    for name in TARGETS:
        path = NOTEBOOKS_DIR / name
        if not path.exists():
            print(f"\n⚠️  Skipping {name} — file not found")
            continue
        patch_notebook(path)

    print("\n" + "=" * 60)
    print("  ✅ All notebooks patched!")
    print()
    print("  Changes made:")
    print("    1. rescale=1./255 → preprocessing_function=preprocess_input")
    print("       (MobileNetV2 now receives pixels in [-1, 1] as designed)")
    print()
    print(f"    2. 'categorical_crossentropy' → CategoricalCrossentropy(")
    print(f"         label_smoothing={LABEL_SMOOTHING})")
    print("       (Penalises overconfidence, improves domain generalisation)")
    print()
    print("  ⚠️  NOTE: You must RETRAIN all models for changes to take effect.")
    print("=" * 60)


if __name__ == "__main__":
    main()
