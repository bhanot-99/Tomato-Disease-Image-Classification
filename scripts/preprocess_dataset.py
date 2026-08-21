#!/usr/bin/env python3
"""
Build the processed PlantVillage tomato splits used by all training notebooks.

Linux/portable replacement for notebooks 01_preprocessing_color, 01b_preprocessing_segmented
and 01c_preprocessing_mixed (which hardcode Windows paths).

Key difference from the original notebooks -- and the reason this script exists as
part of the Issue 1 (routing data-leak) fix:

  The color and segmented folders contain the SAME leaves; segmented filenames are
  just "<base>_final_masked.jpg" versions of "<base>.JPG". The original notebooks
  sampled and split each domain independently, so the same leaf could land in
  color/train and segmented/test. Comparing per-class accuracy across domains to
  decide Strategy E routing is only valid if both domains use the SAME split.
  This script pairs images by base ID and derives one split per class, applied
  identically to color, segmented and mixed.

Protocol (unchanged from the paper): cap at 1000 images/class, 70/15/15
train/val/test, seed 42, resize to 224x224 LANCZOS.
"""
import os
import random
import sys
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLOR_SRC = os.path.join(ROOT, "dataset", "color")
SEG_SRC = os.path.join(ROOT, "dataset", "segmented")

OUT_COLOR = os.path.join(ROOT, "dataset", "processed")
OUT_SEG = os.path.join(ROOT, "dataset", "processed_segmented")
OUT_MIXED = os.path.join(ROOT, "dataset", "processed_mixed")

IMAGE_SIZE = (224, 224)
MAX_IMAGES_PER_CLASS = 1000
SEED = 42
SPLITS = ("train", "val", "test")
EXTS = (".jpg", ".jpeg", ".png")


def list_images(d):
    return sorted(f for f in os.listdir(d) if f.lower().endswith(EXTS))


def base_id(fname):
    """Strip extension and the segmented '_final_masked' suffix."""
    stem = os.path.splitext(fname)[0]
    return stem[: -len("_final_masked")] if stem.endswith("_final_masked") else stem


def save_resized(src, dst):
    with Image.open(src) as im:
        im.convert("RGB").resize(IMAGE_SIZE, Image.LANCZOS).save(dst, quality=95)


def main():
    classes = sorted(
        f for f in os.listdir(COLOR_SRC)
        if f.startswith("Tomato") and os.path.isdir(os.path.join(COLOR_SRC, f))
    )
    print(f"Found {len(classes)} classes\n")

    counts = {out: {s: {} for s in SPLITS} for out in ("color", "segmented", "mixed")}

    for cls in classes:
        color_map = {base_id(f): f for f in list_images(os.path.join(COLOR_SRC, cls))}
        seg_map = {base_id(f): f for f in list_images(os.path.join(SEG_SRC, cls))}

        paired = sorted(set(color_map) & set(seg_map))
        dropped = len(color_map) - len(paired)
        if dropped:
            print(f"  ! {cls}: {dropped} color images have no segmented pair -- skipped")

        # Cap, then one split shared by every domain.
        if len(paired) > MAX_IMAGES_PER_CLASS:
            rng = random.Random(SEED)
            paired = sorted(rng.sample(paired, MAX_IMAGES_PER_CLASS))

        train_ids, temp_ids = train_test_split(paired, test_size=0.30, random_state=SEED)
        val_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=SEED)
        split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}

        for split, ids in split_ids.items():
            # Mixed: half of this split's leaves as color, half as segmented.
            shuffled = list(ids)
            random.Random(SEED).shuffle(shuffled)
            mixed_color = set(shuffled[: len(shuffled) // 2])

            for out_root, domain in ((OUT_COLOR, "color"), (OUT_SEG, "segmented"), (OUT_MIXED, "mixed")):
                dest = os.path.join(out_root, split, cls)
                os.makedirs(dest, exist_ok=True)
                for bid in ids:
                    if domain == "color":
                        src_dir, fname = os.path.join(COLOR_SRC, cls), color_map[bid]
                    elif domain == "segmented":
                        src_dir, fname = os.path.join(SEG_SRC, cls), seg_map[bid]
                    else:
                        use_color = bid in mixed_color
                        src_dir = os.path.join(COLOR_SRC if use_color else SEG_SRC, cls)
                        fname = color_map[bid] if use_color else seg_map[bid]
                    save_resized(os.path.join(src_dir, fname), os.path.join(dest, fname))
                counts[domain][split][cls] = len(ids)

        print(f"  {cls.replace('Tomato___',''):<45} "
              f"train={len(train_ids):>4} val={len(val_ids):>4} test={len(test_ids):>4}")

    print()
    for domain, out in (("color", OUT_COLOR), ("segmented", OUT_SEG), ("mixed", OUT_MIXED)):
        tot = {s: sum(counts[domain][s].values()) for s in SPLITS}
        print(f"{domain:<10} -> train {tot['train']}, val {tot['val']}, test {tot['test']}, "
              f"total {sum(tot.values())}  [{out}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
