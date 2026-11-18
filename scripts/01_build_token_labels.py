"""Step 1 — VISPR polygons -> token-level privacy targets.

For every image and privacy class c:  o_{i,c} = |A_i ∩ M_c| / |A_i|   (C x 576)
and the oracle risk                   P^GT_i  = |A_i ∩ ∪_c M_c| / |A_i| (576)

Output: outputs/labels/{split}.npz  (image_ids, O [float16], p_gt, present) and outputs/labels/classes.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
from tqdm import tqdm

from privprune import vispr
from privprune.config import load_config, work

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
args = ap.parse_args()
cfg = load_config(args.config)
os.makedirs(work(cfg, "labels"), exist_ok=True)

classes_path = work(cfg, "labels", "classes.json")
if os.path.exists(classes_path):
    classes = json.load(open(classes_path))
else:
    classes = vispr.classes_in(vispr.load_split(cfg["paths"]["vispr_root"], "train"))
    json.dump(classes, open(classes_path, "w"), indent=1)
print(f"{len(classes)} privacy classes:", classes)

for split in args.splits:
    recs = vispr.load_split(cfg["paths"]["vispr_root"], split)
    O = np.zeros((len(recs), len(classes), 576), dtype=np.float16)
    p_gt = np.zeros((len(recs), 576), dtype=np.float32)
    present = np.zeros((len(recs), len(classes)), dtype=bool)
    # Polygons are in the annotation's pixel frame; if the JPEG on disk has a different size
    # (e.g. EXIF rotation), the token masks would be misplaced.
    mismatched = [r["image_id"] for r in recs
                  if Image.open(r["image_path"]).size != (r["width"], r["height"])]
    if mismatched:
        print(f"WARNING [{split}] {len(mismatched)} images whose size differs from the annotation, e.g. {mismatched[:5]}")
    for n, rec in enumerate(tqdm(recs, desc=split)):
        o, p, pres = vispr.token_overlaps(rec, classes, cfg["labels"]["supersample"])
        O[n], p_gt[n], present[n] = o, p, pres
    np.savez_compressed(work(cfg, "labels", f"{split}.npz"), image_ids=np.array([r["image_id"] for r in recs]),
                        image_paths=np.array([r["image_path"] for r in recs]), O=O, p_gt=p_gt, present=present)
    frac = (p_gt > cfg["labels"]["delta"]).mean()
    print(f"[{split}] {len(recs)} images | {frac:.1%} of tokens private (>δ) | "
          f"images with any private region: {(p_gt.max(1) > 0).mean():.1%}")
    print("  per-class image counts:", dict(zip(classes, present.sum(0).tolist())))
