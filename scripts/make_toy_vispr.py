"""Build a tiny fake dataset in the exact VISPR-redactions format (random rectangles as 'private'
regions) so the full 01->07 pipeline can be tested before the real dataset is downloaded.

  python scripts/make_toy_vispr.py --images a.jpg b.png ... --out data/toy_vispr
"""
import argparse
import json
import os
import shutil

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--images", nargs="+", required=True)
ap.add_argument("--out", default="data/toy_vispr")
ap.add_argument("--copies", type=int, default=3, help="each image appears this many times with different boxes")
args = ap.parse_args()
rng = np.random.default_rng(0)
classes = ["a105_face_all", "a108_license_plate_all", "a111_name_all", "a109_person_body"]

items = [(p, c) for c in range(args.copies) for p in args.images]
rng.shuffle(items)
n = len(items)
splits = {"train": items[: int(.6 * n)], "val": items[int(.6 * n): int(.8 * n)], "test": items[int(.8 * n):]}
for split, its in splits.items():
    os.makedirs(os.path.join(args.out, "images", f"{split}2017"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "annotations"), exist_ok=True)
    annos = {}
    for j, (p, c) in enumerate(its):
        image_id = f"2017_{split}_{j:04d}"
        rel = f"images/{split}2017/{image_id}.jpg"
        im = Image.open(p).convert("RGB")
        im.save(os.path.join(args.out, rel))
        W, H = im.size
        attrs = []
        for k in range(rng.integers(1, 4)):
            w, h = rng.uniform(.1, .4) * W, rng.uniform(.1, .4) * H
            x, y = rng.uniform(0, W - w), rng.uniform(0, H - h)
            poly = [x, y, x + w, y, x + w, y + h, x, y + h, x, y]
            attrs.append({"instance_id": k, "attr_id": str(rng.choice(classes)), "polygons": [poly]})
        annos[image_id] = {"image_id": image_id, "image_path": rel, "image_height": H, "image_width": W, "attributes": attrs}
    json.dump({"annotations": annos}, open(os.path.join(args.out, "annotations", f"{split}2017.json"), "w"))
    print(split, len(annos))
