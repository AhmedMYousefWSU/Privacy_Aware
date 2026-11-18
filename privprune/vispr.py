"""VISPR-redactions (Orekondy et al., CVPR 2018) loader.

Expected layout (as distributed on the project website):
    <root>/images/{train2017,val2017,test2017}/*.jpg
    <root>/annotations/{train2017,val2017,test2017}.json

Annotation format (from privacy_filters/tools/evaltools/evaluate_simple.py):
    {"annotations": {image_id: {"image_path": "images/val2017/xxx.jpg",
                                "image_height": H, "image_width": W,
                                "attributes": [{"attr_id": "a105_face_all",
                                                "polygons": [[x1, y1, x2, y2, ...], ...], ...}]}}}
"""
import json
import os.path as osp

import numpy as np
from PIL import Image

from . import geometry as G
from .attributes import ATTR_IDS


def annotation_path(root, split):
    for name in (f"{split}2017.json", f"{split}.json"):
        for sub in ("annotations", "annotations/2017_v1", ""):
            p = osp.join(root, sub, name)
            if osp.exists(p):
                return p
    raise FileNotFoundError(f"No annotation json for split '{split}' under {root}")


def load_split(root, split):
    """Returns a list of image records sorted by image_id."""
    with open(annotation_path(root, split)) as f:
        annos = json.load(f)["annotations"]
    recs = []
    for image_id in sorted(annos):
        a = annos[image_id]
        recs.append({
            "image_id": image_id,
            "image_path": osp.join(root, a["image_path"]),
            "width": a["image_width"],
            "height": a["image_height"],
            "attributes": a.get("attributes", []),
        })
    return recs


def token_overlaps(rec, class_ids, supersample=4):
    """Per-class token overlap o_{i,c} and the union overlap P^GT_i.

    Returns (O: (C, 576) float32, p_gt: (576,) float32, present: (C,) bool).
    Assumes the annotation's image_width/height match the image on disk.
    """
    w, h = rec["width"], rec["height"]
    s = max(w, h)
    dx, dy, scale = (s - w) // 2, (s - h) // 2, G.IMAGE_SIZE / s
    polys_by_class = {c: [] for c in class_ids}
    for attr in rec["attributes"]:
        if attr["attr_id"] in polys_by_class:
            for poly in attr["polygons"]:
                polys_by_class[attr["attr_id"]].append(G.transform_polygon(poly, dx, dy, scale))

    O = np.zeros((len(class_ids), G.NUM_TOKENS), dtype=np.float32)
    union = None
    for ci, c in enumerate(class_ids):
        if polys_by_class[c]:
            m = G.rasterize(polys_by_class[c], supersample)
            O[ci] = G.mask_to_token_overlap(m)
            union = m if union is None else (union | m)
    p_gt = G.mask_to_token_overlap(union) if union is not None else np.zeros(G.NUM_TOKENS, np.float32)
    return O, p_gt, O.max(axis=1) > 0


def classes_in(records):
    """Attribute ids that actually occur, in canonical ATTR_IDS order (unknown ids appended)."""
    seen = {a["attr_id"] for r in records for a in r["attributes"]}
    return [a for a in ATTR_IDS if a in seen] + sorted(seen - set(ATTR_IDS))


def load_image(rec):
    return Image.open(rec["image_path"]).convert("RGB")
