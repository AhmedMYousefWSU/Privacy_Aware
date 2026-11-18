"""Step 2 — cache frozen CLIP visual tokens z_i (LLaVA layer -2, 576 x 1024, fp16) per split.

Output: outputs/features/{split}_z.npy (memmap, N x 576 x 1024)  ~1.2 MB / image.
Order matches outputs/labels/{split}.npz.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from privprune import geometry as G
from privprune.config import load_config, work
from privprune.vlm import PrunableLlava

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
ap.add_argument("--batch", type=int, default=32)
args = ap.parse_args()
cfg = load_config(args.config)
os.makedirs(work(cfg, "features"), exist_ok=True)

vlm = PrunableLlava(cfg["model"]["llava"], cfg["model"]["device"], attn_implementation="sdpa")
for split in args.splits:
    paths = np.load(work(cfg, "labels", f"{split}.npz"))["image_paths"]
    out = np.lib.format.open_memmap(work(cfg, "features", f"{split}_z.npy"), mode="w+",
                                    dtype=np.float16, shape=(len(paths), 576, 1024))
    for b in tqdm(range(0, len(paths), args.batch), desc=split):
        imgs = [G.llava_square_image(Image.open(p).convert("RGB"))[0] for p in paths[b:b + args.batch]]
        with torch.no_grad():
            z = vlm.encode(vlm.pixel_values(imgs))
        out[b:b + len(imgs)] = z.cpu().numpy().astype(np.float16)
    out.flush()
