"""Step 0 — sanity-check the whole pipeline on ONE image before touching the dataset.

Checks
  1. our hand-built inputs_embeds (all 576 tokens) == HF processor path (same logits / same answer)
  2. polygon -> token overlap mapping (draws the private tokens)
  3. R from attention and from CLIP, P from a (fake) polygon, soft / hard selection
  4. answers + prefill latency for full vs pruned token sets
Writes outputs/smoke/*.png.

  python scripts/00_smoke_test.py --image path.jpg --question "What color is the car?" \
      --polygon 100,50,220,50,220,180,100,180
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from privprune import geometry as G
from privprune.config import load_config, work
from privprune.selection import normalize, num_keep, select
from privprune.vlm import PROMPT, ClipRelevance, PrunableLlava

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--image", required=True)
ap.add_argument("--question", default="What is in this image? Answer briefly.")
ap.add_argument("--polygon", default=None, help="fake private region x1,y1,x2,y2,... in ORIGINAL pixel coords")
ap.add_argument("--ratio", type=float, default=0.25)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--no-clip", action="store_true")
args = ap.parse_args()
cfg = load_config(args.config)
dev = cfg["model"]["device"]
os.makedirs(work(cfg, "smoke"), exist_ok=True)

img = Image.open(args.image).convert("RGB")
img336, (dx, dy, scale) = G.llava_square_image(img)
vlm = PrunableLlava(cfg["model"]["llava"], dev, attn_implementation="eager")
pv = vlm.pixel_values(img336)
z = vlm.encode(pv)
v = vlm.project(z)[0]
print("z:", tuple(z.shape), "projected:", tuple(v.shape))

# ---- 1. equivalence with the standard HF path
inputs = vlm.processor(images=img336, text=PROMPT.format(question=args.question), return_tensors="pt").to(dev, torch.float16)
with torch.no_grad():
    ref = vlm.model(**inputs).logits[0, -1].float()
    e, _ = vlm.build_embeds(v, args.question)
    ours = vlm.model(inputs_embeds=e).logits[0, -1].float()
diff = (ref - ours).abs().max().item()
print(f"[1] max |logit diff| HF vs ours: {diff:.4f}  (argmax equal: {ref.argmax().item() == ours.argmax().item()})")
assert inputs["input_ids"].shape[1] == e.shape[1], "sequence length mismatch"

# ---- 2. oracle privacy from a polygon
if args.polygon:
    poly = [float(x) for x in args.polygon.split(",")]
    P = G.mask_to_token_overlap(G.rasterize([G.transform_polygon(poly, dx, dy, scale)]))
else:
    P = np.zeros(576, dtype=np.float32)
print(f"[2] tokens with P>0.25: {(P > 0.25).sum()}")

# ---- 3. relevance
R_attn = normalize(vlm.attention_relevance(v, args.question).numpy(), cfg["relevance"]["norm"])
R = {"attention": R_attn}
if not args.no_clip:
    R["clip"] = normalize(ClipRelevance(cfg["model"]["clip"], dev, revision=cfg["model"].get("clip_revision"))(pv, args.question).numpy(), cfg["relevance"]["norm"])

k = num_keep(args.ratio)
sets = {"full": select("full", k), "relevance": select("relevance", k, R=R_attn),
        f"soft λ={args.lam}": select("soft", k, R=R_attn, P=P, lam=args.lam),
        "hard τ=0.5": select("hard", k, R=R_attn, P=P, tau=0.5)}

# ---- 4. answers + latency
for name, keep in sets.items():
    torch.cuda.synchronize(); t0 = time.time()
    ans = vlm.generate(v, args.question, [keep], max_new_tokens=20)[0]
    torch.cuda.synchronize()
    ptr = P[keep].sum() / P.sum() if P.sum() > 0 else float("nan")
    print(f"[4] {name:14s} tokens={len(keep):3d} PTR={ptr:.2f} time={time.time() - t0:.2f}s  -> {ans!r}")

panels = [("image", np.asarray(img336)), ("P (oracle)", P.reshape(24, 24))] + \
         [(f"R {n}", r.reshape(24, 24)) for n, r in R.items()] + \
         [(n, np.asarray(G.render_kept_tokens(img336, keep))) for n, keep in sets.items() if n != "full"]
fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4))
for ax, (title, im) in zip(axes, panels):
    ax.imshow(im, cmap=None if im.ndim == 3 else "magma"); ax.set_title(title, fontsize=9); ax.axis("off")
fig.suptitle(args.question, fontsize=10)
fig.tight_layout()
path = work(cfg, "smoke", "smoke_test.png")
fig.savefig(path, dpi=130)
print("saved", path)
