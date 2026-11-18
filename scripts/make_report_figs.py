"""Report figures from stored results (no GPU needed): mask_check.png and qualitative.{pdf,png}.

Private regions are pixelated (one flat colour per token) for display only.

  python scripts/make_report_figs.py --mask-seed 7 \
      --qual 2017_17231794:a105_face_all:kind+of+place 2017_17477081:a108_license_plate_all:indoors \
             2017_16368107:a111_name_all:main+color
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from privprune import geometry as G
from privprune.selection import hex_to_keep

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="neurips/figs")
ap.add_argument("--mask-seed", type=int, default=7)
ap.add_argument("--qual", nargs="+", required=True, help="image_id:attr_id:words+of+the+neutral+question")
args = ap.parse_args()

lab = np.load("outputs/labels/test.npz")
classes = json.load(open("outputs/labels/classes.json"))
row = {str(i): n for n, i in enumerate(lab["image_ids"])}
NAMES = {"a105_face_all": "face", "a108_license_plate_all": "license plate", "a111_name_all": "name"}


def square(n):
    return G.llava_square_image(Image.open(str(lab["image_paths"][n])).convert("RGB"))[0]


def pixelate(im, p):
    a = np.asarray(im).astype(float).copy()
    for t in np.flatnonzero(p > 0.05):
        y, x = divmod(t, 24)
        blk = a[y * 14:(y + 1) * 14, x * 14:(x + 1) * 14]
        blk[:] = blk.reshape(-1, 3).mean(0)
    return a.astype(np.uint8)


def overlay(ax, a, p):
    ax.imshow(a)
    ax.imshow(np.kron(p.reshape(24, 24), np.ones((14, 14))), cmap="Reds", alpha=.4, vmin=0, vmax=1)


plt.rcParams.update({"font.size": 8})

# ---------------------------------------------------------------- mask_check: random test images
rng = np.random.default_rng(args.mask_seed)
picks = rng.choice(len(lab["image_ids"]), 6, replace=False)
fig, ax = plt.subplots(1, 6, figsize=(18, 3.4))
for a, n in zip(ax, picks):
    overlay(a, pixelate(square(n), lab["p_gt"][n]), lab["p_gt"][n])
    title = ", ".join(classes[c].split("_")[1] for c in np.flatnonzero(lab["present"][n]))
    a.set_title(title if len(title) <= 42 else title[:40] + "…", fontsize=9)
    a.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(args.out, "mask_check.png"), dpi=90)
print("mask_check:", [str(lab["image_ids"][n]) for n in picks])

# ---------------------------------------------------------------- qualitative: stored keep-sets at K=288
ex = [q.split(":") for q in args.qual]
want = {e[0] for e in ex}
rows = {}
for l in open("outputs/results.jsonl"):
    d = json.loads(l)
    if d["image_id"] in want and d["ratio"] in (1.0, 0.5):
        rows.setdefault(d["image_id"], []).append(d)


def pick(rs, typ, m, attr=None, q=None):
    for r in rs:
        if r["type"] == typ and r["method"] == m and (attr is None or r.get("attr_id") == attr) \
                and (q is None or q in r["question"]) and (m != "relevance" or r["r_src"] == "attention") \
                and (m != "soft" or (r["p_src"] == "learned" and r["r_src"] == "attention" and r["param"] == 2.0)):
            return r
    raise KeyError((typ, m, attr, q))


def short(a):
    words = a.replace(",", " ").replace(".", " ").split()
    return " ".join(words[:2])


def shade(a, keep):
    m = np.zeros(576, bool)
    m[keep] = True
    out = a.astype(float)
    out[~np.kron(m.reshape(24, 24), np.ones((14, 14), bool))] *= 0.15
    return out.astype(np.uint8)


fig, ax = plt.subplots(len(ex), 3, figsize=(6.6, 2.33 * len(ex)))
for i, (img, attr, uq) in enumerate(ex):
    uq = uq.replace("+", " ")
    rs, n = rows[img], row[img]
    p = lab["p_gt"][n]
    a = pixelate(square(n), p)
    pf, pr, ps = [pick(rs, "probe_adv", m, attr) for m in ("full", "relevance", "soft")]
    uf = pick(rs, "utility", "full", q=uq)
    ur, us = [pick(rs, "utility", m, q=uf["question"]) for m in ("relevance", "soft")]
    qn = {"kind of place": "scene", "main color": "background", "indoors": "in/out"}.get(uq, "neutral")
    overlay(ax[i, 0], a, p)
    ax[i, 0].set_title(f'All 576 tokens (red: {NAMES.get(attr, attr.split("_", 1)[1].replace("_all", "").replace("_", " "))})\nprobe: "{short(pf["answer"])}"   {qn}: "{short(uf["answer"])}"', fontsize=7)
    ax[i, 1].imshow(shade(a, hex_to_keep(pr["keep_hex"])))
    ax[i, 1].set_title(f'Relevance only (PTR {pr["ptr"]:.2f})\nprobe: "{short(pr["answer"])}"   {qn}: "{short(ur["answer"])}"', fontsize=7)
    ax[i, 2].imshow(shade(a, hex_to_keep(ps["keep_hex"])))
    ax[i, 2].set_title(f'Ours, learned P, λ=2 (PTR {ps["ptr"]:.2f})\nprobe: "{short(ps["answer"])}"   {qn}: "{short(us["answer"])}"', fontsize=7)
for x in ax.ravel():
    x.axis("off")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.out, f"qualitative.{ext}"), dpi=110)
print("qualitative:", [e[0] for e in ex])
