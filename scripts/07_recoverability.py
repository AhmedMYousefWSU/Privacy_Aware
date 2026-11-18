"""Step 7 (privacy metric 2) — can an attacker still recover the private attributes from the kept tokens?

Because ViT tokens are contextual (self-attention mixes information across patches), dropping the
face tokens does not guarantee face information is gone. We train an image-level multi-label
probe  g(mean_{i∈keep} z_i) -> attributes  on train images (with random token dropout so it is
used to seeing subsets) and report class-mean AP on the kept tokens of every configuration.
Lower mAP = less recoverable.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from privprune.config import load_config, work
from privprune.metrics import average_precision
from privprune.selection import hex_to_keep

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--epochs", type=int, default=30)
args = ap.parse_args()
cfg = load_config(args.config)
dev = cfg["model"]["device"]
torch.manual_seed(0)
rng = np.random.default_rng(0)

lab_tr = np.load(work(cfg, "labels", "train.npz"))
z_tr = np.load(work(cfg, "features", "train_z.npy"), mmap_mode="r")
Y_tr = torch.from_numpy(lab_tr["present"].astype(np.float32)).to(dev)
C = Y_tr.shape[1]
probe = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, C)).to(dev)
opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
pos_w = ((len(Y_tr) - Y_tr.sum(0)) / Y_tr.sum(0).clamp(min=1)).clamp(max=50)

bs = 64
for ep in range(args.epochs):
    for b in range(0, len(z_tr), bs):
        idx = np.arange(b, min(b + bs, len(z_tr)))
        z = torch.from_numpy(np.asarray(z_tr[idx])).to(dev).float()
        ratio = rng.choice([1.0, 0.75, 0.5, 0.25, 0.1])
        mask = (torch.rand(z.shape[:2], device=dev) < ratio).float()
        x = (z * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        loss = nn.functional.binary_cross_entropy_with_logits(probe(x), Y_tr[idx], pos_weight=pos_w)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"probe epoch {ep}: loss {loss.item():.4f}")

split = cfg["eval"]["split"]
lab_te = np.load(work(cfg, "labels", f"{split}.npz"))
z_te = np.load(work(cfg, "features", f"{split}_z.npy"), mmap_mode="r")
row_of = {str(i): n for n, i in enumerate(lab_te["image_ids"])}

df = pd.DataFrame(map(json.loads, open(work(cfg, "results.jsonl"))))
df = df[df.type == "utility"].fillna({"r_src": "-", "p_src": "-", "param": -1})
probe.eval()
out = []
for key, g in df.groupby(["method", "r_src", "p_src", "param", "ratio"]):
    scores, labels = [], []
    for image_id, h in zip(g.image_id, g.keep_hex):
        n = row_of[image_id]
        keep = hex_to_keep(h)
        z = torch.from_numpy(np.asarray(z_te[n][keep])).float().to(dev)
        with torch.no_grad():
            s = torch.sigmoid(probe(z.mean(0))) if len(keep) else torch.zeros(C, device=dev)
        scores.append(s.cpu().numpy()); labels.append(lab_te["present"][n])
    S, L = np.stack(scores), np.stack(labels)
    aps = [average_precision(S[:, c], L[:, c]) for c in range(C)]
    out.append(dict(zip(["method", "r_src", "p_src", "param", "ratio"], key), recover_mAP=float(np.nanmean(aps)), n=len(g)))
res = pd.DataFrame(out)
res.to_csv(work(cfg, "recoverability.csv"), index=False, float_format="%.4f")
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
