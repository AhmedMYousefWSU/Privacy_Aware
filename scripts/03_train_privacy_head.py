"""Step 3 — train the per-token privacy head on frozen features (VLM untouched).

Targets are soft overlaps o_{i,c}; loss is class-weighted BCE. Model selection on val
token-level AP of P_i = max_c p_{i,c} against the binary target P^GT_i > δ.

Output: outputs/privacy_head/head.pt, metrics.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from tqdm import tqdm

from privprune.config import load_config, work
from privprune.metrics import average_precision
from privprune.privacy_head import PrivacyHead, compute_pos_weight, soft_bce_loss

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
hp, delta, dev = cfg["privacy_head"], cfg["labels"]["delta"], cfg["model"]["device"]
out_dir = work(cfg, "privacy_head")
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(0)

classes = json.load(open(work(cfg, "labels", "classes.json")))


def load(split):
    lab = np.load(work(cfg, "labels", f"{split}.npz"))
    z = np.load(work(cfg, "features", f"{split}_z.npy"), mmap_mode="r")
    return z, lab["O"], lab["p_gt"]


z_tr, O_tr, _ = load("train")
z_va, O_va, pgt_va = load("val")
C = len(classes)

pos_weight = compute_pos_weight(torch.from_numpy(O_tr.astype(np.float32).sum(axis=(0, 2))), O_tr.shape[0] * 576).to(dev)
head = PrivacyHead(1024, hp["hidden"], C, hp["dropout"]).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
steps = hp["epochs"] * int(np.ceil(len(z_tr) / hp["batch_images"]))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hp["lr"], total_steps=steps)


@torch.no_grad()
def predict(z, bs=64):
    head.eval()
    P, p = [], []
    for b in range(0, len(z), bs):
        Pi, pi = head.privacy_risk(torch.from_numpy(np.asarray(z[b:b + bs])).to(dev))
        P.append(Pi.cpu().numpy()); p.append(pi.cpu().numpy())
    return np.concatenate(P), np.concatenate(p)  # (N,576), (N,576,C)


def evaluate(z, O, p_gt):
    P, p = predict(z)
    res = {"AP_any": average_precision(P.ravel(), (p_gt > delta).ravel())}
    per = {c: average_precision(p[..., ci].ravel(), (O[:, ci] > delta).ravel()) for ci, c in enumerate(classes)}
    res["mAP_classes"] = float(np.nanmean(list(per.values())))
    res["per_class_AP"] = per
    return res


best, history = -1.0, []
for ep in range(hp["epochs"]):
    head.train()
    perm, tot = np.random.permutation(len(z_tr)), 0.0
    for b in tqdm(range(0, len(perm), hp["batch_images"]), desc=f"epoch {ep}", leave=False):
        idx = np.sort(perm[b:b + hp["batch_images"]])  # sorted -> faster memmap reads
        z = torch.from_numpy(np.asarray(z_tr[idx])).to(dev).float().reshape(-1, 1024)
        y = torch.from_numpy(O_tr[idx].astype(np.float32)).to(dev).transpose(1, 2).reshape(-1, C)
        loss = soft_bce_loss(head(z), y, pos_weight)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tot += loss.item() * len(idx)
    val = evaluate(z_va, O_va, pgt_va)
    history.append({"epoch": ep, "train_loss": tot / len(perm), "val_AP_any": val["AP_any"], "val_mAP": val["mAP_classes"]})
    print(history[-1])
    if val["AP_any"] > best:
        best = val["AP_any"]
        torch.save({"state_dict": head.state_dict(), "classes": classes, "hidden": hp["hidden"]}, os.path.join(out_dir, "head.pt"))

ckpt = torch.load(os.path.join(out_dir, "head.pt"))
head.load_state_dict(ckpt["state_dict"])
report = {"history": history, "val": evaluate(z_va, O_va, pgt_va)}
if os.path.exists(work(cfg, "features", "test_z.npy")):
    z_te, O_te, pgt_te = load("test")
    report["test"] = evaluate(z_te, O_te, pgt_te)
json.dump(report, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
print("val:", {k: v for k, v in report["val"].items() if k != "per_class_AP"})
if "test" in report:
    print("test:", {k: v for k, v in report["test"].items() if k != "per_class_AP"})
