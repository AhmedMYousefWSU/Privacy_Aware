"""Step 5 — run every (R source, P source, method, λ/τ, K) configuration on every task.

One output row per (task, configuration) in outputs/results.jsonl; resumable (finished task_ids
are skipped). Identical keep-sets are generated only once; equal-length sets are batched.
"""
import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from privprune import geometry as G
from privprune import metrics as M
from privprune.config import load_config, work
from privprune.privacy_head import PrivacyHead
from privprune.selection import keep_to_hex, normalize, num_keep, select
from privprune.vlm import ClipRelevance, PrunableLlava

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--tasks", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--limit-images", type=int, default=None)
ap.add_argument("--types", nargs="+", default=None, help="only run tasks of these types")
args = ap.parse_args()
cfg = load_config(args.config)
ev, rc, dev = cfg["eval"], cfg["relevance"], cfg["model"]["device"]
tasks_path = args.tasks or work(cfg, "tasks.jsonl")
out_path = args.out or work(cfg, "results.jsonl")

tasks = [dict(t, task_id=i) for i, t in enumerate(json.load(open(tasks_path)) if tasks_path.endswith(".json")
                                                   else map(json.loads, open(tasks_path)))]
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["task_id"] for l in open(out_path)}
by_image = OrderedDict()
for t in tasks:
    if t["task_id"] not in done and (args.types is None or t["type"] in args.types):
        by_image.setdefault(t["image_path"], []).append(t)
if args.limit_images:
    by_image = OrderedDict(list(by_image.items())[:args.limit_images])
print(f"{sum(map(len, by_image.values()))} tasks on {len(by_image)} images to run ({len(done)} already done)")

# ---------------------------------------------------------------- oracle P^GT
lab = np.load(work(cfg, "labels", f"{ev['split']}.npz"))
gt_row = {str(i): n for n, i in enumerate(lab["image_ids"])}

vlm = PrunableLlava(cfg["model"]["llava"], dev, attn_implementation="eager")
head = None
if "learned" in ev["p_sources"]:
    ck = torch.load(work(cfg, "privacy_head", "head.pt"), map_location=dev)
    head = PrivacyHead(1024, ck["hidden"], len(ck["classes"])).to(dev).eval()
    head.load_state_dict(ck["state_dict"])
clip_rel = ClipRelevance(cfg["model"]["clip"], dev, mode=rc["clip_mode"], revision=cfg["model"].get("clip_revision")) if "clip" in ev["r_sources"] else None
layers = range(rc["attn_layers"][0], rc["attn_layers"][1] + 1)


def raw_relevance(src, v, pv, q):
    if src == "attention":
        return vlm.attention_relevance(v, q, layers, rc["attn_query"]).numpy()
    if src == "clip":
        return clip_rel(pv, q).numpy()
    if src == "loo":
        return vlm.loo_relevance(v, q, block=rc["loo_block"])[0].numpy()
    raise ValueError(src)


fout = open(out_path, "a")
for img_n, (image_path, tlist) in enumerate(tqdm(by_image.items(), desc="images")):
    img336, _ = G.llava_square_image(Image.open(image_path).convert("RGB"))
    pv = vlm.pixel_values(img336)
    z = vlm.encode(pv)
    v = vlm.project(z)[0]

    P = {}
    image_id = tlist[0]["image_id"]
    if "gt" in ev["p_sources"] and image_id in gt_row:
        P["gt"] = lab["p_gt"][gt_row[image_id]]
    for ps in ev["p_sources"]:  # "gt_dil<r>": oracle mask grown by r tokens (tests whether context tokens leak)
        if ps.startswith("gt_dil") and image_id in gt_row:
            P[ps] = G.dilate_tokens(lab["p_gt"][gt_row[image_id]], int(ps[len("gt_dil"):]))
    if head is not None:
        P["learned"] = head.privacy_risk(z[0].float())[0].cpu().numpy()
    p_gt = lab["p_gt"][gt_row[image_id]] if image_id in gt_row else None

    R_cache = {}
    rng = np.random.default_rng([ev["seed"], img_n])
    rand_keeps = {r: select("random", num_keep(r), rng=rng) for r in ev["keep_ratios"]}  # shared across tasks

    for t in tlist:
        rq = t.get("relevance_question", t["question"])
        R = {}
        for src in ev["r_sources"]:
            if (src, rq) not in R_cache:
                R_cache[(src, rq)] = normalize(raw_relevance(src, v, pv, rq), rc["norm"])
            R[src] = R_cache[(src, rq)]

        configs = [dict(method="full", r_src=None, p_src=None, param=None, ratio=1.0, keep=select("full", 576))]
        for ratio in ev["keep_ratios"]:
            k = num_keep(ratio)
            configs.append(dict(method="random", r_src=None, p_src=None, param=None, ratio=ratio, keep=rand_keeps[ratio]))
            for rs in R:
                configs.append(dict(method="relevance", r_src=rs, p_src=None, param=0.0, ratio=ratio, keep=select("relevance", k, R=R[rs])))
            for ps in P:
                configs.append(dict(method="privacy_only", r_src=None, p_src=ps, param=None, ratio=ratio, keep=select("privacy_only", k, P=P[ps])))
            for rs in R:
                for ps in P:
                    for lam in ev["lambdas"]:
                        configs.append(dict(method="soft", r_src=rs, p_src=ps, param=lam, ratio=ratio,
                                            keep=select("soft", k, R=R[rs], P=P[ps], lam=lam)))
                    for tau in ev["taus"]:
                        configs.append(dict(method="hard", r_src=rs, p_src=ps, param=tau, ratio=ratio,
                                            keep=select("hard", k, R=R[rs], P=P[ps], tau=tau)))

        uniq = OrderedDict()
        for c in configs:
            c["hex"] = keep_to_hex(c["keep"])
            uniq.setdefault(c["hex"], c["keep"])
        hexes = list(uniq)
        answers = dict(zip(hexes, vlm.generate(v, t["question"], [uniq[h] for h in hexes], ev["max_new_tokens"])))

        for c in configs:
            row = {k: t[k] for k in ("task_id", "image_id", "type", "question") if k in t}
            row.update({k: t[k] for k in ("attr_id", "gt", "answers", "relevance_question") if k in t})
            row.update(method=c["method"], r_src=c["r_src"], p_src=c["p_src"], param=c["param"], ratio=c["ratio"],
                       n_kept=int(len(c["keep"])), answer=answers[c["hex"]], keep_hex=c["hex"])
            if p_gt is not None:
                row["ptr"] = M.private_token_retention(c["keep"], p_gt)
                row["hit"] = M.private_token_hit_rate(c["keep"], p_gt, cfg["labels"]["delta"])
            fout.write(json.dumps(row) + "\n")
        fout.flush()
fout.close()
print("wrote", out_path)
