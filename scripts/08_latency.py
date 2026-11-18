"""Step 8 — efficiency: LLM prefill latency and end-to-end answer latency vs. number of kept visual tokens.

Measures, per image, (a) the LLM prefill forward pass on [prefix | kept v_i | question], and (b) greedy generation
of a short answer, for K in {576, 432, 288, 144, 58}. The vision encoder and projector run once and are
excluded (they are identical for every K). Also times the selection overheads: the attention-relevance pass
(16 LLM layers on all 576 tokens) and the privacy head.

  python scripts/08_latency.py --n-images 30
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
from PIL import Image

from privprune import geometry as G
from privprune.config import load_config, work
from privprune.privacy_head import PrivacyHead
from privprune.selection import normalize, num_keep, select
from privprune.vlm import PrunableLlava

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--n-images", type=int, default=30)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()
cfg = load_config(args.config)
dev = cfg["model"]["device"]

tasks = [json.loads(l) for l in open(work(cfg, "tasks.jsonl"))]
util = [t for t in tasks if t["type"] == "utility"]
seen, chosen = set(), []
for t in util:
    if t["image_path"] not in seen:
        seen.add(t["image_path"]); chosen.append(t)
chosen = chosen[:args.n_images]

vlm = PrunableLlava(cfg["model"]["llava"], dev, attn_implementation="sdpa")
ck = torch.load(work(cfg, "privacy_head", "head.pt"), map_location=dev)
head = PrivacyHead(1024, ck["hidden"], len(ck["classes"])).to(dev).eval()
head.load_state_dict(ck["state_dict"])
layers = range(cfg["relevance"]["attn_layers"][0], cfg["relevance"]["attn_layers"][1] + 1)


def timed(fn, reps):
    fn()  # warm-up
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


ratios = [1.0, 0.75, 0.5, 0.25, 0.1]
rows = []
with torch.no_grad():
    for t in chosen:
        img336, _ = G.llava_square_image(Image.open(t["image_path"]).convert("RGB"))
        z = vlm.encode(vlm.pixel_values(img336)); v = vlm.project(z)[0]
        R = normalize(vlm.attention_relevance(v, t["question"], layers, "text").numpy(), "rank")
        P = head.privacy_risk(z[0].float())[0].cpu().numpy()
        r = {"image": t["image_path"],
             "t_relevance": timed(lambda: vlm.attention_relevance(v, t["question"], layers, "text"), args.reps),
             "t_privacy_head": timed(lambda: head.privacy_risk(z[0].float()), args.reps)}
        for ratio in ratios:
            keep = select("full", 576) if ratio == 1.0 else select("soft", num_keep(ratio), R=R, P=P, lam=1.0)
            emb, _ = vlm.build_embeds(v, t["question"], keep)
            r[f"prefill_{ratio}"] = timed(lambda: vlm.model.language_model(inputs_embeds=emb), args.reps)
            r[f"seqlen_{ratio}"] = int(emb.shape[1])
            r[f"answer_{ratio}"] = timed(lambda: vlm.generate(v, t["question"], [keep], max_new_tokens=10), max(2, args.reps // 2))
        rows.append(r)

out = work(cfg, "latency.json")
json.dump(rows, open(out, "w"), indent=1)
print(f"{len(rows)} images | relevance pass {np.median([r['t_relevance'] for r in rows])*1e3:.1f} ms | "
      f"privacy head {np.median([r['t_privacy_head'] for r in rows])*1e3:.2f} ms")
for ratio in ratios:
    print(f"K={num_keep(ratio) if ratio < 1 else 576:3d}  seq={np.median([r[f'seqlen_{ratio}'] for r in rows]):.0f}  "
          f"prefill={np.median([r[f'prefill_{ratio}'] for r in rows])*1e3:.1f} ms  "
          f"answer(10 tok)={np.median([r[f'answer_{ratio}'] for r in rows])*1e3:.1f} ms")
print("wrote", out)
