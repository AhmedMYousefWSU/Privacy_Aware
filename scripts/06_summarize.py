"""Step 6 — aggregate results.jsonl into the privacy / utility / efficiency table and trade-off plots.

Columns (per method, R source, P source, λ or τ, keep ratio):
  util_agree  utility questions: exact-match agreement with the all-token answer          (↑)
  util_f1     same, token F1                                                             (↑)
  vqa_acc     labelled VQA accuracy (only if --vqa-json tasks were added)                (↑)
  ptr         Private Token Retention on utility tasks, Σ x_i P^GT_i / Σ P^GT_i           (↓)
  hit         fraction of private tokens (P^GT > δ) kept                                 (↓)
  leak_adv    'yes' rate on present-attribute probes, R from the probe (attacker prompt) (↓)
  leak_benign 'yes' rate on present-attribute probes, tokens chosen for a benign question (↓)
  false_yes   'yes' rate on absent-attribute probes (sanity: should stay low)
  n_kept      average number of visual tokens given to the LLM                          (efficiency)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from privprune import metrics as M
from privprune.config import load_config, work

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--results", default=None)
args = ap.parse_args()
cfg = load_config(args.config)
df = pd.DataFrame(map(json.loads, open(args.results or work(cfg, "results.jsonl"))))

full = df[df.method == "full"].set_index("task_id")["answer"]
df["full_answer"] = df.task_id.map(full)
u = df.type == "utility"
df.loc[u, "util_agree"] = [M.exact_match(a, f) for a, f in zip(df.answer[u], df.full_answer[u])]
df.loc[u, "util_f1"] = [M.token_f1(a, f) for a, f in zip(df.answer[u], df.full_answer[u])]
if "answers" in df:
    v = df.type == "vqa"
    df.loc[v, "vqa_acc"] = [M.vqa_accuracy(a, g) for a, g in zip(df.answer[v], df.answers[v])]
df["said_yes"] = [float(M.yes_no(a) == 1) for a in df.answer]
for typ, col in [("probe_adv", "leak_adv"), ("probe_benign", "leak_benign"), ("probe_neg", "false_yes")]:
    df.loc[df.type == typ, col] = df.said_yes[df.type == typ]
df["ptr_u"] = df.ptr.where(u) if "ptr" in df else float("nan")
df["hit_u"] = df.hit.where(u) if "hit" in df else float("nan")

keys = ["method", "r_src", "p_src", "param", "ratio"]
cols = [c for c in ["util_agree", "util_f1", "vqa_acc", "ptr_u", "hit_u", "leak_adv", "leak_benign", "false_yes", "n_kept"] if c in df]
summary = (df.fillna({"r_src": "-", "p_src": "-", "param": -1})
             .groupby(keys)[cols].mean().rename(columns={"ptr_u": "ptr", "hit_u": "hit"}).reset_index())
summary.to_csv(work(cfg, "summary.csv"), index=False, float_format="%.4f")
pd.set_option("display.width", 200, "display.max_rows", 400)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# per-attribute leakage for the adversarial probe (which private classes survive pruning?)
adv = df[df.type == "probe_adv"].fillna({"r_src": "-", "p_src": "-", "param": -1})
adv.groupby(["attr_id", "method", "r_src", "p_src", "param", "ratio"]).said_yes.agg(["mean", "count"]) \
   .reset_index().to_csv(work(cfg, "leakage_per_attribute.csv"), index=False, float_format="%.4f")

# ---------------------------------------------------------------- trade-off plots
s = summary
for ratio in sorted(s.ratio.unique()):
    if ratio == 1.0:
        continue
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, xcol, xlabel in [(axes[0], "ptr", "Private token retention (↓)"), (axes[1], "leak_adv", "Adversarial probe leakage (↓)")]:
        if xcol not in s or "util_agree" not in s:
            continue
        sr = s[s.ratio == ratio]
        for (rs, ps), g in sr[sr.method == "soft"].groupby(["r_src", "p_src"]):
            base = sr[(sr.method == "relevance") & (sr.r_src == rs)]
            g = pd.concat([base, g.sort_values("param")])
            ax.plot(g[xcol], g.util_agree, "-o", ms=4, label=f"soft R={rs} P={ps} (λ: 0→{g.param.max():g})")
        for m, mk in [("random", "x"), ("privacy_only", "s"), ("hard", "^")]:
            g = sr[sr.method == m]
            ax.scatter(g[xcol], g.util_agree, marker=mk, s=40, label=m)
        f = s[s.method == "full"]
        ax.scatter(f[xcol], f.util_agree, marker="*", s=120, c="k", label="full (576 tokens)")
        ax.set_xlabel(xlabel); ax.set_ylabel("Utility: agreement with full answer (↑)"); ax.grid(alpha=.3)
    axes[0].set_title(f"keep ratio {ratio:g} ({int(round(ratio * 576))} tokens)")
    axes[1].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(work(cfg, f"tradeoff_ratio{ratio:g}.png"), dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- GQA utility vs token budget
if "vqa_acc" in s and s.vqa_acc.notna().any():
    g = s[s.vqa_acc.notna()]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    full_acc = g[g.method == "full"].vqa_acc.mean()
    ax.axhline(full_acc, c="k", ls="--", lw=1, label=f"full (576) {full_acc:.3f}")
    for key, gg in g[g.method != "full"].groupby(["method", "r_src", "p_src", "param"]):
        m, rs, ps, prm = key
        if m == "soft" and prm not in (0.5, 1.0, 2.0):
            continue
        lbl = m + ("" if rs == "-" else f" R={rs}") + ("" if ps == "-" else f" P={ps}") + ("" if prm in (-1, 0.0) else f" {'λ' if m == 'soft' else 'τ'}={prm:g}")
        gg = gg.sort_values("ratio")
        ax.plot(gg.ratio * 576, gg.vqa_acc, "-o", ms=3, lw=1, label=lbl)
    ax.set_xlabel("visual tokens kept"); ax.set_ylabel("GQA accuracy (↑)"); ax.grid(alpha=.3)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(work(cfg, "gqa_utility.png"), dpi=150); plt.close(fig)
print("wrote", work(cfg, "summary.csv"), "and tradeoff_*.png")
