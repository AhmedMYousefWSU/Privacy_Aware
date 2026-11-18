"""Compute E6 (dilated oracle) metrics exactly as 06_summarize.py does, without touching outputs/summary.csv,
and write midterm/e6_numbers.tex with one LaTeX macro per table cell.

  python scripts/e6_report_numbers.py outputs/results_dilated.jsonl midterm/e6_numbers.tex
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

from privprune import metrics as M

src, dst = sys.argv[1], sys.argv[2]
df = pd.DataFrame(map(json.loads, open(src)))
full = df[df.method == "full"].set_index("task_id")["answer"]
u = df.type == "utility"
df.loc[u, "util_agree"] = [M.exact_match(a, full[t]) for a, t in zip(df.answer[u], df.task_id[u])]
df["said_yes"] = [float(M.yes_no(a) == 1) for a in df.answer]
df.loc[df.type == "probe_adv", "leak_adv"] = df.said_yes
df.loc[df.type == "probe_benign", "leak_benign"] = df.said_yes
df["ptr_u"] = df.ptr.where(u)

s = df[(df.method == "soft") & (df.param == 2.0)].groupby(["p_src", "ratio"])[["util_agree", "ptr_u", "leak_adv", "leak_benign"]].mean()
print(s.round(3).to_string())
# sanity: the shared baselines must match the main run's summary
base = df[df.method.isin(["relevance", "random"])].groupby(["method", "ratio"])[["util_agree", "leak_adv"]].mean()
print(base.round(3).to_string())

cells = {"a": ("gt_dil1", 0.5), "b": ("gt_dil2", 0.5), "c": ("gt_dil1", 0.25), "d": ("gt_dil2", 0.25)}
with open(dst, "w") as f:
    for k, key in cells.items():
        r = s.loc[key]
        for m, col in [("U", "util_agree"), ("P", "ptr_u"), ("L", "leak_adv"), ("B", "leak_benign")]:
            f.write(f"\\newcommand{{\\Esix{m}{k}}}{{{r[col]:.2f}}}\n")
    f.write("\\input{e6_text}\n")
print("wrote", dst)
