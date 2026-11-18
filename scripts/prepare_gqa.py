"""Download GQA testdev-balanced (lmms-lab/GQA mirror, ~68 MB) and convert it to the --vqa-json format
used by 04_build_tasks.py:  [{"image_id", "image_path", "question", "answers": [answer]}]

  python scripts/prepare_gqa.py --out data/gqa
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import privprune  # noqa: F401  (sets HF_HOME)
import pandas as pd
from huggingface_hub import hf_hub_download
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/gqa")
args = ap.parse_args()
os.makedirs(os.path.join(args.out, "images"), exist_ok=True)

imgs = pd.read_parquet(hf_hub_download("lmms-lab/GQA", "testdev_balanced_images/testdev-00000-of-00001.parquet", repo_type="dataset"))
qs = pd.read_parquet(hf_hub_download("lmms-lab/GQA", "testdev_balanced_instructions/testdev-00000-of-00001.parquet", repo_type="dataset"))
print("image columns:", list(imgs.columns), "| question columns:", list(qs.columns))

paths = {}
for _, r in imgs.iterrows():
    p = os.path.abspath(os.path.join(args.out, "images", f"{r['id']}.jpg"))
    if not os.path.exists(p):
        Image.open(io.BytesIO(r["image"]["bytes"])).convert("RGB").save(p, quality=95)
    paths[str(r["id"])] = p

items = [{"image_id": f"gqa_{r['imageId']}", "image_path": paths[str(r["imageId"])], "question": str(r["question"]),
          "answers": [str(r["answer"])], "qid": str(r["id"]), "gqa_type": str(r["types"]["structural"])}
         for _, r in qs.iterrows() if str(r["imageId"]) in paths]
json.dump(items, open(os.path.join(args.out, "gqa_testdev.json"), "w"))
print(f"{len(paths)} images, {len(items)} questions -> {args.out}/gqa_testdev.json")
