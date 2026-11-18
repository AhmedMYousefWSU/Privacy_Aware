#!/usr/bin/env bash
# Full E1 (oracle P) + E3/E4 (learned P) run with GQA utility, default config.
# Every step is pinned away from logical CPUs 14-15: this workstation's i9-14900K has a faulty
# core there that produces random, impossible Python errors (see README "Hardware note").
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/ahmed.abdelnaby/miniforge3/envs/CIDER/bin/python}
RUN="taskset -c 0-13,16-31 $PY"
CFG=${CFG:-configs/default.yaml}

$RUN scripts/01_build_token_labels.py --config $CFG
$RUN scripts/02_extract_features.py --config $CFG
$RUN scripts/03_train_privacy_head.py --config $CFG
[ -f data/gqa/gqa_testdev.json ] || $RUN scripts/prepare_gqa.py --out data/gqa
$RUN scripts/04_build_tasks.py --config $CFG --vqa-json data/gqa/gqa_testdev.json --max-vqa 300
$RUN scripts/05_evaluate.py --config $CFG
$RUN scripts/06_summarize.py --config $CFG
$RUN scripts/07_recoverability.py --config $CFG
echo "ALL DONE"
