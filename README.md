# Privacy-Aware Visual Token Pruning for VLMs

**Keep the visual tokens that are useful for the current question, and drop the ones that carry
privacy-sensitive content.**

For every visual token *i* of a frozen VLM (LLaVA-1.5-7B, 576 tokens = 24×24 CLIP patches):

| symbol | meaning | depends on the question? | how it is computed |
|---|---|---|---|
| **R_i** ∈ [0,1] | task relevance | **yes** | prompt→token attention (main), CLIP text–patch similarity, leave-block-out (oracle) |
| **P_i** ∈ [0,1] | privacy risk | no | VISPR polygon overlap (oracle) or a small MLP on the frozen token (learned) |
| **S_i** | selection score | — | `S_i = R_i − λ·P_i`, keep top-K (soft); or never keep `P_i ≥ τ` (hard gate) |

Top-K on S is the exact solution of
`max_x Σ x_i R_i − λ Σ x_i P_i  s.t.  Σ x_i = K, x_i∈{0,1}`: task information minus privacy risk
under a token budget.

```
 question q ─────────────────────────────┐
                                         ▼
 image ─► pad-to-square ─► CLIP ViT-L/14-336 ─► z_1..z_576 (1024-d) ─┬─► projector ─► v_i (4096-d)
                                                                     │                 │
                                     privacy head f_P(z_i) ─► P_i    │    R_i ◄─ attention(q → v_i)
                                                                     │                 │
                                          S_i = R_i − λ P_i  ◄───────┴─────────────────┘
                                                   │ top-K
                                                   ▼
                        [USER: ] [v_i for i in keep] [q ASSISTANT:] ─► LLaMA ─► answer
```

The VLM is completely frozen. The only trained component is the privacy head
(1024→256→C, about 0.27M parameters).

---

## Storage

The root disk is full, so everything for this project lives on the second disk:

| what | where |
|---|---|
| project (code, `data/`, `outputs/`) | `/mnt/disk1/ahmed/vlm_privacy_course` (symlinked from `~/Documents/Fedrecover/vlm_privacy_course`) |
| model weights (LLaVA-1.5-7B ~14 GB, CLIP-L/14-336 ~1.7 GB) | `/mnt/disk1/ahmed/hf` (`privprune/__init__.py` sets `HF_HOME` for this project only) |

Expected usage is about 35 GB: ~16 GB of weights, ~10 GB of features for all VISPR splits, plus the images.

### Hardware note (this workstation)
The i9-14900K has a faulty physical core (logical CPUs 14–15). Python processes scheduled on it hit
random, impossible errors, for example `too many values to unpack (expected 0)` inside `sre_compile` or
`transformers` imports, or `keys must be str ... not str` from `json`. Measured failure rates: ~25% unpinned,
67–100% pinned to CPU 14/15, 0/20 with the core excluded. **Run everything as**
`taskset -c 0-13,16-31 python ...` (`scripts/run_full.sh` already does this), and report the CPU to IT:
silent data corruption is possible too, not only crashes.

---

## 1. Methodology (step by step = script by script)

| step | script | what it does |
|---|---|---|
| 0 | `scripts/00_smoke_test.py` | one image: checks that our pruned forward matches HF's LLaVA exactly, then plots R, P and the kept tokens |
| 1 | `scripts/01_build_token_labels.py` | VISPR polygons → per-class token overlap `o_{i,c}` and oracle `P^GT_i` |
| 2 | `scripts/02_extract_features.py` | caches frozen `z_i` (LLaVA vision layer −2, 576×1024 fp16) |
| 3 | `scripts/03_train_privacy_head.py` | trains `f_P` with soft class-weighted BCE; reports token-level AP |
| 4 | `scripts/04_build_tasks.py` | utility questions + privacy probe questions per test image (+ optional GQA/VQAv2) |
| 5 | `scripts/05_evaluate.py` | runs every {R source} × {P source} × {method} × {λ/τ} × {K}; generates answers |
| 6 | `scripts/06_summarize.py` | privacy / utility / efficiency table, per-attribute leakage, trade-off curves |
| 7 | `scripts/07_recoverability.py` | attacker probe: can the private attributes be recovered from the kept tokens? |

Experiment order. **Do the oracle-P experiment first**: if `S = R − λ·P^GT` does not beat relevance-only
pruning on the privacy–utility curve, there is no point training the privacy head.

| experiment | R | P | question it answers |
|---|---|---|---|
| E1 oracle privacy | attention | GT mask | does the idea work at all? |
| E2 oracle upper bound | LOO | GT mask | best achievable trade-off |
| E3 privacy model | attention | learned | cost of imperfect P |
| E4 full method | attention / CLIP | learned | deployable system |
| E5 generalisation | attention | learned (trained on VISPR) | transfer to HR-VISPR |

All of E1–E4 come out of **one run of step 5**, because it loops over `r_sources` × `p_sources`.

---

## 2. Dataset

### Main: VISPR-redactions (Orekondy, Fritz, Schiele — CVPR 2018)
* 8,473 images (train/val/test 2017 splits); about 47.6K instances of 24 localisable privacy attributes, with pixel-level polygons.
* Attributes include face, person, nudity, name, home/current address, phone, email, username,
  license plate, credit card, passport, driver's license, student ID, national ID, signature, fingerprint,
  handwriting, medical history, receipts, tickets, mail, date of birth, date/time, landmark, etc.
  The list is in `privprune/attributes.py`, and the yes/no probe questions are the official ones from
  `visual_redactions/resources/attr_id_to_question.tsv`.
* Project page: <https://resources.mpi-inf.mpg.de/d2/orekondy/redactions/> (now returns 403). The files are
  still live at `https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/{train,val,test}2017.tar.gz`
  (7.8 / 3.2 / 5.6 GB) and `.../v1/annotations/{train,val,test}2017.json`. License: CC BY-NC 4.0 for the
  annotations; the original licenses apply to the images. Already downloaded here: 3,873 / 1,611 / 2,989 images.
* GQA testdev-balanced (398 images, 12,578 questions) is fetched and converted by `scripts/prepare_gqa.py`.
* Put it here:
  ```
  data/vispr_redactions/images/{train2017,val2017,test2017}/*.jpg
  data/vispr_redactions/annotations/{train2017,val2017,test2017}.json
  ```
  The annotation format is `{"annotations": {id: {"image_path", "image_width", "image_height", "attributes":
  [{"attr_id", "polygons": [[x1,y1,x2,y2,...], ...]}]}}}` (`privprune/vispr.py`).

### Secondary
* **HR-VISPR** (~10.1K images, 18 human-related privacy labels, plus object-detection utility labels) for the
  generalisation experiment (E5). It has only image-level privacy labels, so use the learned P and measure
  leakage with probes/recoverability. It cannot measure PTR.
* **GQA testdev-balanced or VQAv2 val subset** for labelled utility. Convert to
  `[{"image_path", "question", "answers": [...]}]` and pass `--vqa-json` to step 4.
  These images have no privacy masks, so they only measure utility.

---

## 3. Computing P (privacy risk)

**Geometry.** LLaVA-1.5 pads the image to a square with the CLIP-mean colour, resizes it to 336×336,
and splits it into 24×24 patches of 14 px, flattened row-major: token `i` covers row `i//24`, column `i%24`.
We apply *the same* pad-and-scale transform to every polygon (`privprune/geometry.py`). We then rasterise
at 4× resolution and average-pool to 24×24:

```
o_{i,c}  = |A_i ∩ M_c| / |A_i|          per-class soft target      (C × 576)
P^GT_i   = |A_i ∩ ∪_c M_c| / |A_i|       oracle privacy risk        (576)
```

> The HF `llava-1.5-7b-hf` processor *centre-crops* by default, which would silently cut off image content
> and shift the masks. We pad to square ourselves (as original LLaVA-1.5 does), so the processor's
> resize and crop are no-ops.

**Learned P.** `p_{i,c} = σ(f_c(z_i))`, with `f` = LayerNorm → Linear(1024,256) → GELU → Dropout → Linear(256,C).
It is multi-label (sigmoid, not softmax) because face ⊂ person, name ⊂ document, and so on.
It is trained on the soft targets `o_{i,c}` with weighted BCE, where `w+_c = neg/pos`, capped at 100, so that
rare classes (passport, credit card) are not swamped by face/person.

```
P_i = max_c p_{i,c}          (all class weights w_c = 1; severity weighting is only an ablation)
```

The input is the pre-projector CLIP feature (1024-d), the same tensor LLaVA consumes. Model selection uses val
token-level AP of `P_i` against `P^GT_i > δ` (δ = 0.25).

## 4. Computing R (task relevance)

All variants are normalised per image and question to [0,1] (`privprune/selection.py:normalize`).

**R1 — attention (main).** Build the sequence `[pre-text | 576 visual | question + "ASSISTANT:"]` and run only
the first 16 LLM layers (a FastV-style cheap pass):

```
R~_i = 1/(|L||H||Q|) Σ_{l∈L} Σ_h Σ_{t∈Q} A^{l,h}_{t,i}        L = layers 2..15, Q = question tokens
```

**R2 — CLIP text–patch similarity.** `R_i = cos(W_v z_i^{MaskCLIP}, e_q)` uses the same ViT-L/14-336 plus its
text tower. The MaskCLIP value-embedding trick is needed because raw CLIP patch tokens localise poorly.
This R needs **no LLM pass**, which matters for the threat model below.

**R3 — leave-block-out (oracle, not deployable).** Generate `y` with all tokens. For each 2×2 block `b`
(144 passes), compute `R_b = max(0, log p(y | all) − log p(y | all \ b))`. Teacher forcing is batched.
Use `loo_block: 1` for true leave-one-out (576 passes).

**Normalisation.** Min–max (as originally proposed) is fragile. LLaVA has *attention-sink* visual tokens with
10–100× the attention of the rest, so min–max squashes almost every R_i to ≈0. λ then becomes much
stronger than intended, and it varies from image to image. The default is therefore **rank** normalisation
(R_i = rank/(N−1), uniform on [0,1]), which gives λ the same meaning everywhere. Min–max is an ablation.

---

## 5. Metrics

| axis | metric | where |
|---|---|---|
| privacy (geometric) | **PTR** = Σ x_i P^GT_i / Σ P^GT_i; hit rate = fraction of tokens with P^GT > δ kept | step 5/6 |
| privacy (behavioural) | **probe leakage** = "yes" rate on the official VISPR question for present attributes (adversarial / benign prompt); false-yes on absent attributes | step 6 |
| privacy (representational) | **recoverability mAP** of a linear attribute probe on mean-pooled kept tokens | step 7 |
| utility | agreement with the full-token answer (EM / F1) on neutral VISPR questions; VQA accuracy on GQA/VQAv2 | step 6 |
| efficiency | tokens kept, prefill latency (smoke test), FLOPs ∝ sequence length | step 0/6 |

Baselines: full (576), random, relevance-only (λ=0), privacy-only (−P), and ours (soft λ ∈ {0.1, 0.25, 0.5, 1, 2};
hard τ ∈ {0.5, 0.8}). Keep ratios are {75, 50, 25, 10}%. An existing pruner (FastV, SparseVLM, VisPruner or
DivPrune) can be added as another `R` source.

---

## 6. Design decisions still open for discussion

1. **Threat model: who must not see the private tokens?**
   * *(a) Output leakage.* The VLM runs locally; we only want the *answer* not to reveal private content.
     Attention-R is fine here.
   * *(b) Split inference / untrusted LLM server.* The vision encoder and scorer run on-device, and only kept tokens
     are sent. **Attention-R breaks this model**, because computing it already feeds all 576 tokens (the faces
     included) to the LLM. Only CLIP-R (or another on-device R) is valid.
   → Recommendation: state (b) as the motivating setting, report CLIP-R as the deployable method, and treat
   attention-R as the stronger-but-leaky reference.

2. **The prompt is adversarial.** R depends on q, so an attacker who asks *"what is the person's name?"* pulls R
   onto exactly the private tokens. With soft scoring, `R − λP` can still keep them when λ is small. The hard
   gate cannot be talked out of it. Step 4 therefore builds two probe variants:
   `probe_adv` (R from the probe itself, worst case) and `probe_benign` (tokens chosen for a neutral task).
   Expected: soft wins on utility, hard wins on adversarial leakage.

3. **Token mixing.** CLIP tokens are contextual. After 23 layers of self-attention, background tokens also
   encode something about the face, so removing the face tokens ≠ removing face information. Step 7
   (recoverability probe) measures how much survives. If it is a lot, one mitigation is to prune *before* the
   later ViT layers or to mask attention inside the ViT — an extension, not needed for the first version.

4. **Pruning ≠ deletion.** The LLM can still infer "there is a person" from the context tokens, which is why
   probe leakage never reaches 0. The absent-attribute false-yes rate controls for the model simply guessing.

5. **Utility questions on VISPR** have no labels, so agreement with the full-token answer is used.
   GQA/VQAv2 provide labelled utility but no privacy masks. Report both.

---

## 7. Running

```bash
conda activate CIDER            # torch 2.4.1 + transformers 4.55 (tested); or: pip install -r requirements.txt
cd /mnt/disk1/ahmed/vlm_privacy_course

# 0. sanity check on any image
python scripts/00_smoke_test.py --image some.jpg --question "What color is the car?" \
       --polygon 100,50,220,50,220,180,100,180        # fake private box, original pixel coords

# optional: full plumbing test on a fake dataset (numbers meaningless)
python scripts/make_toy_vispr.py --images a.jpg b.jpg c.png ... --out data/toy_vispr
for s in 01_build_token_labels 02_extract_features 03_train_privacy_head 04_build_tasks 05_evaluate 06_summarize 07_recoverability; do
  python scripts/$s.py --config configs/toy.yaml; done

# real run
python scripts/01_build_token_labels.py
python scripts/02_extract_features.py              # ~10 GB of features for all splits
python scripts/03_train_privacy_head.py
python scripts/04_build_tasks.py [--vqa-json gqa_testdev.json]
python scripts/05_evaluate.py                      # resumable; start with --limit-images 20
python scripts/06_summarize.py
python scripts/07_recoverability.py
```

To run E1 alone (oracle P) quickly, set `p_sources: [gt]` in `configs/default.yaml` and skip steps 2–3.

Cost (RTX 4090, fp16): each (image, question) needs one truncated attention pass, one CLIP pass, and about
40–60 unique short generations (identical keep-sets are de-duplicated, and equal lengths are batched). Budget
roughly 20–40 s per image with the default grid. `loo` adds about 144 teacher-forced passes per question.

## Layout
```
privprune/            library
  geometry.py         pad/resize, polygon → 24×24 token overlap, visualisation
  vispr.py            VISPR-redactions loader, per-class token overlaps
  vlm.py              PrunableLlava (encode, project, attention-R, LOO-R, pruned generation), ClipRelevance
  privacy_head.py     MLP head + soft weighted BCE
  selection.py        normalisation, soft/hard/baseline selection
  metrics.py          PTR, hit rate, yes/no parsing, EM/F1/VQA acc, AP
  attributes.py       VISPR attribute ids, names, official probe questions
scripts/              00–07 pipeline (see §1)
configs/              default.yaml (real), toy.yaml (plumbing test)
```
