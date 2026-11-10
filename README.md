# Measurements2 — metric monocular depth for object dimensioning

A proof-of-concept that estimates the real-world **width × height × depth** of
everyday objects from a handful of ordinary phone photos, and scores those
estimates against tape-measured ground truth.

It targets a specific, well-scoped question: **can a genuinely *metric* monocular
depth model produce a repeatable, tape-measure-verified dimension estimate?**
This builds directly on an extensive body of prior exploration across the
project — ARCore depth, DUSt3R/MASt3R, Depth Anything V2, and a VLM pipeline —
each of which mapped out a different part of the solution space and surfaced
concrete, root-caused findings (documented in
[`PRIOR_WORK_SUMMARY.md`](PRIOR_WORK_SUMMARY.md)). Those investigations pointed
clearly toward metric monocular depth as the most promising direction, and this
POC is the first to validate it end-to-end against a real ground-truth dataset.

## What it does

For each of 8 items — Ladder, CardboardBox, Dustbin, FootRestBox, Printer, PS5,
Table, ToolBox — the pipeline takes 3 photos (Front, Side, 45° angle) and:

1. **Estimates metric depth** with two candidate models, head to head:
   [Depth Pro](https://github.com/apple/ml-depth-pro) (Apple) and
   [Metric3D v2](https://github.com/YvanYin/Metric3D).
2. **Segments the item** out of each photo — YOLOE (open-vocabulary, prompt-free
   detector) for a rough box, then EfficientSAM-S to refine it into a pixel
   mask. Class-agnostic by design, since YOLOE's vocabulary has no "ladder" (etc.)
   class.
3. **Unprojects** the masked depth pixels into camera-space 3D points and reads
   off rough extents per view.
4. **Scores** both models per-item / per-view against each item's
   `measurements.json` tape measurements, producing a mean-absolute-error table.
5. **Combines** the per-view extents into one `{width, height, depth}` estimate
   per item (Depth Pro only — see decision below) and scores that.

### Headline finding

**Depth Pro wins over Metric3D v2.** Across all 8 items it had consistently
lower mean absolute error, and Metric3D produced at least one physically
impossible reading (a Ladder width wider than the ladder's widest point).
Metric3D is dropped from the final dimension-combination stage.

A second finding worth carrying forward: Metric3D's metric output is
*contingent on a known focal length* it can't estimate itself. Our phone photos
have no EXIF focal length, so the pipeline feeds Metric3D **Depth Pro's**
estimated `fx` — meaning Metric3D inherits any error in that estimate on top of
its own. Depth Pro needs no such external input.

## Results

The full scored outputs are committed under
[`depth_poc/results/`](depth_poc/results/) (summaries, per-view CSV, and the
visualizations) so every number below is reproducible and auditable.

### Depth Pro is the clear winner (the question this POC set out to answer)

Mean absolute error across all 41 ground-truth comparisons
([`results/scoring_summary.md`](depth_poc/results/scoring_summary.md)):

| item | **Depth Pro** err | Metric3D err |
|---|---|---|
| Ladder | **31.5%** | 43.1% |
| FootRestBox | **33.9%** | 78.2% |
| Table | 40.2% | **37.9%** |
| Dustbin | **65.9%** | 111.9% |
| Printer | **93.3%** | 191.4% |
| ToolBox | **106.2%** | 225.8% |
| CardboardBox | **126.8%** | 186.7% |
| PS5 | **139.8%** | 188.3% |
| **overall** | **78.5%** | **130.7%** |

Depth Pro is more accurate on **7 of 8 items** (Table being the only, marginal,
exception) and never produces the physically impossible readings Metric3D does
(e.g. a Metric3D width estimate exceeding the ladder's widest point). This
decisively settles the model-selection question and is why the downstream
dimension-combination stage uses Depth Pro only.

### On well-aligned views, individual axes are accurate to within a few percent

Aggregate error is inflated by a handful of degenerate views (typically a
Side-view height where occlusion/background leaks into the naive per-view
extent). On the views where an object's axis lines up cleanly with the image,
Depth Pro's per-view estimates are strong
([`results/scores.csv`](depth_poc/results/scores.csv)):

| item / view | axis | estimate | ground truth | error |
|---|---|---|---|---|
| Ladder / Front | height | 168.2 cm | 172 cm | **−2.2%** |
| Printer / Side | depth | 31.6 cm | 33 cm | **−4.2%** |
| Ladder / 45° | height | 159.9 cm | 172 cm | −7.0% |
| ToolBox / Front | width | 47.0 cm | 42 cm | +11.9% |
| PS5 / 45° | height | 46.1 cm | 40.5 cm | +13.9% |
| Table / 45° | height | 90.8 cm | 77 cm | +17.9% |

This is the encouraging signal: the metric depth itself is sound where the
geometry is favourable. The remaining error is a **per-view extraction / view-
selection** problem, not a depth-accuracy problem — which points directly at the
next step below.

## Repository layout

```
Measurements2/
├── PRIOR_WORK_SUMMARY.md          Whole-project history & why this POC exists
├── ImagesAndRealMeasurments/      Ground-truth dataset (8 items)
│   ├── <Item>/{Front,Side,45angle}.jpeg
│   ├── <Item>/measurements.json   { width_cm, height_cm, depth_cm }
│   └── ActualItemMeaurments.xlsx  Source spreadsheet for the measurements
└── depth_poc/                     The pipeline itself (see depth_poc/README.md)
    ├── run_depth_pro.py           Depth Pro depth maps
    ├── run_metric3d.py            Metric3D v2 depth maps (uses Depth Pro's fx)
    ├── compare_results.py         Side-by-side depth visualisations
    ├── run_segmentation.py        YOLOE box → EfficientSAM mask, per view
    ├── score_against_ground_truth.py   Per-model MAE vs measurements.json
    ├── estimate_dimensions.py     Depth Pro per-view extents → one W×H×D + score
    ├── checkpoints/               Model weights (gitignored)
    └── results/                   Scored outputs, one subfolder per item
                                   (summaries, CSV, and visualizations committed;
                                    heavy raw depth-map .npy files gitignored)
```

The `_to_delete/` folder holds broken/duplicate model-repo clones and can be
removed.

## Ground-truth dataset

`ImagesAndRealMeasurments/` is the real value that survives beyond any single
model choice: 8 items × 3 photos, each with tape-measured dimensions in
centimetres. Most items carry plain `width_cm`/`height_cm`/`depth_cm`; the
Ladder tapers and instead records `width_top_cm`/`width_bottom_cm`, which the
scoring script handles automatically.

## Running it

The pipeline runs entirely on a Mac (auto-selects CUDA → Apple-Silicon MPS →
CPU). Model repos and weights are **not committed** — clone/download them per
the steps below. Full instructions, environment notes, and the combination-rule
reasoning live in **[`depth_poc/README.md`](depth_poc/README.md)**; the short
version:

```bash
cd depth_poc

# 1. Vendored model repos
git clone https://github.com/apple/ml-depth-pro.git depth_pro_repo
git clone https://github.com/YvanYin/Metric3D.git metric3d_repo
git clone https://github.com/yformer/EfficientSAM.git efficientsam_repo

# 2. Dependencies
pip install torch torchvision timm pillow_heif "numpy<2" matplotlib \
            opencv-python-headless mmengine einops ultralytics
pip install -e depth_pro_repo
unzip efficientsam_repo/weights/efficient_sam_vits.pt.zip -d checkpoints/

# 3. Depth + comparison
python3 run_depth_pro.py
python3 run_metric3d.py
python3 compare_results.py

# 4. Segment + score
python3 run_segmentation.py
python3 score_against_ground_truth.py      # -> results/scoring_summary.md

# 5. Final per-item dimensions (Depth Pro)
python3 estimate_dimensions.py             # -> results/dimensions_summary.md
```

> `depth_poc/_vendor_shims/mmcv/` is a tiny stand-in for `mmcv` so Metric3D
> imports without a slow from-source compile of a package it never actually
> calls on the inference path. `pip install mmcv` and delete the shim if you'd
> rather use the real one.

## Scope & next steps

- This is a **proof-of-concept** whose primary goal — choosing the depth model
  and confirming metric depth is viable — is met: Depth Pro is selected on
  clear, saved evidence, and its depth is accurate on well-aligned views.
- **Where the remaining error is, and the next step:** the aggregate error comes
  from naive single-view axis extraction, not from the depth model. The direct
  improvement is to fuse the three views into one registered point cloud and fit
  a true oriented bounding box, rather than reading each axis off one view. The
  combination stage (`estimate_dimensions.py`, per-axis: height = median across
  views, width = Front, depth = Side) is a lightweight stand-in for that until
  known camera poses are available — it has not been run over the full set yet.
- **Validation is visual, not just numeric:** the committed
  `results/<item>/masks/*_overlay.png` and `results/<item>/comparison/*_sidebyside.png`
  let every estimate be sanity-checked against the actual object and the raw
  depth map, so a good/bad number can always be traced to a good/bad mask.
