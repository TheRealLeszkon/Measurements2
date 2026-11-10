# Depth Pro vs Metric3D v2 — depth comparison across all 8 items

Run two candidate metric monocular depth models on every item's photos
(Ladder, CardboardBox, Dustbin, FootRestBox, Printer, PS5, Table, ToolBox —
every subfolder of `../ImagesAndRealMeasurments/` that has a
`measurements.json`) and score both against the tape-measured ground truth,
before building any 3D geometry / OBB / dimension logic on top.

## Status

Checkpoints are already sitting in `checkpoints/` (`depth_pro.pt`, 1.9GB,
verified byte-for-byte; `metric3d_vit_small.pth`, 150MB). All 4 pipeline
scripts (`run_depth_pro.py`, `run_metric3d.py`, `run_segmentation.py`,
`score_against_ground_truth.py`) loop over every item automatically — no
per-item setup needed, no path to edit. They started out Ladder-only (that
was the first item tried, per the original plan) and were generalized once
the approach was validated there. Everything runs on your Mac — nothing in
this pipeline has been executed in the cloud; only syntax-checked
(`py_compile`), read carefully, and reasoned through against the numbers
your own runs have produced so far.

### 1. Get the two vendored model repos (not committed here — just clone them)

```bash
cd depth_poc
git clone https://github.com/apple/ml-depth-pro.git depth_pro_repo
git clone https://github.com/YvanYin/Metric3D.git metric3d_repo
```

### 2. Install dependencies

```bash
pip install torch torchvision timm pillow_heif "numpy<2" matplotlib opencv-python-headless mmengine einops
pip install -e depth_pro_repo
```

(No need to install real `mmcv` — see the note below on `_vendor_shims/`.)

### 3. Run depth + comparison

```bash
python3 run_depth_pro.py      # writes results/<item>/depth_pro/{name}_depth_m.npy, _depth_vis.png, summary.json
python3 run_metric3d.py       # writes results/<item>/metric3d/... (uses Depth Pro's fx estimate, see note below)
python3 compare_results.py    # writes results/<item>/comparison/{name}_sidebyside.png, comparison_table.{json,md}
```

Both scripts auto-detect the best available device (CUDA, then Apple
Silicon MPS, then CPU) — no manual edit needed. Each processes all 8 items
(24 photos) in one run.

### 4. Segment each item and score both models against the tape measurements

This stage masks each depth map down to just the item, unprojects the
masked pixels into camera-space 3D points, and compares the resulting rough
extents against each item's `measurements.json`.

First, get the two extra models it needs:

```bash
git clone https://github.com/yformer/EfficientSAM.git efficientsam_repo
unzip efficientsam_repo/weights/efficient_sam_vits.pt.zip -d checkpoints/
pip install ultralytics
```

`yoloe-11s-seg-pf.pt` (the detector) auto-downloads to `checkpoints/` on
first run — no manual step needed for it.

Then run, in order:

```bash
python3 run_segmentation.py            # writes results/<item>/masks/{name}_mask.npy, _overlay.png, summary.json
python3 score_against_ground_truth.py  # writes results/<item>/scoring/scoring.{json,md} + results/scoring_summary.md
```

`run_segmentation.py` uses YOLOE (open-vocabulary, prompt-free detector) to
get a rough bounding box, then EfficientSAM-S to refine that box into a
pixel mask. It's class-agnostic by design — YOLOE's fixed vocabulary won't
reliably contain each item's exact name (it has no "ladder" class at all,
for instance) — so the box is picked by excluding whole-scene labels (walls,
floors, rooms, by keyword and by size >55% of the frame) and then preferring
a class-name keyword match when one exists (e.g. "storage box" for
CardboardBox), falling back to highest confidence otherwise. An earlier
version additionally required the box to span most of the frame's height and
reach near the floor; that assumption came from the ladder's specific
framing and rejected almost every detection on the other 7 (smaller,
further-back) items, so it's gone. **Check the `results/<item>/masks/*_overlay.png`
images** after running — that's the real sanity check, not the class label.
A view can still come back skipped if literally every detection is
scene-level; that's expected, correct behavior (a missing view beats a
wrong mask), not a bug.

`score_against_ground_truth.py` prints a per-item, per-view, per-model error
table and an overall mean-absolute-error score for each model, both overall
(`results/scoring_summary.md`) and per item. Height is checked against every
view that got a mask (the camera was held level), width only against the
Front view, and depth/thickness only against the Side view. A view
`run_segmentation.py` skipped is simply left out of that item's table, not
filled with a wrong number.

### 5. Combine into one dimension estimate per item (Depth Pro only)

**Decision: Depth Pro, not Metric3D v2.** Across all 8 items,
`results/scoring_summary.md` showed Depth Pro with consistently lower mean
absolute error than Metric3D, and Metric3D produced at least one physically
impossible reading (a Front-view width estimate for the Ladder that exceeded
the ladder's own widest point). Metric3D is no longer part of the pipeline
from this stage on.

Everything through step 4 only produced per-view numbers (e.g. "the Front
view says width=X"). This step is the first one that actually answers "what
are this item's dimensions": it combines Depth Pro's per-view masked extents
into one `{width_cm, height_cm, depth_cm}` per item and scores that combined
number against ground truth.

```bash
python3 estimate_dimensions.py   # writes results/<item>/dimensions/estimate.{json,md}, box_visualization.png
                                  # + results/dimensions_summary.md (cross-item summary)
```

Combination rule (see the script's docstring for the full reasoning): height
is the median across every view that has a mask (most view-invariant, since
the camera was held level); width comes from the Front view specifically
(it's the only view where the image's horizontal axis lines up with the
object's true width axis); depth/thickness comes from the Side view for the
same reason on that axis. This is **not** a true multi-view 3D fusion — there
are no known camera poses linking the 3 shots together, so it can't be. It's
the best combination the existing per-view data supports; a real oriented
bounding box from a registered point cloud would be the next step up if this
number isn't accurate enough.

`box_visualization.png` per item draws the estimated box as a wireframe over
the Front-view point cloud, so you can eyeball whether the box actually
looks like it's wrapping the object before trusting the number.

## Why Metric3D runs after Depth Pro

Depth Pro estimates its own focal length from the image (part of its output).
Metric3D v2 does not — it needs a known pixel focal length (`fx`) supplied up
front to de-canonicalize its output into real metric units
(`canonical_to_real_scale = fx / 1000.0`, straight from the official
`hubconf.py`). Our phone photos have no EXIF focal length, so `run_metric3d.py`
reads Depth Pro's estimated `fx` for each image from
`results/<item>/depth_pro/summary.json` and uses that. This is itself a real
finding worth carrying forward: Metric3D's metric-ness is contingent on an
external intrinsic estimate that Depth Pro doesn't need. If Depth Pro's fx
estimate is off, Metric3D's depth inherits that error on top of its own.

## Layout

- `depth_pro_repo/`, `metric3d_repo/` — vendored official repos, git-cloned per step 1 above (gitignored — don't commit these into the project repo, just re-clone)
- `_vendor_shims/mmcv/` — a ~15-line stand-in for the real `mmcv` package (see note below); `run_metric3d.py` puts it on `sys.path` automatically
- `checkpoints/` — model weights (already here: `depth_pro.pt`, `metric3d_vit_small.pth`; gitignored, don't commit them)
- `../ImagesAndRealMeasurments/<item>/` — 8 item folders, each with 3 photos (45angle, Front, Side) plus `measurements.json` ground truth, reused from the sibling folder rather than duplicated here
- `results/<item>/` — outputs of the scripts above, one subfolder per item (gitignored, regenerable)
- `results/scoring_summary.md` — cross-item summary table written by `score_against_ground_truth.py`
- `efficientsam_repo/` — vendored EfficientSAM repo, git-cloned per step 4 above (gitignored — don't commit, just re-clone)

## Environment notes

- Metric3D's `mono/utils/comm.py` imports `mmcv.utils.collect_env` at module
  load time but never calls it on the inference path. Real `mmcv` needs a
  slow from-source compile (no prebuilt wheel was reachable in the sandbox
  this was authored in) purely for that unused import, so `_vendor_shims/mmcv/`
  stands in for it: `Config`/`DictAction` are just re-exported from `mmengine`
  (already a real dependency), and `collect_env` is a no-op. Nothing on the
  inference path depends on real mmcv's behavior. If you'd rather use the
  real package, `pip install mmcv` and delete `_vendor_shims/`.
- Both scripts run fp16 (`torch.half`) by default and pick CUDA → MPS → CPU
  automatically. Expect Depth Pro (ViT-L, ~1536px processing) to be
  noticeably slower per image than Metric3D v2 ViT-Small regardless of
  device.
- Ground truth format: every item has `width_cm`/`height_cm`/`depth_cm`
  except Ladder, which tapers and instead has `width_top_cm`/
  `width_bottom_cm` (no plain `width_cm`) — `score_against_ground_truth.py`
  handles both shapes automatically.
