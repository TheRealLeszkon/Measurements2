# PackingSurveyImproved — Prior Work Summary

*Compiled 2026-08-22 for the Measurements2 dimension-estimation POC. Sources: the docs and code inside each experiment folder listed below. All paths are relative to `PackingSurveyImproved/` unless noted.*

## TL;DR — what to take away before starting new experiments

1. **No experiment in this entire project tree has a single logged, tape-measure-verified, on-device W×H×D accuracy number.** Every pipeline below reaches "code-complete" or "ran once informally," then stops short of a repeatable, saved ground-truth result. Closing that gap is the actual open problem.
2. **ARCore depth was tried extensively (3 separate app iterations) and had two concrete, root-caused bugs**, both now understood/fixed in code but never re-validated:
   - Z-depth vs. straight-line range confusion — up to ~20% error at frame corners (`ARSelectionTest2`).
   - `DEPTH16` raw-bits misdecode (13-bit mm + 3-bit confidence read as pure mm) causing ~8m depth inflation, root-caused to a real "Height ≈ 626 cm" blow-up (`ARTest2`).
3. **DUSt3R and MASt3R were both tried** (`Metrics-Finding-Tests`) but produced no saved accuracy numbers — scripts are fully interactive (`input()` prompts), results only ever printed to stdout, never logged to a file. The project brief's "not sufficiently reliable" verdict on DUSt3R is anecdotal, not backed by a surviving error table.
4. **YOLO/COCO detectors have no generic "box" class** — tap-to-select on a plain cardboard box fails outright with class-based detectors. This bug, found in `ARSelectionTest`, is why `ARTest3` pivoted to **MobileSAM** (promptable, class-agnostic) for the interactive-selection step.
5. **EfficientSAM vs FastSAM has the one hard, saved benchmark number in the whole tree**: EfficientSAM averaged **14.5 ms/mask vs FastSAM's 484.3 ms/mask (~33× faster)**, consistent across 4 scenes and 3 detector sources (`Yolo-test/benchmark_results.csv`). This is the direct evidence behind the decision to use YOLO→EfficientSAM on-device.
6. **Depth Anything V2 (monocular, relative depth) was tried and explicitly rejected for measurement** — used only as a visual "thermal-style" effect. Documented reason: per-frame min/max normalization causes flicker and the output is relative depth, not metric — "fine for a visual effect, not for measurement" (`ImageProcessingWithOpenCV/ThermalDepth/README.md`).
7. **A parallel non-geometric approach exists**: a 4-agent VLM pipeline (Observer→Estimator→Critic→Judge) that reasons over photos to estimate dimensions with confidence labels, no depth/point-cloud math at all. Framework is built but ground truth is still a placeholder — never benchmarked for real.

---

## App context

- The companion Android survey app runs **YOLO → EfficientSAM on-device** for immediate feedback while capturing, and leaves measurement and pricing to the server, so capturing the next room is never blocked.
- No server-side measurement pipeline existed when this was written: nothing turned masks into real-world dimensions. That is the gap Measurements2 is meant to fill.

## `test/` — experiment superproject

`test/` is a git superproject; submodules are `ARTest`, `camera-sub`, `camera-test`, `Metrics-Finding-Tests`, `SAMTiral`, `Yolo-test` (plus plain folders `ARSelectionTest`, `ARSelectionTest2`, `ARTest2`, `ARTest3`, `Measurements2`). `test/.git.old-seg-and-measurement-tests` is a renamed, archived pre-reset `.git` dir (last commit: "removed models from commit") — old history, not source of truth.

### ARTest — first ARCore-depth-only prototype
- Pipeline: Camera → ARCore depth → point cloud → object isolation (nearest-point sphere) → temporal EMA accumulator → **gravity-aligned AABB (no PCA)** → L/W/H. Deliberately no ML at all ("No TensorFlow / No MediaPipe / No OpenCV").
- Design note: PCA/min-area-rect explicitly deferred "until measurements prove it matters."
- **Never validated on hardware** — no depth-capable device was available at the time; pipeline compiled but produced no real accuracy numbers.
- Its `CLAUDE.md` set the project's minimalism philosophy: smallest amount of software, avoid mathematical complexity unless it demonstrably improves accuracy.

### ARTest2 — most-developed AR line: multi-view ARCore + YOLO11-Seg + PCA/OBB
- Single-model choice: **YOLO11-Seg** (not YOLO+EfficientSAM) "to halve on-device complexity" for a PoC, run via ONNX Runtime Mobile (NNAPI).
- Masked depth → world point cloud accumulated across viewpoints (1.5cm-voxel bounded grid). **PCA OBB used for debug display only** — authoritative geometry is **gravity-aligned height (percentile floor/top) + min-area-rectangle horizontal footprint**, explicitly not PCA, because "PCA finds the principal directions of the point distribution... not guaranteed to correspond to the physical semantic axes of an object."
- **Concrete failure, root-caused** (`docs/PROGRESS.md`, Increment 4): first real multi-view scan of a box on a cluttered floor produced **Height ≈ 626 cm, Footprint 1177×580 cm**, target misclassified as "toilet 64%." Root cause: target selection picked the largest detection independently every frame, so a misdetection with an oversized mask (or a switched target) let room-scale depth into the accumulator. Fixed with a `TargetTracker` (persistent identity via class+IoU+center) + `TargetIsolation` (0.9 m radius gate) + later tap-to-select + world-space centroid lock.
- **Second concrete bug**, found via a depth↔RGB reprojection diagnostic (`docs/CAPTURE_FIRST.md`): ARCore's `acquireDepthImage16Bits()` packs 13-bit mm + 3-bit confidence; the code was reading all 16 bits as raw mm, adding +8192 mm whenever a confidence bit was set — "a plausible contributor to the earlier 'Height ≈ 626 cm' blow-ups." Fixed via a shared `Depth16.kt` decoder (`0x1FFF` mask).
- Status: all 7 milestones code-complete, unit-tested (16+ JVM tests, synthetic-box test recovers W/H/D within a few cm in simulation), but **real-device tape-measure accuracy was never completed** — blocked on device availability and lighting (`INSUFFICIENT_LIGHT` sessions). `docs/development-process.md` states outright: "Previous experiments showed significant measurement errors — the purpose of this iteration is to determine whether multi-view accumulation materially reduces those errors," confirming single-frame PCA-OBB measurement was known-bad before this iteration.
- Reusable engineering rules from `docs/rules.md`: never use raw min/max without outlier rejection; never treat one frame as an independent measurement; localize the failing pipeline stage (seg→depth→alignment→pose→accumulation→outliers→floor→footprint→stabilization) before touching the geometry algorithm; never claim hardware/accuracy validation without evidence.

### ARSelectionTest / ARSelectionTest2 — narrow probes (selection UX and raw depth characterization)
- **ARSelectionTest**: isolates only "tap/drag-select an object, keep it anchored as you walk" — not a measurement pipeline. v0.1 used COCO-class YOLO for tap-to-mask and **failed outright because COCO has no "box" class**. Fixed in v0.2 (class-agnostic depth region-grow, verified on-device: `extent=20×16×14cm`) and refined in v0.3 (drag-rectangle hybrid: YOLO mask if ≥50% inside the rectangle, else rectangle-bounded region-grow); v0.3 built but not yet verified on-device.
- **ARSelectionTest2 ("ARDepth")**: isolates only ARCore depth characterization, no ML. Verified on-device (OnePlus Nord 5, matte office scene): **raw ARCore depth is Z-depth (perpendicular to the camera plane), not straight-line range** — they diverge ~0.7–1% near image center up to **~20% at screen corners** (e.g. Z=1.078 m → true range 1.292 m at a far corner). A manual calibration slider was added as a stop-gap; a full tape-measure sweep at 0.5/1/2/3 m was left as future work and, per the survey, never done.

### ARTest3 ("AR Measure") — most sophisticated AR pipeline: MobileSAM + visual hull
- Three phases, all coded: (1) tap → **MobileSAM** (ONNX encoder+decoder, replacing YOLO/COCO specifically to fix the "no box class" problem found in ARSelectionTest) → coarse PCA OBB → ARCore world anchor. (2) guided 3-view capture (0°/45°/90°, parallax-gated ≥20°) with per-view re-segmentation and keyframe caching. (3) **visual-hull voxel carving** (intersection of 3 silhouette cones) → depth fusion filtered to inside the hull → cuboid fit (PCA orthogonal axes + robust percentile extents) → interactive corner-drag calibration UI.
- Status: all 6 milestones code-complete, 93 MB APK builds and installs/launches, but **the full tap→SAM→anchor→capture→measure flow was never human-verified on device**, and at last note the MobileSAM ONNX assets themselves hadn't been exported/generated yet (needs the MobileSAM pip package + network access).

### camera-sub / camera-test — plumbing only, no measurement logic
- `camera-sub`: plain photo-collection app (system camera intent + Room DB, MVVM) — explicitly "NOT a camera application."
- `camera-test`: bare CameraX permission/preview scaffold.
- Both reusable as UI plumbing; neither contains any measurement code.

### Metrics-Finding-Tests — the DUSt3R / MASt3R experiments
- Vendored `dust3r/` and `mast3r/` (naver) run against 3 photos of an object (`images/box2/*.jpeg`), optional `rembg` background removal, producing an **unscaled** point cloud → axis-aligned bbox. The user manually supplies one known real dimension to recover metric scale (`scale_factor = real_measurement / raw_dimension`), then the script prompts for tape-measure ground truth on all 3 axes and prints absolute + % error — but only to stdout, interactively, nothing logged to disk.
- `benchmark_mast3r.py` repeats the protocol with MASt3R's sparse global alignment (matching-based) instead of DUSt3R's global aligner, using the **metric-trained checkpoint** `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`, for side-by-side comparison.
- **No accuracy numbers survive on disk** — only point-cloud output files (`output_dust3r.ply`, `output_dust3r_object.ply`) prove the scripts were run.
- A companion **`arcore_capture/`** Android app (Kotlin, no AI) exports ARCore pose+intrinsics+images (`camera_poses.json`, `camera_intrinsics.json`, `metadata.json`) in DUSt3R/MASt3R-consumable format — an attempt to feed DUSt3R *known* ARCore camera poses instead of its own estimated (unscaled, possibly less accurate) poses. One real captured session exists in `arcore_data/capture_001/`.
- Several `DepthCamera_*.json` dumps at the repo root (dated 2026-07-23), not inspected in detail — likely raw ARCore depth captures.

### SAMTiral — minimal, effectively abandoned
- Empty README. Only fetches the original Meta `segment-anything` ViT-B checkpoint via `uv sync`; no custom code beyond the vendored package. Superseded by `Yolo-test`'s broader SAM-variant benchmark and by ARTest3's move to MobileSAM.

### Yolo-test — the benchmark behind the EfficientSAM decision
- `benchmark.py` systematically times SAM variants (SAM ViT-B/L/H, MobileSAM, FastSAM, EfficientSAM, SAM2) for load time, inference time, mask count, auto-skipping models with missing checkpoints.
- **Hard numbers** (`benchmark_results.csv`): EfficientSAM averaged **14.5 ms/mask** vs FastSAM's **484.3 ms/mask** (~33× faster), consistent across 4 test scenes (Desk, Family-room-2, FamilyRoom, Hall) and 3 detector sources (Objects365, LVIS, YOLO-World). This is the evidence behind ARTest2's PLAN.md claim and the survey app's on-device pipeline.
- `filter_benchmark.py`: a geometry-only post-filter (no AI) removing tiny/sliver detections and parent-child-contained duplicates (e.g. "drawer inside a table," ≥90% containment) before any downstream stage. Logged example: FamilyRoom/LVIS — 32 raw detections → 20 after geometry filter → 13 after parent-child filter.
- `tools/export/README.md`: documents the ONNX export pipeline actually used on-device — **YOLOe** (`yoloe-11s-seg-pf.onnx`, official Ultralytics exporter) and **EfficientSAM** (`efficient_sam_vits_encoder/decoder.onnx`, custom split export — EfficientSAM has no official ONNX exporter). Both convert cleanly to ONNX Runtime Mobile at opset 17. This is the direct source of the ONNX assets used in ARTest2/ARTest3.

## `AgenticApproaches/` — VLM-based, non-geometric approach

### MultipleVLMs (`vlm-dimension-agent/`)
- Deliberately **no classical CV / depth / point clouds**. A 4-agent pipeline (Observer → Estimator → Critic → Judge, each a distinct prompt) reasons over 3–5 photos to estimate W/H/D, labeling each value observed/inferred/assumed, output as JSON with ranges, confidence, and warnings.
- Backends: `transformers` (CUDA/CPU/MPS) or `mlx` (Apple Silicon 4-bit — "the way to run locally on a Mac"). Candidate models: Qwen2.5-VL-7B, Qwen3-VL-8B/30B, InternVL3-8B/14B, MiniCPM-V-4.5, SmolVLM-256M (smoke-test only).
- Explicit stated limitation: "accuracy hinges on the presence of genuine, correctly-identified scale references. Without one, absolute size is largely an educated guess... **Depth is the weakest dimension** (the far side is usually unseen)."
- **No real accuracy results exist** — `dataset/object_001/ground_truth.json` is explicitly marked `"_placeholder": true` with a note to replace it with real measurements before trusting benchmark numbers. A `benchmark.py` (MAPE, range-hit-rate, confidence-vs-error) exists but has apparently never been run against real ground truth. Framed in its own README as exploring "whether classical computer vision / geometric reconstruction is actually necessary" — i.e. a genuine alternative track worth revisiting, not a dead end.

### smolVLM
- Empty README; contains `smolVLM.py`, `main.py`. Almost certainly an earlier, superseded standalone SmolVLM experiment predating the MultipleVLMs framework (which already includes an `smolvlm-256m` entry described as a "plumbing smoke test only"). No further documentation to extract.

## `ImageProcessingWithOpenCV/`

- Top-level scripts (`contours.py`, `transformations.py`, `smoothing.py`, `colour_spaces.py`, `draw.py`, `color_channel_split_merge.py`, `main.py`) are basic OpenCV tutorial/learning material — contour detection, geometric transforms, blur, color-space conversion, channel split/merge. Not measurement pipeline code.
- **`ThermalDepth/` — directly relevant to the depth-estimation mission.** Android app running **Depth Anything V2 Small (ViT-S)**, exported to ONNX, on-device via ONNX Runtime (NNAPI/CPU), used to colorize live camera frames as a pseudo-thermal visual effect. Explicit, load-bearing caveat in its README: **"This is not a thermal camera. Colors represent estimated relative depth, not temperature or physical distance."** And under "Compromises": "Per-frame min/max normalization causes mild global flicker as the scene's depth range changes; fine for a visual effect, **not for measurement**."
- **Takeaway for Measurements2**: Depth Anything V2 in its base form produces only *relative*, per-frame-normalized depth, not metric depth — it would need a separate scale-recovery step before it could feed a W×H×D pipeline (the same scale-recovery problem DUSt3R/MASt3R hit in `Metrics-Finding-Tests`). This is consistent with the project brief's instruction to look at genuinely *metric* depth models (Depth Pro, Metric3D v2) instead.

## `android-architecture-guide`

- `features/processing/CLAUDE.md` — architecture-only guardrails: AI inference (YOLO, SAM) must be isolated from UI, run async, be cancellable and replaceable, never run inside composables. No measurement-approach-specific content found.

---

## Open threads worth revisiting

- Neither ARCore-depth line (ARTest/ARTest2/ARTest3) nor the DUSt3R/MASt3R line ever produced a saved, repeatable tape-measure accuracy result — despite two rounds of real bug-fixing (Z-depth/range confusion, DEPTH16 misdecode) that plausibly explain much of the earlier "unreliable" results.
- The VLM (no-geometry) approach in `AgenticApproaches/MultipleVLMs` is unfinished but architecturally interesting as a fallback/complement, especially given its self-aware acknowledgment that depth is its weakest axis — worth benchmarking against the new metric-depth pipeline once both have real ground truth (which now exists for 8 objects in `test/Measurements2/ImagesAndRealMeasurments/*/measurements.json`).
- No prior experiment used a genuinely metric monocular depth model (Depth Pro, Metric3D v2) as specified in this project's brief — Depth Anything V2 (relative only) is the closest prior art, and it was explicitly ruled out for measurement use.
