#!/usr/bin/env python3
"""Dimension estimation: combine Depth Pro's per-view masked extents into one
final {width_cm, height_cm, depth_cm} estimate per item, and score that
single combined estimate against the tape-measured ground truth.

This is the next stage after run_depth_pro.py / run_segmentation.py /
score_against_ground_truth.py -- those produced per-view numbers (e.g.
"Front view says width=X, Side view says depth=Y") but never combined them
into one answer for the object. This script is that combination step: the
actual "estimate this item's dimensions" deliverable, not just a model
comparison.

Model choice: Depth Pro only. Comparing against Metric3D v2 across all 8
items (see results/scoring_summary.md) showed Depth Pro consistently lower
mean absolute error, and Metric3D produced at least one physically
impossible reading (Ladder Front-view width exceeding the ladder's own
widest point) that Depth Pro did not. Metric3D is left out of this stage.

Combination rule (deliberately simple -- this is NOT a true multi-view 3D
fusion / SFM registration, which would need known camera poses between the
3 shots and none exist here; see the docstring note on LIMITATIONS below):
- height_cm: median across every view that has a mask. Height is the most
  view-invariant of the 3 axes (camera was held level in every shot), so
  combining multiple independent estimates should be the most reliable
  single number this pipeline can produce.
- width_cm: taken from the Front view specifically -- it's the only view
  where the image's horizontal axis lines up with the object's true width
  axis. 45angle and Side both foreshorten or hide it.
- depth_cm (thickness/front-to-back): taken from the Side view specifically,
  for the same reason in the other axis.
- If the view a dimension depends on has no mask (run_segmentation.py
  skipped it), that dimension is reported as null with a reason, not
  guessed.

LIMITATIONS (be upfront about these, don't oversell the number):
- Camera-space per-view extents, not a gravity-aligned oriented bounding box
  computed from a registered 3D point cloud. Works because this dataset's
  Front/Side shots are close to axis-aligned with the object by convention
  (photographer stood square-on), not because the pipeline verified alignment.
- No cross-view outlier rejection beyond each view's own 2nd-98th percentile
  clip (see robust_extent()).
"""
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

BASE = Path(__file__).parent
ITEMS_DIR = BASE.parent / "ImagesAndRealMeasurments"
RESULTS_DIR = BASE / "results"
MODEL = "depth_pro"


def iter_items():
    for item_dir in sorted(ITEMS_DIR.iterdir()):
        if item_dir.is_dir() and (item_dir / "measurements.json").exists():
            yield item_dir


def robust_extent(values, lo=2, hi=98):
    if len(values) < 10:
        return float("nan")
    p_lo, p_hi = np.percentile(values, [lo, hi])
    return float(p_hi - p_lo)


def unproject_extents(depth, mask, fx, fy, cx, cy):
    ys, xs = np.where(mask)
    z = depth[ys, xs].astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    ys, xs, z = ys[valid], xs[valid], z[valid]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return {
        "n_points": int(len(z)),
        "width_m": robust_extent(x),
        "height_m": robust_extent(y),
        "depth_m": robust_extent(z),
        "points_xyz": np.stack([x, y, z], axis=1),
    }


def pct_err(est, gt):
    if est is None or gt is None or gt == 0:
        return None
    return 100.0 * (est - gt) / gt


def gt_width_targets(gt):
    if "width_cm" in gt:
        return {"width_cm": gt["width_cm"]}
    targets = {}
    if "width_bottom_cm" in gt:
        targets["bottom"] = gt["width_bottom_cm"]
    if "width_top_cm" in gt:
        targets["top"] = gt["width_top_cm"]
    return targets


def per_view_extents(item_dir, item):
    """Same per-view unprojection score_against_ground_truth.py does, but
    Depth Pro only, and it also hands back the point clouds for the box
    visualization below."""
    masks_dir = RESULTS_DIR / item / "masks"
    depth_dir = RESULTS_DIR / item / MODEL
    focal_path = depth_dir / "summary.json"
    if not focal_path.exists():
        return {}, {}
    focal = json.loads(focal_path.read_text())

    views = {}
    pointclouds = {}
    for view in ["45angle", "Front", "Side"]:
        mask_path = masks_dir / f"{view}_mask.npy"
        depth_path = depth_dir / f"{view}_depth_m.npy"
        if not mask_path.exists() or not depth_path.exists() or view not in focal:
            continue
        mask = np.load(mask_path)
        depth = np.load(depth_path)
        fx = focal[view]["focal_length_px_estimated"]
        h, w = mask.shape
        cx, cy = w / 2.0, h / 2.0
        ext = unproject_extents(depth, mask, fx, fx, cx, cy)
        pointclouds[view] = ext.pop("points_xyz")
        views[view] = ext
    return views, pointclouds


def combine(views):
    """The actual combination rule -- see module docstring."""
    height_candidates = {v: views[v]["height_m"] * 100 for v in views if not np.isnan(views[v]["height_m"])}
    height_cm = float(np.median(list(height_candidates.values()))) if height_candidates else None

    width_cm, width_source = None, None
    if "Front" in views and not np.isnan(views["Front"]["width_m"]):
        width_cm = views["Front"]["width_m"] * 100
        width_source = "Front"

    depth_cm, depth_source = None, None
    if "Side" in views and not np.isnan(views["Side"]["depth_m"]):
        depth_cm = views["Side"]["depth_m"] * 100
        depth_source = "Side"

    return {
        "width_cm": width_cm,
        "width_source": width_source,
        "height_cm": height_cm,
        "height_source_views": list(height_candidates.keys()),
        "height_per_view_cm": height_candidates,
        "depth_cm": depth_cm,
        "depth_source": depth_source,
    }


def draw_box(ax, width_m, height_m, depth_m, color="red"):
    """Wireframe of the estimated box, centered at the origin in
    (X=width, Z=depth, -Y=height) plot axes to match the point-cloud scatter."""
    if width_m is None or height_m is None or depth_m is None:
        return
    x0, x1 = -width_m / 2, width_m / 2
    z0, z1 = 0, depth_m
    y0, y1 = -height_m / 2, height_m / 2  # plotted as -Y already
    corners = np.array([[x, z, y] for x in (x0, x1) for z in (z0, z1) for y in (y0, y1)])
    edges = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 6), (5, 7), (6, 7)]
    for i, j in edges:
        ax.plot(*zip(corners[i], corners[j]), color=color, linewidth=1.5)


def process_item(item_dir):
    item = item_dir.name
    gt = json.loads((item_dir / "measurements.json").read_text())
    views, pointclouds = per_view_extents(item_dir, item)
    if not views:
        print(f"[{item}] no {MODEL} results/masks found -- run the earlier pipeline steps first. Skipping.")
        return None

    est = combine(views)
    width_targets = gt_width_targets(gt)
    result = {
        "item": item,
        "ground_truth_cm": gt,
        "estimate_cm": {k: est[k] for k in ("width_cm", "height_cm", "depth_cm")},
        "sources": {
            "width": est["width_source"],
            "height_views_used": est["height_source_views"],
            "height_per_view_cm": est["height_per_view_cm"],
            "depth": est["depth_source"],
        },
        "errors_pct": {
            "height": pct_err(est["height_cm"], gt.get("height_cm")),
            "depth": pct_err(est["depth_cm"], gt.get("depth_cm")),
        },
    }
    for label, target in width_targets.items():
        result["errors_pct"][f"width_vs_{label}"] = pct_err(est["width_cm"], target)

    out_dir = RESULTS_DIR / item / "dimensions"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "estimate.json", "w") as f:
        json.dump(result, f, indent=2)

    def fmt(v):
        return f"{v:.1f}cm" if v is not None else "n/a"

    def fmt_err(v):
        return f"{v:+.1f}%" if v is not None else "n/a"

    lines = [
        f"# {item} — dimension estimate ({MODEL})", "",
        f"Ground truth (cm): {gt}", "",
        f"- width: {fmt(est['width_cm'])} (from {est['width_source'] or 'no mask available'}) "
        + "  ".join(f"err vs {k.split('width_vs_')[-1]}={fmt_err(v)}"
                     for k, v in result['errors_pct'].items() if k.startswith('width_vs_')),
        f"- height: {fmt(est['height_cm'])} (median of {est['height_source_views']}, "
        f"per-view: {{{', '.join(f'{k}: {v:.1f}cm' for k, v in est['height_per_view_cm'].items())}}}) "
        f"err={fmt_err(result['errors_pct']['height'])}",
        f"- depth: {fmt(est['depth_cm'])} (from {est['depth_source'] or 'no mask available'}) "
        f"err={fmt_err(result['errors_pct']['depth'])}",
    ]
    (out_dir / "estimate.md").write_text("\n".join(lines))
    print(f"[{item}] width={fmt(est['width_cm'])} height={fmt(est['height_cm'])} depth={fmt(est['depth_cm'])}  "
          f"(gt: {gt})")

    if "Front" in pointclouds:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111, projection="3d")
        pts = pointclouds["Front"]
        sample = pts[np.random.choice(len(pts), min(4000, len(pts)), replace=False)]
        ax.scatter(sample[:, 0], sample[:, 2], -sample[:, 1], s=1, alpha=0.3)
        if est["width_cm"] is not None and est["height_cm"] is not None and est["depth_cm"] is not None:
            draw_box(ax, est["width_cm"] / 100, est["height_cm"] / 100, est["depth_cm"] / 100)
        ax.set_title(f"{item} — estimated box over Front-view point cloud")
        ax.set_xlabel("X (width, m)"); ax.set_ylabel("Z (depth, m)"); ax.set_zlabel("-Y (height, m)")
        fig.tight_layout()
        fig.savefig(out_dir / "box_visualization.png", dpi=130)
        plt.close(fig)

    return result


def main():
    results = []
    for item_dir in iter_items():
        r = process_item(item_dir)
        if r:
            results.append(r)

    if not results:
        print("\nNo items produced an estimate -- nothing to summarize.")
        return

    errs = []
    for r in results:
        for v in r["errors_pct"].values():
            if v is not None:
                errs.append(abs(v))
    overall = np.mean(errs) if errs else None

    lines = [f"# Dimension estimation summary ({MODEL})", "",
             f"**Overall mean abs error across {len(errs)} GT comparisons: "
             f"{overall:.1f}%**" if overall is not None else "No comparisons available.", "",
             "| item | width_cm (err) | height_cm (err) | depth_cm (err) |",
             "|---|---|---|---|"]
    for r in results:
        est = r["estimate_cm"]
        errp = r["errors_pct"]
        w_err = next((v for k, v in errp.items() if k.startswith("width_vs_")), None)

        def cell(val, err):
            if val is None:
                return "n/a"
            err_s = f" ({err:+.1f}%)" if err is not None else ""
            return f"{val:.1f}{err_s}"

        lines.append(
            f"| {r['item']} | {cell(est['width_cm'], w_err)} | "
            f"{cell(est['height_cm'], errp['height'])} | {cell(est['depth_cm'], errp['depth'])} |"
        )
    (RESULTS_DIR / "dimensions_summary.md").write_text("\n".join(lines))
    print(f"\nSaved dimension estimates to results/<item>/dimensions/ and results/dimensions_summary.md")
    if overall is not None:
        print(f"Overall mean abs error: {overall:.1f}%")


if __name__ == "__main__":
    main()
