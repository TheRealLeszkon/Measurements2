#!/usr/bin/env python3
"""Mask each depth model's output to just the object, unproject the masked
pixels into camera-space 3D points (pinhole model, per project instructions:
Z=depth, X=(x-cx)*Z/fx, Y=(y-cy)*Z/fy), and compare the resulting rough
extents against the tape-measured ground truth -- for every item, not just
one.

This is deliberately crude -- axis-aligned extents in *camera* space, not a
proper gravity-aligned / PCA oriented bounding box (that's the next pipeline
stage). Height (Y-extent) is roughly view-invariant since the camera was
held level in all shots, so it's the most trustworthy quick check. Width
(X-extent) is only meaningful from the Front view; depth/thickness
(Z-extent) only from the Side view -- the 45angle view mixes both and isn't
scored, just reported for reference. A view with no mask (run_segmentation.py
skipped it -- see that script's docstring for why that's better than a wrong
number) is simply left out of the table for that item, not filled with a bad
value.
"""
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

BASE = Path(__file__).parent
ITEMS_DIR = BASE.parent / "ImagesAndRealMeasurments"
RESULTS_DIR = BASE / "results"

MODELS = ["depth_pro", "metric3d"]


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
    if gt is None or gt == 0 or est != est:  # nan check
        return None
    return 100.0 * (est - gt) / gt


def gt_width_targets(gt):
    """Ladder has width_top_cm/width_bottom_cm instead of a single width_cm
    (it tapers). Everything else has a plain width_cm. Return a dict of
    {label: value} to compare the Front-view apparent width against."""
    if "width_cm" in gt:
        return {"width_cm": gt["width_cm"]}
    targets = {}
    if "width_bottom_cm" in gt:
        targets["bottom"] = gt["width_bottom_cm"]
    if "width_top_cm" in gt:
        targets["top"] = gt["width_top_cm"]
    return targets


def score_item(item_dir):
    item = item_dir.name
    masks_dir = RESULTS_DIR / item / "masks"
    out_dir = RESULTS_DIR / item / "scoring"
    gt_path = item_dir / "measurements.json"
    dp_dir = RESULTS_DIR / item / "depth_pro"
    m3d_dir = RESULTS_DIR / item / "metric3d"

    if not (dp_dir / "summary.json").exists() or not (m3d_dir / "summary.json").exists():
        print(f"[{item}] missing depth_pro/metric3d results -- run run_depth_pro.py / "
              f"run_metric3d.py first. Skipping.")
        return []
    if not masks_dir.exists():
        print(f"[{item}] no masks -- run run_segmentation.py first. Skipping.")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    gt = json.loads(gt_path.read_text())
    gt_height = gt.get("height_cm")
    gt_depth = gt.get("depth_cm")
    width_targets = gt_width_targets(gt)
    print(f"\n########## {item} -- ground truth (cm): {gt} ##########")

    dp_focal = json.loads((dp_dir / "summary.json").read_text())

    rows = []
    pointclouds = {}
    for view in ["45angle", "Front", "Side"]:
        mask_path = masks_dir / f"{view}_mask.npy"
        if not mask_path.exists():
            print(f"  [{item}/{view}] no mask, skipping (either not run yet, or "
                  f"run_segmentation.py couldn't find a confident box for this view)")
            continue
        if view not in dp_focal:
            print(f"  [{item}/{view}] no depth_pro focal-length entry, skipping")
            continue
        mask = np.load(mask_path)
        fx = dp_focal[view]["focal_length_px_estimated"]
        fy = fx
        h, w = mask.shape
        cx, cy = w / 2.0, h / 2.0

        for model_name in MODELS:
            depth_path = RESULTS_DIR / item / model_name / f"{view}_depth_m.npy"
            if not depth_path.exists():
                continue
            depth = np.load(depth_path)
            ext = unproject_extents(depth, mask, fx, fy, cx, cy)
            pointclouds[(view, model_name)] = ext.pop("points_xyz")

            row = {
                "item": item,
                "view": view,
                "model": model_name,
                "n_mask_points": ext["n_points"],
                "apparent_width_cm": ext["width_m"] * 100,
                "apparent_height_cm": ext["height_m"] * 100,
                "apparent_depth_cm": ext["depth_m"] * 100,
                "height_err_pct": pct_err(ext["height_m"] * 100, gt_height),
            }
            if view == "Front":
                for label, target in width_targets.items():
                    row[f"width_err_pct_vs_{label}"] = pct_err(ext["width_m"] * 100, target)
            if view == "Side":
                row["depth_err_pct"] = pct_err(ext["depth_m"] * 100, gt_depth)
            rows.append(row)
            print(json.dumps(row, indent=2))

    with open(out_dir / "scoring.json", "w") as f:
        json.dump({"rows": rows, "ground_truth_cm": gt}, f, indent=2)

    def fmt_pct(v):
        return f"{v:+.1f}%" if v is not None else "-"

    width_cols = sorted({k for r in rows for k in r if k.startswith("width_err_pct_vs_")})
    header = (["view", "model", "n_pts", "width_cm", "height_cm", "depth_cm", "height_err%"]
               + width_cols + ["depth_err%"])
    lines = [f"Ground truth (cm): {gt}", "",
             "| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    for r in rows:
        cells = [
            r["view"], r["model"], str(r["n_mask_points"]),
            f"{r['apparent_width_cm']:.1f}", f"{r['apparent_height_cm']:.1f}", f"{r['apparent_depth_cm']:.1f}",
            fmt_pct(r["height_err_pct"]),
        ]
        cells += [fmt_pct(r.get(c)) for c in width_cols]
        cells.append(fmt_pct(r.get("depth_err_pct")))
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "scoring.md").write_text("\n".join(lines))

    if ("Front", "depth_pro") in pointclouds and ("Front", "metric3d") in pointclouds:
        fig = plt.figure(figsize=(12, 6))
        for i, model_name in enumerate(["depth_pro", "metric3d"]):
            pts = pointclouds[("Front", model_name)]
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            sample = pts[np.random.choice(len(pts), min(4000, len(pts)), replace=False)]
            ax.scatter(sample[:, 0], sample[:, 2], -sample[:, 1], s=1, alpha=0.4)
            ax.set_title(f"{item} Front view -- {model_name}")
            ax.set_xlabel("X (width, m)")
            ax.set_ylabel("Z (depth, m)")
            ax.set_zlabel("-Y (height, m)")
        fig.tight_layout()
        fig.savefig(out_dir / "pointcloud_sanity_check_front.png", dpi=130)
        plt.close(fig)

    print(f"[{item}] saved scoring to {out_dir}")
    return rows


def main():
    all_rows = []
    for item_dir in iter_items():
        all_rows.extend(score_item(item_dir))

    if not all_rows:
        print("\nNo scoring rows produced across any item -- nothing to summarize.")
        return

    # Cross-item overall scorecard: mean absolute error across every GT
    # comparison actually available (height in all views scored, width in
    # Front, depth in Side), per model, plus a per-item breakdown.
    summary_lines = ["# Cross-item scoring summary", ""]
    for model_name in MODELS:
        model_rows = [r for r in all_rows if r["model"] == model_name]
        errs = []
        for r in model_rows:
            if r.get("height_err_pct") is not None:
                errs.append(abs(r["height_err_pct"]))
            for k, v in r.items():
                if k.startswith("width_err_pct_vs_") and v is not None:
                    errs.append(abs(v))
            if r.get("depth_err_pct") is not None:
                errs.append(abs(r["depth_err_pct"]))
        if errs:
            line = f"{model_name}: mean abs error across {len(errs)} GT comparisons (all items) = {np.mean(errs):.1f}%"
            print(line)
            summary_lines.append(f"- **{line}**")

    summary_lines.append("")
    summary_lines.append("## Per-item mean abs error")
    summary_lines.append("")
    summary_lines.append("| item | depth_pro mean abs err% | metric3d mean abs err% |")
    summary_lines.append("|---|---|---|")
    items = sorted({r["item"] for r in all_rows})
    for item in items:
        cells = [item]
        for model_name in MODELS:
            errs = []
            for r in all_rows:
                if r["item"] != item or r["model"] != model_name:
                    continue
                if r.get("height_err_pct") is not None:
                    errs.append(abs(r["height_err_pct"]))
                for k, v in r.items():
                    if k.startswith("width_err_pct_vs_") and v is not None:
                        errs.append(abs(v))
                if r.get("depth_err_pct") is not None:
                    errs.append(abs(r["depth_err_pct"]))
            cells.append(f"{np.mean(errs):.1f}%" if errs else "-")
        summary_lines.append("| " + " | ".join(cells) + " |")

    (RESULTS_DIR / "scoring_summary.md").write_text("\n".join(summary_lines))
    print(f"\nSaved cross-item summary to {RESULTS_DIR / 'scoring_summary.md'}")


if __name__ == "__main__":
    main()
