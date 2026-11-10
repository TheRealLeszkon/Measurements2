#!/usr/bin/env python3
"""Build a side-by-side comparison of Depth Pro vs Metric3D v2 depth for every
item's photos: RGB | Depth Pro | Metric3D v2, plus a stats table and a couple
of quick sanity checks against each item's known ground truth
(test/Measurements2/ImagesAndRealMeasurments/<item>/measurements.json).

Loops over every subfolder of ../ImagesAndRealMeasurments that has a
measurements.json (all 8 items, not just one).
"""
import json
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

BASE = Path(__file__).parent
ITEMS_DIR = BASE.parent / "ImagesAndRealMeasurments"
RESULTS_DIR = BASE / "results"


def iter_items():
    for item_dir in sorted(ITEMS_DIR.iterdir()):
        if item_dir.is_dir() and (item_dir / "measurements.json").exists():
            yield item_dir


def load_summary(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def compare_item(item_dir):
    item = item_dir.name
    dp_dir = RESULTS_DIR / item / "depth_pro"
    m3d_dir = RESULTS_DIR / item / "metric3d"
    out_dir = RESULTS_DIR / item / "comparison"
    gt_path = item_dir / "measurements.json"

    dp_summary = load_summary(dp_dir / "summary.json")
    m3d_summary = load_summary(m3d_dir / "summary.json")
    if not dp_summary or not m3d_summary:
        print(f"[{item}] need both results/{item}/depth_pro/summary.json and "
              f"results/{item}/metric3d/summary.json -- run run_depth_pro.py and "
              f"run_metric3d.py first. Skipping.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    gt = json.loads(gt_path.read_text()) if gt_path.exists() else None
    if gt:
        print(f"[{item}] ground truth (tape measure) for reference: {gt}")

    names = sorted(set(dp_summary) & set(m3d_summary))
    rows = []
    for name in names:
        img = np.array(Image.open(item_dir / f"{name}.jpeg").convert("RGB"))
        dp_vis = np.array(Image.open(dp_dir / f"{name}_depth_vis.png"))
        m3d_vis = np.array(Image.open(m3d_dir / f"{name}_depth_vis.png"))
        dp_depth = np.load(dp_dir / f"{name}_depth_m.npy")
        m3d_depth = np.load(m3d_dir / f"{name}_depth_m.npy")

        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        axes[0].imshow(img); axes[0].set_title(f"{item}/{name} - RGB"); axes[0].axis("off")
        axes[1].imshow(dp_vis); axes[1].set_title(
            f"Depth Pro\nmedian={dp_summary[name]['depth_median_m']:.2f}m  "
            f"fx={dp_summary[name]['focal_length_px_estimated']:.0f}px"
        ); axes[1].axis("off")
        axes[2].imshow(m3d_vis); axes[2].set_title(
            f"Metric3D v2\nmedian={m3d_summary[name]['depth_median_m']:.2f}m  "
            f"fx={m3d_summary[name]['focal_length_px_used']:.0f}px ({m3d_summary[name]['focal_length_source']})"
        ); axes[2].axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_sidebyside.png", dpi=130)
        plt.close(fig)

        # Pixel-space agreement: correlate the two depth maps (resized to match
        # if needed) as a rough cross-model sanity check -- not an accuracy
        # measure (no ground-truth depth map exists), just "do they agree on
        # relative structure".
        if dp_depth.shape != m3d_depth.shape:
            from PIL import Image as _I
            m3d_depth_rs = np.array(
                _I.fromarray(m3d_depth).resize(dp_depth.shape[::-1], _I.BILINEAR)
            )
        else:
            m3d_depth_rs = m3d_depth
        corr = float(np.corrcoef(dp_depth.flatten(), m3d_depth_rs.flatten())[0, 1])

        rows.append({
            "view": name,
            "depth_pro_median_m": dp_summary[name]["depth_median_m"],
            "depth_pro_min_m": dp_summary[name]["depth_min_m"],
            "depth_pro_max_m": dp_summary[name]["depth_max_m"],
            "depth_pro_focal_px": dp_summary[name]["focal_length_px_estimated"],
            "depth_pro_seconds": dp_summary[name]["inference_seconds"],
            "metric3d_median_m": m3d_summary[name]["depth_median_m"],
            "metric3d_min_m": m3d_summary[name]["depth_min_m"],
            "metric3d_max_m": m3d_summary[name]["depth_max_m"],
            "metric3d_focal_px_used": m3d_summary[name]["focal_length_px_used"],
            "metric3d_seconds": m3d_summary[name]["inference_seconds"],
            "pixelwise_correlation": corr,
        })
        print(f"[{item}] {name}: DepthPro median={rows[-1]['depth_pro_median_m']:.3f}m  "
              f"Metric3D median={rows[-1]['metric3d_median_m']:.3f}m  "
              f"corr={corr:.3f}")

    with open(out_dir / "comparison_table.json", "w") as f:
        json.dump({"rows": rows, "ground_truth_cm": gt}, f, indent=2)

    lines = ["| view | DP median (m) | DP range (m) | DP fx (px) | DP sec | "
             "M3D median (m) | M3D range (m) | M3D fx (px) | M3D sec | pixel corr |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['view']} | {r['depth_pro_median_m']:.2f} | "
            f"{r['depth_pro_min_m']:.2f}-{r['depth_pro_max_m']:.2f} | "
            f"{r['depth_pro_focal_px']:.0f} | {r['depth_pro_seconds']:.1f} | "
            f"{r['metric3d_median_m']:.2f} | "
            f"{r['metric3d_min_m']:.2f}-{r['metric3d_max_m']:.2f} | "
            f"{r['metric3d_focal_px_used']:.0f} | {r['metric3d_seconds']:.1f} | "
            f"{r['pixelwise_correlation']:.3f} |"
        )
    (out_dir / "comparison_table.md").write_text("\n".join(lines))
    print(f"[{item}] saved comparison to {out_dir}")


def main():
    for item_dir in iter_items():
        compare_item(item_dir)


if __name__ == "__main__":
    main()
