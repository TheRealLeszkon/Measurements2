#!/usr/bin/env python3
"""Run Apple Depth Pro on every item's photos and save raw + visualized depth.

Loops over every subfolder of ../ImagesAndRealMeasurments that has a
measurements.json (so it picks up all 8 items -- Ladder, CardboardBox,
Dustbin, FootRestBox, Printer, PS5, Table, ToolBox -- not just one)."""
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt

REPO = Path(__file__).parent / "depth_pro_repo"
sys.path.insert(0, str(REPO / "src"))

from depth_pro import create_model_and_transforms, load_rgb  # noqa: E402
from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT  # noqa: E402
import dataclasses  # noqa: E402

CKPT = Path(__file__).parent / "checkpoints" / "depth_pro.pt"
ITEMS_DIR = Path(__file__).parent.parent / "ImagesAndRealMeasurments"
OUT_DIR = Path(__file__).parent / "results"


def get_torch_device() -> torch.device:
    """Prefer GPU acceleration when it's actually available (matches the
    official depth_pro CLI). On Apple Silicon this picks MPS."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_items():
    for item_dir in sorted(ITEMS_DIR.iterdir()):
        if item_dir.is_dir() and (item_dir / "measurements.json").exists():
            yield item_dir


def main():
    if not CKPT.exists():
        print(f"MISSING CHECKPOINT: {CKPT}")
        sys.exit(2)

    device = get_torch_device()
    config = dataclasses.replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(CKPT))

    print(f"Loading Depth Pro model ({device}, fp16)...")
    t0 = time.time()
    # fp32 on CPU peaked at ~6GB RSS on a single 1536px image and got OOM-killed
    # in the sandbox this was first tried in; torch.half roughly halves both
    # weight and activation memory (the only precision value depth_pro.py's
    # model.half() branch actually triggers on) and is what the official CLI
    # defaults to anyway. On MPS/CUDA this also runs meaningfully faster.
    model, transform = create_model_and_transforms(
        config=config, device=device, precision=torch.half
    )
    model.eval()
    print(f"  loaded in {time.time() - t0:.1f}s")

    for item_dir in iter_items():
        item = item_dir.name
        out_dir = OUT_DIR / item / "depth_pro"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {}
        print(f"\n########## {item} ##########")
        for img_path in sorted(item_dir.glob("*.jpeg")):
            name = img_path.stem
            print(f"\n=== {item}/{name} ===")
            image, _, f_px = load_rgb(str(img_path))
            t0 = time.time()
            with torch.no_grad():
                prediction = model.infer(transform(image), f_px=f_px)
            dt = time.time() - t0
            depth = prediction["depth"].detach().cpu().float().numpy().squeeze()  # meters
            focal_px = None
            if prediction.get("focallength_px") is not None:
                focal_px = float(prediction["focallength_px"].detach().cpu().item())
            del prediction
            gc.collect()

            stats = {
                "inference_seconds": dt,
                "focal_length_px_estimated": focal_px,
                "focal_length_px_from_exif": f_px,
                "image_size_wh": list(image.shape[1::-1]) if hasattr(image, "shape") else None,
                "depth_min_m": float(np.nanmin(depth)),
                "depth_max_m": float(np.nanmax(depth)),
                "depth_median_m": float(np.nanmedian(depth)),
                "depth_mean_m": float(np.nanmean(depth)),
            }
            print(json.dumps(stats, indent=2))
            summary[name] = stats

            np.save(out_dir / f"{name}_depth_m.npy", depth)

            inverse_depth = 1.0 / depth
            max_invdepth = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth = max(1 / 250, inverse_depth.min())
            inv_norm = (inverse_depth - min_invdepth) / (max_invdepth - min_invdepth)
            cmap = plt.get_cmap("turbo")
            color = (cmap(inv_norm)[..., :3] * 255).astype(np.uint8)
            Image.fromarray(color).save(out_dir / f"{name}_depth_vis.png")

            # Write incrementally so partial progress survives if a later image OOMs.
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    print(f"\nSaved results to {OUT_DIR}/<item>/depth_pro/")


if __name__ == "__main__":
    main()
