#!/usr/bin/env python3
"""Run Metric3D v2 (ViT-Small) on every item's photos and save raw + visualized
depth.

Loops over every subfolder of ../ImagesAndRealMeasurments that has a
measurements.json (all 8 items, not just one).

Metric3D v2 needs a known pixel focal length (fx) to de-canonicalize its output
into real metric depth (canonical_to_real_scale = fx / 1000.0, per the official
hubconf.py example). Our phone photos carry no EXIF focal length, so unlike
Depth Pro (which estimates fx itself), Metric3D here is fed the fx that Depth
Pro estimated for the same image -- see results/<item>/depth_pro/summary.json.
This dependency asymmetry is itself one of the findings of this comparison.
"""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

REPO = Path(__file__).parent / "metric3d_repo"
sys.path.insert(0, str(Path(__file__).parent / "_vendor_shims"))  # local mmcv shim, see README
sys.path.insert(0, str(REPO))

from mmengine import Config  # noqa: E402
from mono.model.monodepth_model import get_configured_monodepth_model  # noqa: E402

CKPT = Path(__file__).parent / "checkpoints" / "metric3d_vit_small.pth"
CFG_FILE = REPO / "mono/configs/HourglassDecoder/vit.raft5.small.py"
ITEMS_DIR = Path(__file__).parent.parent / "ImagesAndRealMeasurments"
OUT_DIR = Path(__file__).parent / "results"

INPUT_SIZE = (616, 1064)  # (h, w), per vit.raft5.small.py data_basic.crop_size


def get_torch_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def iter_items():
    for item_dir in sorted(ITEMS_DIR.iterdir()):
        if item_dir.is_dir() and (item_dir / "measurements.json").exists():
            yield item_dir


def preprocess(rgb_origin, fx, fy, cx, cy):
    h, w = rgb_origin.shape[:2]
    scale = min(INPUT_SIZE[0] / h, INPUT_SIZE[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    intrinsic = [fx * scale, fy * scale, cx * scale, cy * scale]
    padding = [123.675, 116.28, 103.53]
    h2, w2 = rgb.shape[:2]
    pad_h = INPUT_SIZE[0] - h2
    pad_w = INPUT_SIZE[1] - w2
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(
        rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
        cv2.BORDER_CONSTANT, value=padding,
    )
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb_t = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb_t = torch.div((rgb_t - mean), std)
    rgb_t = rgb_t[None, :, :, :]
    return rgb_t, pad_info, intrinsic


def main():
    if not CKPT.exists():
        print(f"MISSING CHECKPOINT: {CKPT}")
        sys.exit(2)

    device = get_torch_device()
    print(f"Building Metric3D v2 ViT-Small model ({device})...")
    cfg = Config.fromfile(str(CFG_FILE))
    model = get_configured_monodepth_model(cfg)
    state = torch.load(str(CKPT), map_location="cpu")
    sd = state["model_state_dict"] if "model_state_dict" in state else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded checkpoint: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
    model.to(device)
    model.eval()

    for item_dir in iter_items():
        item = item_dir.name
        out_dir = OUT_DIR / item / "metric3d"
        out_dir.mkdir(parents=True, exist_ok=True)
        dp_summary_path = OUT_DIR / item / "depth_pro" / "summary.json"
        dp_summary = json.loads(dp_summary_path.read_text()) if dp_summary_path.exists() else {}
        if not dp_summary:
            print(f"WARNING [{item}]: no Depth Pro summary found -- run run_depth_pro.py "
                  f"first so Metric3D has a real focal length to use. Falling back to a "
                  f"generic 26mm-equivalent assumption.")

        summary = {}
        print(f"\n########## {item} ##########")
        for img_path in sorted(item_dir.glob("*.jpeg")):
            name = img_path.stem
            print(f"\n=== {item}/{name} ===")
            rgb_origin = np.array(Image.open(img_path).convert("RGB"))
            h, w = rgb_origin.shape[:2]

            dp_stats = dp_summary.get(name, {})
            fx = dp_stats.get("focal_length_px_estimated")
            if fx is None:
                fx = 26.0 / 36.0 * max(w, h)  # generic 26mm-equiv fallback
            cx, cy = w / 2.0, h / 2.0

            rgb_t, pad_info, intrinsic = preprocess(rgb_origin, fx, fx, cx, cy)
            rgb_t = rgb_t.to(device)

            t0 = time.time()
            with torch.no_grad():
                pred_depth, confidence, output_dict = model.inference({"input": rgb_t})
            dt = time.time() - t0

            pred_depth = pred_depth.squeeze()
            pred_depth = pred_depth[
                pad_info[0]: pred_depth.shape[0] - pad_info[1],
                pad_info[2]: pred_depth.shape[1] - pad_info[3],
            ]
            pred_depth = torch.nn.functional.interpolate(
                pred_depth[None, None, :, :], (h, w), mode="bilinear"
            ).squeeze()

            canonical_to_real_scale = intrinsic[0] / 1000.0
            pred_depth = pred_depth * canonical_to_real_scale
            pred_depth = torch.clamp(pred_depth, 0, 300)
            depth = pred_depth.cpu().numpy()

            stats = {
                "inference_seconds": dt,
                "focal_length_px_used": float(fx),
                "focal_length_source": "depth_pro_estimate" if name in dp_summary else "generic_26mm_fallback",
                "image_size_wh": [w, h],
                "depth_min_m": float(np.nanmin(depth)),
                "depth_max_m": float(np.nanmax(depth)),
                "depth_median_m": float(np.nanmedian(depth)),
                "depth_mean_m": float(np.nanmean(depth)),
            }
            print(json.dumps(stats, indent=2))
            summary[name] = stats

            np.save(out_dir / f"{name}_depth_m.npy", depth)

            inverse_depth = 1.0 / np.clip(depth, 1e-3, None)
            max_invdepth = min(inverse_depth.max(), 1 / 0.1)
            min_invdepth = max(1 / 250, inverse_depth.min())
            inv_norm = (inverse_depth - min_invdepth) / (max_invdepth - min_invdepth)
            cmap = plt.get_cmap("turbo")
            color = (cmap(np.clip(inv_norm, 0, 1))[..., :3] * 255).astype(np.uint8)
            Image.fromarray(color).save(out_dir / f"{name}_depth_vis.png")

        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    print(f"\nSaved results to {OUT_DIR}/<item>/metric3d/")


if __name__ == "__main__":
    main()
