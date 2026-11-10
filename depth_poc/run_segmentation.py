#!/usr/bin/env python3
"""Segment the object in each item's 3 photos: YOLOE (prompt-free, open-vocab
detector -- avoids the "COCO has no box/ladder class" trap this project hit
before, see PRIOR_WORK_SUMMARY.md) for a bounding box, then EfficientSAM-S
refines that box into a precise mask. This is the project's established
YOLOE -> EfficientSAM pattern (see Yolo-test).

Loops over every subfolder of ../ImagesAndRealMeasurments that has a
measurements.json (all 8 items: Ladder, CardboardBox, Dustbin, FootRestBox,
Printer, PS5, Table, ToolBox -- not just one).

Output per item: results/<item>/masks/{name}_mask.npy (bool, same H,W as the
depth maps) and results/<item>/masks/{name}_overlay.png for a visual sanity
check.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

BASE = Path(__file__).parent
ITEMS_DIR = BASE.parent / "ImagesAndRealMeasurments"
OUT_DIR = BASE / "results"
YOLOE_CKPT = BASE / "checkpoints" / "yoloe-11s-seg-pf.pt"
EFFICIENTSAM_CKPT = BASE / "checkpoints" / "efficient_sam_vits.pt"
EFFICIENTSAM_REPO = BASE / "efficientsam_repo"

sys.path.insert(0, str(EFFICIENTSAM_REPO))


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


def item_keywords(item_name):
    """Split a folder name like 'CardboardBox' or 'FootRestBox' into lowercase
    keywords ('cardboard', 'box' / 'foot', 'rest', 'box') for a loose name
    match against YOLOE's open-vocab class guesses. Short/numeric fragments
    (e.g. the "5" in "PS5") are dropped since they're too likely to false-match."""
    words = re.findall(r"[A-Z]?[a-z]+", item_name)
    return [w.lower() for w in words if len(w) > 2]


SCENE_WORDS = (
    "room", "floor", "wall", "interior", "house", "sky", "ceiling", "gym",
    "laboratory", "hospital", "corridor", "hallway", "corner", "staircase",
)


def pick_object_box(result, image_hw, keywords):
    """From a YOLOE Results object, pick the box most likely to be the item
    being measured.

    Verified against the real photos first, not shipped blind -- this
    heuristic went through 4 failed attempts before this one (see git
    history / PRIOR_WORK_SUMMARY.md for the full story):

    1. Naive "highest confidence" or "largest box" both fail: highest-
       confidence can be a small background object, largest-area can be a
       whole-scene label ("clean room", "home interior") that isn't a real
       object at all.
    2/3. Two versions of a "full-body object standing on the floor" filter
       (hard-cutoff, then a softened score) required spanning >50% of frame
       height AND reaching >75% toward the floor. That overfit to how the
       ladder specifically fills its frame: run against the other 7 items
       (photographed smaller/further back) it rejected almost every
       detection.
    4. Dropping the floor-position requirement entirely (keep only a scene-
       label blocklist + size cap) fixed that, but overcorrected: verified
       against the real run's overlay images, several items (Printer,
       CardboardBox, ToolBox, PS5) picked a small red decoy object that
       recurs in the background of several shots, because it sometimes gets
       a cleaner single-object label than the real item does. The one part
       of the old geometry check that *was* valid turned out to be floor
       position, not size -- this dataset really is shot with the item
       resting on the floor, camera roughly level, so genuine picks were
       confirmed (via detector_box_xyxy in results/<item>/masks/summary.json)
       to consistently reach at least into the lower half of the frame,
       while every confirmed-wrong pick stayed entirely in the upper ~30%.

    Fix: keep the scene-label blocklist (SCENE_WORDS) and size cap (>55% of
    frame area), and bring back a *loose* floor-position requirement (box
    bottom edge past the vertical midpoint of the frame) without the old
    height-span requirement -- an item can be small in frame and still be
    resting on the floor. Among what survives, prefer a class-name keyword
    match (e.g. "storage box" for CardboardBox), else highest confidence.
    Returns None (skip the view) when nothing survives -- a missing view
    beats a wrong mask.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, []
    h, w = image_hw
    image_area = h * w
    names = result.names
    detections = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
        conf = float(boxes.conf[i].item())
        xyxy = boxes.xyxy[i].tolist()
        area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
        detections.append({"cls_name": cls_name, "conf": conf, "xyxy": xyxy, "area": area})

    plausible = [
        d for d in detections
        if d["area"] / image_area < 0.55
        and d["xyxy"][3] / h > 0.5   # bottom edge reaches at least the lower half of the frame
        and not any(sw in d["cls_name"].lower() for sw in SCENE_WORDS)
    ]
    if not plausible:
        return None, detections

    name_matches = [d for d in plausible if any(kw in d["cls_name"].lower() for kw in keywords)]
    best = max(name_matches or plausible, key=lambda d: d["conf"])
    return best, detections


def main():
    if not YOLOE_CKPT.parent.exists():
        YOLOE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    if not EFFICIENTSAM_CKPT.exists():
        print(f"MISSING: {EFFICIENTSAM_CKPT}\n"
              f"git clone https://github.com/yformer/EfficientSAM.git {EFFICIENTSAM_REPO.name}\n"
              f"then unzip {EFFICIENTSAM_REPO.name}/weights/efficient_sam_vits.pt.zip into checkpoints/")
        sys.exit(2)
    if not (EFFICIENTSAM_REPO / "efficient_sam").exists():
        print(f"MISSING: {EFFICIENTSAM_REPO}/efficient_sam -- "
              f"git clone https://github.com/yformer/EfficientSAM.git {EFFICIENTSAM_REPO.name}")
        sys.exit(2)

    from efficient_sam.efficient_sam import build_efficient_sam  # noqa: E402
    from ultralytics import YOLOE  # noqa: E402

    device = get_torch_device()

    print(f"Loading YOLOE-11s-seg-pf (detector) and EfficientSAM-S (mask refiner) on {device}...")
    detector = YOLOE(str(YOLOE_CKPT))  # auto-downloads to this path if missing
    segmenter = build_efficient_sam(
        encoder_patch_embed_dim=384, encoder_num_heads=6, checkpoint=str(EFFICIENTSAM_CKPT)
    ).eval().to(device)

    for item_dir in iter_items():
        item = item_dir.name
        keywords = item_keywords(item)
        out_dir = OUT_DIR / item / "masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {}
        print(f"\n########## {item} (keywords={keywords}) ##########")

        for img_path in sorted(item_dir.glob("*.jpeg")):
            name = img_path.stem
            print(f"\n=== {item}/{name} ===")
            pil_image = Image.open(img_path).convert("RGB")
            rgb_np = np.array(pil_image)

            results = detector.predict(str(img_path), verbose=False)
            best, all_detections = pick_object_box(results[0], image_hw=rgb_np.shape[:2], keywords=keywords)
            print("  detections:", [(d["cls_name"], round(d["conf"], 2)) for d in all_detections])
            if best is None:
                print("  NO DETECTIONS PASSED THE FILTER -- skipping (no mask written for this view)")
                summary[name] = {
                    "detector_class": None,
                    "skipped": True,
                    "all_detections": [
                        {"cls_name": d["cls_name"], "conf": round(d["conf"], 3), "xyxy": d["xyxy"]}
                        for d in all_detections
                    ],
                }
                with open(out_dir / "summary.json", "w") as f:
                    json.dump(summary, f, indent=2)
                continue
            print(f"  picked: {best['cls_name']} conf={best['conf']:.2f} box={best['xyxy']}")

            x1, y1, x2, y2 = best["xyxy"]
            input_points = torch.tensor([[[[x1, y1], [x2, y2]]]], dtype=torch.float32, device=device)
            input_labels = torch.tensor([[[2, 3]]], dtype=torch.float32, device=device)
            image_tensor = transforms.ToTensor()(rgb_np)[None, ...].to(device)

            with torch.no_grad():
                predicted_logits, predicted_iou = segmenter(image_tensor, input_points, input_labels)
            sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
            predicted_logits = torch.take_along_dim(predicted_logits, sorted_ids[..., None, None], dim=2)
            mask = torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().numpy()

            mask_px = int(mask.sum())
            summary[name] = {
                "detector_class": best["cls_name"],
                "detector_conf": best["conf"],
                "detector_box_xyxy": best["xyxy"],
                "skipped": False,
                "all_detections": [
                    {"cls_name": d["cls_name"], "conf": round(d["conf"], 3), "xyxy": d["xyxy"]}
                    for d in all_detections
                ],
                "mask_pixel_count": mask_px,
                "mask_fraction_of_image": mask_px / mask.size,
            }
            print(f"  mask: {mask_px} px ({100 * mask_px / mask.size:.1f}% of image)")

            np.save(out_dir / f"{name}_mask.npy", mask)

            overlay = rgb_np.copy()
            overlay[mask] = (overlay[mask] * 0.4 + np.array([255, 60, 60]) * 0.6).astype(np.uint8)
            Image.fromarray(overlay).save(out_dir / f"{name}_overlay.png")

            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    print(f"\nSaved masks to {OUT_DIR}/<item>/masks/")


if __name__ == "__main__":
    main()
