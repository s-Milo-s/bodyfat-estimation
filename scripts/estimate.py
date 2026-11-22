"""End-to-end: front + side silhouettes + gender/age/height/weight -> measurements -> body fat %.

Run from the project root, e.g.
  venv/bin/python scripts/estimate.py --front a.png --side b.png --gender male --age 30 --height 178 --weight 80
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch
from PIL import Image

from bodym_model import BodyNet, extra_features, get_device, load_masks
from nhanes_train import OUT as STAGE2_PATH
from nhanes_train import predict as predict_body_fat
from nhanes_train import rfm
from bodym_train import CKPT as STAGE1_PATH
from segment import photo_to_mask


def check_mask(path):
    w, h = Image.open(path).size
    if abs(w / h - 0.75) > 0.05:
        print(f"warning: {path} is {w}x{h}; BodyM masks are 720x960 (3:4), results may be off")


def estimate(front, side, gender, age, height_cm, weight_kg):
    ck = torch.load(STAGE1_PATH, weights_only=False)
    targets, mean, std = ck["targets"], ck["mean"], ck["std"]
    device = get_device()
    model = BodyNet(len(targets), pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    meta = pd.DataFrame([{"is_male": int(gender == "male"), "height_cm": height_cm}])
    with torch.no_grad():
        pred = model(
            (load_masks([front]).float() / 255).to(device),
            (load_masks([side]).float() / 255).to(device),
            extra_features(meta).to(device),
        ).cpu().numpy()[0]
    measurements = dict(zip(targets, (pred * std + mean).tolist()))

    stage2 = joblib.load(STAGE2_PATH)
    male = int(gender == "male")
    body_fat = predict_body_fat(
        stage2, male, age, height_cm, weight_kg, measurements["waist"], measurements["hip"]
    )
    rfm_row = pd.DataFrame([dict(male=male, height_cm=height_cm, waist_cm=measurements["waist"])])
    return measurements, body_fat, float(rfm(rfm_row)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--front", required=True)
    ap.add_argument("--side", required=True)
    ap.add_argument("--gender", choices=["male", "female"], required=True)
    ap.add_argument("--age", type=float, required=True)
    ap.add_argument("--height", type=float, required=True, help="cm")
    ap.add_argument("--weight", type=float, required=True, help="kg")
    ap.add_argument("--raw", action="store_true",
                    help="--front/--side are normal photos; segment them into masks first")
    args = ap.parse_args()

    front, side = Path(args.front).resolve(), Path(args.side).resolve()
    for p in (front, side):
        if not p.exists():
            raise SystemExit(f"not found: {p}")

    if args.raw:
        out_dir = Path(__file__).resolve().parent.parent / "outputs" / "masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        for view, p in (("front", front), ("side", side)):
            mask, warnings = photo_to_mask(p, view)
            saved[view] = out_dir / f"{p.stem}_{view}_mask.png"
            mask.save(saved[view])
            for w in warnings:
                print(f"warning ({view}): {w}")
        print(f"segmentation masks saved in {out_dir} (check them by eye)\n")
        front, side = saved["front"], saved["side"]
    else:
        for p in (front, side):
            check_mask(p)

    m, bf, rfm_bf = estimate(front, side, args.gender, args.age, args.height, args.weight)
    print("Predicted measurements (cm):")
    for k, v in m.items():
        print(f"  {k:20s} {v:6.1f}")
    print(f"\nBody fat (trained model): {bf:.1f}%")
    print(f"Body fat (RFM formula):   {rfm_bf:.1f}%")


if __name__ == "__main__":
    main()
