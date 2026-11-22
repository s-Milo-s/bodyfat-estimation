"""End-to-end: front + side photos/silhouettes + gender/age/height/weight -> measurements -> body fat %.

Run from the project root, e.g.
  venv/bin/python scripts/estimate.py --raw --front a.jpg --side b.jpg --gender male --age 30 --height 178 --weight 80
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
import torch
from PIL import Image

from bodym_model import BodyNet, extra_features, get_device, mask_to_tensor
from bodym_train import CKPT as STAGE1_PATH
from nhanes_train import OUT as STAGE2_PATH
from nhanes_train import predict as predict_body_fat
from nhanes_train import rfm
from segment import image_to_mask


class Estimator:
    """Loads both stages once; safe to reuse for many requests."""

    def __init__(self):
        ck = torch.load(STAGE1_PATH, weights_only=False, map_location="cpu")
        self.targets, self.mean, self.std = ck["targets"], ck["mean"], ck["std"]
        self.device = get_device()
        self.model = BodyNet(len(self.targets), pretrained=False).to(self.device)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.stage2 = joblib.load(STAGE2_PATH)

    def from_masks(self, front_mask, side_mask, gender, age, height_cm, weight_kg):
        male = int(gender == "male")
        meta = pd.DataFrame([{"is_male": male, "height_cm": height_cm}])
        with torch.no_grad():
            pred = self.model(
                (mask_to_tensor(front_mask).unsqueeze(0).float() / 255).to(self.device),
                (mask_to_tensor(side_mask).unsqueeze(0).float() / 255).to(self.device),
                extra_features(meta).to(self.device),
            ).cpu().numpy()[0]
        measurements = dict(zip(self.targets, (pred * self.std + self.mean).tolist()))

        body_fat = predict_body_fat(
            self.stage2, male, age, height_cm, weight_kg, measurements["waist"], measurements["hip"]
        )
        rfm_row = pd.DataFrame([dict(male=male, height_cm=height_cm, waist_cm=measurements["waist"])])
        return {"measurements": measurements, "body_fat": body_fat, "rfm": float(rfm(rfm_row)[0])}

    def from_photos(self, front_img, side_img, gender, age, height_cm, weight_kg):
        """PIL photos in, result dict out. Nothing is written to disk."""
        front_mask, w1 = image_to_mask(front_img, "front")
        side_mask, w2 = image_to_mask(side_img, "side")
        result = self.from_masks(front_mask, side_mask, gender, age, height_cm, weight_kg)
        result["warnings"] = [f"front: {w}" for w in w1] + [f"side: {w}" for w in w2]
        result["masks"] = (front_mask, side_mask)
        return result


def check_mask(path):
    w, h = Image.open(path).size
    if abs(w / h - 0.75) > 0.05:
        print(f"warning: {path} is {w}x{h}; BodyM masks are 720x960 (3:4), results may be off")


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

    est = Estimator()
    if args.raw:
        result = est.from_photos(
            Image.open(front), Image.open(side), args.gender, args.age, args.height, args.weight
        )
        out_dir = Path(__file__).resolve().parent.parent / "outputs" / "masks"
        out_dir.mkdir(parents=True, exist_ok=True)
        for view, p, mask in (("front", front, result["masks"][0]), ("side", side, result["masks"][1])):
            mask.save(out_dir / f"{p.stem}_{view}_mask.png")
        for w in result["warnings"]:
            print("warning:", w)
        print(f"segmentation masks saved in {out_dir} (check them by eye)\n")
    else:
        for p in (front, side):
            check_mask(p)
        result = est.from_masks(
            Image.open(front), Image.open(side), args.gender, args.age, args.height, args.weight
        )

    print("Predicted measurements (cm):")
    for k, v in result["measurements"].items():
        print(f"  {k:20s} {v:6.1f}")
    print(f"\nBody fat (trained model): {result['body_fat']:.1f}%")
    print(f"Body fat (RFM formula):   {result['rfm']:.1f}%")


if __name__ == "__main__":
    main()
