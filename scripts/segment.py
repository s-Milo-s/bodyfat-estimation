"""Phone photo -> BodyM-style silhouette mask (720x960, white person on black).

  venv/bin/python scripts/segment.py --photo me_front.jpg --view front --out front_mask.png
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy import ndimage
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

CANVAS_W, CANVAS_H = 720, 960
PERSON_CLASS = 15  # Pascal VOC label used by the torchvision weights
WORK_LONG_SIDE = 1280

# Median framing of BodyM masks: (person bbox height px, top margin px, bbox center x px)
FRAMING = {"front": (836, 65, 365), "side": (861, 57, 360)}

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT).eval()
    return _model


def prepare_image(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    scale = WORK_LONG_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    return img


def person_probability(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0) / 255
    s = 520 / min(img.size)
    x = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
    with torch.no_grad():
        logits = _get_model()((x - mean) / std)["out"]
    prob = logits.softmax(1)[:, PERSON_CLASS:PERSON_CLASS + 1]
    prob = F.interpolate(prob, size=(img.height, img.width), mode="bilinear", align_corners=False)
    return prob[0, 0].numpy()


def clean_mask(prob):
    """Threshold, keep the largest person blob, drop specks and tiny holes."""
    mask = prob > 0.5
    labels, n = ndimage.label(mask)
    warnings = []
    if n == 0:
        return mask, ["no person found"]
    sizes = ndimage.sum(mask, labels, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    if len(order) > 1 and sizes[order[1]] > 0.3 * sizes[order[0]]:
        warnings.append("more than one person in the photo; using the largest")
    mask = labels == (order[0] + 1)
    holes, nh = ndimage.label(~mask)
    if nh:
        hole_sizes = ndimage.sum(~mask, holes, range(1, nh + 1))
        for i, s in enumerate(hole_sizes, 1):
            if s < 0.002 * mask.size:
                mask[holes == i] = True
    return mask, warnings


def is_cut_off(mask):
    ys = np.where(mask.any(1))[0]
    return ys.min() <= 1 or ys.max() >= mask.shape[0] - 2


def check_quality(mask):
    ys, xs = np.where(mask)
    warnings = []
    h, w = mask.shape
    if mask.mean() < 0.05:
        warnings.append("person is small in the photo; stand closer or crop")
    if xs.min() <= 1 or xs.max() >= w - 2:
        warnings.append("body touches the side edge; arms or body may be cut off")
    return warnings


def normalize_mask(mask, view):
    """Crop to the person and place on a 720x960 canvas using BodyM's median framing."""
    target_h, top, cx = FRAMING[view]
    ys, xs = np.where(mask)
    crop = Image.fromarray((mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 255).astype(np.uint8))
    scale = min(target_h / crop.height, (CANVAS_W - 20) / crop.width)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.LANCZOS)
    canvas = Image.new("L", (CANVAS_W, CANVAS_H), 0)
    canvas.paste(crop, (round(cx - crop.width / 2), top))
    return canvas.point(lambda p: 255 if p > 127 else 0)


def image_to_mask(img, view):
    """PIL photo -> (BodyM-style PIL mask, warnings). Raises ValueError if no person is found."""
    mask, warnings = clean_mask(person_probability(prepare_image(img)))
    if not mask.any():
        raise ValueError(f"no person found in the {view} photo")
    if is_cut_off(mask):
        # a cropped body gets stretched to full height by normalize_mask, which inflates every width
        raise ValueError(
            f"Your head or feet are cut off in the {view} photo. Step back so your whole body, "
            "head to toes, fits in the frame."
        )
    warnings += check_quality(mask)
    return normalize_mask(mask, view), warnings


def photo_to_mask(path, view):
    return image_to_mask(Image.open(path), view)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--photo", required=True)
    ap.add_argument("--view", choices=list(FRAMING), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    mask, warnings = photo_to_mask(args.photo, args.view)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mask.save(args.out)
    for w in warnings:
        print("warning:", w)
    print("saved", args.out)


if __name__ == "__main__":
    main()
