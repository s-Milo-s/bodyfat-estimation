"""Dataset and model for silhouettes (front + side) -> body measurements."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.models import ResNet18_Weights, resnet18

ROOT = Path(__file__).resolve().parent.parent / "data" / "bodym"
SIZE = (256, 192)  # (height, width); masks are 960x720


def mask_to_tensor(img):
    arr = np.array(img.convert("L").resize((SIZE[1], SIZE[0]), Image.BILINEAR))
    return torch.from_numpy(arr).unsqueeze(0)  # 1,H,W uint8


def load_masks(paths):
    return torch.stack([mask_to_tensor(Image.open(ROOT / p)) for p in paths])  # N,1,H,W


def extra_features(df):
    height = (df.height_cm.values - 170.0) / 10.0
    return torch.tensor(np.stack([df.is_male.values, height], 1), dtype=torch.float32)


class BodyMDataset(Dataset):
    def __init__(self, csv, targets, mean, std, augment=False):
        df = pd.read_csv(csv)
        self.df = df
        self.front = load_masks(df.front)
        self.side = load_masks(df.side)
        self.extra = extra_features(df)
        y = df[targets].values.astype(np.float32)
        self.y = torch.from_numpy((y - mean) / std)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def _prep(self, img):
        img = img.float() / 255.0
        if self.augment:
            img = TF.affine(
                img,
                angle=float(np.random.uniform(-3, 3)),
                translate=[int(np.random.randint(-8, 9)), int(np.random.randint(-8, 9))],
                scale=1.0,
                shear=[0.0, 0.0],
            )
        return img

    def __getitem__(self, i):
        return self._prep(self.front[i]), self._prep(self.side[i]), self.extra[i], self.y[i]


class BodyNet(nn.Module):
    """Shared ResNet18 backbone over both views, concatenated with gender + height."""

    def __init__(self, n_out, pretrained=True):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512 * 2 + 2, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, n_out)
        )

    @staticmethod
    def _norm(x):
        return (x.repeat(1, 3, 1, 1) - 0.45) / 0.225

    def forward(self, front, side, extra):
        b = front.shape[0]
        f = self.backbone(self._norm(torch.cat([front, side])))
        return self.head(torch.cat([f[:b], f[b:], extra], 1))


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
