"""Evaluate the trained model against a gender+height-only baseline. Run from the project root."""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader

from bodym_model import ROOT, BodyMDataset, BodyNet, extra_features, get_device
from bodym_train import CKPT


def rfm(is_male, height_cm, waist_cm):
    return np.where(is_male == 1, 64, 76) - 20 * height_cm / waist_cm


def predict(model, ds, device):
    model.eval()
    out = []
    with torch.no_grad():
        for front, side, extra, _ in DataLoader(ds, batch_size=64):
            out.append(model(front.to(device), side.to(device), extra.to(device)).cpu())
    return torch.cat(out).numpy()


def main():
    ck = torch.load(CKPT, weights_only=False)
    targets, mean, std = ck["targets"], ck["mean"], ck["std"]
    device = get_device()
    model = BodyNet(len(targets), pretrained=False).to(device)
    model.load_state_dict(ck["model"])

    proc = ROOT / "processed"
    train_df = pd.read_csv(proc / "train.csv")
    baseline = Ridge(alpha=1.0).fit(extra_features(train_df).numpy(), train_df[targets].values)

    rows, rfm_rows = {}, {}
    for name in ["val", "testA", "testB"]:
        ds = BodyMDataset(proc / f"{name}.csv", targets, mean, std)
        df = ds.df
        truth = df[targets].values
        pred = predict(model, ds, device) * std + mean
        base = baseline.predict(extra_features(df).numpy())
        rows[f"{name} model"] = np.abs(pred - truth).mean(0)
        rows[f"{name} baseline"] = np.abs(base - truth).mean(0)

        w = targets.index("waist")
        true_rfm = rfm(df.is_male.values, df.height_cm.values, truth[:, w])
        rfm_rows[name] = (
            np.abs(rfm(df.is_male.values, df.height_cm.values, pred[:, w]) - true_rfm).mean(),
            np.abs(rfm(df.is_male.values, df.height_cm.values, base[:, w]) - true_rfm).mean(),
        )

    table = pd.DataFrame(rows, index=targets).round(2)
    table.loc["MEAN"] = table.mean().round(2)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("MAE in cm (lower is better; baseline uses gender+height only)\n")
    print(table)
    print("\nRFM error vs RFM from true waist (percentage points)")
    for name, (m, b) in rfm_rows.items():
        print(f"  {name:6s} model={m:.2f}  baseline={b:.2f}")


if __name__ == "__main__":
    main()
