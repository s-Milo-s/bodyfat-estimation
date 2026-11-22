"""Train silhouettes -> body measurements. Run from the project root."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from bodym_model import ROOT, BodyMDataset, BodyNet, get_device
from bodym_prepare_data import TARGETS

CKPT = ROOT.parent.parent / "models" / "bodynet.pt"


def run_eval(model, loader, std, device):
    model.eval()
    err = 0.0
    n = 0
    with torch.no_grad():
        for front, side, extra, y in loader:
            pred = model(front.to(device), side.to(device), extra.to(device)).cpu()
            err += ((pred - y).abs() * std).sum(0)
            n += len(y)
    return err / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--limit", type=int, default=0, help="use only N train rows (smoke test)")
    args = ap.parse_args()

    proc = ROOT / "processed"
    train_df = pd.read_csv(proc / "train.csv")
    if args.limit:
        train_df = train_df.sample(args.limit, random_state=0)
        train_df.to_csv(proc / "_smoke.csv", index=False)
    mean = train_df[TARGETS].values.astype("float32").mean(0)
    std = train_df[TARGETS].values.astype("float32").std(0)

    train_csv = proc / ("_smoke.csv" if args.limit else "train.csv")
    train_ds = BodyMDataset(train_csv, TARGETS, mean, std, augment=True)
    val_ds = BodyMDataset(proc / "val.csv", TARGETS, mean, std)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64)

    device = get_device()
    print(f"device={device} train={len(train_ds)} val={len(val_ds)}")
    model = BodyNet(len(TARGETS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl)
    )
    loss_fn = torch.nn.SmoothL1Loss()
    std_t = torch.from_numpy(std)

    CKPT.parent.mkdir(exist_ok=True)
    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for front, side, extra, y in train_dl:
            front, side, extra, y = (t.to(device) for t in (front, side, extra, y))
            loss = loss_fn(model(front, side, extra), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(y)
        val_mae = run_eval(model, val_dl, std_t, device)
        waist = val_mae[TARGETS.index("waist")].item()
        print(f"epoch {epoch:2d} train_loss={total / len(train_ds):.4f} "
              f"val_mae_mean={val_mae.mean():.2f}cm waist={waist:.2f}cm", flush=True)
        if val_mae.mean().item() < best:
            best = val_mae.mean().item()
            torch.save({"model": model.state_dict(), "mean": mean, "std": std,
                        "targets": TARGETS}, CKPT)
    print(f"best val mean MAE {best:.2f}cm -> {CKPT}")


if __name__ == "__main__":
    main()
