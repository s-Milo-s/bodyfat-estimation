"""Build per-photo CSVs for BodyM: train/val (split by subject) plus testA/testB."""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent / "data" / "bodym"
OUT = ROOT / "processed"

# height is an input (user-provided), so it is not a prediction target
TARGETS = [
    "ankle", "arm-length", "bicep", "calf", "chest", "forearm", "hip",
    "leg-length", "shoulder-breadth", "shoulder-to-crotch", "thigh", "waist", "wrist",
]


def load_split(split):
    d = ROOT / split
    meta = pd.read_csv(d / "hwg_metadata.csv")
    meas = pd.read_csv(d / "measurements.csv").drop(columns=["height"])
    photos = pd.read_csv(d / "subject_to_photo_map.csv")

    df = photos.merge(meta, on="subject_id", how="inner").merge(meas, on="subject_id", how="inner")
    assert len(df) == len(photos), f"{split}: lost rows in merge ({len(df)} of {len(photos)})"

    df["front"] = split + "/mask/" + df.photo_id + ".png"
    df["side"] = split + "/mask_left/" + df.photo_id + ".png"
    df["is_male"] = (df.gender == "male").astype(int)

    missing = [p for p in pd.concat([df.front, df.side]) if not (ROOT / p).exists()]
    assert not missing, f"{split}: {len(missing)} missing images, e.g. {missing[0]}"
    return df


def main():
    OUT.mkdir(exist_ok=True)

    train_all = load_split("train")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    tr_idx, va_idx = next(splitter.split(train_all, groups=train_all.subject_id))
    sets = {
        "train": train_all.iloc[tr_idx],
        "val": train_all.iloc[va_idx],
        "testA": load_split("testA"),
        "testB": load_split("testB"),
    }

    overlap = set(sets["train"].subject_id) & set(sets["val"].subject_id)
    assert not overlap, "subject leakage between train and val"

    for name, df in sets.items():
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(f"{name:6s} photos={len(df):5d} subjects={df.subject_id.nunique():4d} "
              f"male={df.is_male.mean():.0%} height={df.height_cm.mean():.1f}cm "
              f"waist={df.waist.mean():.1f}cm")


if __name__ == "__main__":
    main()
