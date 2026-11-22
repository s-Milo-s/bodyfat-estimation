"""Stage 2: measurements -> DEXA body fat %, trained on NHANES 2017-18. Run from the project root."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data" / "nhanes"
OUT = PROJECT / "models" / "bodyfat_nhanes.joblib"

# DEXA is only available up to age 59
MIN_AGE, MAX_AGE = 18, 59
INPUTS = ["male", "age", "height_cm", "weight_kg", "waist_cm", "hip_cm"]


def load_nhanes():
    bmx = pd.read_sas(DATA / "BMX_J.XPT")
    dxx = pd.read_sas(DATA / "DXX_J.XPT")
    demo = pd.read_sas(DATA / "DEMO_J.XPT")
    df = (
        demo[["SEQN", "RIAGENDR", "RIDAGEYR"]]
        .merge(bmx[["SEQN", "BMXHT", "BMXWT", "BMXWAIST", "BMXHIP"]], on="SEQN")
        .merge(dxx[["SEQN", "DXDTOPF"]], on="SEQN")
        .dropna()
    )
    df = df.rename(columns={
        "RIDAGEYR": "age", "BMXHT": "height_cm", "BMXWT": "weight_kg",
        "BMXWAIST": "waist_cm", "BMXHIP": "hip_cm", "DXDTOPF": "body_fat_pct",
    })
    df["male"] = (df.RIAGENDR == 1).astype(int)
    return df[(df.age >= MIN_AGE) & (df.age <= MAX_AGE)].reset_index(drop=True)


def add_features(df):
    out = df[INPUTS].copy()
    out["bmi"] = df.weight_kg / (df.height_cm / 100) ** 2
    out["waist_to_height"] = df.waist_cm / df.height_cm
    out["waist_to_hip"] = df.waist_cm / df.hip_cm
    return out


def rfm(df):
    return np.where(df.male == 1, 64, 76) - 20 * df.height_cm / df.waist_cm


def predict(bundle, male, age, height_cm, weight_kg, waist_cm, hip_cm):
    """Predict body fat % for one person (used by the end-to-end pipeline)."""
    row = pd.DataFrame([dict(male=male, age=age, height_cm=height_cm, weight_kg=weight_kg,
                             waist_cm=waist_cm, hip_cm=hip_cm)])
    return float(bundle["model"].predict(add_features(row))[0])


def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def main():
    df = load_nhanes()
    train, test = train_test_split(df, test_size=0.2, random_state=0, stratify=df.male)
    print(f"adults={len(df)} train={len(train)} test={len(test)}")

    X_train, y_train = add_features(train), train.body_fat_pct
    X_test, y_test = add_features(test), test.body_fat_pct

    candidates = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0
        ),
    }

    cv = {}
    for name, model in candidates.items():
        cv[name] = mae(y_train, cross_val_predict(model, X_train, y_train, cv=5))
    best_name = min(cv, key=cv.get)
    best = candidates[best_name].fit(X_train, y_train)

    sex_mean = train.groupby("male").body_fat_pct.mean()
    print(f"\n5-fold CV on train (MAE, points): " + ", ".join(f"{k}={v:.2f}" for k, v in cv.items()))
    print(f"selected: {best_name}\n")

    results = {
        "guess avg per sex": mae(y_test, test.male.map(sex_mean)),
        "RFM formula": mae(y_test, rfm(test)),
        "trained model": mae(y_test, best.predict(X_test)),
    }
    print("Held-out test MAE vs DEXA (percentage points, lower is better)")
    for k, v in results.items():
        print(f"  {k:18s} {v:.2f}")

    for label, sex in [("female", 0), ("male", 1)]:
        m = test.male == sex
        print(f"  {label}: RFM {mae(y_test[m], rfm(test[m])):.2f} | "
              f"model {mae(y_test[m], best.predict(X_test[m])):.2f}")

    OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"model": best, "inputs": INPUTS, "name": best_name}, OUT)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
