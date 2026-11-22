# Body Fat Estimation from Photos

**Project Path:** `~/development/bodyfat-estimation`  
**Status:** Research/Prototyping  
**Last Updated:** 2026-11-21

## Current Plan (POC, non-commercial, updated 2026-11-21)

Two stages. Only stage 1 is being built now.

1. **Stage 1: silhouettes -> body measurements** (BodyM, CC-BY-NC, POC only)
   - Inputs: front mask + side mask + gender + height. (BodyM has no age, so age is not a stage 1 input.)
   - Outputs: 13 measurements (all BodyM measurements except height, which is an input).
   - Model: shared pretrained ResNet18 over both views + gender/height -> MLP.
2. **Stage 2: measurements -> body fat** (later)
   - For now: Relative Fat Mass, `64 - 20 * (height / waist)` male, `76 - 20 * (height / waist)` female (cm, same units).
   - Later: train on NHANES (DEXA `DXDTOPF` = real total % fat) using measurements that exist in both datasets.
   - Navy method needs neck; BodyM has none and NHANES 2017-18 has none either. Would need a different NHANES cycle that includes neck.

### Stage 1 results (ResNet18, 15 epochs, ~20 min on MPS; log in `logs/eval_*.txt`)
Mean absolute error in cm (baseline = ridge on gender+height only):

| Measurement | val | testA | testB | testB baseline |
|-------------|-----|-------|-------|----------------|
| waist | 2.55 | 2.13 | 3.55 | 10.50 |
| hip | 2.04 | 2.10 | 2.85 | 8.02 |
| chest | 2.10 | 2.05 | 3.01 | 9.27 |
| mean of 13 | 1.16 | 1.14 | 1.57 | 3.70 |

RFM computed from predicted waist differs from RFM from true waist by ~1.0-1.5 points (baseline 3.7-4.6).
This only measures waist-error propagation, not accuracy against real body fat. RFM itself is an approximation of DEXA.

### Stage 2 results (`scripts/nhanes_train.py` -> `models/bodyfat_nhanes.joblib`)
Decision: train on real DEXA body fat (`DXDTOPF`), NOT on formula outputs (a model trained on RFM labels can never beat RFM). RFM is the baseline to beat.
Inputs: gender, age, height, weight, waist, hip (the app asks for weight, age, height, gender; photos give waist/hip).
Data: 2,424 NHANES adults aged 18-59, 80/20 split (485 held-out test).

| Method | Test MAE vs DEXA (points) |
|--------|---------------------------|
| Guess average per sex | 5.38 |
| RFM formula | 3.12 (female 3.61, male 2.59) |
| Trained (gradient boosting) | 2.57 (female 2.65, male 2.50) |

Ridge was nearly as good in CV (2.70 vs 2.64), so the gain comes mostly from adding weight/age/hip, not model complexity.
Goal later: lower error. 1 point is probably not reachable with tape measurements alone.

### Findings that matter
- `train_full.csv` `body_fat_pct` is NOT ground truth. It correlates 0.93 with a BMI formula. Do not train on it.
- BodyM has no body-fat truth at all. NHANES (DEXA) is the only real body-fat label we have.
- Each BodyM subject has 1-27 captures. Always split by `subject_id`, never by image.
- BodyM S3 layout: `train|testA|testB / mask` (front), `mask_left` (side), plus `hwg_metadata.csv`, `measurements.csv`, `subject_to_photo_map.csv`.
- testA = 87 subjects / 1,684 photos, testB = 400 subjects / 1,160 photos. Report them separately.
- NHANES 2017-18 (`BMX_J`) overlap with BodyM: height, weight, waist, hip (direct); `BMXARMC` ~ bicep (approx).
  `BMXARML`/`BMXLEG` are upper-arm/upper-leg lengths, not comparable. No neck or thigh.

### Pipeline files (`~/development/bodyfat-estimation/scripts/`)
| File | Purpose |
|------|---------|
| `bodym_prepare_data.py` | Merge CSVs, split train/val by subject, write `data/bodym/processed/*.csv` |
| `bodym_model.py` | Dataset (masks cached in memory) + `BodyNet` |
| `bodym_train.py` | Train (`--epochs`, `--limit` for smoke test), saves `models/bodynet.pt`, logs in `logs/` |
| `bodym_evaluate.py` | MAE in cm per measurement vs gender+height-only baseline; RFM error |
| `nhanes_train.py` | Stage 2: measurements + age/weight -> DEXA body fat, saves `models/bodyfat_nhanes.joblib` |
| `segment.py` | Phone photo -> BodyM-style mask (torchvision DeepLabV3 person class, largest person, re-framed to 720x960) |
| `estimate.py` | End-to-end MVP: `--front --side --gender --age --height --weight` -> measurements + body fat %. Add `--raw` to pass normal photos (masks saved to `outputs/masks/` for eyeballing) |

End-to-end example:
`venv/bin/python scripts/estimate.py --front f.png --side s.png --gender male --age 30 --height 178 --weight 80`
With real photos: `... estimate.py --raw --front f.jpg --side s.jpg ...`

### Segmentation step (built, NOT yet validated on real selfies)
- BodyM masks are pure 0/255, person bbox ~836px tall (front) / ~861px (side) on 720x960, centered. `segment.py` rescales any cutout to that framing.
- Re-framing test: messing up scale/position of BodyM masks then re-normalizing changed predicted waist/hip by ~0.6 cm only.
- Side masks in BodyM face both directions (~54% right by a toe heuristic), so orientation is not enforced.
- Warns on: no person, several people, small person, head/feet/sides touching the photo edge.
- Known failure modes: baggy clothing (BodyM subjects wore tight clothes) inflates measurements; occlusion cuts parts out; back views / reused images give nonsense (a baggy-fleece back view as both views predicted waist 107 cm for a 68 kg 168 cm woman).
- Photo guidance for the app: full body in frame (feet visible), plain background if possible, tight clothing, arms slightly away from the body (front), stand straight, camera at about waist/chest height, one person only.
- Next: test on real photos of the user and compare predicted waist/hip to a tape measure.

Run from project root with `venv/bin/python scripts/<file>.py`. Stack: PyTorch on Apple MPS.

---

## Goal

Build a model that estimates body fat percentage from phone photos (selfies).

## The Problem

Body fat % is hard to measure at home. BMI ≠ body fat (BMI is just weight/height²). The Navy formula uses tape measurements (waist, hip, neck, height) but requires manual measuring.

**Target:** Photo → Body Fat %

---

## Key Decisions Made

### 1. Navy Formula for Ground Truth
- Men: Uses waist - neck + height
- Women: Uses waist + hip - neck + height
- Validated against DEXA scans in literature

### 2. Synthetic Data Approach (ATTEMPTED - LIMITED)
Tried MPFB2/MakeHuman for generating varied body types.

**Problem:** MPFB2 morph targets are capped and don't produce realistic fat distribution:
- Max "weight" target doesn't create belly bulge
- Even values of 3.0 show minimal difference
- Based on artistic models, not real body scans

**Files created:**
- `scripts/generate_body.py` - Blender script for MPFB2 body generation
- `scripts/extract_measurements.py` - Measurement extraction from mesh
- `scripts/preprocess.py` - Image preprocessing (grayscale + edge)
- `data/renders/` - Test renders

### 3. Real Data Exploration

**BodyM Dataset (Amazon):**
- 2,018 real subjects with silhouettes + 14 measurements
- Front + side poses
- BMI range: 10.6 to 62.8 (excellent variety)
- **License: CC-BY-NC 4.0 (NON-COMMERCIAL ONLY)**
- Downloaded to: `data/bodym/`

**NHANES:**
- 3,658 people with DEXA body fat % + tape measurements
- Public domain
- No photos
- Downloaded to: `data/nhanes/`

---

## Licensing Problem

| Dataset/Tool | Body Variation | Commercial Use |
|--------------|---------------|----------------|
| SMPL/SMPL-X | Excellent | ❌ No |
| BodyM | Excellent | ❌ No |
| MPFB2 | Poor | ✅ Yes |
| NHANES | N/A (no photos) | ✅ Yes |

**Core issue:** Good body variation data comes from real scans → non-commercial licenses.

---

## Technical Stack

- **Blender 5.3** + MPFB2 extension (installed)
- **Python venv** at `bodyfat-estimation/venv/`
- **AWS CLI** installed for S3 access
- **Dependencies:** pandas, pyreadstat, numpy

---

## Data Files

```
bodyfat-estimation/
├── data/
│   ├── bodym/
│   │   ├── train_metadata.csv      # 2,018 subjects (gender, height, weight)
│   │   ├── train_measurements.csv  # 14 body measurements per subject
│   │   ├── train_photo_map.csv     # Maps subjects to photo IDs
│   │   ├── train_full.csv          # Merged with calculated body fat %
│   │   └── samples/                # ~20 sample silhouette images
│   ├── nhanes/
│   │   ├── BMX_J.XPT              # Body measurements (waist, hip)
│   │   ├── DXX_J.XPT              # DEXA body fat %
│   │   └── DEMO_J.XPT             # Demographics
│   ├── mpfb_assets/               # MPFB2 asset pack (clothes, hair, etc.)
│   └── renders/                   # Test renders from MPFB2
├── scripts/
│   ├── generate_body.py           # Blender body generation
│   ├── extract_measurements.py    # Mesh → measurements
│   └── preprocess.py              # Image preprocessing
└── venv/                          # Python virtual environment
```

---

## Options for Commercial Use

1. **Collect your own photos** - Partner with gym/fitness app, get consent
2. **License a commercial dataset** - Pay for rights
3. **Measurements-only product** - User inputs measurements → body fat (no photos)
4. **Hybrid** - Train on non-commercial, fine-tune on small owned dataset

---

## Next Steps (When Resumed)

1. **Decide commercial path** - What's the actual product?
2. **If going photo route:**
   - Need to collect/license photos with body measurements
   - Or find commercial-friendly body scan data
3. **If going measurements-only:**
   - Can use NHANES (public domain) directly
   - Build simple calculator with Navy formula
   - Add ML to improve on Navy formula using NHANES DEXA data

---

## Code Snippets

### Navy Body Fat Formula (Python)
```python
import math

def navy_body_fat(gender, waist_cm, hip_cm, neck_cm, height_cm):
    if gender == 'male':
        bf = 495 / (1.0324 - 0.19077 * math.log10(waist_cm - neck_cm) + 0.15456 * math.log10(height_cm)) - 450
    else:
        bf = 495 / (1.29579 - 0.35004 * math.log10(waist_cm + hip_cm - neck_cm) + 0.22100 * math.log10(height_cm)) - 450
    return max(5, min(60, bf))
```

### Run MPFB2 Body Generation
```bash
cd ~/development/bodyfat-estimation
blender --background --python scripts/generate_body.py -- --output data/renders/test --bmi 30 --sex male
```

### Access BodyM on AWS
```bash
aws s3 ls --no-sign-request s3://amazon-bodym/
aws s3 cp --no-sign-request s3://amazon-bodym/train/measurements.csv ./
```

---

## References

- BodyM Paper: "Human Body Measurement Estimation with Adversarial Augmentation"
- MPFB2 Docs: https://static.makehumancommunity.org/mpfb.html
- Navy Formula: Hodgdon & Beckett (1984)
- NHANES: https://wwwn.cdc.gov/nchs/nhanes/
