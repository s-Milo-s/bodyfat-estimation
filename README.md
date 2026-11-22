# Body Fat Estimation from Photos (POC)

Estimate body fat % from two phone photos (front + side) plus gender, age, height and weight.

**Non-commercial proof of concept.** Stage 1 is trained on the BodyM dataset (CC-BY-NC 4.0), so the
trained weights are non-commercial too.

## How it works

```
front photo ─┐                                                    ┌─ gender, age
side photo  ─┴─> segment.py ─> silhouettes ─> Stage 1 (ResNet18) ─┤  height, weight
                 (DeepLabV3)    (720x960)      13 body measurements │
                                               (waist, hip, ...)    ▼
                                                              Stage 2 (NHANES model)
                                                                    │
                                                                    ▼
                                                              body fat %
```

| Stage | Trained on | Input | Output |
|-------|-----------|-------|--------|
| Segmentation | pretrained (torchvision DeepLabV3, no training here) | phone photo | BodyM-style silhouette |
| 1. Silhouettes -> measurements | BodyM (2,018 people, 6,134 captures) | front + side mask, gender, height | 13 measurements in cm |
| 2. Measurements -> body fat | NHANES 2017-18 with DEXA scans (2,424 adults, 18-59) | gender, age, height, weight, waist, hip | DEXA-style body fat % |

Stage 2 is trained on real DEXA body fat, with the RFM formula (`64 - 20*height/waist` male,
`76 - 20*height/waist` female) as the baseline to beat. Training on formula outputs would cap it at the formula.

## Results so far

Stage 1, mean absolute error in cm (held-out data; baseline uses gender + height only):

| | val | testA | testB | testB baseline |
|---|---|---|---|---|
| waist | 2.55 | 2.13 | 3.55 | 10.50 |
| hip | 2.04 | 2.10 | 2.85 | 8.02 |
| mean of 13 | 1.16 | 1.14 | 1.57 | 3.70 |

Stage 2, mean absolute error vs DEXA body fat (485 held-out adults):

| Method | Error (points) |
|---|---|
| Guess the average per sex | 5.38 |
| RFM formula | 3.12 |
| Trained model | 2.57 |

**Not yet validated:** the full photo -> body fat path on real phone photos. No dataset has both photos and
body fat, so end-to-end accuracy can only be checked against tape measurements. The two stages' errors also add up.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Apple Silicon uses the MPS GPU automatically; otherwise it falls back to CPU.

### Data (not in the repo)

BodyM (needs the AWS CLI, no account required):
```bash
mkdir -p data/bodym
for s in train testA testB; do
  aws s3 sync --no-sign-request s3://amazon-bodym/$s/ data/bodym/$s/
done
```

NHANES 2017-18:
```bash
mkdir -p data/nhanes && cd data/nhanes
for f in BMX_J DXX_J DEMO_J; do
  curl -O https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/$f.XPT
done
```

## Run

All commands run from the project root.

```bash
# Stage 1: prepare data (split by subject), train (~20 min on Apple GPU), evaluate
venv/bin/python scripts/bodym_prepare_data.py
venv/bin/python scripts/bodym_train.py --epochs 15
venv/bin/python scripts/bodym_evaluate.py

# Stage 2 (seconds)
venv/bin/python scripts/nhanes_train.py

# End to end from normal photos
venv/bin/python scripts/estimate.py --raw \
  --front front.jpg --side side.jpg \
  --gender male --age 30 --height 178 --weight 80
```

`--raw` saves the generated silhouettes to `outputs/masks/`. Look at them before trusting a result.
Without `--raw`, `--front`/`--side` must already be BodyM-style masks (white person on black, 3:4 portrait).

## Project structure

```
scripts/
  bodym_prepare_data.py   merge BodyM CSVs, split train/val by subject
  bodym_model.py          dataset + BodyNet (shared ResNet18 over both views)
  bodym_train.py          stage 1 training
  bodym_evaluate.py       stage 1 evaluation vs gender+height baseline
  nhanes_train.py         stage 2 training, compared against the RFM formula
  segment.py              phone photo -> BodyM-style silhouette
  estimate.py             end-to-end inference
  generate_body.py, batch_generate.py, extract_measurements.py,
  preprocess.py, test_blender.py
                          early synthetic-data experiment (Blender + MPFB2); abandoned,
                          kept for reference only
docs/project-notes.md     decisions, findings and caveats
logs/                     training and evaluation output
data/, models/, outputs/  not in git
```

## Photo guidance

Full body in frame with feet visible, one person, tight clothing, arms slightly away from the body in
the front photo, standing straight, camera at about waist/chest height, plain background if possible.
Baggy clothing makes the model overestimate measurements.

## Known limitations

- Trained on clean research silhouettes; real phone photos are untested.
- Waist/hip error is about 2-3.5 cm, which adds about 1 point of body fat error on top of stage 2.
- NHANES waist is tape-measured, BodyM waist comes from 3D scans; they may differ slightly in definition.
- NHANES DEXA covers ages 8-59; stage 2 is trained on 18-59 only.
- Neck is not available in BodyM or NHANES 2017-18, so the Navy formula is not used.

## Data licenses

- **BodyM:** CC-BY-NC 4.0, non-commercial only
- **NHANES:** public domain
- **torchvision DeepLabV3 weights:** see torchvision's model license terms before any commercial use
