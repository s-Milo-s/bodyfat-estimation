# Body Fat Estimation from Photos

Estimate body fat percentage from phone photos using synthetic training data.

## Pipeline Overview

1. **Synthetic Data Generation** - Render bodies using Anny + Blender/MPFB2
2. **Measurement Extraction** - Compute circumferences from 3D mesh
3. **NHANES Bridge** - Map measurements → body fat using CDC data
4. **Train Model** - Photo → measurements → body fat

## Setup

### Prerequisites

1. **Blender 4.x** (with Python 3.11)
   ```bash
   # macOS
   brew install --cask blender
   ```

2. **MPFB2** (MakeHuman Plugin for Blender)
   - Download from: https://static.makehumancommunity.org/mpfb/releases/
   - Install as Blender addon: Edit → Preferences → Add-ons → Install

3. **Python environment** (for preprocessing & training)
   ```bash
   cd bodyfat-estimation
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

### Project Structure

```
bodyfat-estimation/
├── scripts/
│   ├── generate_body.py      # Blender script - generate & render one body
│   ├── batch_generate.py     # Run generation in batch
│   ├── extract_measurements.py
│   └── preprocess.py         # Grayscale + edge enhancement
├── data/
│   ├── renders/              # Raw rendered images
│   └── processed/            # Preprocessed images ready for training
├── models/                   # Trained model checkpoints
└── notebooks/                # Exploration & analysis
```

## Quick Start

### 1. Test Blender setup
```bash
blender --background --python scripts/test_blender.py
```

### 2. Generate a single test body
```bash
blender --background --python scripts/generate_body.py -- --output data/renders/test
```

### 3. Preprocess images
```bash
python scripts/preprocess.py --input data/renders --output data/processed
```

## Image Specs

- **Dimensions:** 512x682 (3:4 portrait, matches iPhone selfie)
- **Views:** Front (A-pose) + Side (90°)
- **Format:** PNG with alpha channel (raw), grayscale PNG (processed)

## Licenses

- **Anny body model:** Apache 2.0
- **MPFB2 code:** GPLv3
- **MPFB2 assets:** CC0
- **NHANES data:** Public domain
