# Dataset Curation and YAMNet Training

## Overview

This document covers the dataset half of the Chatak ML pipeline. **The simulator repo
handles dataset creation only.** Training lives in
[anamtya-tech/yamnet](https://github.com/anamtya-tech/yamnet).

| Stage | Where | Key file |
|-------|-------|----------|
| 1. Run simulations | simulator | `simulator.py` |
| 2. Analyse + curate WAV datasets | simulator | `yamnet_dataset_curator.py` |
| 3. Fine-tune YAMNet | yamnet repo | `training/train_yamnet.py` |
| 4. Export to TFLite | yamnet repo | `training/export_finetuned.py` |
| 5. Deploy to ODAS | chatak-odas | swap `.tflite` + `custom_class_map.csv` |

---

## Architecture

### Components

#### 1. YAMNet Dataset Curator (`yamnet_dataset_curator.py`)

Extracts labelled WAV files from ODAS analysis results for YAMNet fine-tuning:

- **Spatial + temporal alignment gate**: only saves samples that are both
  direction-matched (angular error ≤ threshold) *and* temporally matched to a
  ground-truth scene source.  Unmatched samples go to an `unknown` dataset for
  optional manual review.
- **Issue-based filtering**: within clean matches, saves only samples that have a
  training-worthy issue — unclassified, low confidence, or label mismatch — so
  correctly-classified events are skipped.
- **Track-level WAV stitching**: groups all `.bin` sidecars for the same
  `track_id` and reconstructs a single longer WAV (~0.76 s per bin), improving
  YAMNet input quality.
- **TensorFlow-compatible output**: WAV files at 16 kHz mono + `labels.csv`
  (`filename`, `label`, `fold`) ready for `training/data_loader.py`.

#### 2. Audio Reconstructor (`audio_reconstructor.py`)

Converts ODAS `.bin` sidecar files (96 × 257 float32 magnitude spectra) back to
audio waveforms via inverse STFT (Griffin-Lim when phase is unavailable).

#### 3. Dataset Configurator UI (`configurator.py` — `DatasetConfigurator` class)

Streamlit panel (**🎯 YAMNet Datasets**) for:
- Creating / switching active dataset
- Adjusting curation criteria (confidence threshold, angle threshold,
  include-mismatches flags)
- Triggering curation from the most recent analysis run
- Viewing per-label sample counts and the master `labels.csv`

---

## Workflow

### Step 1 — Build a dataset (simulator)

```
Scene Configurator → Audio Renderer → ODAS Simulator → Results Analyzer
                                                                ↓
                                                   YAMNet Datasets panel
                                                   (yamnet_dataset_curator)
                                                                ↓
                             outputs/yamnet_datasets/yamnet_train_001/
                               audio/<wav files>
                               labels.csv
```

1. Run a simulation (Scene → Render → ODAS Simulator).
2. Analyse the run in **📊 Results Analyzer**.
3. Open **🎯 YAMNet Datasets** and click **Curate from last analysis**.
4. Repeat for multiple scenes; the curator appends to `labels.csv` deduplicating
   by filename.
5. Aim for ≥ 100 samples per class before training.

### Step 2 — Fine-tune YAMNet (yamnet repo)

Once you have a dataset, switch to the **yamnet** repo:

```bash
cd ~/yamnet && source tfenv/bin/activate

# First time: export the base SavedModel
python integration/export_yamnet_core.py
cp -r integration/yamnet_core model_store/base/yamnet_core_savedmodel

# Fine-tune
python training/train_yamnet.py \
    --dataset ~/simulator/outputs/yamnet_datasets/yamnet_train_001 \
    --savedmodel model_store/base/yamnet_core_savedmodel \
    --phase1-epochs 20 \
    --phase2-epochs 30
```

See [anamtya-tech/yamnet — training/](https://github.com/anamtya-tech/yamnet/tree/main/training)
for full documentation.

### Step 3 — Export and deploy (yamnet repo)

```bash
python training/export_finetuned.py \
    --checkpoint model_store/checkpoints/chatak_yamnet_<ts> \
    --version v1.0.0

cp model_store/releases/v1.0.0/chatak_yamnet_v1.0.0.tflite ~/sodas/
cp model_store/releases/v1.0.0/custom_class_map.csv        ~/sodas/
# Update raw.model_path and raw.class_map_path in your ODAS .cfg
```

---

## Dataset Organisation

```
simulator/outputs/
└── yamnet_datasets/
    ├── curator_config.json            # Active dataset + curation criteria
    ├── yamnet_train_001/              # Primary fine-tuning dataset
    │   ├── audio/                     # Reconstructed WAV files (16 kHz mono)
    │   │   └── <run_id>_<idx>_t<ts>_<N>bins_dir<az>deg_<label>.wav
    │   ├── spectrograms/              # Mel-spectrogram PNG visualisations
    │   ├── metadata/                  # Per-run detail CSVs
    │   └── labels.csv                 # Master label file (all runs, appended)
    └── yamnet_unknown_001/            # Unmatched samples for manual review
        └── (same structure)
```

### `labels.csv` Schema

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | WAV filename (unique key) |
| `run_id` | str | Simulation run identifier |
| `timestamp` | float | Detection time (ODAS `timeStamp × 0.008` s) |
| `label` | str | Ground truth class (from scene config) |
| `yamnet_class` | str | ODAS top-1 event class name at curation time |
| `yamnet_confidence` | float | Peak single-hop confidence |
| `yamnet_votes` | int | Hops that agreed on the winning class (0–6) |
| `yamnet_ambiguous` | bool | True when #2 candidate ties #1 on hop votes |
| `top_k_candidates` | str | Top-5: `Name(Nv,conf)\|...` pipe-separated |
| `ground_truth` | str | GT label (same as `label` for clean matches) |
| `curation_reason` | str | Tags: `unclassified`, `low_confidence_0.xx`, `mismatch_yamnet:X_gt:Y`, `ambiguous_topk` |
| `activity` | float | ODAS track activity score |
| `n_stitched_bins` | int | Number of `.bin` sidecars stitched |
| `stitched_duration_s` | float | Reconstructed audio duration (s) |
| `position` | dict | ODAS 3D position `{x,y,z}` |
| `angular_error` | float | Degrees between ODAS direction and GT source |
| `dataset_type` | str | `training` or `unknown` |
| `clean_match` | bool | True when GT-matched and routed to training |
| `manual_verification_needed` | bool | Flagged for human review |
| `fold` | str | Train/val/test split (default `train`; reassigned by `train_yamnet.py` if not set) |

### Audio WAV naming

```
{run_id}_{idx:04d}_t{timestamp}_{N}bins_dir{az:.0f}deg_{label}.wav
```

Example: `run_20260301_111706_0000_t3_520_2bins_dir9deg_Wolfhowl.wav`

- `2bins` → 2 `.bin` sidecars stitched → ~1.53 s audio
- `dir9deg` → 9° angular error from GT source

---

## Curation Settings

### Confidence threshold

The **YAMNet confidence threshold** slider (default **0.75**) controls which
samples are sent to the training dataset:

- **Save when `yamnet_conf < threshold`** — captures events the base model
  doesn't know; fine-tuning needed.
- **Skip when `yamnet_conf ≥ threshold`** — base model already handles this;
  no training benefit.

Persists to `curator_config.json`.

### Angular match threshold

Only samples with `angular_error ≤ direction_threshold_deg` (default **15°**)
are considered spatially aligned and eligible for the training dataset.

### Track-based WAV stitching

| Track bins | Audio duration | Notes |
|-----------|---------------|-------|
| 1 | ~0.76 s | Minimum — one evaluation window |
| 2 | ~1.53 s | |
| 3 | ~2.30 s | |
| 6 | ~4.60 s | Common for 5-second clips |

Stitching yields longer audio, improving YAMNet fine-tuning quality
(YAMNet's ideal minimum is ~0.96 s).

---

---

## Spectrogram Visualisation (March 13, 2026)

All spectrogram displays — both the static PNG (saved by `yamnet_dataset_curator.py`) and the interactive Plotly heatmap (in `dataset_visualizer.py`) — were updated from linear magnitude to **dB scale** with the **Turbo** colormap.

### Why the Change
The ODAS beamformer outputs linear magnitude spectra. In a typical frame the dynamic range spans 60–80 dB. Displaying raw linear values compresses all meaningful structure into the top 1% of the colour scale — the image appears almost entirely black.

### What Changed

| Property | Before | After |
|----------|--------|-------|
| Scale | Linear magnitude | dB: `20·log₁₀(max(v, 1e-6))` |
| Colormap | `magma` / `Viridis` | `Turbo` |
| Clip range | Full min/max | 5th–99th percentile |
| X axis | Frame index | Time in seconds (`frame × 128/16000`) |
| Y axis | Bin index 0–256 | Frequency in Hz (0–8000) |
| Background | White | Dark (`#0e1117`) — matches Streamlit dark theme |
| Color bar | `Magnitude` | `dB` |

### Regenerating Existing PNGs
To re-render the PNG spectrograms for an existing dataset without re-running a full simulation:

```python
import sys, types, importlib, numpy as np, matplotlib
matplotlib.use('Agg')
from pathlib import Path
sys.path.insert(0, '/home/azureuser/simulator')
mod = importlib.import_module('yamnet_dataset_curator')
Curator = mod.YAMNetDatasetCurator
curator = Curator.__new__(Curator)
curator._save_spectrogram_plot = types.MethodType(Curator._save_spectrogram_plot, curator)

for bin_file in Path('outputs/yamnet_datasets/yamnet_train_001/bins').glob('*.bin'):
    raw = np.fromfile(bin_file, dtype=np.float32)
    frames = raw.size // 257
    data = raw[:frames*257].reshape(frames, 257)
    label = bin_file.stem.rsplit('_', 1)[-1]
    spec_path = bin_file.parent.parent / 'spectrograms' / (bin_file.stem + '.png')
    curator._save_spectrogram_plot(data, spec_path, label)
```

---

## Troubleshooting

**No samples curated after analysis**
- Check angular threshold — tighten or loosen in Dataset Settings.
- Verify `.bin` sidecars exist: `ls ~/sodas/ClassifierLogs/` — should contain `*.bin` files alongside the JSON logs.
- Lower the confidence threshold if base YAMNet is already performing well (saves fewer samples).

**Labels.csv has many `unknown` label rows**
- The scene config may not have sources covering all detected directions.
- Check ground truth JSON paths in the run manifest.

**Very short WAV files (< 0.5 s)**
- Track fired only once — single `.bin`. Still usable but suboptimal.
- Run longer simulations (≥ 10 s per source) for more bins per track.

---

## Related Repos

| Repo | Role |
|------|------|
| [anamtya-tech/simulator](https://github.com/anamtya-tech/simulator) | **This repo** — dataset curation |
| [anamtya-tech/yamnet](https://github.com/anamtya-tech/yamnet) | YAMNet fine-tuning + TFLite export |
| [anamtya-tech/chatak-odas](https://github.com/anamtya-tech/chatak-odas) | ODAS C fork — runtime inference |

