# Dataset Curation and Model Training System

## Overview

This system implements a complete machine learning pipeline for audio source classification using ODAS (Open embeddeD Audition System) output data. The pipeline includes:

1. **Dataset Curation**: Automatically save high-confidence matched spectrograms with labels
2. **Lightweight CNN Model**: EfficientNet-inspired 1D CNN optimized for Raspberry Pi
3. **Incremental Training**: Train a single model that improves over time with new data
4. **Intelligent Analysis**: Use model predictions to reduce reliance on ground truth matching

## Architecture

### Components

#### 1. Dataset Manager (`dataset_manager.py`)
Manages dataset organization and curation:
- **Configurable naming**: `train1`, `train2`, etc.
- **Automatic organization**: Each run saves to separate CSV files within the active dataset
- **Confidence filtering**: Only saves samples with confidence >= threshold (default 0.85)
- **Metadata tracking**: Tracks runs processed, sample counts, and label distributions

**Key Features:**
- Multiple dataset support for different experiments
- Dataset merging capabilities
- Statistics and visualization
- Easy loading for training

#### 2. Model Trainer (`model_trainer.py`)
Implements the lightweight CNN and training logic:

**Model Architecture:**
- Input: 1024 frequency bins from ODAS output
- Depthwise separable convolutions for efficiency
- Squeeze-and-Excitation blocks for feature recalibration
- ~100K parameters (optimized for Raspberry Pi)
- Output: Softmax over audio source classes

**Training Features:**
- Data augmentation (noise injection)
- Early stopping with patience
- Learning rate scheduling
- Checkpoint saving with metadata
- Training history visualization
- TorchScript export for deployment

#### 3. Model Interface (`model_interface.py`)
Streamlit interface for all ML operations:

**Tabs:**
1. **Dataset Management**: View, create, manage datasets
2. **Train Model**: Configure and train models
3. **Evaluate Model**: Confusion matrix, classification report, confidence analysis
4. **Deploy Model**: Set active model for inference

#### 4. Updated Analyzer (`analyzer.py`)
Enhanced with model integration:

**Analysis Modes:**
- **Without Model**: Ground truth matching only (original behavior)
- **With Model**: Hybrid approach using both ground truth and model predictions

**Intelligent Data Curation:**
- High-confidence ground truth matches → Save to dataset (already validated)
- Model predictions with high confidence → Use for classification, don't retrain
- Mismatches between model and ground truth → Flag for training (potential improvement)
- Low confidence predictions → Flag for training (model uncertain)

## Workflow

### Phase 1: Initial Dataset Creation

1. **Run Simulations**:
   ```
   Scene Configurator → Audio Renderer → ODAS Simulator
   ```

2. **Analyze Results**:
   - Go to "Results Analyzer"
   - Select a run
   - Enable "Save to dataset" in settings
   - Run analysis
   - High-confidence matches automatically saved to `dataset/train1/`

3. **Repeat** for multiple scenes to build initial dataset

### Phase 2: Initial Model Training

1. **Go to Model Training** → Dataset Management:
   - View dataset stats
   - Check label distribution
   - Ensure sufficient samples (recommend 500+ per class)

2. **Train Model** tab:
   - Select dataset (e.g., `train1`)
   - Load dataset
   - Configure hyperparameters:
     - Epochs: 50-100
     - Learning Rate: 0.001
     - Validation Split: 0.2
     - Early Stopping Patience: 10
   - Start training
   - Monitor training curves

3. **Evaluate Model**:
   - Check confusion matrix
   - Review classification report
   - Analyze confidence distribution
   - Goal: >90% validation accuracy, >0.85 avg confidence

4. **Deploy Model**:
   - Set as "Active Model"
   - Now analyzer will use this model for predictions

### Phase 3: Incremental Learning

With an active model deployed, the analyzer becomes smarter:

#### Analysis Behavior:

**For each detection:**

1. **Ground Truth Matching** (if in known time window):
   - Calculate angular distance
   - Compute confidence score
   
2. **Model Prediction**:
   - Feed 1024 bins to model
   - Get predicted label and confidence

3. **Decision Logic**:

   | Scenario | Action | Save to Dataset? |
   |----------|--------|------------------|
   | GT match (conf > 0.85) & Model agrees | Use GT label | ✅ Yes (reinforcement) |
   | GT match (conf > 0.85) & Model disagrees | Use GT label, flag for training | ✅ Yes (correction) |
   | GT match (conf < 0.85) & Model confident | Use Model prediction | ❌ No (trust model) |
   | GT match (conf < 0.85) & Model uncertain | Use GT label, flag for training | ✅ Yes (both uncertain) |
   | Unknown & Model confident | Use Model prediction | ❌ No (trust model) |
   | Unknown & Model uncertain | Use Model prediction, flag for training | ✅ Yes (expand knowledge) |

**Benefits:**
- **Reduces manual labeling**: Model handles confident predictions
- **Focuses training**: Only saves samples that improve model
- **Continuous improvement**: Model learns from disagreements and uncertainties

#### Retraining Cycle:

1. **Accumulate New Data**:
   - Run multiple analyses
   - New training samples automatically collected in active dataset
   - Create new dataset for each training cycle: `train2`, `train3`, etc.

2. **Retrain Model**:
   - Go to Model Training
   - Select new dataset (or merge multiple datasets)
   - Load existing model: Check "Continue from existing model"
   - Select last best model
   - Train with lower learning rate (e.g., 0.0001)
   - Fewer epochs (20-30)

3. **Evaluate Improvement**:
   - Compare new model vs old model on test dataset
   - Check if accuracy improved
   - Verify confidence distribution shifted higher

4. **Deploy if Better**:
   - Set new model as active
   - Old model kept as backup

## Dataset Organization

```
simulator/outputs/
├── yamnet_datasets/
│   ├── curator_config.json            # Active dataset target + curation criteria
│   ├── yamnet_train_001/              # Primary YAMNet fine-tuning dataset
│   │   ├── audio/                     # Reconstructed WAV files (1 per ODAS track)
│   │   │   └── <run_id>_<idx>_t<ts>_<N>bins_dir<az>deg_<label>.wav
│   │   ├── spectrograms/              # Mel-spectrogram PNG visualisations
│   │   ├── metadata/                  # Per-run metadata CSVs
│   │   └── labels.csv                 # Master label file (all runs)
│   └── yamnet_unknown_001/            # Samples with no GT match (unknown class)
│       └── (same structure)
├── dataset/
│   └── train1/                        # Legacy lightweight-CNN dataset
│       └── <run_id>.csv
└── models/
    ├── active_model.pth
    └── ...
```

### YAMNet `labels.csv` Schema

The master label file written by `YAMNetDatasetCurator`:

| Column | Type | Description |
|--------|------|-------------|
| `filename` | str | WAV filename (unique key) |
| `run_id` | str | Simulation run identifier |
| `timestamp` | float | Detection time in seconds (ODAS `timeStamp × 0.008`) |
| `label` | str | Ground truth class (from scene config) |
| `yamnet_class` | str | ODAS top-1 event class name |
| `yamnet_confidence` | float | Peak single-hop confidence (`event_max_confidence`) |
| `yamnet_votes` | int | Hops that agreed on the winning class (0–6) |
| `yamnet_ambiguous` | bool | True when #2 candidate ties #1 on hop votes |
| `top_k_candidates` | str | Top-5 candidates: `Name(Nv,conf)\|...` pipe-separated |
| `ground_truth` | str | GT label (same as `label` for matched samples) |
| `curation_reason` | str | Comma-separated tags e.g. `unclassified,low_confidence_0.00,mismatch_yamnet:X_gt:Y,ambiguous_topk` |
| `activity` | float | ODAS track activity score |
| `n_stitched_bins` | int | Number of `.bin` sidecars stitched into this WAV |
| `stitched_duration_s` | float | Reconstructed audio duration in seconds |
| `position` | dict | ODAS 3D position `{x,y,z}` |
| `confidence` | float | Spatial angular match confidence |
| `angular_error` | float | Degrees between ODAS direction and GT source |
| `dataset_type` | str | `training` or `unknown` |
| `clean_match` | bool | True when GT-matched and routed to training |
| `manual_verification_needed` | bool | Flagged for human review |
| `fold` | str | Train/val split fold (default `train`) |

### Audio WAV Naming Convention

```
{run_id}_{idx:04d}_t{timestamp}_{N}bins_dir{az_error:.0f}deg_{label}.wav
```

Example: `wolf_frog_ele_20260301_015542_run_20260301_111706_0000_t3_520_2bins_dir9deg_Wolfhowl.wav`

- `t3_520` = timestamp 3.520 s
- `2bins` = 2 `.bin` sidecars stitched → ~1.5 s audio
- `dir9deg` = 9° angular error from GT source

### Track-Based WAV Stitching

Each ODAS track can produce multiple `.bin` sidecars (one per 48 ms YAMNet
evaluation).  The curator **groups all `.bin` files for the same `track_id`**
and reconstructs them into a single WAV:

| Track bins | Audio duration | Notes |
|-----------|---------------|-------|
| 1 | ~0.76 s | Minimum — one evaluation |
| 2 | ~1.53 s | |
| 3 | ~2.30 s | |
| 6 | ~4.60 s | Common for 5-second sound clips |

This yields significantly longer, higher-quality YAMNet training inputs
compared to per-bin WAVs.

### YAMNet Confidence Threshold

The **YAMNet confidence threshold** slider in the Dataset Curation Settings
expander controls which samples the curator saves:

- **Save when `yamnet_conf < threshold`** — captures events the base model
  doesn't know (needs training).
- **Skip when `yamnet_conf ≥ threshold`** — the base model already classifies
  this correctly; no fine-tuning benefit.
- Default: **0.75**.  Range: 0.0–1.0 (step 0.05).  Persists to
  `curator_config.json`.

This is separate from the spatial angular match confidence (which determines
whether a detection is ground-truth matched at all).

## Model Details

### Architecture: LightweightAudioClassifier

```
Input: (batch, 1024) frequency bins
    ↓
Reshape: (batch, 1, 1024)
    ↓
Stem: Conv1d(1→32, k=7, s=2) + BN + ReLU
    ↓
MBConv Block 1: 32→32 channels (expand 2x)
    ↓
MBConv Block 2: 32→48 channels (expand 4x, stride 2)
    ↓
MBConv Block 3: 48→48 channels (expand 4x)
    ↓
MBConv Block 4: 48→64 channels (expand 4x, stride 2)
    ↓
MBConv Block 5: 64→64 channels (expand 4x)
    ↓
Global Average Pooling
    ↓
Dropout(0.3)
    ↓
Linear(64 → num_classes)
    ↓
Output: (batch, num_classes) logits
```

**Parameters**: ~100K (optimized for edge devices)

**Inference Speed** (estimated):
- Raspberry Pi 4: ~5-10ms per sample
- GPU: <1ms per sample

### Training Hyperparameters (Recommended)

**Initial Training:**
```python
num_epochs = 50-100
learning_rate = 0.001
batch_size = 32
dropout = 0.3
test_split = 0.2
patience = 10
optimizer = Adam
scheduler = ReduceLROnPlateau(factor=0.5, patience=5)
```

**Incremental Training:**
```python
num_epochs = 20-30
learning_rate = 0.0001  # Lower LR
batch_size = 32
load_existing = True
```

## Usage Examples

### Example 1: Create Initial Dataset

```python
# 1. Run analyzer on first simulation
# Settings:
# - Angle Threshold: 10°
# - Save to dataset: ✓
# - Confidence threshold: 0.85

# Result: 150 samples saved to train1

# 2. Repeat for 5 more simulations with different scenes
# Result: 900 total samples in train1 (150 per run)

# 3. Check dataset
# Go to Model Training → Dataset Management
# View "train1": 900 samples, 6 classes, balanced distribution
```

### Example 2: Train Initial Model

```python
# Model Training → Train Model
# 1. Select "train1"
# 2. Load Dataset
# 3. Configure:
#    - Epochs: 50
#    - Learning Rate: 0.001
#    - Validation Split: 0.2
#    - Patience: 10
# 4. Start Training

# Results after 35 epochs (early stop):
# - Train Accuracy: 94.5%
# - Val Accuracy: 91.2%
# - Avg Confidence: 0.89

# 5. Evaluate Model
# - Confusion matrix: Good separation between classes
# - Low confidence samples: 8% (acceptable)

# 6. Deploy Model
# - Set as Active Model ✓
```

### Example 3: Incremental Learning Cycle

```python
# With active model deployed:

# 1. Run 10 new simulations
# 2. Analyze each with "Save to dataset" enabled
# 3. Model automatically:
#    - Classifies 70% with high confidence (not saved again)
#    - Flags 20% as mismatches/uncertain (saved to train2)
#    - Leaves 10% as ground truth (saved to train2)
# Result: 180 new samples in train2 (focused on model weaknesses)

# 4. Train incrementally:
#    - Select train2
#    - Load existing model ✓
#    - LR: 0.0001
#    - Epochs: 20
# Result: Val accuracy improved to 93.1%

# 5. Deploy new model
# 6. Repeat cycle...
```

## Performance Tips

### For Raspberry Pi Deployment:

1. **Use TorchScript Export**:
   - Model Interface → Deploy → Export for Raspberry Pi
   - Optimized inference graph

2. **Reduce Batch Size**:
   - For real-time: Batch size = 1
   - For batch processing: Batch size = 4-8

3. **Quantization** (optional):
   ```python
   # Post-training quantization
   quantized_model = torch.quantization.quantize_dynamic(
       model, {torch.nn.Linear}, dtype=torch.qint8
   )
   # Reduces model size and improves speed
   ```

### For Training Speed:

1. **Use GPU if available**: Automatically detected
2. **Increase batch size**: 64 or 128 if memory allows
3. **Reduce early stopping patience**: If dataset is large
4. **Use data augmentation**: Helps with small datasets

## Monitoring and Debugging

### Check Dataset Quality:

```python
# Dataset Management → View Details
# Look for:
# - Balanced classes (similar sample counts)
# - High confidence scores (>0.85 average)
# - Low angular errors (<5° average)
# - Diverse time coverage
```

### Check Model Performance:

```python
# Evaluate Model → Confusion Matrix
# Red flags:
# - Off-diagonal entries: Class confusion
# - Low confidence: Model uncertain
# - Poor recall for some classes: Need more samples

# Solutions:
# - Add more samples for confused classes
# - Check if classes are acoustically similar
# - Increase model capacity or training time
```

### Troubleshooting:

**Problem**: Model accuracy stuck at 70%
- **Check**: Dataset label quality - are ground truth labels correct?
- **Check**: Class imbalance - some classes have very few samples?
- **Solution**: Clean dataset, add more samples, balance classes

**Problem**: Model overfits (train 99%, val 75%)
- **Check**: Dataset too small or not diverse enough
- **Solution**: Add dropout, more data augmentation, early stopping

**Problem**: Model always predicts one class
- **Check**: Severe class imbalance in training data
- **Solution**: Balance dataset, use class weights in loss function

## Advanced Features

### Dataset Merging:

Combine multiple datasets for comprehensive training:

```python
# Dataset Management
# Merge datasets: train1, train2, train3
# Output: train_merged_20250130
# Use for final model training
```

### Confidence Threshold Tuning:

Experiment with different thresholds:
- **High (0.90)**: Only very confident samples → smaller, cleaner dataset
- **Medium (0.85)**: Balanced approach → recommended
- **Low (0.75)**: More samples, some noise → use if data-scarce

### Model Ensemble (Future):

Train multiple models on different datasets, average predictions for higher confidence.

## Conclusion

This system enables continuous improvement of audio source classification through:
1. ✅ Automated dataset curation from simulation runs
2. ✅ Efficient model architecture for edge deployment
3. ✅ Intelligent training data selection using model feedback
4. ✅ Easy retraining and deployment workflow

The key insight: **Use model predictions to reduce manual labeling while focusing training on the model's weaknesses** - creating a virtuous cycle of improvement.
