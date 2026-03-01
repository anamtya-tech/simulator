# YAMNet Fine-Tuning Dataset Curation System

## Overview

This system automatically curates training datasets for fine-tuning YAMNet based on ODAS output analysis. It identifies samples where YAMNet predictions can be improved and prepares them in TensorFlow-ready format.

## Architecture

### Core Components

1. **YAMNetDatasetCurator** (`yamnet_dataset_curator.py`)
   - Main dataset management and curation logic
   - Extracts samples needing improvement
   - Saves in TensorFlow Hub YAMNet format
   - Manages multiple named datasets

2. **AudioReconstructor** (`audio_reconstructor.py`)
   - Reconstructs audio from ODAS frequency bins (1024 bins)
   - Implements Griffin-Lim algorithm for phase reconstruction
   - Uses overlap-add for temporal continuity
   - Produces 16kHz mono WAV files

3. **DatasetVisualizer** (`dataset_visualizer.py`)
   - Interactive Streamlit interface for dataset exploration
   - Audio playback in browser
   - Spectrogram visualization
   - Filtering and analytics

4. **DatasetConfigurator** (`configurator.py`)
   - UI for dataset management
   - Curation settings configuration
   - Dataset merging and preparation
   - Comprehensive usage guide

## Data Flow

```
ODAS Output (bins + YAMNet predictions)
    ↓
Analyzer (compares with ground truth)
    ↓
Curation Filter (mismatches, low confidence, unclassified)
    ↓
Audio Reconstruction (bins → WAV)
    ↓
Dataset Storage (audio/ + metadata/)
    ↓
TensorFlow Dataset (train/val/test splits)
    ↓
YAMNet Fine-Tuning
```

## Dataset Structure

Each dataset contains:

```
outputs/yamnet_datasets/yamnet_train_001/
├── audio/                          # Reconstructed WAV files (16kHz mono)
│   ├── run_001_0000_t0_245_wolf.wav
│   ├── run_001_0001_t0_312_elephant.wav
│   └── ...
├── spectrograms/                   # Visual representations for review
│   ├── run_001_0000_t0_245_wolf.png
│   └── ...
├── metadata/                       # Per-run metadata CSVs
│   ├── run_001_20260213_143025.csv
│   └── ...
└── labels.csv                      # Master label file for training
```

### labels.csv Format

Compatible with TensorFlow Hub YAMNet:

| Column | Description |
|--------|-------------|
| filename | Audio filename (e.g., run_001_0000_t0_245_wolf.wav) |
| label | Ground truth label |
| fold | train/val/test split |
| yamnet_class | YAMNet's prediction |
| yamnet_confidence | YAMNet's confidence (0-1) |
| ground_truth | Original ground truth label |
| curation_reason | Why this sample was selected |
| run_id | Source simulation run |
| timestamp | Time in simulation |
| activity | ODAS activity level |
| position | Source position (x, y, z) |

## Curation Criteria

Samples are selected when:

1. **Mismatches**: YAMNet prediction ≠ ground truth
   - Example: YAMNet says "Dog" but ground truth is "Wolf"
   - Reason: `mismatch_yamnet:Dog_gt:Wolf`

2. **Unclassified**: YAMNet didn't provide a classification
   - Example: class_id = -1
   - Reason: `unclassified`

3. **Low Confidence**: YAMNet confidence below threshold
   - Example: confidence < 0.7
   - Reason: `low_confidence_0.45`

4. **Minimum Activity**: Filters out silent/quiet samples
   - Default: activity >= 0.3

## Usage Workflow

### 1. Configure Curation Settings

In Streamlit app: **🎯 YAMNet Datasets → ⚙️ Settings**

```python
# Example configuration
{
    'include_mismatches': True,      # Save prediction mismatches
    'include_unclassified': True,    # Save unclassified samples
    'include_low_confidence': True,  # Save low confidence predictions
    'confidence_threshold': 0.7,     # Threshold for "low confidence"
    'min_activity': 0.3              # Minimum ODAS activity level
}
```

### 2. Run Simulations with Curation Enabled

In Streamlit app: **📊 Results Analyzer**

1. Enable "Save to YAMNet dataset"
2. Select active dataset (or create new)
3. Run analysis
4. Samples automatically curated

```python
# Programmatic usage
from yamnet_dataset_curator import YAMNetDatasetCurator

curator = YAMNetDatasetCurator('outputs/yamnet_datasets')
curator.set_active_dataset('yamnet_train_001')

# Curate from analysis results
stats = curator.curate_from_analysis(analysis_results, run_id='run_001')
print(f"Saved {stats['saved']} samples")
```

### 3. Review Dataset

In Streamlit app: **🎯 YAMNet Datasets → 📈 Visualizer**

- Browse samples with metadata
- Play audio directly in browser
- View spectrograms
- Filter by label, confidence, reason
- Analyze distribution

### 4. Prepare for Training

In Streamlit app: **🎯 YAMNet Datasets → 📊 Datasets**

Click "Prepare for TensorFlow" to:
- Create train/val/test splits (70/15/15 by default)
- Generate separate CSVs for each split
- Stratify by label for balanced splits

```python
# Programmatic usage
result = curator.create_tensorflow_dataset(
    'yamnet_train_001',
    train_val_test_split=(0.7, 0.15, 0.15)
)

print(result)
# {
#     'dataset_path': 'outputs/yamnet_datasets/yamnet_train_001',
#     'train_csv': '.../ train_labels.csv',
#     'val_csv': '.../val_labels.csv',
#     'test_csv': '.../test_labels.csv',
#     'splits': {'train': 700, 'val': 150, 'test': 150}
# }
```

### 5. Fine-Tune YAMNet

```python
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
from pathlib import Path

# Load YAMNet
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load dataset
dataset_path = Path('outputs/yamnet_datasets/yamnet_train_001')
train_df = pd.read_csv(dataset_path / 'train_labels.csv')

# Create TensorFlow dataset
def load_audio_file(filename):
    audio_path = dataset_path / 'audio' / filename
    audio_binary = tf.io.read_file(str(audio_path))
    waveform, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1)
    waveform = tf.squeeze(waveform, axis=-1)
    return waveform

def load_and_preprocess(row):
    waveform = load_audio_file(row['filename'])
    # Get YAMNet embeddings
    scores, embeddings, spectrogram = yamnet_model(waveform)
    # Use embeddings for transfer learning
    return embeddings, row['label']

# Build classifier on top of YAMNet embeddings
num_classes = len(train_df['label'].unique())
classifier = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(None, 1024)),  # YAMNet embeddings
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

classifier.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train
# ... (implement dataset loading and training loop)
```

## Audio Reconstruction Details

### Challenge
ODAS provides only **magnitude spectrum** (1024 frequency bins), not the full complex spectrum needed for perfect audio reconstruction.

### Solution: Griffin-Lim Algorithm

1. **Phase Initialization**: Random phase values
2. **Iterative Refinement**:
   ```
   for iteration in range(n_iter):
       audio = inverse_fft(magnitude * e^(iθ))
       spectrum = fft(audio)
       θ = angle(spectrum)  # Keep new phase
       # Reapply original magnitude
   ```
3. **Overlap-Add**: For multi-frame sequences

### Quality Considerations

- Reconstructed audio is sufficient for training
- Not perfect for human listening (no original phase information)
- Higher iteration count = better quality but slower
- Alternative: Use original audio if available (not implemented yet)

### Parameters

```python
AudioReconstructor(
    sample_rate=16000,      # YAMNet requires 16kHz
    n_fft=1024,             # Matches ODAS bins
    hop_length=128,         # ~8ms frames
    use_griffin_lim=True    # vs random phase
)
```

## Dataset Management

### Multiple Datasets

Track different experiments:
```
yamnet_train_001  # Initial dataset
yamnet_train_002  # More diverse sources
yamnet_train_003  # Different confidence threshold
merged_20260213   # Combined datasets
```

### Active Dataset

Only one dataset is "active" for curation at a time. Set via:
- UI: **🎯 YAMNet Datasets → 📊 Datasets → Set as Active**
- Code: `curator.set_active_dataset('yamnet_train_002')`

### Merging Datasets

Combine multiple datasets:
```python
curator.merge_datasets(
    ['yamnet_train_001', 'yamnet_train_002'],
    'merged_comprehensive'
)
```

## Configuration Files

### curator_config.json

Located at: `outputs/yamnet_datasets/curator_config.json`

```json
{
    "created_at": "2026-02-13T...",
    "active_dataset": "yamnet_train_001",
    "datasets": {
        "yamnet_train_001": {
            "created_at": "2026-02-13T...",
            "sample_count": 1250,
            "samples_by_label": {
                "wolf": 320,
                "elephant": 410,
                "frog": 520
            },
            "runs_processed": [
                {
                    "run_id": "run_001",
                    "timestamp": "2026-02-13T...",
                    "samples_added": 45
                }
            ]
        }
    },
    "curation_criteria": {
        "include_mismatches": true,
        "include_low_confidence": true,
        "confidence_threshold": 0.7,
        "include_unclassified": true,
        "min_activity": 0.3
    },
    "audio_params": {
        "sample_rate": 16000,
        "target_duration": 1.0,
        "overlap_frames": 5
    }
}
```

## Integration with Analyzer

The analyzer automatically curates when:

1. "Save to YAMNet dataset" is enabled
2. Analysis completes successfully
3. Samples meet curation criteria

```python
# In analyzer.py
if save_to_yamnet:
    yamnet_stats = self.yamnet_curator.curate_from_analysis(
        results,
        run_id
    )
    st.success(f"Curated {yamnet_stats['saved']} samples")
```

## API Reference

### YAMNetDatasetCurator

```python
curator = YAMNetDatasetCurator(output_dir='outputs/yamnet_datasets')

# Dataset management
curator.get_active_dataset()
curator.set_active_dataset(name)
curator.list_datasets()
curator.get_dataset_stats(name)

# Curation
curator.curate_from_analysis(analysis_results, run_id)

# Preparation
curator.create_tensorflow_dataset(name, train_val_test_split=(0.7, 0.15, 0.15))

# Merging
curator.merge_datasets(dataset_names, new_name)
```

### AudioReconstructor

```python
reconstructor = AudioReconstructor(
    sample_rate=16000,
    use_griffin_lim=True
)

# Single frame
audio = reconstructor.reconstruct_single_frame(bins)

# Multiple frames
audio = reconstructor.reconstruct_multi_frame(frames_bins, overlap_frames=3)

# From detections
result = reconstructor.reconstruct_from_detections(
    detections,
    target_duration=1.0
)

# Save
reconstructor.save_audio(audio, 'output.wav')
```

## Best Practices

1. **Diverse Data**: Run multiple simulations with different:
   - Source positions
   - Audio samples
   - Noise conditions
   - Timing overlaps

2. **Balanced Labels**: Aim for similar sample counts per class
   - Monitor in Dataset Visualizer
   - Adjust scene configurations if needed

3. **Quality Control**: Review samples before training
   - Use Dataset Visualizer audio playback
   - Check for label errors
   - Verify spectrograms

4. **Iterative Improvement**:
   ```
   1. Train initial model
   2. Identify problematic classes
   3. Curate more data for those classes
   4. Retrain
   5. Repeat
   ```

5. **Confidence Thresholds**: Adjust based on YAMNet performance
   - Start with 0.7
   - Lower if too few samples
   - Raise if too many low-quality samples

6. **Activity Filtering**: Skip silent samples
   - Default 0.3 works well
   - Increase if too many noise samples
   - Decrease if missing valid samples

## Troubleshooting

### No Samples Curated

**Possible causes:**
- YAMNet is performing too well (all high confidence matches)
- Activity threshold too high
- Curation criteria too strict

**Solutions:**
- Lower confidence threshold
- Check YAMNet classifications in analysis results
- Enable all curation options

### Audio Quality Poor

**Possible causes:**
- Low Griffin-Lim iterations
- Poor ODAS bin quality
- Low activity samples

**Solutions:**
- Increase activity threshold
- Check ODAS configuration
- Review source audio quality

### Imbalanced Dataset

**Possible causes:**
- Some source types more common
- Some classes easier for YAMNet
- Uneven scene configuration

**Solutions:**
- Create scenes with underrepresented classes
- Use dataset merging to balance
- Adjust sampling in training

### Out of Memory

**Possible causes:**
- Too many samples processed at once
- Large audio files

**Solutions:**
- Process runs individually
- Use batch reconstruction
- Increase system RAM

## Future Enhancements

1. **Direct Audio Storage**: Store original audio instead of reconstructed
2. **Advanced Filtering**: Spatial filtering, SNR-based selection
3. **Active Learning**: Suggest samples most valuable for training
4. **Augmentation**: Automatic data augmentation during curation
5. **Model Integration**: Direct fine-tuning from interface
6. **Metrics Tracking**: Track improvement over iterations

## References

- [TensorFlow Hub YAMNet](https://tfhub.dev/google/yamnet/1)
- [YAMNet Tutorial](https://www.tensorflow.org/hub/tutorials/yamnet)
- [Griffin-Lim Algorithm](https://ieeexplore.ieee.org/document/1164317)
- [AudioSet Dataset](https://research.google.com/audioset/)

## Questions?

This system bridges ODAS analysis with YAMNet fine-tuning. The key insight is using mismatches and low-confidence predictions to identify where YAMNet needs improvement, then automatically preparing training data in the correct format.
