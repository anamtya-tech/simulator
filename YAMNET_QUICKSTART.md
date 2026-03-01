# YAMNet Fine-Tuning Quick Start

## 5-Minute Setup

### 1. Run a Simulation (if you haven't already)

```bash
# Start Streamlit app
streamlit run app.py

# Navigate to: ⚙️ ODAS Simulator
# Select a scene and run
```

### 2. Analyze Results with YAMNet Curation

1. Navigate to: **📊 Results Analyzer**
2. Enable: "Save to YAMNet dataset" ✅
3. Select YAMNet dataset: `yamnet_train_001` (or create new)
4. Click: **🔍 Analyze Run**

The system will automatically:
- Compare YAMNet predictions with ground truth
- Identify mismatches and low-confidence samples
- Reconstruct audio from frequency bins
- Save to dataset

### 3. Review Curated Data

Navigate to: **🎯 YAMNet Datasets → 📈 Visualizer**

- Browse samples
- Play audio 🔊
- View spectrograms 📊
- Check label distribution

### 4. Prepare for Training

Navigate to: **🎯 YAMNet Datasets → 📊 Datasets**

Click: **Prepare for TensorFlow** on your dataset

This creates:
- `train_labels.csv` (70%)
- `val_labels.csv` (15%)
- `test_labels.csv` (15%)

### 5. Fine-Tune YAMNet (External)

```python
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

# Load YAMNet
yamnet = hub.load('https://tfhub.dev/google/yamnet/1')

# Load your dataset
df = pd.read_csv('outputs/yamnet_datasets/yamnet_train_001/train_labels.csv')

# ... implement fine-tuning (see full README)
```

## What Gets Curated?

✅ **Mismatches**: YAMNet said "Dog", ground truth is "Wolf"  
✅ **Unclassified**: YAMNet didn't classify  
✅ **Low Confidence**: YAMNet confidence < 70%  
❌ **Silent Samples**: Activity < 30%  

## Typical Workflow

```
1. Run 5-10 simulations ──┐
                          │
2. Analyze each one ──────┤
   with curation enabled  │
                          │
3. Accumulate samples ────┘
   in one dataset

4. Review in visualizer
   └─> Filter, listen, verify

5. Prepare for TensorFlow
   └─> Creates train/val/test splits

6. Fine-tune YAMNet
   └─> Transfer learning on embeddings

7. Deploy improved model
   └─> Update ODAS configuration

8. Repeat!
```

## Directory Structure After Setup

```
outputs/
├── yamnet_datasets/
│   ├── curator_config.json          # Configuration
│   └── yamnet_train_001/             # Your dataset
│       ├── audio/                    # WAV files
│       ├── spectrograms/             # Visualizations
│       ├── metadata/                 # Per-run CSVs
│       ├── labels.csv                # Master labels
│       ├── train_labels.csv          # Training split
│       ├── val_labels.csv            # Validation split
│       └── test_labels.csv           # Test split
├── runs/                             # Simulation runs
└── analysis/                         # Analysis results
```

## Common Commands

### Create New Dataset
```python
# Via UI: 🎯 YAMNet Datasets → 📊 Datasets → Create New Dataset

# Or programmatically:
from yamnet_dataset_curator import YAMNetDatasetCurator
curator = YAMNetDatasetCurator()
curator.set_active_dataset('yamnet_train_002')
```

### Merge Datasets
```python
# Via UI: 🎯 YAMNet Datasets → 📊 Datasets → Merge Datasets

# Or programmatically:
curator.merge_datasets(
    ['yamnet_train_001', 'yamnet_train_002'],
    'yamnet_merged_comprehensive'
)
```

### Adjust Curation Settings
```python
# Via UI: 🎯 YAMNet Datasets → ⚙️ Settings

# Or edit: outputs/yamnet_datasets/curator_config.json
```

## Quick Tips

💡 **Tip 1**: Start with default settings, they work well  
💡 **Tip 2**: Aim for 500-1000 samples per class minimum  
💡 **Tip 3**: Use visualizer to catch label errors early  
💡 **Tip 4**: Run diverse scenes to get varied data  
💡 **Tip 5**: Lower confidence threshold if too few samples  

## Verify It's Working

After running analysis with curation:

1. Check: `outputs/yamnet_datasets/yamnet_train_001/audio/`
   - Should have `.wav` files

2. Check: `outputs/yamnet_datasets/yamnet_train_001/labels.csv`
   - Should have rows with your samples

3. Use Visualizer:
   - Should see samples listed
   - Audio should play

## Troubleshooting

### No samples curated?
- Check if YAMNet is classifying (see analyzer results)
- Lower confidence threshold (Settings)
- Check activity threshold

### Audio won't play?
- Check file exists in `audio/` folder
- Verify WAV format (16kHz mono)
- Check browser audio permissions

### Labels look wrong?
- Verify ground truth in scene file
- Check YAMNet predictions in analyzer
- Use visualizer filters to investigate

## Next Steps

1. **Read full documentation**: `YAMNET_FINETUNING_README.md`
2. **Explore visualizer**: Play with filters and analytics
3. **Collect more data**: Run multiple diverse simulations
4. **Prepare training**: When you have enough samples
5. **Fine-tune**: Follow TensorFlow YAMNet tutorial

## Support

For questions or issues:
- Check: `YAMNET_FINETUNING_README.md` (comprehensive guide)
- Review: `outputs/yamnet_datasets/curator_config.json`
- Inspect: Dataset Visualizer analytics

Happy fine-tuning! 🎯🎵
