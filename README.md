# Audio Simulation & Classification Pipeline

A Streamlit-based application for generating synthetic training data for directional audio classification using ODAS (Open embeddeD Audition System).

## Overview

This pipeline creates labelled datasets for fine-tuning YAMNet by:
1. **Configuring** synthetic audio scenes with known source positions
2. **Rendering** multi-channel audio using room acoustics simulation
3. **Processing** with ODAS to detect and classify sound source tracks
4. **Matching** detected tracks to ground truth sources
5. **Curating** labelled WAV datasets for YAMNet fine-tuning in the [yamnet repo](https://github.com/anamtya-tech/yamnet)

## Components

### 1. Sources Library (`config/sources.csv`)
- Catalog of 42 audio source types (directional and ambient)
- 75 variants per source type in `audio_cache/forest/audio/`
- Examples: Axe, Chainsaw, Lion, Helicopter (directional); Bird, Wind, Fire (ambient)

### 2. Scene Configurator (`configurator.py`)
- **Easy Mode**: Auto-generate random scenes with configurable parameters
- **Manual Mode**: Fine-tune individual source positions and timing
- Spatial constraints: 10-500m radius, -2 to 100m height
- Time constraints: Fixed start/end times with manual override
- Output: JSON scene files in `config/scenes/`

### 3. Audio Renderer (`renderer.py`)
- Uses **pyroomacoustics** for spatial audio simulation
- Simulates ReSpeaker USB 4 Mic Array (4-mic circular, 64mm diameter)
- Generates 6-channel raw PCM audio (16kHz, S16_LE)
  - Channels 1,6: Zeros (no mic)
  - Channels 2-5: 4 microphone signals
- Handles directional sources (point sources) and ambient sources (omnidirectional overlay)
- Output: `.raw` files in `simulator/outputs/`

### 4. ODAS Simulator (`simulator.py`)
- Orchestrates socket server (`vm_socket_emit.py`) and ODAS binary (`odaslive`)
- Streams raw audio via TCP socket (port 10000)
- ODAS processes audio with:
  - Sound Source Tracking (SST) — Kalman filter
  - 257-bin magnitude spectrum per detection (stored as `.bin` sidecar files)
  - Position estimates (x, y, z) and activity scores
- Output: JSON files in `~/sodas/ClassifierLogs/` (set by `classifier_log_dir` in the ODAS config)
  - `sst_session_live.json_<timestamp>`: Frame-by-frame detections with frequency bins
  - `sst_classify_events_<timestamp>.json`: Event classifications

### 5. Results Analyzer (`analyzer.py`)
- Loads ground truth scene configuration
- Parses ODAS output (detections with 257-bin magnitude spectra + track metadata)
- **Matches** detections to sources using angular threshold (default 10°)
- Calculates statistics: precision, recall per source
- Feeds matched results to `YAMNetDatasetCurator` for WAV extraction
- Visualizes angular error distribution and label distribution

### 6. YAMNet Dataset Curator (`yamnet_dataset_curator.py`)
- Filters spatially + temporally aligned matches
- Reconstructs audio from ODAS `.bin` sidecar spectra (via `audio_reconstructor.py`)
- Stitches multiple `.bin` files per track for longer WAVs
- Saves 16 kHz mono WAVs + `labels.csv` in YAMNet-compatible format
- Output: `outputs/yamnet_datasets/<dataset_name>/`

## Installation

### Prerequisites
```bash
# System dependencies
sudo apt-get install -y python3.12-dev build-essential

# Python environment (using uv)
cd simulator
uv pip install -r requirements.txt
```

### Required Packages
- streamlit >= 1.30.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.11.0
- librosa >= 0.10.0
- soundfile >= 0.12.0
- matplotlib >= 3.7.0
- pyroomacoustics >= 0.7.0

## Usage

### Launch Application
```bash
cd /home/azureuser/simulator
streamlit run app.py
```

### Workflow

#### Step 1: Configure Scene
1. Navigate to **"🎨 Scene Configurator"**
2. Choose mode:
   - **Easy Mode**: Set number of sources, radius, duration → Generate
   - **Manual Mode**: Add/edit sources individually, customize positions
3. Save scene with a descriptive name

#### Step 2: Render Audio
1. Navigate to **"🔊 Audio Renderer"**
2. Select saved scene configuration
3. Click **"Render Audio"**
4. Wait for pyroomacoustics simulation to complete
5. Output: `{date}_{scene_name}_ChatakX_sim.raw` + metadata JSON

#### Step 3: Run ODAS Simulation
1. Navigate to **"⚙️ ODAS Simulator"**
2. Select rendered audio file
3. Configure socket port (default 10000)
4. Click **"▶️ Run Simulation"**
5. Monitor logs (socket server + ODAS output)
6. Wait for completion (duration = scene duration)
7. Output: SST and classify_events JSON in `~/sodas/ClassifierLogs/`

#### Step 4: Analyze Results
1. Navigate to **"📊 Results Analyzer"**
2. Select simulation run
3. Adjust angle threshold (1-45°, default 10°)
4. Click **"📊 Analyze"**
5. Review statistics:
   - Detection recall per source
   - Angular error distribution
   - Label distribution

#### Step 5: Curate YAMNet Dataset
1. Navigate to **"🎯 YAMNet Datasets"**
2. Select or create a dataset (e.g., `yamnet_train_001`)
3. Adjust curation settings (confidence threshold, angle threshold)
4. Click **"Curate from last analysis"**
5. Repeat steps 1–5 for multiple scenes
6. When ready, train in the [yamnet repo](https://github.com/anamtya-tech/yamnet):
   ```bash
   python training/train_yamnet.py \
       --dataset ~/simulator/outputs/yamnet_datasets/yamnet_train_001 \
       --savedmodel model_store/base/yamnet_core_savedmodel
   ```

## Output Format

### Scene Configuration JSON
```json
{
  "scene_name": "forest_test_01",
  "duration": 60,
  "max_radius": 200,
  "max_height": 50,
  "directional_sources": [
    {
      "label": "Axe",
      "wav_path": "/home/azureuser/audio_cache/forest/audio/axe23.wav",
      "x": 45.2, "y": -12.8, "z": 1.5,
      "azimuth": -15.8, "distance": 47.0, "height": 1.5,
      "start_time": 5.0, "end_time": 25.0
    }
  ],
  "ambient_sources": [
    {
      "label": "Bird",
      "wav_path": "/home/azureuser/audio_cache/forest/audio/bird07.wav",
      "start_time": 0, "end_time": 60
    }
  ]
}
```

## Architecture

### Data Flow
```
sources.csv → Scene Config → Pyroomacoustics → 6ch Raw Audio
                                                      ↓
                                            Socket Server (port 10000)
                                                      ↓
                                            ODAS (odaslive)
                                                      ↓
                                        SST JSON (detections + .bin sidecars)
                                                      ↓
            Ground Truth Scene Config ← Analyzer → YAMNetDatasetCurator
                                                      ↓
                                        outputs/yamnet_datasets/<name>/
                                          audio/*.wav + labels.csv
                                                      ↓
                                        yamnet repo: train_yamnet.py
                                                      ↓
                                        model_store/releases/*.tflite → ODAS
```

### File Structure
```
simulator/
├── app.py                        # Main Streamlit app
├── configurator.py               # Scene + dataset configuration
├── renderer.py                   # Audio rendering with pyroomacoustics
├── simulator.py                  # ODAS orchestration
├── analyzer.py                   # Detection matching and analysis
├── yamnet_dataset_curator.py     # WAV extraction + labels.csv generation
├── audio_reconstructor.py        # Spectra .bin → audio waveform
├── yamnet_helper/                # Python YAMNet spectrum classifier (reference)
├── requirements.txt
├── README.md
└── outputs/
    ├── *.raw                     # Rendered audio files
    ├── *_metadata.json           # Render metadata
    ├── runs/
    │   └── run_*.json            # Simulation run manifests
    └── yamnet_datasets/
        ├── curator_config.json
        └── yamnet_train_001/
            ├── audio/*.wav       # 16 kHz mono WAV training clips
            ├── spectrograms/
            ├── metadata/
            └── labels.csv        # Master label file for training

config/
├── sources.csv                   # Audio source catalog
└── scenes/
    └── *.json                    # Saved scene configurations

~/sodas/ClassifierLogs/           # set by classifier_log_dir in chatak-odas
├── sst_session_live.json_*       # ODAS detections with .bin sidecar refs
└── sst_classify_events_*         # ODAS event classifications
```

## Configuration

### Spatial Parameters
- **Radius Range**: 10-500 meters (configurable per scene)
- **Height Range**: -2 to 100 meters (configurable per scene)
- **Mic Array**: 4-mic circular at origin, 64mm diameter

### Matching Parameters
- **Angular Threshold**: 1-45 degrees (default 10°)
- Uses spherical coordinates for position comparison
- Calculates angular distance between detection and source vectors

### ODAS Parameters
- **Sample Rate**: 16 kHz (fixed)
- **Hop Size**: 128 samples (~8ms frames)
- **Frame Size**: 256 samples
- **Tracking**: Kalman filter with dynamic source addition
- **Frequency Bins**: 257 per detection (FFT size 512, half spectrum, stored as `.bin` sidecars)

## Troubleshooting

### Import Errors
```bash
# Verify all packages installed
python3 -c "import streamlit, numpy, pandas, scipy, librosa, soundfile, matplotlib, pyroomacoustics"
```

### ODAS Not Found
```bash
# Check ODAS binary exists (chatak-odas fork)
ls -l ~/z_odas_newbeamform/build/bin/odaslive

# Rebuild if needed
cd ~/z_odas_newbeamform
cmake -B build && cmake --build build -j$(nproc)
```

### Socket Connection Errors
- Ensure port 10000 is not in use: `lsof -i :10000`
- Check socket server logs in terminal output
- Verify `vm_socket_emit.py` has execute permissions

### Empty Datasets
- Increase angle threshold if too few matches
- Check ODAS logs for detection activity
- Verify audio files exist and are valid
- Ensure scene duration matches audio length

## Future Enhancements

- [ ] Multi-scene batch processing
- [ ] Real-time visualisation of detections
- [ ] Manual label correction UI for `yamnet_unknown` samples
- [ ] Automatic fold assignment (train/val/test) in the curator
- [ ] GPU acceleration for audio rendering
- [ ] Reverb and room parameter customisation
- [ ] SNR-based filtering for low-quality reconstructions

## License

This project builds upon ODAS (MIT License) and pyroomacoustics (MIT License).

## References

- ODAS fork (Chatak): https://github.com/anamtya-tech/chatak-odas
- YAMNet training: https://github.com/anamtya-tech/yamnet
- ODAS upstream: https://github.com/introlab/odas
- Pyroomacoustics: https://github.com/LCAV/pyroomacoustics
- ReSpeaker Mic Array: https://wiki.seeedstudio.com/ReSpeaker_Mic_Array_v2.0/
