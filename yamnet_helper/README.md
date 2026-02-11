# YAMNet Spectrum Classifier

Python implementation that parallels the C++ YAMNet integration in ODAS. This module accepts pre-computed magnitude spectra (257 bins per frame) and performs audio classification using YAMNet.

## Overview

This implementation mirrors the behavior of the C++ code in `/home/azureuser/z_odas_newbeamform/src/yamnet/`, providing:

- **Spectrum-based input**: Takes 257-bin magnitude spectra (matching ODAS FFT output)
- **Frame accumulation**: Buffers 96 frames for classification
- **Overlapping windows**: 50% overlap (48 frames) for temporal continuity
- **TensorFlow Lite inference**: Uses the same TFLite model as C++ implementation

## Files

- `yamnet_spectrum_classifier.py` - Core classifier implementation
- `audio_classifier_demo.py` - Demo script for processing audio files
- `test_spectrum_classifier.py` - Unit tests with synthetic data
- `__init__.py` - Module initialization

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 16,000 Hz | Audio sample rate |
| Frame Length | 400 samples | Window size for each frame |
| Frame Step | 160 samples | Hop size (~10ms) |
| FFT Size | 512 | FFT window size |
| Spectrum Bins | 257 | Output bins (FFT_SIZE//2 + 1) |
| Mel Bins | 64 | Mel-frequency bins |
| Patch Frames | 96 | Frames per classification |
| Patch Hop | 48 | Frames between classifications |
| Classes | 521 | AudioSet classes |

## Installation

```bash
# Install required packages
pip install numpy tensorflow librosa matplotlib
```

## Usage

### 1. Direct Spectrum Input (Parallel to C++ API)

```python
from yamnet_spectrum_classifier import YAMNetSpectrumClassifier

# Initialize classifier
classifier = YAMNetSpectrumClassifier(
    model_path='/home/azureuser/z_odas_newbeamform/models/yamnet_core.tflite',
    class_map_path='/home/azureuser/z_odas_newbeamform/models/yamnet_class_map.csv'
)

# Frame-by-frame processing
for spectrum in magnitude_spectra:  # shape: (257,)
    result = classifier.add_frame(spectrum)
    
    if result:
        class_id, class_name, confidence = result
        print(f"{class_name}: {confidence:.3f}")

# Or classify a full patch directly
patch = np.array(magnitude_spectra[-96:])  # shape: (96, 257)
class_id, class_name, confidence = classifier.classify_patch(patch)
```

### 2. Audio File Processing

```bash
# Process an audio file with default settings
python audio_classifier_demo.py /path/to/audio.wav

# Specify model paths
python audio_classifier_demo.py audio.wav \
    --model /path/to/yamnet_core.tflite \
    --class-map /path/to/yamnet_class_map.csv

# Limit duration and specify output directory
python audio_classifier_demo.py audio.wav \
    --duration 30.0 \
    --output-dir ./results

# Skip plotting (useful for batch processing)
python audio_classifier_demo.py audio.wav --no-plot
```

### 3. Run Tests

```bash
# Run all unit tests
python test_spectrum_classifier.py
```

## Example Output

```
Processing: example.wav
  Loaded: 480000 samples (30.00s) at 16000 Hz
  Processing 2995 frames...
    [  0.96s] Speech                         (conf: 0.824)
    [  1.92s] Speech                         (conf: 0.791)
    [  2.88s] Music                          (conf: 0.653)
    ...

Total predictions: 30

SUMMARY
======================================================================
Total predictions: 30
Time span: 0.96s to 29.76s

Top 5 detected classes:
  Speech                        :  18 ( 60.0%)
  Music                         :   8 ( 26.7%)
  Silence                       :   3 ( 10.0%)
  Inside, small room            :   1 (  3.3%)

Average confidence: 0.712
```

## API Reference

### YAMNetSpectrumClassifier

#### Methods

**`__init__(model_path, class_map_path)`**
- Initialize classifier with TFLite model and class map

**`add_frame(magnitude_spectrum)`**
- Add one 257-bin spectrum frame
- Returns `(class_id, class_name, confidence)` when ready, `None` otherwise
- Automatically manages buffer and triggers classification at 96 frames and every 48 frames thereafter

**`classify_patch(patch)`**
- Classify a full 96×257 patch immediately
- `patch`: numpy array of shape (96, 257)
- Returns `(class_id, class_name, confidence)`

**`reset()`**
- Clear frame buffer and reset state

**`get_class_name(class_id)`**
- Get human-readable class name from ID

**`spectrum_to_mel(magnitude_spectrum)`**
- Convert 257-bin magnitude spectrum to 64-bin mel-frequency spectrum

### compute_magnitude_spectrum

```python
def compute_magnitude_spectrum(audio_segment, window_size=512, hop_size=160)
```

Compute magnitude spectrum from audio samples.

- `audio_segment`: Audio samples (numpy array)
- `window_size`: FFT window size (default: 512)
- Returns: 257-bin magnitude spectrum

## Comparison with C++ Implementation

| Feature | C++ (ODAS) | Python (This) |
|---------|-----------|---------------|
| Input Format | 257-bin spectra | 257-bin spectra |
| Mel Filterbank | ✓ Custom | ✓ Custom (matching) |
| TFLite Inference | ✓ C API | ✓ Python API |
| Frame Buffering | ✓ 96 frames | ✓ 96 frames |
| Overlap Strategy | ✓ 48 frames (50%) | ✓ 48 frames (50%) |
| Classification Smoothing | ✓ Streak-based | ✗ Not implemented* |

*Smoothing can be added at application level using the same logic from mod_sst.c

## Integration with ODAS

This Python implementation can be used to:

1. **Test ODAS spectral output**: Verify that spectra from ODAS produce expected classifications
2. **Offline analysis**: Process recorded spectral data from ODAS logs
3. **Prototyping**: Test classification strategies before implementing in C++
4. **Debugging**: Compare Python and C++ outputs for validation

## Example: Processing ODAS Spectral Output

```python
import numpy as np
import json
from yamnet_spectrum_classifier import YAMNetSpectrumClassifier

# Initialize classifier
classifier = YAMNetSpectrumClassifier(model_path, class_map_path)

# Load spectral data from ODAS (e.g., from JSON logs)
with open('odas_spectra.json', 'r') as f:
    spectra_log = json.load(f)

# Process each track's spectra
for track_id, track_data in spectra_log.items():
    classifier.reset()
    
    for spectrum in track_data['spectra']:  # List of 257-bin arrays
        result = classifier.add_frame(np.array(spectrum))
        
        if result:
            class_id, class_name, confidence = result
            print(f"Track {track_id}: {class_name} ({confidence:.3f})")
```

## Visualization

The demo script generates two types of plots:

1. **Analysis Plot**: 
   - Audio waveform
   - Confidence over time
   - Top detected classes (bar chart)

2. **Timeline Plot**:
   - Color-coded classification timeline
   - Opacity indicates confidence
   - Shows temporal evolution of classifications

## Troubleshooting

**Issue**: Model file not found
```
Error: Model file not found: /path/to/yamnet_core.tflite
```
**Solution**: Ensure model files are in the expected location or specify correct paths

**Issue**: Wrong input shape
```
AssertionError: Expected shape (257,), got (256,)
```
**Solution**: Ensure FFT produces 257 bins (FFT_SIZE=512 → 257 output bins)

**Issue**: No predictions generated
```
No predictions to plot!
```
**Solution**: Audio must be at least 96 frames (0.96s at 16kHz) for first prediction

## License

Matches the license of ODAS and YAMNet components.

## References

- [YAMNet Paper](https://arxiv.org/abs/1912.01227)
- [ODAS - Open embeddeD Audition System](https://github.com/introlab/odas)
- [AudioSet](https://research.google.com/audioset/)
