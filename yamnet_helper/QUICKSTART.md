# YAMNet Blackbox Classifier - Quick Start Guide

## Overview

This module provides a simple blackbox interface to YAMNet audio classification. Just feed in magnitude spectra (257 bins per frame) and get audio class predictions.

## Installation

```bash
# Activate your virtual environment
source ~/.venv/bin/activate

# Install dependencies (if not already installed)
pip install numpy tensorflow librosa matplotlib
```

## Quick Start

### 1. Simple Frame-by-Frame Classification

```python
from simple_demo import SimpleYAMNetClassifier
import numpy as np

# Initialize classifier
classifier = SimpleYAMNetClassifier()

# Generate or load your spectra (257 bins each)
for spectrum in your_spectra:  # Each spectrum is np.array of shape (257,)
    result = classifier.predict(spectrum)
    
    if result:  # Classification ready every 96 frames (with 48-frame overlap)
        print(f"Detected: {result['class']} (confidence: {result['confidence']:.3f})")
```

### 2. Batch Classification (96 frames at once)

```python
# If you have a full batch of 96 frames already
spectra_batch = np.array(your_96_spectra)  # Shape: (96, 257)

result = classifier.predict_batch(spectra_batch)
print(f"Class: {result['class']}, Confidence: {result['confidence']:.3f}")
```

## Running the Demos

### Run All Demos
```bash
cd /home/azureuser/simulator/yamnet_helper
python simple_demo.py
```

### Test Basic Functionality
```bash
python test_spectrum_classifier.py
```

### Process Audio File with Visualization
```bash
# Process an audio file and generate plots
python audio_classifier_demo.py /path/to/audio.wav

# With options
python audio_classifier_demo.py audio.wav \
    --duration 30.0 \
    --output-dir ./my_results
```

## Input Format

**Magnitude Spectrum**: 257 float values (output of 512-point FFT)

```python
# From audio samples
audio_window = audio[start:start+512]  # 512 samples
windowed = audio_window * np.hanning(512)
spectrum = np.abs(np.fft.rfft(windowed, n=512))  # 257 bins
```

## Output Format

```python
{
    'class_id': 137,              # Integer class ID (0-520)
    'class': 'Music',             # Human-readable class name
    'confidence': 0.824           # Confidence score (0.0-1.0)
}
```

## Key Features

✅ **Simple API**: Just feed spectra, get predictions  
✅ **Automatic buffering**: Handles 96-frame accumulation internally  
✅ **50% overlap**: Predictions every 48 frames for temporal continuity  
✅ **521 classes**: Full AudioSet ontology  
✅ **Flexible**: Works with pre-computed spectra or live audio

## Architecture

```
Audio → FFT (512) → Magnitude (257 bins) → YAMNet → Class Prediction
         ↓                    ↓                        ↓
      Window            Accumulate               Every 96 frames
    (10ms hop)           96 frames              (48 frame overlap)
```

## Common Use Cases

### 1. Real-time Audio Classification
```python
classifier = SimpleYAMNetClassifier()

# In your audio processing loop
while True:
    audio_chunk = get_audio_chunk()  # Get 512 samples
    spectrum = compute_spectrum(audio_chunk)
    
    result = classifier.predict(spectrum)
    if result:
        handle_classification(result)
```

### 2. Batch Processing Files
```python
classifier = SimpleYAMNetClassifier()

for audio_file in audio_files:
    classifier.reset()  # Reset for each new file
    
    spectra = extract_spectra(audio_file)
    for spectrum in spectra:
        result = classifier.predict(spectrum)
        if result:
            log_result(audio_file, result)
```

### 3. Integration with ODAS
```python
# Process ODAS spectral output
classifier = SimpleYAMNetClassifier()

for track_id, track_data in odas_tracks.items():
    classifier.reset()
    
    for spectrum in track_data['spectra']:
        result = classifier.predict(spectrum)
        if result:
            print(f"Track {track_id}: {result['class']}")
```

## Performance Notes

- **First prediction**: After accumulating 96 frames (~0.96 seconds)
- **Subsequent predictions**: Every 48 frames (~0.48 seconds)
- **Latency**: ~1 second initial, then ~0.5 second updates
- **Throughput**: Can process thousands of frames per second (depending on CPU)

## File Structure

```
yamnet_helper/
├── __init__.py                      # Module initialization
├── yamnet_spectrum_classifier.py    # Pure Python implementation
├── yamnet_c_wrapper.py              # C++ wrapper (optional)
├── simple_demo.py                   # Simple blackbox demo ← START HERE
├── audio_classifier_demo.py         # Full audio processing demo
├── test_spectrum_classifier.py      # Unit tests
└── README.md                        # Full documentation
```

## Troubleshooting

**Q: No predictions appearing?**  
A: Need at least 96 frames. Check that you're feeding spectra continuously.

**Q: Wrong spectrum size?**  
A: Must be exactly 257 bins. Use 512-point FFT: `np.fft.rfft(signal, n=512)`

**Q: Want to use C++ implementation?**  
A: Compile the ODAS yamnet module to shared library and the wrapper will auto-detect it.

**Q: Memory usage growing?**  
A: Call `classifier.reset()` between different audio sources.

## Next Steps

- **Integrate with your pipeline**: Use `SimpleYAMNetClassifier` in your code
- **Visualize results**: Use `audio_classifier_demo.py` to see predictions over time
- **Customize**: Modify confidence thresholds or implement smoothing logic
- **Scale**: Process multiple tracks in parallel with separate classifier instances

## Support

For issues or questions:
1. Check the full README.md for detailed documentation
2. Run test_spectrum_classifier.py to verify installation
3. Try simple_demo.py to see expected behavior
