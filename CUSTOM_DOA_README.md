# Custom Direction-of-Arrival (DOA) Processor

A custom implementation of direction-of-arrival estimation for microphone array audio, designed to process rendered audio files and compare results with SODAS/ODAS output.

## Overview

This processor implements a phase-based DOA estimation algorithm using the following approach:

1. **Frame-based Processing**: 8ms frames (128 samples @ 16kHz) with 50% overlap
2. **FFT-based Peak Detection**: Identify frequency peaks above noise threshold
3. **Phase Difference Analysis**: Calculate direction using GCC-PHAT approach
4. **Confidence Window Validation**: Temporal validation across n frames
5. **Output Averaging**: Average frequency and direction over validated frames

## Algorithm Details

### 1. Input Processing
- Reads 6-channel raw PCM files (S16_LE format)
- Extracts 4 microphone channels (channels 2-5)
- Processes in 8ms frames with 4ms hop (50% overlap)

### 2. Frequency Domain Analysis
```
For each frame:
  - Apply Hann window
  - Compute FFT (256 points with zero-padding)
  - Identify peaks above SNR threshold
  - Filter by frequency range (200-6000 Hz)
```

### 3. Direction Estimation (GCC-PHAT)
```
For each peak:
  - Calculate phase differences between mic pairs:
    * Left-Right (Mic 1-3): X-axis component
    * Back-Front (Mic 2-4): Y-axis component
  - Convert phase to TDOA (Time Difference of Arrival)
  - Estimate azimuth angle (0°=front, 90°=right, 180°=back, 270°=left)
  - Calculate directional unit vector (x, y, z)
```

### 4. Confidence Window Validation
```
Window size: 8 frames (64ms)
For each detection:
  - Check if similar detection exists in previous frames
  - Similarity criteria:
    * Frequency within ±100 Hz
    * Direction within ±20 degrees
  - If present in ≥62.5% of frames (5/8):
    → Accept as valid detection
  - Else: Reject as spurious/noise
```

### 5. Output Generation
- Average frequency over validated frames
- Average direction (x, y, z) over validated frames
- Output confidence score (0-1)
- Frame count for each detection

## Microphone Array Geometry

ReSpeaker USB 4 Mic Array (planar configuration):

```
        Front (Mic 4)
             +Y
              |
              |
    Left -----+----- Right
   (Mic 1)    |    (Mic 3)
         -X       +X
              |
              |
        Back (Mic 2)
             -Y
```

- Mic 1 (Ch 2): [-0.032, 0.000, 0.000] m (Left)
- Mic 2 (Ch 3): [0.000, -0.032, 0.000] m (Back)
- Mic 3 (Ch 4): [0.032, 0.000, 0.000] m (Right)
- Mic 4 (Ch 5): [0.000, 0.032, 0.000] m (Front)

## Usage

### 1. Command Line Interface

Process a single raw audio file:

```bash
cd /home/azureuser/simulator
python test_custom_doa.py outputs/renders/your_audio.raw
```

This will:
- Process the audio file
- Generate a JSON output file (`your_audio_custom_doa.json`)
- Print summary statistics and detected sources

### 2. Python API

```python
from custom_doa_processor import process_audio_file, ProcessingConfig

# Use default configuration
results = process_audio_file('input.raw', 'output.json')

# Custom configuration
config = ProcessingConfig(
    frame_size=128,
    confidence_window_size=10,
    min_peak_snr=15.0,
    direction_tolerance=15.0
)
results = process_audio_file('input.raw', 'output.json', config)
```

### 3. Streamlit Web Interface

```bash
cd /home/azureuser/simulator
streamlit run app.py
```

Then navigate to "🔬 Custom DOA Processor" in the sidebar.

## Configuration Parameters

### Frame Processing
- `frame_size`: Frame size in samples (default: 128 = 8ms @ 16kHz)
- `hop_size`: Hop size in samples (default: 64 = 4ms, 50% overlap)
- `fft_size`: FFT size with zero-padding (default: 256)

### Peak Detection
- `min_peak_snr`: Minimum SNR in dB above noise floor (default: 12.0)
- `min_frequency`: Minimum frequency to analyze in Hz (default: 200)
- `max_frequency`: Maximum frequency to analyze in Hz (default: 6000)
- `peak_prominence`: Minimum peak prominence in dB (default: 3.0)

### Confidence Window
- `confidence_window_size`: Number of frames for validation (default: 8 = 64ms)
- `confidence_threshold`: Minimum ratio for validation (default: 0.625 = 62.5%)
- `direction_tolerance`: Angular tolerance in degrees (default: 20.0)
- `frequency_tolerance`: Frequency tolerance in Hz (default: 100.0)

## Output Format

The processor generates a JSON file with the following structure:

```json
{
  "metadata": {
    "file": "input.raw",
    "duration": 10.0,
    "frames_processed": 2500,
    "total_peaks_detected": 5000,
    "total_peaks_validated": 1200,
    "validation_rate": 0.24
  },
  "config": {
    "frame_size": 128,
    "confidence_window": 8,
    ...
  },
  "frames": [
    {
      "frame": 0,
      "time": 0.0,
      "detections": [
        {
          "frequency": 1234.5,
          "azimuth": 45.0,
          "elevation": 0.0,
          "x": 0.707,
          "y": 0.707,
          "z": 0.0,
          "energy": -20.5,
          "confidence": 0.875,
          "frame_count": 7
        }
      ]
    }
  ],
  "summary": {
    "total_detections": 1200,
    "unique_sources": 5,
    "sources": [
      {
        "frequency": 1234.5,
        "azimuth": 45.0,
        "occurrence_count": 450,
        "avg_confidence": 0.82
      }
    ]
  }
}
```

## Comparison with SODAS

Key differences:

| Feature | Custom DOA | SODAS/ODAS |
|---------|-----------|------------|
| Algorithm | GCC-PHAT phase difference | SRP-PHAT beamforming |
| Processing | Frame-by-frame (8ms) | Configurable |
| Validation | Confidence window | Kalman filter tracking |
| Output | All validated peaks | Tracked sources |
| Frequency Info | Per detection | Not in standard output |
| Elevation | Assumed 0° (planar array) | Full 3D if available |

### When to Use Custom DOA

✅ **Use Custom DOA when:**
- Need detailed frequency information per detection
- Want to understand frame-by-frame detections
- Need to tune detection sensitivity
- Debugging microphone array issues
- Comparing different processing approaches

⚠️ **Use SODAS/ODAS when:**
- Need production-ready tracking
- Require smooth trajectories (Kalman filtering)
- Need real-time performance
- Want standard SST output format

## Performance Characteristics

### Computational Complexity
- Frame processing: O(N log N) for FFT
- Peak detection: O(N) per frame
- Confidence validation: O(W × P) where W=window size, P=peaks per frame

### Typical Performance
- Processing speed: ~50-100x real-time on modern CPU
- Memory usage: ~100MB for 10s audio file
- Output size: ~1-10 MB per minute of audio (depends on detections)

## Tuning Guide

### Increase Sensitivity (More Detections)
- Decrease `min_peak_snr` (e.g., 10 dB)
- Decrease `confidence_threshold` (e.g., 0.5)
- Increase `direction_tolerance` (e.g., 30°)

### Increase Specificity (Fewer False Positives)
- Increase `min_peak_snr` (e.g., 15 dB)
- Increase `confidence_threshold` (e.g., 0.75)
- Increase `confidence_window_size` (e.g., 12 frames)
- Decrease `direction_tolerance` (e.g., 15°)

### Improve Frequency Resolution
- Increase `fft_size` (e.g., 512 or 1024)
- Note: Increases processing time

### Improve Temporal Resolution
- Decrease `hop_size` (e.g., 32 samples = 2ms)
- Note: Increases frame count and processing time

## Limitations

1. **Planar Array**: Cannot accurately estimate elevation (assumes horizontal plane)
2. **Phase Wrapping**: May fail for very high frequencies (>8kHz) with this mic spacing
3. **Near-Field**: Assumes far-field sources (>1m from array)
4. **Single Source per Frequency**: Cannot separate multiple sources at same frequency
5. **No Tracking**: Does not maintain source identity across time (unlike Kalman filters)

## Future Enhancements

Potential improvements:

1. **Kalman Filter Tracking**: Add source tracking for smooth trajectories
2. **MUSIC/ESPRIT**: Higher-resolution DOA estimation
3. **Multi-source Separation**: Handle multiple sources per frequency bin
4. **Adaptive Thresholds**: Automatic SNR and confidence tuning
5. **GPU Acceleration**: Parallel processing for real-time performance

## Files

- `custom_doa_processor.py`: Core processing logic
- `custom_simulator.py`: Streamlit UI integration
- `test_custom_doa.py`: Command-line testing tool
- `CUSTOM_DOA_README.md`: This documentation

## References

- GCC-PHAT: Knapp & Carter (1976) "The Generalized Correlation Method for Estimation of Time Delay"
- SRP-PHAT: DiBiase et al. (2001) "Robust Localization in Reverberant Rooms"
- Microphone Array Processing: Benesty et al. "Springer Handbook of Speech Processing"

## Support

For issues or questions:
1. Check configuration parameters
2. Review output JSON for validation rates
3. Compare with SODAS output using the comparison interface
4. Adjust thresholds based on your specific use case
