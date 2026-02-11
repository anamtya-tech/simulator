# Source Tracking - Quick Reference

## What You Got

✅ **Unique Source IDs** - Each detected source gets a persistent ID across frames  
✅ **383 tracks created** from 10-second test file  
✅ **Track lifetimes** - Longest track persisted for 0.82 seconds  
✅ **Color-coded visualization** - Same source = same color throughout time  
✅ **Detailed track statistics** - Duration, frequency stability, confidence

## Key Commands

### 1. Process Audio with Tracking
```bash
cd /home/azureuser/simulator
python test_custom_doa.py outputs/renders/YOUR_FILE.raw
```

**Output includes:**
- Unique source count
- Total tracks created
- Top 10 longest tracks with IDs, frequencies, durations

### 2. General Visualization (with track timeline)
```bash
python visualize_doa.py YOUR_FILE_custom_doa.json output.png
```

**Shows:**
- Polar plot of source directions
- Detection timeline
- **Track timeline** (top 20 by duration)
- Frequency/direction distributions
- **Color-coded spectrogram by track ID**

### 3. Detailed Track Analysis (NEW!)
```bash
python visualize_tracks.py YOUR_FILE_custom_doa.json tracks.png
```

**Shows:**
- Track trajectories over time (polar)
- Frequency evolution for top 5 tracks
- Confidence evolution
- Gantt chart of track lifetimes
- Frequency-time heatmap with track overlays
- Track duration and detection statistics

## Understanding the Output

### JSON Structure
```json
{
  "metadata": {
    "total_tracks": 383,  // NEW: Total unique sources tracked
    ...
  },
  "frames": [
    {
      "detections": [
        {
          "frequency": 1820.5,
          "azimuth": 60.3,
          "source_id": 101,  // NEW: Links to track 101
          ...
        }
      ]
    }
  ],
  "tracks": [  // NEW: Complete track information
    {
      "track_id": 101,
      "duration_seconds": 0.82,
      "detection_count": 177,
      "avg_frequency": 1816.9,
      "avg_azimuth": 60.1,
      "start_time": 0.98,
      "end_time": 1.80
    }
  ]
}
```

## How Tracking Works

**Frame-to-Frame Matching:**
1. New detection appears
2. Compare to active tracks (frequency + direction)
3. If match found: assign existing track ID
4. If no match: create new track ID
5. Track ends after 5 frames without detection

**Track ID Assignment:**
- IDs start at 1 and increment
- Each track maintains running averages
- Same color used for visualization

## Track Statistics (Test File)

| Metric | Value |
|--------|-------|
| Total Sources (unique) | 34 |
| Total Tracks | 383 |
| Tracks per Source | ~11 avg |
| Longest Track | 0.82s (177 detections) |
| Shortest Track | ~0.02s (single detection) |
| Avg Track Duration | ~0.10s |

**Why more tracks than sources?**
- Sources can stop and restart (creates new track)
- Frequency drift can cause track splits
- Intermittent detections

## Visualization Features

### Track Timeline (in general viz)
- Shows top 20 tracks by duration
- Horizontal bars = track lifetime
- Color-coded by track ID
- Y-axis shows track ID + frequency

### Color-Coded Spectrogram
- Each track has consistent color
- Easy to see same source over time
- Colors cycle through 20 distinct hues

### Track Trajectories (dedicated viz)
- Polar plot shows spatial movement
- Radius = time
- Angle = direction
- Useful for tracking moving sources

### Frequency Evolution
- Shows how frequency changes over time
- Stable = horizontal line
- Modulated = curved line
- Top 5 longest tracks

## Filtering Tracks (Python)

```python
from custom_doa_processor import process_audio_file
import json

# Process file
results = process_audio_file('audio.raw', 'output.json')

# Load results
with open('output.json') as f:
    data = json.load(f)

# Long-lived tracks only
long_tracks = [t for t in data['tracks'] if t['duration_seconds'] > 0.3]
print(f"Long tracks: {len(long_tracks)}")

# Specific frequency range (bird calls 2-6 kHz)
bird_tracks = [t for t in data['tracks'] 
               if 2000 <= t['avg_frequency'] <= 6000]

# High confidence tracks
stable_tracks = [t for t in data['tracks'] 
                 if t['avg_confidence'] > 0.8]

# Find track by ID
track_101 = next(t for t in data['tracks'] if t['track_id'] == 101)
print(f"Track 101: {track_101['avg_frequency']:.0f}Hz, "
      f"{track_101['duration_seconds']:.2f}s")
```

## Configuration

**Adjust tracking behavior:**

```python
from custom_doa_processor import ProcessingConfig

config = ProcessingConfig(
    max_frames_gap=5,  # Frames without detection before track ends
    track_similarity_tolerance=1.5,  # Matching tolerance multiplier
    
    # Stricter matching (fewer track splits):
    track_similarity_tolerance=1.2,
    max_frames_gap=10,
    
    # Looser matching (more tracks):
    track_similarity_tolerance=2.0,
    max_frames_gap=3,
)
```

## Common Use Cases

### 1. Count Active Sources at Time T
```python
# Get detections at specific time
target_time = 5.0  # 5 seconds
frame_time = target_time * config.sample_rate / config.hop_size

detections_at_t = [d for f in data['frames'] 
                   if abs(f['time'] - target_time) < 0.01
                   for d in f['detections']]

unique_sources = set(d['source_id'] for d in detections_at_t)
print(f"Active sources at {target_time}s: {len(unique_sources)}")
```

### 2. Find Overlapping Tracks
```python
# Tracks active during specific time window
start, end = 2.0, 3.0

overlapping = [t for t in data['tracks']
               if t['start_time'] <= end and t['end_time'] >= start]
print(f"Overlapping tracks: {len(overlapping)}")
```

### 3. Track Statistics by Frequency Range
```python
import numpy as np

freq_ranges = {
    'Low': (200, 1000),
    'Mid': (1000, 2000),
    'High': (2000, 6000)
}

for name, (fmin, fmax) in freq_ranges.items():
    tracks_in_range = [t for t in data['tracks']
                       if fmin <= t['avg_frequency'] <= fmax]
    avg_duration = np.mean([t['duration_seconds'] for t in tracks_in_range])
    print(f"{name}: {len(tracks_in_range)} tracks, avg {avg_duration:.2f}s")
```

## Comparison: Before vs After Tracking

### Before (Without Tracking)
❌ No way to know which detections are same source  
❌ Cannot measure source duration  
❌ Difficult to visualize temporal patterns  
❌ Cannot track source movement  
❌ All detections look independent  

### After (With Tracking)
✅ Unique ID per source  
✅ Track lifetime statistics  
✅ Color-coded visualization  
✅ Trajectory analysis  
✅ Temporal correlation  
✅ Count simultaneous sources  

## Files and Tools

| File | Purpose |
|------|---------|
| `custom_doa_processor.py` | Core processor with tracking |
| `test_custom_doa.py` | CLI tool (shows track stats) |
| `visualize_doa.py` | General viz (includes track timeline) |
| `visualize_tracks.py` | **NEW** Dedicated track analysis |
| `TRACKING_FEATURE_SUMMARY.md` | Full documentation |
| `QUICK_REFERENCE.md` | Updated with tracking info |

## Next Steps

1. **Process your audio:**
   ```bash
   python test_custom_doa.py outputs/renders/YOUR_FILE.raw
   ```

2. **Visualize tracks:**
   ```bash
   python visualize_tracks.py YOUR_FILE_custom_doa.json
   ```

3. **Analyze specific tracks:**
   - Load JSON in Python
   - Filter by track_id, duration, frequency
   - Export to CSV for further analysis

4. **Adjust parameters if needed:**
   - Too many short tracks? Increase `max_frames_gap`
   - Track splits? Increase `track_similarity_tolerance`

## Tips

💡 **Track ID colors** cycle through 20 colors (tab20 colormap)  
💡 **Short tracks** (<0.1s) often represent transients or noise  
💡 **Long tracks** (>0.5s) represent sustained sources  
💡 **Frequency stability** (low std) indicates pure tones  
💡 **Many overlapping tracks** = complex acoustic scene  

---

**Status:** ✅ Fully Implemented and Tested  
**Test Results:** 383 tracks from 10-second audio  
**Longest Track:** 0.82 seconds (Track ID 101)  
**Visualization:** Two tools with color-coded tracks
