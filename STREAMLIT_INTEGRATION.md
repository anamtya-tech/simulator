# Streamlit App Integration - Classifier Output

## Overview
The Streamlit app (`/home/azureuser/simulator/app.py`) has been updated to fully integrate YAMNet classifier output from ODAS into the Results Analyzer module.

## Changes Made

### 1. Analyzer Module Updates (`analyzer.py`)

#### Detection Parsing Enhanced
The `_parse_odas_output()` method now extracts YAMNet classification data from each detection:

```python
detection = {
    # ... existing fields ...
    'class_id': src.get('class_id', -1),
    'class_name': src.get('class_name', 'unclassified'),
    'class_confidence': src.get('class_confidence', 0.0),
    'class_timestamp': src.get('class_timestamp', 0),
    'track_id': src.get('id', 0),
    'track_tag': src.get('tag', ''),
    'track_type': src.get('type', '')
}
```

#### Display Summary Enhanced
Added a new "🎯 YAMNet Classification Statistics" section showing:
- Classified detections count
- Average classification confidence
- Unique classes detected
- Classification distribution table
- Ground truth vs YAMNet predictions comparison

#### HTML Report Enhanced
Added YAMNet classification section to interactive reports with:
- Overall classification statistics
- Classification distribution table with percentages
- Per-class confidence metrics

### 2. Python Parser (`odas_classifier_parser.py`)
Created a utility module for parsing classifier output:
- `OdasClassifierParser` class for reading JSON files
- `TrackClassification` dataclass for structured data
- Filter methods for class name and confidence
- Session summary generator

### 3. Test Suite (`test_classifier_integration.py`)
Created comprehensive test script to verify:
- Parser can read classifier output
- All required fields are present
- Analyzer integration works correctly
- Classification data is properly structured

## Usage Workflow

### 1. Run Simulation
```bash
streamlit run /home/azureuser/simulator/app.py
```

Navigate to **⚙️ ODAS Simulator** and run a simulation.

### 2. View Results
Navigate to **📊 Results Analyzer**:

1. Select the run from the dropdown
2. Click "🔍 Analyze Run"
3. View the enhanced statistics including:
   - Traditional detection metrics
   - **NEW: YAMNet Classification Statistics**
   - Ground truth vs predicted class comparison
4. Click "🔍 Open Report (Full Page)" to see interactive HTML report

### 3. What You'll See

#### Summary Statistics (Enhanced)
```
📊 Analysis Summary
├── Total Detections: 150
├── Match Rate: 85.3%
├── Avg Error: 3.45°
└── Avg Confidence: 0.876

🎯 YAMNet Classification Statistics
├── Classified Detections: 142
├── Avg Classification Confidence: 0.891
└── Unique Classes Detected: 5

Classification Distribution:
┌──────────────┬───────┬──────────┬─────────────┐
│ Class        │ Count │ Avg Conf │ Min/Max     │
├──────────────┼───────┼──────────┼─────────────┤
│ Speech       │ 78    │ 0.923    │ 0.812/0.987 │
│ Music        │ 34    │ 0.856    │ 0.732/0.945 │
│ Dog          │ 18    │ 0.887    │ 0.798/0.956 │
│ Bird         │ 8     │ 0.834    │ 0.723/0.912 │
│ Traffic      │ 4     │ 0.792    │ 0.702/0.865 │
└──────────────┴───────┴──────────┴─────────────┘
```

#### Ground Truth vs YAMNet Comparison
Shows time-aligned comparison of:
- Ground truth labels from scene config
- YAMNet predicted class
- Classification confidence
- Angular error

#### Interactive HTML Report
Includes new section with:
- Classification statistics cards
- Distribution table with percentages
- Per-class confidence metrics

## Testing

Run the test suite to verify integration:

```bash
python3 /home/azureuser/simulator/test_classifier_integration.py
```

Expected output:
```
🧪 ODAS Classifier Output Test Suite

============================================================
Testing ODAS Classifier Output Parser
============================================================

✅ Found session file: sst_session_live.json_20260209_122345.json
✅ Parsed 1250 frames
✅ Tracks in first frame: 3

📊 Sample Track Data:
   Track ID: 1
   Direction: (0.456, 0.234, 0.123)
   YAMNet Class Name: Speech
   YAMNet Confidence: 0.923
   ✅ All required fields present

📈 Session Summary:
   Total frames: 1250
   Unique tracks: 4
   
🎯 Classification Distribution:
   Speech: 892
   Music: 234
   Dog: 89
   Bird: 35

============================================================
Test Summary
============================================================
Parser Test: ✅ PASS
Analyzer Integration: ✅ PASS

✅ All tests passed!
```

## File Structure

```
simulator/
├── app.py                           # Main Streamlit app (navigation)
├── analyzer.py                      # ✅ UPDATED - Added YAMNet stats
├── odas_classifier_parser.py        # ✅ NEW - Parser utility
├── test_classifier_integration.py   # ✅ NEW - Test suite
├── simulator.py                     # Simulation runner (references classifier files)
└── outputs/
    ├── runs/                        # Run metadata files
    └── analysis/                    # Analysis results with YAMNet data
        ├── {run_id}_analysis.json   # ✅ Contains classification data
        ├── {run_id}_report.html     # ✅ Interactive report with YAMNet section
        └── {run_id}_dataset.csv     # Training dataset

z_odas_newbeamform/
└── src/module/mod_sst.c            # ✅ UPDATED - Exports classification data

sodas/
└── local_socket.cfg                # ✅ UPDATED - enable_classifier_output = "enabled"
```

## Configuration Check

Ensure classifier output is enabled:

```bash
grep "enable_classifier_output" /home/azureuser/sodas/local_socket.cfg
```

Should show:
```
enable_classifier_output = "enabled";
```

## Output Files

When a simulation runs with `enable_classifier_output = "enabled"`:

1. **Session Live File** (`/home/chatak/z_odas/ClassifierLogs/sst_session_live.json_*.json`)
   - One line per frame (8ms intervals)
   - Each line contains timestamp, detections with direction, and YAMNet classification

2. **Fingerprint File** (`/home/chatak/z_odas/ClassifierLogs/sst_session_live_*.json`)
   - Simplified spectral fingerprints

The analyzer automatically reads these files and displays:
- Detection metrics
- Classification accuracy
- Ground truth comparison
- Interactive visualizations

## Benefits

### For Analysis
- See which classes are being detected
- Validate YAMNet predictions against ground truth
- Identify misclassifications for model improvement
- Track classification confidence over time

### For Training
- High-confidence matches saved to dataset automatically
- Ground truth + YAMNet prediction for validation
- Filter by confidence threshold
- Identify samples needing manual review

### For Debugging
- Verify YAMNet is classifying correctly
- Compare predicted vs expected classes
- Track classification performance per source type
- Identify problem areas (low confidence, mismatches)

## Troubleshooting

### No Classification Data Shown
1. Check config: `enable_classifier_output = "enabled"`
2. Rebuild ODAS: `cd /home/azureuser/z_odas_newbeamform/build && make`
3. Verify models exist: `ls /home/azureuser/z_odas_newbeamform/models/`
4. Run test: `python3 /home/azureuser/simulator/test_classifier_integration.py`

### Classification Shows "unclassified"
- Tracks need 96 frames before classification
- Short simulations may not accumulate enough frames
- Check track duration and frame counts

### Parser Errors
- Check JSON file format with: `head -1 /home/chatak/z_odas/ClassifierLogs/sst_session_live.json_*.json | python3 -m json.tool`
- Verify file permissions
- Check disk space

## Next Steps

1. **Run a simulation** to generate new data with classification
2. **Analyze results** to see YAMNet performance
3. **Compare predictions** against ground truth labels
4. **Curate dataset** using high-confidence matches
5. **Retrain models** if needed based on misclassifications

## Implementation Complete

✅ Configuration flag added
✅ JSON output enhanced with classification
✅ Analyzer updated to parse classification
✅ Display enhanced with YAMNet statistics
✅ HTML reports include classification section
✅ Python parser utility created
✅ Test suite implemented
✅ Documentation complete

The system is now ready to use classifier output in the analysis workflow!
