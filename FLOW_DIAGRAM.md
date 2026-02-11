# Visual Flow: ODAS Classifier to Streamlit Results

```
┌─────────────────────────────────────────────────────────────────┐
│                    ODAS PROCESSING PIPELINE                      │
└─────────────────────────────────────────────────────────────────┘

1. AUDIO INPUT
   └─> Raw audio from simulation
       └─> /home/azureuser/simulator/outputs/renders/*.raw

2. ODAS PROCESSING (z_odas_newbeamform)
   └─> mod_sst.c processes audio
       ├─> Tracks sound sources
       ├─> Accumulates 96 frames of spectral data per track
       └─> YAMNet classifies each 96-frame patch
           └─> Returns: class_name, class_id, confidence

3. JSON OUTPUT (if enable_classifier_output = "enabled")
   └─> /home/chatak/z_odas/ClassifierLogs/sst_session_live.json_*.json
       └─> Each line = 1 frame (8ms)
           └─> {
                 "timeStamp": 123456789,
                 "src": [{
                   "id": 1,
                   "x": 0.5, "y": 0.3, "z": 0.1,    ← Direction
                   "activity": 0.85,
                   "class_id": 42,                   ← YAMNet
                   "class_name": "Speech",           ← YAMNet
                   "class_confidence": 0.923,        ← YAMNet
                   "class_timestamp": 123456789,     ← YAMNet
                   "bins": [...],
                   "fingerprint": [...]
                 }]
               }

4. SIMULATION RUN (simulator.py)
   └─> Creates run file
       └─> /home/azureuser/simulator/outputs/runs/{run_id}.json
           └─> {
                 "run_id": "...",
                 "scene_name": "...",
                 "session_live_file": "/home/chatak/.../sst_session_live.json_*.json",
                 "scene_file": "...",
                 ...
               }

5. STREAMLIT APP (app.py)
   ┌────────────────────────────────────────────────┐
   │         Navigation: 📊 Results Analyzer        │
   └────────────────────────────────────────────────┘
   
   ┌─> analyzer.py
   │   └─> render()
   │       ├─> Load run file
   │       ├─> _analyze_run()
   │       │   ├─> _parse_odas_output()  ← READS JSON
   │       │   │   └─> Extracts:
   │       │   │       ├─> x, y, z (direction)
   │       │   │       ├─> class_id
   │       │   │       ├─> class_name        ✅ NEW
   │       │   │       ├─> class_confidence  ✅ NEW
   │       │   │       └─> class_timestamp   ✅ NEW
   │       │   │
   │       │   └─> Match to ground truth sources
   │       │
   │       ├─> _display_summary()  ← SHOWS STATS
   │       │   ├─> Detection metrics
   │       │   └─> 🎯 YAMNet Classification Statistics  ✅ NEW
   │       │       ├─> Classified detections count
   │       │       ├─> Avg classification confidence
   │       │       ├─> Unique classes detected
   │       │       ├─> Classification distribution table
   │       │       └─> Ground truth vs YAMNet comparison
   │       │
   │       └─> _generate_html_report()  ← INTERACTIVE REPORT
   │           └─> Includes YAMNet section with:  ✅ NEW
   │               ├─> Classification stats cards
   │               ├─> Distribution table
   │               └─> Per-class confidence metrics
   │
   └─> User sees:
       ┌────────────────────────────────────────┐
       │ 📊 Analysis Summary                    │
       ├────────────────────────────────────────┤
       │ Total Detections: 150                  │
       │ Match Rate: 85.3%                      │
       │ Avg Error: 3.45°                       │
       └────────────────────────────────────────┘
       
       ┌────────────────────────────────────────┐
       │ 🎯 YAMNet Classification Statistics    │ ✅ NEW
       ├────────────────────────────────────────┤
       │ Classified Detections: 142             │
       │ Avg Classification Confidence: 0.891   │
       │ Unique Classes Detected: 5             │
       │                                        │
       │ Classification Distribution:           │
       │ ┌────────┬───────┬─────────┐          │
       │ │ Class  │ Count │ Avg Conf│          │
       │ ├────────┼───────┼─────────┤          │
       │ │ Speech │  78   │  0.923  │          │
       │ │ Music  │  34   │  0.856  │          │
       │ │ Dog    │  18   │  0.887  │          │
       │ └────────┴───────┴─────────┘          │
       │                                        │
       │ Ground Truth vs YAMNet Predictions:    │
       │ ┌───────┬────────────┬──────────┐     │
       │ │ Time  │ Ground T.  │ YAMNet   │     │
       │ ├───────┼────────────┼──────────┤     │
       │ │ 0.50s │ Speech     │ Speech   │ ✓   │
       │ │ 1.23s │ Music      │ Music    │ ✓   │
       │ │ 2.45s │ Dog        │ Animal   │ ≈   │
       │ └───────┴────────────┴──────────┘     │
       └────────────────────────────────────────┘
       
       [🔍 Open Report (Full Page)]  ← Interactive HTML
```

## Data Flow Summary

```
Audio → ODAS → YAMNet → JSON → Simulator → Analyzer → Streamlit UI
  ↓       ↓       ↓       ↓        ↓          ↓           ↓
Raw    Track  Classify Write   Create     Parse      Display
       +      +        to       Run        +          Stats
       SST    96       File     File       Match      +
              frames                                  Report
```

## Key Files Modified

```
z_odas_newbeamform/
├── include/odas/module/mod_sst.h
│   └─> Added: enable_classifier_output flag
├── src/module/mod_sst.c
│   ├─> Enhanced: dump_track_buffers_to_json()
│   │   └─> Now includes: class_id, class_name, class_confidence, class_timestamp
│   └─> Uncommented: JSON output calls (conditional on flag)
└── demo/odaslive/parameters.c
    └─> Added: config parsing for enable_classifier_output

sodas/
└── local_socket.cfg
    └─> Added: enable_classifier_output = "enabled"

simulator/
├── analyzer.py                      ✅ UPDATED
│   ├─> _parse_odas_output()        ← Extracts classification data
│   ├─> _display_summary()          ← Shows YAMNet statistics
│   └─> _generate_html_report()     ← Includes classification section
├── odas_classifier_parser.py       ✅ NEW
│   └─> Utility for parsing classifier output
├── test_classifier_integration.py  ✅ NEW
│   └─> Tests parser and analyzer integration
└── STREAMLIT_INTEGRATION.md        ✅ NEW
    └─> Complete documentation
```

## Expected User Experience

1. **Configure**: Set `enable_classifier_output = "enabled"` in config
2. **Build**: Run `make` in ODAS build directory
3. **Run Simulation**: Use "⚙️ ODAS Simulator" in Streamlit
4. **Analyze**: Navigate to "📊 Results Analyzer"
5. **View Results**: See detection metrics + YAMNet classification stats
6. **Download**: Get HTML report with interactive plots + classification data

## Before vs After

### Before
```
📊 Results Analyzer
├─> Detection metrics (angular error, match rate)
├─> Per-source statistics
├─> Timeline visualization
└─> 3D spatial plots
```

### After ✅
```
📊 Results Analyzer
├─> Detection metrics (angular error, match rate)
├─> Per-source statistics
├─> 🎯 YAMNet Classification Statistics        ← NEW!
│   ├─> Classified detections count
│   ├─> Classification confidence
│   ├─> Class distribution table
│   └─> Ground truth vs predicted comparison
├─> Timeline visualization (with classification)
└─> 3D spatial plots (with classification)
```

## Integration Complete ✅

All components are connected and working together to provide:
- Real-time YAMNet classification in ODAS
- Structured JSON output with classification data
- Parser utilities for easy data access
- Enhanced Streamlit UI with classification statistics
- Interactive HTML reports with classification insights
- Test suite for validation

Ready to use!
