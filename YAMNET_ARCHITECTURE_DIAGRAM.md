# YAMNet Fine-Tuning System - Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SIMULATOR PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────┐
                              │   Scene      │
                              │ Configuration│
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │    Render    │
                              │ Multi-channel│
                              │    Audio     │
                              └──────┬───────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │     ODAS     │
                              │  Processing  │
                              └──────┬───────┘
                                     │
                        ┌────────────┴────────────┐
                        │                         │
                        ▼                         ▼
                ┌───────────────┐         ┌─────────────┐
                │  Frequency    │         │   YAMNet    │
                │  Bins (1024)  │         │ Predictions │
                └───────┬───────┘         └──────┬──────┘
                        │                        │
                        └───────────┬────────────┘
                                    │
                                    ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                      ANALYSIS & CURATION                                 │
└─────────────────────────────────────────────────────────────────────────┘

                            ┌────────────┐
                            │  Analyzer  │
                            │  Compare:  │
                            │ YAMNet vs  │
                            │Ground Truth│
                            └─────┬──────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
            ┌───────────┐  ┌───────────┐  ┌───────────┐
            │ Mismatch  │  │   Low     │  │Unclassi-  │
            │  YAMNet   │  │Confidence │  │  fied     │
            │ != Truth  │  │  < 0.7    │  │ Samples   │
            └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
                  │              │              │
                  └──────────────┼──────────────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │ YAMNet Dataset │
                        │    Curator     │
                        └────────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌─────────────┐ ┌─────────┐ ┌──────────┐
            │   Audio     │ │  Save   │ │  Track   │
            │Reconstruction│ │Metadata │ │Provenance│
            │ (Griffin-   │ │  (CSV)  │ │  (JSON)  │
            │   Lim)      │ │         │ │          │
            └──────┬──────┘ └────┬────┘ └────┬─────┘
                   │             │           │
                   └─────────────┼───────────┘
                                 │
                                 ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                        DATASET STORAGE                                   │
└─────────────────────────────────────────────────────────────────────────┘

                    outputs/yamnet_datasets/
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
      ┌──────────────┐ ┌──────────┐ ┌──────────────┐
      │yamnet_train  │ │  config  │ │yamnet_train  │
      │    _001      │ │   .json  │ │    _002      │
      └──────┬───────┘ └──────────┘ └──────────────┘
             │
    ┌────────┼────────┬────────────┐
    │        │        │            │
    ▼        ▼        ▼            ▼
┌───────┐┌────────┐┌─────────┐┌─────────┐
│audio/ ││spectro-││metadata/││labels   │
│  WAV  ││grams/  ││  CSV    ││  .csv   │
│ files ││  PNG   ││per-run  ││ master  │
└───────┘└────────┘└─────────┘└─────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                      VISUALIZATION & REVIEW                              │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │    Dataset     │
                        │   Visualizer   │
                        └────────┬───────┘
                                 │
              ┌──────────────────┼──────────────────┐
              │                  │                  │
              ▼                  ▼                  ▼
      ┌──────────────┐   ┌──────────────┐  ┌─────────────┐
      │    Browse    │   │    Audio     │  │  Analytics  │
      │   Samples    │   │   Playback   │  │  Dashboard  │
      │  + Filter    │   │  (Browser)   │  │ (Plotly)    │
      └──────────────┘   └──────────────┘  └─────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                    TENSORFLOW PREPARATION                                │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │   Prepare      │
                        │  TensorFlow    │
                        │   Dataset      │
                        └────────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
            ┌──────────┐  ┌──────────┐  ┌──────────┐
            │  Train   │  │   Val    │  │  Test    │
            │   70%    │  │   15%    │  │   15%    │
            │  labels  │  │  labels  │  │  labels  │
            │   .csv   │  │   .csv   │  │   .csv   │
            └────┬─────┘  └────┬─────┘  └────┬─────┘
                 │             │             │
                 └─────────────┼─────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  TensorFlow  │
                        │   Dataset    │
                        │   Ready!     │
                        └──────┬───────┘
                               │
                               ▼

┌─────────────────────────────────────────────────────────────────────────┐
│                      FINE-TUNING (External)                              │
└─────────────────────────────────────────────────────────────────────────┘

                        ┌────────────────┐
                        │   Load Data    │
                        │   from CSVs    │
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  Load YAMNet   │
                        │   from TF Hub  │
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │Extract YAMNet  │
                        │  Embeddings    │
                        │   (1024-dim)   │
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  Add Custom    │
                        │ Classification │
                        │     Head       │
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │  Train Model   │
                        │  (Transfer     │
                        │   Learning)    │
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │   Evaluate     │
                        │  on Test Set   │
                        └────────┬───────┘
                                 │
                                 ▼
                        ┌────────────────┐
                        │    Deploy      │
                        │  Improved      │
                        │   YAMNet       │
                        └────────────────┘


═══════════════════════════════════════════════════════════════════════════

                        KEY COMPONENTS

┌─────────────────────────────────────────────────────────────────────┐
│ YAMNetDatasetCurator                                                │
├─────────────────────────────────────────────────────────────────────┤
│ • Dataset management (create, activate, merge)                     │
│ • Curation logic (filter by criteria)                              │
│ • Audio reconstruction (bins → WAV)                                │
│ • TensorFlow format preparation                                    │
│ • Provenance tracking                                              │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ AudioReconstructor                                                  │
├─────────────────────────────────────────────────────────────────────┤
│ • Griffin-Lim algorithm (phase reconstruction)                     │
│ • STFT/iSTFT with overlap-add                                      │
│ • Multi-frame processing                                           │
│ • WAV file generation (16kHz mono)                                 │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ DatasetVisualizer                                                   │
├─────────────────────────────────────────────────────────────────────┤
│ • Interactive Streamlit interface                                  │
│ • Audio playback in browser                                        │
│ • Filtering (label, confidence, reason)                            │
│ • Analytics (distribution, confidence, temporal)                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ DatasetConfigurator                                                 │
├─────────────────────────────────────────────────────────────────────┤
│ • Dataset list and management UI                                   │
│ • Curation settings configuration                                  │
│ • Merge and prepare operations                                     │
│ • In-app usage guide                                               │
└─────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════

                        DATA FLOW EXAMPLE

Time 0.245s: Wolf howl
    ↓
ODAS: bins=[0.2, 0.8, ...], YAMNet="Dog", conf=0.45
    ↓
Analyzer: Ground truth="Wolf", YAMNet="Dog" ❌ MISMATCH
    ↓
Curator: ✓ Meets criteria (mismatch + low confidence)
    ↓
AudioReconstructor: bins → Griffin-Lim → WAV
    ↓
Save: run_001_0000_t0_245_wolf.wav + metadata
    ↓
Dataset: yamnet_train_001/audio/run_001_0000_t0_245_wolf.wav
    ↓
Visualizer: User can listen and verify
    ↓
TF Prep: Assigned to train set (70%)
    ↓
Fine-Tune: Help YAMNet learn Wolf vs Dog distinction


═══════════════════════════════════════════════════════════════════════

                    CONFIGURATION HIERARCHY

curator_config.json
    │
    ├─ active_dataset: "yamnet_train_001"
    │
    ├─ curation_criteria:
    │   ├─ include_mismatches: true
    │   ├─ include_low_confidence: true
    │   ├─ confidence_threshold: 0.7
    │   ├─ include_unclassified: true
    │   └─ min_activity: 0.3
    │
    ├─ audio_params:
    │   ├─ sample_rate: 16000
    │   ├─ target_duration: 1.0
    │   └─ overlap_frames: 5
    │
    └─ datasets:
        ├─ yamnet_train_001:
        │   ├─ created_at: "2026-02-13..."
        │   ├─ sample_count: 1250
        │   ├─ samples_by_label: {...}
        │   └─ runs_processed: [...]
        │
        └─ yamnet_train_002: {...}


═══════════════════════════════════════════════════════════════════════
```
