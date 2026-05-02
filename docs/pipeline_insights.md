# Pipeline Insights — Lessons Learned
> Last updated: 2026-05-02  
> Covers the full chain: Scene → Renderer → ODAS → Analyzer → Dataset → YAMNet  
> For ODAS SST parameter specifics see [odas_sst_tuning.md](odas_sst_tuning.md).  
> For planned experiments see [experiments.md](experiments.md).

---

## 1. Domain Gap: The Biggest Unresolved Risk

### What it is

Training data and deployment data come from different distributions:

| Stage | Audio seen by YAMNet | Source |
|-------|----------------------|--------|
| **GT Dataset Builder** | Raw rendered `.f32` clips, sliced by GT time window | Pre-beamformed multichannel → downmixed to mono |
| **Post-ODAS Curator** | Griffin-Lim reconstruction from ODAS `.bin` spectrograms | Post-beamformed, spatially filtered, compressed |
| **Live Raspberry Pi** | Post-beamformed WAV from `odaslive` SSS stage | Same pipeline as post-ODAS curator |

A model trained only on GT clips has never seen beamformer colouration, null steering artefacts, or Griffin-Lim compression noise. These artefacts are present on every clip the deployed model will classify.

### Why it matters in practice

- Bear call clips from the GT dataset are clean, full-bandwidth renderings.
- The same bear call after ODAS beamforming has spectral smearing, residual sidelobe suppression, and possible phase artefacts at low frequencies.
- YAMNet's embedding layer is trained on AudioSet; it is relatively robust to mild processing, but heavy beamformer colouration can shift the embedding far enough to confuse class boundaries.

### Mitigation strategy

1. **EXP-B2 / EXP-B3**: Train on post-ODAS curator clips only. Compare F1 to GT-trained model on the same held-out ODAS test set.
2. **EXP-B5**: Mixed training — combine GT and post-ODAS clips. Expected to generalise better than either alone.
3. Long term: replace Griffin-Lim with direct WAV output from the ODAS SSS stage (requires patching `odaslive` to write `.wav` files per track). This would eliminate the reconstruction artefact entirely.

---

## 2. Ambient Capture Contains Directional Components

### Observation

The raw ambient capture used in scenes (e.g. `forest_background.wav`) is a real field recording made with a directional microphone. When this is replayed as the "background" channel in the renderer and fed to ODAS, it does **not** arrive as diffuse noise — it has residual directional structure that the GCC-PHAT beamformer interprets as a localised source.

### Consequence

- ODAS consistently birth spurious tracks at the arrival direction of the dominant ambient component (usually 0°–30° from mic forward axis, depending on the recording geometry).
- This means even with `gainMin=0.40` and `N_prob=6`, ~0.63 FP/s remain that are driven by the ambient capture itself, not by geometry artefacts.
- Training YAMNet on clips from these spurious tracks without special handling teaches the model to classify ambient noise as animal species.

### Current workaround

`curate_ambient_as_background()` in `yamnet_dataset_curator.py`: re-labels all detections from an ambient-only run (no GT events) as `background`. This gives YAMNet explicit hard negatives that look exactly like the deployment FPs it will need to suppress.

### Longer-term fix

Replace directional ambient files with synthetic diffuse noise (pink noise convolved with a spherical impulse response). This would make the ambient genuinely non-directional and reduce the structural FP floor. See EXP-A5 in `experiments.md`.

---

## 3. Per-Label Detection Rate Varies Wildly

### Observed spread (Balanced preset, forest_animals scene)

| Label | Events detected | Miss rate | Notes |
|-------|----------------|-----------|-------|
| drone_bebop | ~97% | ~3% | Loud, long, spectrally unique |
| frog | ~75% | ~25% | Short calls, but repetitive |
| elephant | ~70% | ~30% | Long events, wide spectral spread |
| bear | ~38% | ~62% | Short bursts, often below gainMin |
| drone_binary | ~25–33% | ~67–75% | Volume/distance issue in scene config |

### Root causes

1. **Event duration** — `N_prob=6` requires 48ms of sustained beamformer evidence. Animal calls shorter than ~100ms are unreliable even if loud.
2. **Volume in scene config** — sources at lower dBFS are more affected by `gainMin`. `drone_binary` and `bear` are likely set too quietly relative to the ambient level.
3. **Spectral overlap with ambient** — frog and insect calls overlap heavily with wind/leaf-rustle frequency bands, confusing SSL.

### What to try

- Increase source volume for low-detection labels (+3–6 dB) in the scene configurator.
- Use High-Recall preset specifically for bear/drone_binary scenes to build their training datasets.
- Add a per-label `N_prob` override to the config (not currently supported by ODAS — would require C code change).

---

## 4. Training Dataset Strategy

### The two-dataset problem

The system produces two kinds of datasets, and they should be used differently:

| Dataset type | Built by | Audio quality | FP contamination | Best used for |
|---|---|---|---|---|
| **GT Dataset** | `gt_dataset_builder.py` | High (direct render) | None (GT-windowed) | Baseline training, class coverage |
| **Post-ODAS Dataset** | `yamnet_dataset_curator.py` | Lower (Griffin-Lim) | Present | Deployment-distribution training, hard negatives |

### Recommended training mix (hypothesis, to be validated by EXP-B)

```
GT clips (all labels, 70/15/15 split by source file)
  + Post-ODAS clips (same scene, deduplicated against GT windows)
  + Ambient-only hard negatives (label = background)
```

The source-file grouped split is critical: clips from the same source WAV must never span train and test folds. Griffin-Lim chunking of a single ODAS track produces many clips from one acoustic event; putting some in train and some in test inflates reported accuracy (data leakage).

### Per-class clip cap

Large datasets tend to be heavily dominated by `background` or whichever class has the most long events. Capping training clips per class (`max_clips_per_class` in the Fine-Tune UI) prevents the loss from being dominated by one label. Empirically, 200–500 clips per class produces good results; beyond that, additional clips yield diminishing returns unless they add acoustic diversity.

---

## 5. ODAS Output Is Not a Clean Event Segmentation

### What ODAS actually outputs

ODAS SST outputs a **continuous stream of JSON frames** at 8ms intervals. Each frame may contain 0–4 tracked source positions. There is no concept of "event start" or "event end" at the ODAS level — only frame-by-frame track updates.

The analyzer synthesises events by:
1. Grouping consecutive frames where a given track ID is active within a 1s gap.
2. Matching those synthetic events against GT time windows.

### Implication for dataset curation

A single 10-second animal call may produce 50–80 individual 960ms clips (from the 8ms-hop beamformer spectrogram). These clips are **not independent** — they are overlapping windows of the same acoustic event. The stratified split must group them by source event (or at minimum by source WAV basename) or the model will appear to generalise when it has actually memorised spectral fingerprints of individual recordings.

This is addressed in `yamnet_finetuner.py → _stratified_split()` and `_cap_train_clips()` via the `source_wav` column.

---

## 6. Evaluation Metrics Cheat Sheet

| Metric | Computed by | What it measures |
|---|---|---|
| **Event Precision** | `analyzer.py` | Of all ODAS tracks matched to a GT window, what fraction are correct label + direction |
| **Event Recall** | `analyzer.py` | Of all GT events in the scene, what fraction had at least one matched ODAS track |
| **Quiet FP/s** | `analyzer.py` | ODAS tracks during periods with zero active GT sources, normalised by silent duration |
| **FP/min (deployment)** | `yamnet_dataset_curator.compute_deployment_metrics()` | FPs per minute, direction-aware, from the classifier perspective |
| **Correct class + direction %** | `compute_deployment_metrics()` | Of all classifier hits, fraction where both label and azimuth sector are correct |
| **Angular error** | `analyzer.py` | Mean absolute azimuth difference between matched ODAS track and GT source |

### What "matched" means

A detection is "matched" to a GT event if:
- The detection azimuth is within `angle_threshold` (default 30°) of the GT source azimuth, AND
- The detection timestamp falls within the GT event's `[start_time - time_pre, end_time + time_post]` window.

Tightening `angle_threshold` to 15° gives a more honest picture of directional accuracy but will reduce apparent recall for sources near the 30° border.

---

## 7. App Page Reference

| Page | Key controls | Output |
|---|---|---|
| **Scene Configurator** | Source labels, positions, volumes, timing; ambient file | `outputs/scenes/*.json` |
| **Audio Renderer** | Scene file, duration, warmup silence, ambient level | `outputs/renders/*.raw` + metadata JSON |
| **ODAS Simulator** | Preset, Advanced sliders, Experiment tag | `outputs/runs/*.json` (SST JSON frames) |
| **Results Analyzer** | Angle threshold, time windows, Dataset Curation Settings | `outputs/analysis/` (JSON + HTML report + labels.csv) |
| **YAMNet Datasets** | (display only) | — |
| **GT Dataset Builder** | Scene file, clip length, stride | `outputs/gt_datasets/` |
| **Fine-Tune YAMNet** | Dataset selection, max clips/class, BG injection, epochs | `checkpoints/` (`.keras` model) |

### Key file relationships

```
scene.json
  └── renderer → render.raw + render_meta.json
        └── simulator → run.json  (SST frames)
              └── analyzer → analysis.json + report.html + labels.csv
                    └── yamnet_finetuner → staging dir → training → checkpoint.keras
```

All downstream steps embed the `run_id` and `scene_name` so the provenance chain is always traceable.
