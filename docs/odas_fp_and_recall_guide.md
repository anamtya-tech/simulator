# ODAS False Positives & Missed Events — Practical Guide
> Scene used for all data: `forest_animals_20260409_103128` (1220s, 285 GT events, 7 labels, real ambient at −10 dBFS)
> Best config run: RUN6 (`N_prob=6, theta_prob=0.65`) — see [odas_sst_tuning.md](odas_sst_tuning.md)

---

## Part 1: False Positives

### 1.1 What Was Actually Causing the FPs

**Original assumption:** Kalman tails — tracks that live on after a source has stopped.

**Actual finding:** Only **3.7% of FPs were Kalman tails**. Reducing `N_inactive` (from 150 in the wrong config to 30) didn't fix anything meaningful.

**True root cause:** Track birth was too easy.
- `Pnew=0.6` + `N_prob=1` = ODAS confirmed a new track from **any single 8ms SSL peak**
- Real ambient noise (~−10 dBFS from a forest capture) routinely produces beamformer peaks above threshold at array-geometry hotspot azimuths
- **84% of FPs had no nearby GT event** — they were born from ambient noise, not tails of real events

**FP hotspot azimuths (structural):**

| Azimuth | FP count | Cause |
|---------|----------|-------|
| −90° | 285 | Inter-mic axis (Left–Right pair) |
| −120° | 187 | Diagonal axis artifact |
| 0° | 139 | Inter-mic axis (Front–Back pair) |
| +90° | 90 | Inter-mic axis |
| +180° | 29 | Rear axis |

These azimuths correspond to the square array's geometric axes where GCC-PHAT inter-mic delay = 0, producing phantom correlation peaks.

---

### 1.2 What Was Already Fixed (SST Tuning)

Config changes reduced FP/s from **1.34 → 0.63** (53% reduction):

```
Pnew:          0.6  → 0.06    ← single biggest lever
theta_new:     0.40 → 0.80
N_prob:        1    → 6       ← second biggest lever
theta_prob:    0.50 → 0.65
Pfalse:        0.4  → 0.1
gainMin:       0.25 → 0.40
theta_inactive:0.60 → 0.80
```

---

### 1.3 Remaining FP Floor: ~0.63 FP/s

Even with the best config (RUN6), 0.63 spurious tracks/second remain during quiet periods. This is **structural and cannot be eliminated by SST parameter tuning alone**. Options below.

---

### 1.4 What Else Can Be Done (Beyond SST Tuning)

#### A) Post-processing: Suppress Hotspot Detections (Recommended)

The FP azimuths are highly predictable: −90°, 0°, −120°, +90°. A simple post-processing filter in the analyzer can flag or remove detections that:
1. Cluster within ±5° of known hotspot azimuths
2. Occur during periods with no corroborating event from other directions

**In `analyzer.py`:**
```python
ARRAY_HOTSPOTS_DEG = [-90, -120, 0, +90, +180]
HOTSPOT_TOLERANCE_DEG = 5

def is_hotspot_fp(azimuth_deg):
    return any(abs(azimuth_deg - h) < HOTSPOT_TOLERANCE_DEG for h in ARRAY_HOTSPOTS_DEG)
```

Apply this flag to low-energy detections to filter them before downstream classification.

**Caution:** Real sources CAN appear at these azimuths. Only suppress if: short duration (< 2 frames) OR energy is low AND no corroborating YAMNet class vote.

---

#### B) YAMNet Downstream Filtering

Ghost tracks from ambient noise classify as **wind, background, or other non-target classes** by YAMNet, not as bears or drones. After classification:
- Keep only tracks where YAMNet confidence ≥ threshold for a target class
- Raise `min_event_votes` from 1 → 4 in the config (requires 4/6 rolling-hop agreement)

This is the **highest-impact practical fix** for deployed inference. It doesn't reduce the raw ODAS track count, but eliminates FPs from appearing in the event log.

Config change:
```
min_event_votes = 4;   # was 1
```

---

#### C) Spatial Filter (Elevation Restriction)

If your deployment scenario allows: sources of interest (animals, drones) are always near the horizon (±30° elevation). Floor/ceiling reflections and sky-path artifacts can be excluded.

```
spatialfilters = ({
    direction = ( 0.0, 0.0, 1.0 );   # point upward
    angle = ( 60.0, 90.0 );           # accept within 30° of horizontal (60–90° from vertical)
});
```

**Note:** This only helps if the phantom SSL peaks have elevation ≠ 0°. For a flat array, all sources (real and phantom) are reported at elevation ≈ 0°, so this filter would have limited effect without a 3D array.

---

#### D) Hardware: Non-Planar Array

The hotspot problem is geometric. A non-planar (3D) array — tetrahedral, pyramid, or L-shaped — breaks the axis symmetry that creates phantom GCC-PHAT peaks. No configuration tuning required.

This is a hardware-level fix, not immediately actionable, but relevant if deploying a new array version.

---

#### E) Increase `probMin` and `nPots` Together

If only 2 simultaneous sources are ever expected, reduce `nPots=4` → `nPots=2`. This restricts SSL to report only the top 2 peaks per frame, suppressing weaker phantom peaks.

```
nPots = 2;
probMin = 0.35;
```

---

### 1.5 FP Rate Targets by Use Case

| Use case | Acceptable FP/s | Recommended approach |
|----------|----------------|---------------------|
| Dataset collection (more data is better) | 0.5–1.0 | Current config is fine |
| Deployed event detection | < 0.1 | YAMNet filtering + hotspot suppression |
| Triggered recording | < 0.05 | All of the above + `min_event_votes=4` |

---

## Part 2: Missed Events

### 2.1 Per-Label Breakdown (RUN6, best config)

| Label | Detected | Total | Miss rate | Median missed duration | Primary cause |
|-------|----------|-------|-----------|----------------------|---------------|
| `bear` | 17/45 | 45 | **62%** | 2.0s | Quiet/distant, irregular vocalisation |
| `elephant` | 36/42 | 42 | **14%** | 2.0s | ✅ Best performer — loud, distinctive |
| `frog` | 18/41 | 41 | **56%** | 3.7s | Low amplitude, wide spatial spread |
| `lion` | 27/41 | 41 | **34%** | 1.8s | Moderate — mixed results |
| `drone_bebop` | 38/39 | 39 | **3%** | 1.0s | ✅ Loud, continuous tone → easy |
| `drone_binary` | 11/36 | 36 | **69%** | 1.0s | Quietest drone variant |
| `drone_membo` | 26/41 | 41 | **37%** | 1.0s | Medium |

**Key patterns:**
- Drones with continuous tones → high detection (bebop = 97%)
- Short, quiet, or intermittent calls → high miss rate (bear, frog, drone_binary)
- Missed event duration median ≈ 1–3.7s — these are real calls, not sub-second artefacts

---

### 2.2 Why Events Are Missed

**Primary cause: `N_prob=6` confirmation window**

A source must sustain SSL confidence ≥ `theta_prob=0.65` for 6 consecutive frames (48ms) to be born as a track. Short or quiet events fail to sustain this long enough.

```
Event starts → SSL sees it for 3 frames → goes quiet for 1 frame → restarts count → ...
Result: event ends before accumulating N_prob consecutive frames → NEVER BORN
```

**Secondary causes:**
1. **Low SNR at source distance** — SSL confidence stays below `gainMin=0.40` for the entire event
2. **Source overlaps with another strong source** — Kalman allocates the direction to the wrong track
3. **Array blind spot** — flat array cannot resolve elevation, so a source at a very steep angle may not correlate well

---

### 2.3 What Can Be Done

#### A) Tune `N_prob` + `theta_prob` Per Use Case

There is a fundamental tradeoff: longer confirmation window → fewer FPs → more missed short events.

**Options:**

| N_prob | theta_prob | FP/s | Detection rate | Best for |
|--------|-----------|------|---------------|----------|
| 1 | 0.50 | 1.34 | ~85% | Ground truth data collection only |
| 4 | 0.70 | ~0.8 | ~72% | Balanced (shorter events) |
| 6 | 0.65 | 0.63 | ~61% | **Current — reasonable balance** |
| 8 | 0.70 | ~0.4 | ~53% | Low FP at cost of many misses |

To recover `drone_binary` and `bear` (short/quiet events), consider reducing to:
```
N_prob = 4;
theta_prob = 0.70;   # MUST stay high if N_prob is reduced — see coupling warning
```

---

#### B) Lower `gainMin` for Long-Range Sources

`gainMin=0.40` blocks SSL candidates with beamformer gain below 0.40. Quiet or distant sources may genuinely be below this. Try:
```
gainMin = 0.30;
```
This will increase FP/s slightly but may recover missed low-SNR events.

---

#### C) Raise `Pnew` for Dense-Source Scenarios

If the scene has many simultaneous sources and missing events is worse than false positives:
```
Pnew = 0.15;    # or 0.20
```
But this will increase FP/s. Use only if downstream YAMNet filtering is in place.

---

#### D) Per-Label `N_prob` Workaround

ODAS does not natively support per-source confirmation windows. However, the post-processing pipeline (`analyzer.py`) can apply different acceptance criteria per YAMNet class:

```python
# Relax post-processing rules for known hard-to-detect classes
STRICT_CLASSES = ['drone_bebop', 'elephant']  # already high detection
LENIENT_CLASSES = ['bear', 'frog', 'drone_binary']  # high miss rate

# Accept shorter tracks for lenient classes if YAMNet confidence is high
if yamnet_class in LENIENT_CLASSES and yamnet_confidence > 0.85:
    accept_track(min_duration_frames=2)  # vs default minimum
```

---

#### E) Scene-Level Fixes

Some misses are **not fixable by ODAS tuning** — they are rendering artefacts:

- `drone_binary` at 69% miss rate: this is the quietest drone variant. Check scene placement — if rendered at >30m distance, the signal may genuinely be below the noise floor. Verify in the render config.
- `frog` at 3.7s median missed duration: frogs often have low-amplitude, short calls. The render may need `frog` gain increased by 3–6 dB.
- `bear`: investigate whether the render correctly places bear calls within array range.

To check rendered SPL per event, use:
```python
python3 analyzer.py --run <run_dir> --check-snr-per-label
```

---

#### F) Multi-Run Voting (Ensemble)

Run ODAS twice with different configs and union the events:
- **Run A:** `N_prob=6, theta_prob=0.65` → low FP, good precision
- **Run B:** `N_prob=3, theta_prob=0.75` → higher recall for short events

Union detected events, deduplicate by time+azimuth proximity. This is a simulation/analysis strategy, not suitable for live deployment.

---

### 2.4 Detection Rate Targets

| Use case | Target detection rate | Approach |
|----------|----------------------|----------|
| Dataset curator (label coverage) | > 75% | `N_prob=3`, `theta_prob=0.70`, downstream YAMNet filter |
| Deployed alert system | > 80% on target species | Species-specific post-processing rules |
| Research (missed events acceptable) | Current 61% baseline | Fine-tune scene render gains per label |

---

## Part 3: Summary Decision Tree

```
Are FPs the main problem?
  ├─ YES: Is FP/s > 0.8?
  │         YES → Raise Pnew (reduce), raise N_prob, raise theta_new
  │         NO  → Use YAMNet downstream filtering (min_event_votes=4)
  │               OR post-process: suppress hotspot azimuths
  └─ NO: Are missed events the main problem?
          ├─ Short events missed (< 2s) → Reduce N_prob to 3–4, keep theta_prob ≥ 0.70
          ├─ Quiet events missed        → Lower gainMin to 0.30–0.35
          ├─ Specific labels missing    → Check scene render SPL, may need gain increase
          └─ All labels missing equally → Pnew too low — raise it
```

---

*See also: [odas_config_reference.md](odas_config_reference.md) for full parameter explanations, [odas_sst_tuning.md](odas_sst_tuning.md) for run history.*
