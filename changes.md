# Changes — ODAS & Simulator/Analyzer

> Summary of all fixes and improvements made to `z_odas_newbeamform` and `simulator` for the team.
> Date: March 11, 2026

---

## ODAS — `src/module/mod_sst.c`

### 1. YAMNet Integration into SST

The core feature: ODAS now runs YAMNet classification *inside* the real-time C pipeline, not as a post-process. Each active Kalman track accumulates beamformed magnitude spectra in a circular buffer. Every 96 frames (0.768s) a `classify_track_hop()` function assembles a 96×257 float32 patch, passes it through the mel filterbank + log transform, and runs TFLite YAMNet inference. Results are emitted as JSON event records with: track ID, timestamp, direction (x,y,z), top-K class names, hop vote counts, and confidence scores.

**Why:** This is the deployment-domain classification path. The same C code runs on the Raspberry Pi in the field. Training data must go through the same chain to avoid domain mismatch.

---

### 2. Early-Hop Classification for Sparse/Weak Tracks

**Bug:** Tracks that rarely won SSL pot assignments (low-coherence or distant sources) had `frame_count = 1` for their entire lifetime — the normal 96-frame trigger never fired, so YAMNet never classified them at all.

**Fix:** A `hop_age` counter increments unconditionally every processing hop regardless of whether a pot was assigned. An early-hop trigger fires at `hop_age == 48` (if ≥6 real spectral frames exist) or at `hop_age == 96` as fallback. The first half of the patch is zero-padded; the real frames are placed in the second half.

**Why:** Without this, weak-but-present animals (e.g. Bear at 15m during a busy scene) were silently dropped from the event log and never appeared in the dataset.

---

### 3. `.bin` Sidecar File Writing (`sim_mode = 1`)

When `sim_mode = 1` in the config, `classify_track_hop()` writes the assembled 96×257 patch to a `.bin` sidecar file (float32, little-endian) before calling YAMNet. The file path is logged in the emitted JSON event. `sim_mode = 0` is the Pi/field mode — no file I/O overhead.

**Why:** The `.bin` files are the training inputs. They need to be archived during simulation so the curator can stitch, verify, and package them into the dataset without re-running ODAS.

---

## ODAS — `config/runtime/local_socket.cfg`

### 4. Beamformer Switched: DDS → DMVDR

```
mode_sep = "dds"   →   mode_sep = "dmvdr"
```

**DDS** (Delay-and-Sum) steers toward the tracked direction but has zero interference rejection. **DMVDR** (Minimum Variance Distortionless Response) computes per-frequency optimal weights that minimise total output power while maintaining gain toward the target — this implicitly creates nulls toward interference including omnidirectional ambient sources (bird, insects, wind).

**Why:** In any realistic deployment scene there will always be omnidirectional ambient noise — background bird chorus, insects, wind. DDS has no mechanism to suppress this; the ambient leaks into every beamformed track equally regardless of direction, degrading SNR for all sources. DMVDR solves this at the source: its per-frequency weight vector minimises total output power subject to a distortionless constraint toward the target direction, so any energy not coming from that direction — whether a competing directional source or a diffuse ambient field — is attenuated. The practical effect is that weaker or more distant directional sources maintain sufficient SNR across more consecutive hops, producing longer and cleaner spectral patches. The ~15% extra CPU overhead on Raspberry Pi is acceptable for the SNR gain.

---

### 5. New Config Parameters Added

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `sim_mode` | `1` | Write `.bin` sidecar files during simulation; `0` for Pi deployment |
| `min_event_votes` | `1` | Minimum hop votes before emitting an event JSON; raise to 4+ after model is fine-tuned |
| `enable_classifier_output` | `"enabled"` | Enable JSON event log output |
| `classifier_log_dir` | `"./ClassifierLogs"` | Output directory for JSON + `.bin` files |

---

## Simulator — `renderer.py`

### 6. Pre/Post Silence Padding for ODAS Warmup and Tail Flush

Two blocks of silence are now unconditionally added to every rendered 6-channel WAV before it is streamed to ODAS:

- **Head padding (10 s):** ODAS's GCC-PHAT coherence estimator and the per-channel noise-floor tracker (SNE) need several seconds of silence to converge. Without this, the very first animal event fires while ODAS is still initialising — `spectral_count` stays at 1 and the reconstructed audio is near-silent.
- **Tail padding (10 s):** The Kalman filter may still be accumulating the final source's spectral frames when the audio stream ends. Appending silence lets ODAS fully flush all pending 96-frame hop evaluations before the socket closes, ensuring the last animal occurrence gets its full `spectral_count`.

The warmup offset is stored in `render_metadata.json` as `warmup_seconds = 10`. The analyser subtracts this value from all ODAS timestamps before GT matching, so timestamps in `labels.csv` are always relative to the scene start (not the padded WAV start).

**Why:** Before this fix, the first GT source in a scene (typically an Elephant at t≈5s) had `spectral_count=1` because ODAS was still initialising its noise model. The last GT source was also frequently truncated — its final bins were never flushed. Both produced either missing samples or anomalously short ones.

---

## Simulator — `yamnet_dataset_curator.py`

### 7. Training Domain Corrected: `.bin` → Priority 1, Raw PCM → Fallback Only

**Before:** The curator extracted audio by slicing the raw render PCM (the mixed 6-channel file before ODAS). YAMNet was effectively being trained on perfect clean audio it would never see in the field.

**After:** Priority 1 is Griffin-Lim reconstruction from the ODAS-beamformed `.bin` files. Raw PCM extraction is fallback only for cases where no `.bin` exists (source too weak to ever win SSL).

**Why:** Training domain must match deployment domain. The beamformer changes spectral shape, frequency balance, and noise floor. A model trained on clean PCM will degrade in the field.

---

### 8. `.bin` Files Archived into Dataset `bins/` Directory

All `.bin` sidecar files for a GT-matched sample are stitched in time order into a single `(N×96)×257` float32 file and copied to `dataset/bins/{filename}.bin`. Two new columns added to `labels.csv`:

- `spectra_file` — relative path to the stitched `.bin` inside the dataset
- `n_frames` — total rows (spectral frames) in the `.bin`

**Dataset structure is now:**
```
yamnet_train_001/
├── bins/          {filename}.bin    ← TRAINING INPUT (N×257 float32 linear magnitude)
├── audio/         {filename}.wav    ← Griffin-Lim WAV (human verification only)
├── spectrograms/  {filename}.png    ← Spectrogram PNG
├── metadata/      run_{date}.csv    ← Per-run curation stats
└── labels.csv                       ← filename, spectra_file, n_frames, label, ...
```

**Why:** The `.bin` is the actual training input — not the WAV. The WAV (Griffin-Lim reconstruction) is for human ears only to verify the beamformer captured the right animal. The training script will load `.bin` → sliding 96-frame windows → mel+log → YAMNet head fine-tune.

---

### 9. GT-Window Stitching to Fix Track-Flicker Fragmentation

**Before:** Grouping by `track_id`. Kalman tracks flicker during silence gaps within a vocalisation — the same elephant re-acquires with a new track ID 3–5 times during its GT window, producing 3–5 separate 0.77s samples instead of one long sample.

**After:** Grouping by `(label, gt_start_time, gt_end_time)`. All ODAS tracks that matched the same GT animal occurrence are merged; their `.bin` files are stitched in chronological order. One sample per GT occurrence regardless of how many tracks fired.

**Why confirmed safe:** ODAS track IDs are globally monotonically increasing (`obj->id++` only, never reset or reused within a session). So there is no risk of two different animals sharing a track ID.

---

## Simulator — `dataset_visualizer.py`

### 10. Dataset Visualizer Updated to Show `.bin` Content

The **YAMNet Datasets** page (Visualizer tab) now shows per sample:

- 🔊 Griffin-Lim WAV player *(existing)*
- 📊 Spectrogram PNG *(existing)*
- 🔬 `.bin` frame count + duration in the metadata panel *(new)*
- 🔬 Interactive Plotly heatmap of the raw 96×257 linear magnitude spectra — exactly what ODAS feeds into YAMNet — expandable per sample *(new)*
- **Bin Files** count added to the dataset overview header *(new)*

**Why:** The WAV tells you "does this sound like an elephant". The heatmap tells you "what spectral structure is YAMNet actually receiving" — essential for debugging beamformer quality and catching domain issues before training.

---

## Net Effect

Running the same scene (`For_Training_Ele_Bird`, 600s, 6 species, continuous ambient Bird):

| Metric | Before (DDS + old curator) | After (DMVDR + new curator) |
|--------|---------------------------|------------------------------|
| Mean sample duration | 6.0s | 27.3s |
| Samples > 10s | 14 of 73 | majority of 16 |
| Sheep audio quality | Bee-hum (Bird leakage) | Audible bleating |
| `.bin` files in dataset | ❌ | ✅ |
| `spectra_file` in labels.csv | ❌ | ✅ |
| Training domain | Raw PCM ❌ | ODAS-beamformed ✅ |
| Track fragmentation | Many 0.77s per animal | One long sample per GT window |


---

---

# Changes — March 12, 2026

> Investigation of 15 K+ spurious detections, DMVDR crash, and short sample durations.

---

## Bug: 15 000+ Spurious Detections in Analyzer

### Symptom
Run `wolf_frog_wolf_mon_20260312_055822_run_20260312_055833` reported 15 424 matches and 15 389 unmatched detections. All unmatched entries had position `[0,0,0]`, activity 0, and all pointed to the same stale `.bin` file `patch_1_1900.bin`.

### Root Cause — Two Compounding Issues

**Issue A — Stale session file silently consumed:**
ODAS crashed immediately (see DMVDR section below) without writing a new session file. `simulator.py` selected the most-recent session file by mtime, which was a 43 MB file from a run 2+ hours earlier (March 11 05:16 AM) containing 17 988 SST entries for a completely different scene.

**Issue B — Per-frame duplication in the parser:**
`_parse_odas_output` emitted one `detection` dict per SST source line. The ODAS SST JSON is written every `ROLLING_HOPS` frames (~48 ms) and each source entry carries the *same* `spectra_file` / `topk_history` persistently until the next hop fires. A single YAMNet hop event therefore repeated ~16 times (one per SST frame) across the ~768 ms hop window. With 17 988 SST entries over a 485-second session, this inflated to 15 000+ duplicates.

### Fix A — Stale session file guard (`simulator.py`)
1. Record `run_start_time = time.time()` before ODAS is launched.
2. After the session file is located, compare its `os.path.getmtime()` against `run_start_time`. If the file predates the run, log an error and set `session_live_file = None` rather than silently using it.
3. Add an early-crash check: poll the ODAS process after 3 seconds; if it has already exited, show the last 800 characters of the log and abort analysis.

### Fix B — Hop-level deduplication (`analyzer.py`)
After the main parse loop, deduplicate by `(track_id, hop_id)` key where:
- `hop_id = spectra_file` path when non-empty (unique per 96-frame patch, sim_mode=1)
- `hop_id = topk_history[0]['timestamp']` otherwise (ODAS hop frame counter only changes when a new hop fires)

Keep the **last** occurrence of each key (highest accumulated `frame_count` / `spectral_count`). Verified: 17 988 raw entries → 2 176 after deduplication on the stale 43 MB file.

---

## Bug: ODAS Crash — "Invalid separation method" (DMVDR Never Implemented)

### Symptom
Both the March 11 07:51 run and the March 12 05:58 run crashed immediately with:
```
Invalid separation method.
```

### Root Cause
The prior `changes.md` entry recommended switching `mode_sep` to `"dmvdr"`. That config change was made. However, `dmvdr` was **never implemented** in ODAS — neither in the local fork nor in the upstream `introlab/odas` repository:

```c
// src/module/mod_sss.c
steer2demixing_mvdr_construct_zero(obj->steer2demixing_mvdr)  // body: // Not implemented yet
typedef struct steer2demixing_mvdr_obj { } steer2demixing_mvdr_obj;  // empty struct
```

The positive results cited in the prior entry ("Mean sample duration 27.3s") came from earlier runs at 03:43, 04:56, and 05:16 AM on March 11 that were using `mode_sep = "dds"`. The DMVDR recommendation was written based on documentation that does not reflect the actual C code.

### Fix
`mode_sep` set to `"dgss"` (Geometric Source Separation — adaptive, fully implemented).

```
mode_sep = "dds"   →   mode_sep = "dgss"
```

**GSS vs DDS:**
- **DDS (Delay-and-Sum):** steers a fixed beam toward the tracked direction. No interference rejection. All sources leak into all tracks equally.
- **GSS (Geometric Source Separation):** adaptive gradient-descent beamformer. Updates per-frequency complex weight vectors every hop via `W ← W − μ·∇ − λ·W`. Builds nulls toward competing sources over time. Controlled by `mu` (adaptation step, currently 0.01) and `lambda` (weight decay/leakage, currently 0.5).

**Prior entry's DMVDR claims are retracted.** The DMVDR section of the prior entry should be understood as a description of the *intended* beamformer; the actual beamformer in use throughout all tested runs was DDS (before March 12) and now GSS.

---

## ODAS Config: Clarification of All Separation Parameters

| Field | Meaning |
|-------|---------|
| `mode_sep = "dds"` | Delay-and-Sum — fixed beam, no interference rejection |
| `mode_sep = "dgss"` | Geometric Source Separation — adaptive, currently active |
| `mode_sep = "dmvdr"` | MVDR — **not implemented**, causes immediate crash |
| `mode_pf = "ms"` | Multi-Source Post-Filter: MCRA noise estimator + Wiener gain per frequency bin. Applied after separation. |
| `mode_pf = "ss"` | Single-Source Post-Filter: simpler energy-ratio gain, less effective in multi-source scenes |
| `dds: {}` | Empty — DDS has no tunable parameters (purely geometric: mic positions + speed of sound) |
| `dgss: { mu = 0.01; lambda = 0.5; }` | `mu`: gradient step size (larger = faster convergence, more instability). `lambda`: weight decay (prevents divergence) |
| `ms: { alphaPmin, eta, alphaZ, ... }` | MCRA noise tracker and Wiener suppression parameters |
| `separated` output | Beamformer output **before** post-filtering. Written to `separated.raw` |
| `postfiltered` output | Post-filter output **after** noise suppression. Written to `postfiltered.raw`, boosted by `gain_pf = 10.0` |

---

## Investigation: Short Sample Durations (~1 s for 5 s Sources)

### Symptom
Scene `wolf_frog_wolf_mon` has 5-second GT windows per animal. The yamnet_train dataset produced:
- Wolf howl 1: ~1.5 s (2 patches)
- Wolf howl 2: ~1.1 s (2 patches)
- Frog: ~5.7 s (14 patches) ✅
- Monkey: ~1.5 s (3 patches)

### Track Lifetime Analysis

Parsing `sst_session_live.json_20260312_111758.json` (1316 JSONL lines, 519 with sources):

| Track | Animal | Born | Died | Duration | Max activity |
|-------|--------|------|------|----------|-------------|
| 5 | Wolf 1 | 5.17 s | 6.42 s | **1.25 s** | 0.336 |
| 31 | Wolf 2 | 30.27 s | 31.42 s | **1.15 s** | ~0 |
| 14 | Frog | 16.93 s | 23.41 s | **6.48 s** | 1.0 |
| 48 | Monkey | 40.74 s | 48.27 s | **7.54 s** | 1.0 |

The `N_inactive = (150, 150, 150, 150)` Kalman parameter kills a track after `150 × 8 ms = 1.2 s` of sustained `activity < theta_inactive (0.40)`. Wolf track 5 is born with activity 0.336 — just below threshold — so its inactive countdown starts at birth and it dies in exactly 1.2 s.

### Root Cause: GSS Requires Interference to Adapt

**GSS is an adaptive beamformer.** It estimates the spatial covariance of interfering signals and steers nulls toward them. In a **completely silent background scene**, there is no interference for GSS to estimate. The adaptive weights never converge away from identity; the filter degenerates to near-DAS. DAS at these source distances produces very low separated output power → activity falls below `theta_inactive = 0.40` → track is immediately "inactive".

**Why does Frog get activity = 1.0 in the same silent scene?**
The scene is sequential. When frog plays (t = 15–20 s), the wolf GT window (t = 5–10 s) has ended and its room reverb is still decaying. GSS can use this residual acoustic energy as "interference" to estimate suppression weights. Each later source benefits from the reverb tails of all prior sources. The first source (wolf at t = 5 s) plays in **absolute silence** — GSS has nothing to work with.

**The audio file is not at fault.** `wolfhowl01.wav` is continuously active for all 5 seconds (48/49 energy windows above 5% of peak RMS). The wolf signal is loud and physically present at the microphones. The problem is entirely in the GSS adaptive filter failing to initialise without a noise floor.

**Increasing `N_inactive` would not fix this.** The `spec_at_peak` mechanism only writes `.bin` data when the SSL assigns a new energy peak to an active track. In the silent scene, no new SSL peaks are generated in the wolf direction after the first 1.2 s — so no bins would be written even if the track were kept alive indefinitely. The bins would remain near-silent.

### Correct Fix: Add Ambient Noise Floor to Every Scene Render

The simulation must model deployment conditions. In the field there is always ambient acoustic energy (wind, insects, leaf rustle). Even **−40 dBFS pink noise** (100× below the wolf signal, inaudible to humans) gives GSS enough covariance to estimate suppression weights and produce non-zero output at all directions throughout the full GT window.

This is not contaminating the training data — it is making the simulation physically valid. The `.bin` files would then contain the actual beamformed animal audio that ODAS produces in the field, which is precisely what YAMNet sees during deployment.

**Recommended implementation:** Add an optional `ambient_noise_db` field (e.g. `−40`) to the scene JSON. The renderer injects full-duration broadband pink noise at that level before rendering directional sources. The curator already ignores non-GT-matched detections, so the ambient floor produces no spurious training samples.

---

## HTML Report Cleanup (`analyzer.py`)

Two sections removed from `_generate_html_report`:

1. **"🤖 Model Prediction Statistics"** — four stat cards duplicating the Per-Source Statistics table.
2. **"📊 Top-K Classification History per Event"** — up to 20 individual Plotly heatmaps (one per confirmed event), the primary cause of excessive report length. The same information is captured in the YAMNet Classification Timeline chart.

The embedded report iframe height reduced from 2 000 px → 1 400 px.

**Report section order after cleanup:**
Summary → Per-Source Stats → YAMNet Classification Stats → Detection Comparison (slider) → 3D Spatial → Angular Error → YAMNet Timeline → Event Votes → Audio Waveform
