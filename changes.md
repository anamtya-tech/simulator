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
