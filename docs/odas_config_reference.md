# ODAS Configuration Reference
> Active config: `/home/azureuser/z_odas_newbeamform/config/runtime/local_socket.cfg`
> Backup: `local_socket.cfg.bak_20260416`
> ⚠️ Do NOT edit `/home/azureuser/z_odas/re6_sockets.cfg` — that file is not used by the simulator.

---

## Pipeline Overview

```
RAW audio input
    │  (6-ch PCM socket @ 16kHz / 8ms hops)
    ▼
MAPPING  — select which channels are real mics
    │  (4 mic channels → ch1=Left, ch2=Back, ch3=Right, ch4=Front)
    ▼
SNE  — stationary noise floor estimation
    │  (subtracts stable ambient hum/wind from beamformer)
    ▼
SSL  — sound source localisation (GCC-PHAT beamforming)
    │  (scans sphere, emits up to nPots candidate directions per 8ms frame)
    ▼
SST  — sound source tracking (Kalman filter)
    │  (births, maintains and kills persistent directional tracks)
    ▼
SSS  — sound source separation (beamforming + post-filter)
    │  (steers nulls toward noise, boosts tracked direction)
    ▼
CLASSIFY  — per-track YAMNet audio classification (custom extension)
```

**Key timing constant:** `hopSize=128` @ 16 kHz = **8 ms per frame**. All frame-count parameters (`N_prob`, `N_inactive`, `L`) convert to real time by multiplying by 8ms.

---

## `raw:` — Audio Input

| Parameter | Current value | Plain-English meaning |
|-----------|--------------|----------------------|
| `fS` | `16000` | Sample rate in Hz. Must match the hardware clock and all downstream stages. |
| `hopSize` | `128` | Samples per processing step → **8ms per frame**. The "heartbeat" of ODAS. |
| `nBits` | `16` | PCM bit depth (S16LE). Standard for most USB mics. |
| `nChannels` | `6` | Total channels in the USB audio stream. The ReSpeaker outputs 6 channels but only 4 are real mics (see `mapping`). |
| `interface.type` | `"socket"` | Where to read audio from. `"socket"` = simulator pipe. Change to `"pulseaudio"` for live Pi deployment. |
| `interface.port` | `10000` | TCP port the simulator uses to stream rendered audio. |
| `record` | `0` | `1` = save raw input to disk for debugging. |
| `model_path` | `.../models` | Path to YAMNet `.tflite` model files used by the classifier extension. |

---

## `mapping:` — Channel Selection

```
map: (2, 3, 4, 5)   ← 1-indexed channel numbers from the 6-ch stream
```

The 6-channel ReSpeaker USB stream layout:
```
ch0 (ignored) = playback loopback
ch1 = Left mic   ← selected (map index 2 = 1-indexed ch1)
ch2 = Back mic   ← selected
ch3 = Right mic  ← selected
ch4 = Front mic  ← selected
ch5 (ignored) = processed mix output
```

**Why 1-indexed?** ODAS uses 1-based channel numbering. `map:(2,3,4,5)` selects the 2nd, 3rd, 4th, and 5th channels (i.e. ch1–ch4 in 0-indexed terms).

---

## `general:` — Physical Constants and Array Geometry

| Parameter | Current value | Plain-English meaning |
|-----------|--------------|----------------------|
| `epsilon` | `1E-20` | Tiny number added to denominators to prevent division by zero. Don't change. |
| `size.hopSize` | `128` | Must match `raw.hopSize`. |
| `size.frameSize` | `512` | FFT window size in samples = 32ms frequency resolution. Larger = better frequency resolution but more latency. |
| `samplerate.mu` | `16000` | Expected sample rate. Bayesian mean — allows for minor clock drift. |
| `samplerate.sigma2` | `0.01` | How much clock drift is tolerated. |
| `speedofsound.mu` | `343.0` | Speed of sound in m/s (~20°C). Affects all TDOA calculations. |
| `speedofsound.sigma2` | `25.0` | Uncertainty in speed of sound. Allows for ~±5°C temperature variation. Increase for outdoor deployments with large temperature swings. |
| `nThetas` | `181` | Number of elevation scan steps (1° per step). Determines vertical angular resolution. |
| `gainMin` | `0.40` ★ | **Hard gate on SSL candidates before they reach SST.** See tuning notes below. |

### `mics` — Microphone Array Geometry

This is a **square** layout, 32mm spacing, all mics in the horizontal plane (z=0):

```
              Front (+y)
                 [5]
                  │
  Left (-x) [2]──────[4] Right (+x)
                  │
                 [3]
              Back (-y)
```

Each mic entry:
- `mu = (x, y, z)` — Position in metres relative to array centre
- `sigma2` — 3×3 position uncertainty covariance matrix (all zeros = perfectly known positions)
- `direction = (0,0,0)` — Mic directivity axis. `(0,0,0)` = omnidirectional
- `angle = (0.0, 360.0)` — Angular acceptance range. `360°` = accept from all directions

**⚠️ Flat array limitation:** All mics have `z=0`. ODAS cannot resolve elevation angle. All sources are reported at elevation ≈ 0° regardless of their true height. Sources directly above the array are ambiguous.

### `spatialfilters`

Restricts ODAS to only track sources within a cone:
- `direction` = unit vector pointing toward the region of interest
- `angle = (inner_deg, outer_deg)` = cone half-angles

Current setting `direction=(0,0,0)`, `angle=(0,360)` = **no filtering, accept all directions**.

**Deployment tip:** If mounted outdoors on a post and floor reflections are a problem:
```
direction = (0, 0, 1)   # point upward (sky)
angle = (0.0, 80.0)     # accept everything within 80° of vertical
```
This would cut sources arriving from below the horizon (floor, mount reflections).

### `gainMin` tuning detail ★

`gainMin` is a hard threshold on the beamformer output gain. Any SSL candidate direction with gain below this value is **discarded before SST even sees it**.

| Value | Effect |
|-------|--------|
| 0.25 (original) | Very permissive — ambient noise peaks constantly pass through → ~1.34 FP/s |
| 0.40 (current) | Blocks most ambient peaks, passes real sound events → ~0.63 FP/s |
| 0.55+ | Risk of missing quiet/distant sources |

---

## `sne:` — Stationary Noise Estimation

Estimates and subtracts the **stable background noise floor** (fans, hum, wind) from the beamformer before SSL.

| Parameter | Value | Plain-English meaning |
|-----------|-------|----------------------|
| `b` | `3` | Number of frequency bands for noise estimation. |
| `alphaS` | `0.1` | Signal smoothing: how quickly the signal estimate adapts. 0=instant, 1=frozen. |
| `L` | `150` | Noise estimation window in frames = **1.2 seconds**. Ambient noise is averaged over this window. Increase to 300+ if ambient is highly variable (gusty wind, intermittent rain). |
| `delta` | `3.0` | Oversubtraction factor — how aggressively noise is subtracted. Higher = more removal but risk of signal distortion. |
| `alphaD` | `0.1` | Noise estimate smoothing. How quickly the noise floor estimate adapts. |

---

## `ssl:` — Sound Source Localisation

SSL scans the unit sphere (all directions) using GCC-PHAT to find directions with highest inter-mic correlation. It reports up to `nPots` candidate directions per frame.

| Parameter | Value | Plain-English meaning |
|-----------|-------|----------------------|
| `nPots` | `4` | Max simultaneous source candidates SSL reports per frame. Set to ≥ expected simultaneous sources. Higher = more CPU. |
| `nMatches` | `10` | Number of SSL candidates passed to SST per frame. |
| `probMin` | `0.25` | Minimum SSL probability for a candidate to be forwarded to SST. Complementary to `gainMin`. |
| `nRefinedLevels` | `1` | Number of sphere-refinement passes. More = more precise direction estimate but slower. |
| `interpRate` | `4` | Sub-degree interpolation rate for finer angle estimates. |
| `scans` | `level 2 + level 4` | Two-pass search: coarse (level 2) then fine (level 4). |

### ⚠️ GCC-PHAT Array Geometry Artifact

This square array has 4 inter-mic axes at **0°, 90°, 180°, 270°**. GCC-PHAT naturally produces phantom SSL peaks along these axes, even in silence, because the inter-mic delay is exactly zero. This creates the FP hotspots seen at **−90°: 285 FPs, −120°: 187, 0°: 139, +90°: 90** in our test run. The SST birth parameters (`Pnew`, `N_prob`, `theta_new`) are the primary mitigation — they require candidates to be sustained before a track is born.

---

## `sst:` — Sound Source Tracking ★ (most critical section)

SST maintains persistent directional tracks using a Kalman filter. Each track represents one estimated source direction over time.

### Track Lifecycle

```
SSL emits candidate (azimuth, probability)
    │
    ▼ gate: probability > theta_new AND gainMin passed?
    │
    ▼ confirmation window: N_prob consecutive frames ≥ theta_prob?
    │
    ▼ TRACK BORN (confirmed, appears in output)
    │
    ▼ Kalman update each frame (predict + correct with SSL)
    │
    ▼ SSL support drops below theta_inactive for N_inactive frames?
    │
    ▼ TRACK DIES (removed from output)
```

### Birth Parameters (most impactful for FP rate)

| Parameter | Original | Current ★ | Plain-English meaning |
|-----------|----------|-----------|----------------------|
| `Pnew` | `0.6` | `0.06` | **Prior probability per frame that a new source has just appeared.** 0.6 = ODAS assumes sources appear 60% of the time → trusts any SSL peak. 0.06 = expects new sources only occasionally → sceptical of noise spikes. **This was the #1 FP cause.** |
| `theta_new` | `0.40` | `0.80` | Minimum SSL probability for a candidate to start the confirmation window. 0.40 let weak ambient peaks enter the pipeline. 0.80 requires strong confident SSL peaks. |
| `N_prob` | `1` | `6` | **Number of consecutive frames the candidate must sustain confidence** before being promoted to a confirmed track. 1 frame = 8ms — any noise spike confirms instantly. 6 frames = 48ms — requires sustained signal. |
| `theta_prob` | `0.50` | `0.65` | Minimum confidence required for each of the N_prob confirmation frames. |
| `Pfalse` | `0.4` | `0.1` | Prior probability that any SSL candidate is a false alarm. 0.4 = ODAS expected 40% of candidates to be noise — too lenient. 0.1 = tighter scepticism. |

**⚠️ Critical coupling: `N_prob` + `theta_prob` interact non-linearly.**

| N_prob | theta_prob | Result |
|--------|-----------|--------|
| 1 | 0.50 | Confirms on any single 8ms noise burst → 1.34 FP/s (**original**) |
| 4 | 0.55 | Short window + low threshold = accumulates noise across 4 easy chances → **worse than original** at hotspots |
| 6 | 0.65 | Balanced — ambient noise rarely sustains 48ms at 65% confidence → 0.63 FP/s (**current**) |
| 8 | 0.70 | Best FP suppression but missed 34 more events vs current |

Rule of thumb: **if N_prob ≤ 4, theta_prob must be ≥ 0.70 to matter**. Loose threshold + short window is counterproductive.

### Active Track Parameters

| Parameter | Value | Plain-English meaning |
|-----------|-------|----------------------|
| `Ptrack` | `0.85` | Probability an existing track continues to the next frame. Higher = harder to kill a track (more persistent). |
| `active.mu` | `0.3` | Expected SSL gain of an actively-tracked source. Used in Bayesian update. |
| `active.sigma2` | `0.0025` | Variance on that expectation. |
| `inactive.mu` | `0.2` | Expected SSL gain of a coasting track (source temporarily silent). |
| `sigmaR2_active` | `0.015` | Measurement noise for Kalman position update. Higher = trust SSL less, rely on Kalman prediction. |
| `sigmaR2_prob` | `0.0025` | Measurement noise for probability updates. |

### Death Parameters

| Parameter | Current ★ | Plain-English meaning |
|-----------|-----------|----------------------|
| `N_inactive` | `(30,30,30,30)` | Frames a track can coast without SSL support before dying = **240ms ghost persistence**. One value per track age bucket. Reducing this was our first hypothesis for fixing FPs — but only 3.7% of FPs were Kalman tails. Reducing it mainly hurt TP. |
| `theta_inactive` | `0.80` (was 0.60) | Probability threshold below which a track is considered "inactive" and starts the death countdown. |

### Kalman Filter

| Parameter | Value | Plain-English meaning |
|-----------|-------|----------------------|
| `sigmaQ` | `0.004` | **Process noise — how much the Kalman filter allows direction to drift between frames.** LOW (0.001) = source assumed stationary → ghost tracks pin exactly to array axis hotspots. HIGH (0.05) = source can move fast → track drifts. 0.004 is appropriate for animals and slow-moving drones. |

### Custom Extension Parameters

| Parameter | Value | Plain-English meaning |
|-----------|-------|----------------------|
| `enable_classifier_output` | `"enabled"` | Write per-track YAMNet classification JSON to `classifier_log_dir`. |
| `classifier_log_dir` | `"./ClassifierLogs"` | Output directory for classifier JSONs (relative to where odaslive is run from). |
| `sim_mode` | `1` | `1` = write 96×257 float32 `.bin` STFT patch files per hop (simulator/training). `0` = skip .bin files (production deployment). **Set to 0 for Pi deployment** to avoid filling disk. |
| `min_event_votes` | `1` | Minimum YAMNet agreement (out of 6-hop rolling window) before emitting an event. `1` = emit on any classification (good for data collection). `4+` = majority vote (good for deployed inference). |

---

## `sss:` — Sound Source Separation

Steers a beam toward the tracked source while placing nulls toward interference.

| Parameter | Value | Plain-English meaning |
|-----------|-------|----------------------|
| `mode_sep` | `"dgss"` | Separation algorithm. `"dds"` = simple delay-and-sum. `"dgss"` = GSC hybrid (better noise rejection). `"dmvdr"` = minimum variance (best quality, most CPU). |
| `mode_pf` | `"ms"` | Post-filter type. `"ms"` = masking spectral subtraction. `"ss"` = standard spectral subtraction. |
| `ssbShiftHz` | `0` | Frequency shift before ODAS to work around the 2680 Hz aliasing limit of 32mm inter-mic spacing. 0 = disabled. Would be needed for sources (e.g. wolves at 375 Hz) where the fundamental is below the array's resolving limit. |
| `gain_sep` | `1.0` | Linear gain on separated output. 1.0 = unity. |
| `gain_pf` | `10.0` | Linear gain after post-filter (+20 dB). Applied to boost separated signal for YAMNet. |
| `dgss.mu` | `0.01` | GSC adaptive filter step size. Smaller = more stable but slower to adapt. |
| `dgss.lambda` | `0.5` | GSC regularisation. Higher = more robust when sources are poorly separated. |

Post-filter (`ms`) controls spectral suppression of residual noise in separated signal. Most values are safe to leave at defaults.

---

## `classify:` — Legacy TDOA Classifier (not used)

This is ODAS's built-in classification stage based on inter-channel time differences. It is **disabled** (output goes to `blackhole`). Classification in this pipeline is handled by the YAMNet custom extension in SST (`enable_classifier_output`). These parameters do not affect detection results.

---

## Parameter Quick Reference Card

```
MOST IMPACTFUL (tune these first):
  Pnew          = 0.06    # How often do you expect new sources to appear? (0.02–0.20)
  N_prob        = 6       # How many frames must a source sustain before confirming? (3–10)
  theta_new     = 0.80    # How strong must an SSL peak be to start confirmation? (0.60–0.90)
  theta_prob    = 0.65    # How strong per confirmation frame? (0.55–0.80)
  gainMin       = 0.40    # Minimum beamformer gain gate (0.25–0.55)

MODERATE IMPACT:
  Pfalse        = 0.10    # How sceptical should ODAS be of each SSL candidate?
  theta_inactive= 0.80    # When does a track start dying?
  N_inactive    = 30      # How many frames before a coasting track is removed? (×8ms)
  sigmaQ        = 0.004   # How much can a source direction drift per frame?

LOW IMPACT (leave at defaults unless you have specific reason):
  Ptrack, sigmaR2_*, active.mu/sigma2, inactive.mu/sigma2, SNE params, SSL scan params
```

---

*Last updated: 2026-04-16. See also: [odas_sst_tuning.md](odas_sst_tuning.md) for run history, [odas_fp_and_recall_guide.md](odas_fp_and_recall_guide.md) for strategies.*
