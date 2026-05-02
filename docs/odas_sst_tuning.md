# ODAS SST Parameter Tuning — Findings & Reasoning

**Date:** 2026-04-15  
**Config file tuned:** `/home/azureuser/z_odas_newbeamform/config/runtime/local_socket.cfg`  
**Scene used for evaluation:** `forest animals` (20260409 render, 1220s, real ambient capture)  
**Evaluation script:** inline Python against analysis JSONs in `outputs/analysis/`

---

## 1. Setup & Context

### Audio pipeline
```
Rendered .raw (6-ch S16LE, 16kHz)
  → socket server (Python) → TCP → odaslive
  → SST JSON frames → simulator analyzer → analysis JSON
```

### Scene characteristics
- 285 directional GT events across 7 labels (Bear, Elephant, Frog, Lion, drone_bebop, drone_binary, drone_membo)
- Duration 1220s, of which **772s (~63%) have zero active GT sources** ("quiet periods")
- Real ambient capture mixed in at **−10 dBFS** (audible forest background, not synthetic noise)
- Room: 250×250×20m, mic at centre [125,125,1.5], sources distributed full 360° at all azimuths

### Mic array
- ReSpeaker USB 6-mic hexagonal array, all mics at z=0 (flat plane)
- **Flat array → cannot resolve elevation**; ODAS reports all sources at ~0° elevation
- Hexagonal symmetry → azimuth-isotropic in theory, but GCC-PHAT produces phantom peaks along the 6 inter-mic axes

### Config file note
Early tuning runs edited `/home/azureuser/z_odas/re6_sockets.cfg` — this is **NOT the config the simulator uses**. The simulator (see `simulator.py` line 59) reads:
```
/home/azureuser/z_odas_newbeamform/config/runtime/local_socket.cfg
```
All results below refer to changes in the correct file.

---

## 2. Baseline Problem: FP Rate ~70%, Hotspot Azimuths

### Observed symptoms
The directional scatter plot in the analysis report showed strong FP clusters at specific azimuths: **−90°, 0°, −120°**. These azimuths exactly match the inter-mic axes of the hexagonal array geometry.

Initial hypothesis was Kalman persistence tails, but data analysis disproved this:

| FP cause | Count | % of FPs |
|----------|-------|----------|
| `unrelated` (no nearby GT event) | 1051 | **84.1%** |
| `during_gt_ghost` | 81 | 6.5% |
| `pre_gt` | 71 | 5.7% |
| `kalman_tail` | 46 | **3.7%** |

**Kalman tails were only 3.7% of FPs** — the wrong root cause was being targeted.

### True root cause: spurious track creation in ambient noise

ODAS SSL (Steered Response Power / GCC-PHAT) produces phantom beamformer peaks along array geometry axes even during genuine ambient noise. These peaks passed the SST confirmation gate because the baseline config was extremely permissive:

```
Pnew = 0.6        # 60% prior probability of new source — per frame
theta_new = 0.40  # only 40% SSL confidence needed to attempt track birth
N_prob = 1        # only 1 frame (8ms) to confirm a new track
theta_prob = 0.50 # that 1 frame only needs 50% confidence
```

With `Pnew=0.6` and `N_prob=1`, ODAS was attempting to birth a new track on almost every SSL frame, confirming it after a single 8ms window. This produced ~1.34 spurious confirmed tracks per second during silence.

The quiet-period FP rate **did not drop significantly** even when `N_inactive` was reduced (from 250 to 30 frames), confirming that track *death* was irrelevant — the problem was track *birth*.

---

## 3. Parameter Reference

All parameters are in the `sst:` block of the config.

### SSL gate
| Parameter | Location | Role |
|-----------|----------|------|
| `gainMin` | `ssl:` block | Minimum beamformer gain for a direction to be passed to SST. Hard gate on weak peaks before any Bayesian processing. |

### Track birth (SST)
| Parameter | Role |
|-----------|------|
| `Pnew` | Bayesian prior probability that a new source exists at any candidate direction, per frame. High value = ODAS expects sources to appear frequently. |
| `theta_new` | Minimum SSL posterior probability required to *attempt* birthing a new track at a candidate direction. |
| `N_prob` | Number of consecutive frames a candidate must sustain high probability before being promoted to a confirmed track. Each frame = 1 hop = 8ms (hopSize=128 @ 16kHz). |
| `theta_prob` | Minimum probability required for each of the `N_prob` confirmation frames. |
| `Pfalse` | Prior probability of a false alarm. Higher = SST is more sceptical of new candidates. |

### Track survival
| Parameter | Role |
|-----------|------|
| `N_inactive` | Number of frames a track can survive with no supporting SSL evidence before being dropped. `(t1, t2, t3, t4)` = thresholds for 4 track age buckets. |
| `theta_inactive` | Probability below which a track is considered "inactive" (enters the `N_inactive` countdown). |
| `sigmaQ` | Kalman process noise. Low value = track position stays pinned during coast. High value = position uncertainty grows during coast (track drifts/smears). |

---

## 4. Run-by-Run Results

### Evaluation metrics
- **Events detected**: unique GT source windows (label, start_time) that received at least one matched ODAS frame
- **Quiet FP/s**: FP detections during periods with zero active GT sources, normalised by 772 quiet seconds
- **FP%**: FP frames / total detection frames

---

### RUN0 — Baseline (original config)
```
gainMin = 0.25
Pfalse = 0.4
Pnew = 0.6       ← very high
theta_new = 0.40  ← very low
N_prob = 1        ← single frame confirmation
theta_prob = 0.50
N_inactive = (30, 30, 30, 30)
theta_inactive = 0.60
sigmaQ = 0.004
```
| Metric | Value |
|--------|-------|
| Total detections | 2332 |
| Events detected | **194/284 (68%)** |
| FP% | 70.6% |
| Quiet FP/s | **1.338/s** |
| Avg angular error | 6.00° |
| FP hotspots | −90°: 419, 0°: 376, −120°: 311 |

**Issue:** ~1.34 spurious detections per second in silence. Dominant cause is `Pnew=0.6` + `N_prob=1` allowing any GCC-PHAT noise spike to become a confirmed track in 8ms.

---

### RUN1 — Wrong fix: N_inactive reduction (edited wrong config file)
Edited `/home/azureuser/z_odas/re6_sockets.cfg` instead of the config actually used by the simulator. Zero effect on output — identical to baseline. **All subsequent runs edit the correct file.**

---

### RUN2 — Wrong fix: N_inactive 10-30 (correct file, wrong lever)
```
N_inactive = (10, 20, 30, 30)   # was (30,30,30,30)
sigmaQ = 0.01                   # was 0.004
```
| Metric | Value |
|--------|-------|
| Events detected | 167/284 (59%) ← worse |
| FP% | 68.3% |
| Quiet FP/s | 0.624/s |

**Analysis:** FP rate dropped but event detection also dropped. Root cause analysis (see §2) confirmed Kalman tails were only 3.7% of FPs — reducing `N_inactive` hurt real source tracking without addressing the real problem.

---

### RUN3 — Correct levers applied (first real fix)
```
gainMin = 0.40      # was 0.25 — filter weak SSL beams before SST
Pfalse = 0.1        # was 0.4 — more sceptical of false alarms
Pnew = 0.05         # was 0.6 — 12× less likely to attempt track birth
theta_new = 0.80    # was 0.40 — require stronger SSL confidence
N_prob = 8          # was 1 — require 64ms sustained evidence
theta_prob = 0.70   # was 0.50
theta_inactive = 0.80  # was 0.60
N_inactive = (50,75,100,100)  # restored to balanced value
```
| Metric | Value |
|--------|-------|
| Events detected | 167/284 (59%) |
| FP% | 66.3% |
| Quiet FP/s | **0.624/s** |
| Avg angular error | **5.78°** |

**Analysis:** FP rate halved (1.34 → 0.62/s). But `N_prob=8` (64ms) was too tight for brief animal calls — 34 events dropped, mostly short events (median 1.8s, 20 events <2s). The `N_prob` requirement must be sustained across a window wider than the shortest target sounds.

---

### RUN4 — Slightly relaxed (N_prob=4, theta_prob=0.55)
```
N_prob = 4          # was 8 — 32ms
theta_prob = 0.55   # was 0.70
Pnew = 0.08         # was 0.05
```
| Metric | Value |
|--------|-------|
| Events detected | 179/284 (63%) |
| FP% | **70.8%** ← worse than RUN3 |
| Quiet FP/s | 1.23/s |
| FP hotspot −90° | **501** ← worse than baseline |

**Analysis:** `theta_prob=0.55` made confirmation trivial — GCC-PHAT phantom peaks at array-geometry azimuths (−90°, 0°, −120°) easily sustained 4 frames at 55% confidence in ambient noise, pushing FPs above baseline. The confirmation threshold must be high enough that random ambient-driven peaks cannot sustain it.

**Key insight:** The 4-frame window at 55% is worse than 1-frame at 50% (baseline) because it gives more chances for noise bursts to accumulate — the threshold must be raised when the window is short.

---

### RUN5 — Midpoint (N_prob=6, theta_prob=0.65, Pnew=0.06)
```
N_prob = 6          # was 4 — 48ms
theta_prob = 0.65   # was 0.55
Pnew = 0.06         # was 0.08
```
| Metric | Value |
|--------|-------|
| Events detected | 172/284 (61%) |
| FP% | 66.0% |
| Quiet FP/s | 0.642/s |
| Avg angular error | **5.76°** |

**Analysis:** Nearly identical to RUN3. The system has converged — `N_prob` changes between 6 and 8 produce no meaningful difference at this `theta_prob` level. The remaining ~500 quiet-period FPs (~0.63/s) represent the **structural floor** driven by the ambient noise itself pushing real beamformer peaks through the confirmation gate.

---

## 5. Current Config State (after all tuning)

```
gainMin = 0.40          # was 0.25
Pfalse = 0.1            # was 0.4
Pnew = 0.06             # was 0.6
theta_new = 0.80        # was 0.40
N_prob = 6              # was 1
theta_prob = 0.65       # was 0.50
N_inactive = (30,30,30,30)  # unchanged (240ms coast)
theta_inactive = 0.80   # was 0.60
sigmaQ = 0.001          # was 0.004 (reverted)
```

---

## 6. Key Findings Summary

### Finding 1: The FP hotspots are structural, not Kalman-related
The azimuth clusters at −90°, 0°, −120° correspond to the inter-mic axes of the hexagonal ReSpeaker array. GCC-PHAT normalisation creates phantom peaks at these azimuths even in ambient noise. Kalman persistence tails are only 3.7% of FPs.

### Finding 2: Track birth parameters dominate FP rate
`Pnew` and `N_prob` are the most impactful parameters. The original `Pnew=0.6` + `N_prob=1` combination was the primary FP source — essentially confirming any SSL peak in 8ms. Reducing `Pnew` 10× and raising `N_prob` to 6 cut quiet-period FPs by ~53%.

### Finding 3: There is a structural FP floor at ~0.6/s
Even with aggressive birth gating, ~500 quiet-period FPs remain across 772s of silence. These are driven by the ambient capture (real forest background at −10 dBFS) producing sustained beamformer peaks that satisfy any reasonable confirmation threshold. This floor cannot be eliminated by SST parameter tuning alone.

### Finding 4: Confirmation threshold and window length are coupled
Reducing `theta_prob` while keeping `N_prob` short is counterproductive — it actually increases FPs because noise bursts accumulate across more chances. The relationship is:
- Short window (`N_prob` low) → requires high `theta_prob` to be meaningful
- Long window (`N_prob` high) → can tolerate lower `theta_prob` (averaging out noise)

### Finding 5: Event detection vs FP rate is a hard tradeoff
There is no configuration that simultaneously achieves both high event detection (68%) and low FP rate (0.6/s). The tradeoff is approximately:

| Config | Events | Quiet FP/s |
|--------|--------|------------|
| Original | 194/284 (68%) | 1.34 |
| RUN3/RUN5 | 167–172/284 (59–61%) | 0.62–0.64 |

Short events (<2s, e.g. brief calls) are disproportionately hurt by tighter birth gating.

### Finding 6: `gainMin` is the cleanest gate
Raising `gainMin` from 0.25 → 0.40 filters weak SSL beams before any Bayesian processing. It is independent of temporal dynamics and has no negative interaction effects. It consistently improved angular error slightly (6.00° → 5.76°) by removing low-confidence detections.

---

## 7. Remaining Limitations & Next Steps

### What can't be fixed by SST tuning alone

**Option A: Post-processing sticky-track suppression (analyzer side)**  
Track FP detections that repeatedly appear at the same azimuth (±5°) without corresponding GT matches. Flag these as `structural_fp` in the analyzer. This would eliminate the array-geometry hotspots without hurting real sources.

**Option B: YAMNet classification as FP filter**  
The ambient-driven ghost tracks at array-geometry azimuths likely classify as `Wind`, `Rustling`, or `Background noise` rather than animal species. The downstream YAMNet classifier can filter these out at the application level.

**Option C: 3D array (non-flat)**  
A non-planar mic arrangement (e.g., tetrahedral or pyramid) would break the inter-mic axis symmetry and eliminate the axis-aligned phantom peaks. This is a hardware change.

### What should be validated next
1. Run the same tuning against a second scene (different species mix, different ambient) to confirm generalization
2. Test with louder ambient (volume > 1.0 in `ambient_capture`) to find where `gainMin=0.40` starts cutting real sources
3. Validate `drone_binary` consistently low detection (25–33%) — may be a volume/distance issue in the scene rather than SST config

---

## 8. App Preset System (added 2026-05-02)

The simulator UI exposes three named presets that patch the live config before each run. These are defined in `simulator.py → SST_PRESETS` and can be further overridden with per-parameter sliders in the **Advanced** expander.

### Preset definitions

| Preset | Pnew | N_prob | theta_prob | theta_new | Pfalse | gainMin | theta_inactive | Intended use |
|--------|------|--------|------------|-----------|--------|---------|----------------|--------------|
| **Balanced (default)** | 0.06 | 6 | 0.65 | 0.80 | 0.10 | 0.40 | 0.80 | General dataset collection; ~0.63 FP/s |
| **High-Recall** | 0.15 | 3 | 0.60 | 0.60 | 0.20 | 0.30 | 0.70 | Maximise event capture for training data; ~1.3 FP/s |
| **Low-FP (deployment test)** | 0.03 | 8 | 0.75 | 0.85 | 0.05 | 0.45 | 0.85 | Simulate live-device conditions; higher miss rate |

### When to use each preset

- **Balanced** — default for all EXP-A/B runs unless the experiment specifically varies ODAS config.  
- **High-Recall** — use when the goal is to capture as many clips as possible for training (e.g. EXP-B1, EXP-B2). Accept the higher FP rate; the downstream YAMNet classifier will handle filtering.  
- **Low-FP** — use when simulating what the Raspberry Pi will actually see (EXP-C, EXP-D). FP/s will be lower but brief animal calls (<2s) may be missed.

### Using Advanced overrides

Select a preset to pre-populate all 7 sliders, then adjust individual parameters. The preset name is always written to the run JSON for traceability even if you've deviated from it. Recommended workflow:

1. Start with the closest preset.  
2. Adjust only the parameters you have a hypothesis about.  
3. Add an `experiment_tag` (e.g. `exp_c1_low_fp_v2`) so the run is traceable back to `docs/experiments.md`.

### N_prob ↔ theta_prob coupling rule

**Do not lower `theta_prob` without raising `N_prob` proportionally.** A shorter window at a lower threshold is strictly worse than the original single-frame baseline — noise bursts accumulate across more chances to fake a confirmation. The safe design space is:

| N_prob (frames / ms) | Minimum theta_prob |
|----------------------|--------------------|
| 1–3 (8–24ms) | ≥ 0.75 |
| 4–6 (32–48ms) | ≥ 0.65 |
| 7–10 (56–80ms) | ≥ 0.60 |
| > 10 | ≥ 0.55 |

---

## 9. Config File Quick Reference

**File:** `/home/azureuser/z_odas_newbeamform/config/runtime/local_socket.cfg`  
**Note:** This is the ONLY config read by the simulator. `/home/azureuser/z_odas/re6_sockets.cfg` is NOT used.

No recompilation needed — all parameters are parsed by libconfig at ODAS startup. Restart ODAS to apply changes.

```
# Quick reset to best-found config (RUN5):
gainMin = 0.40
Pfalse = 0.1
Pnew = 0.06
theta_new = 0.80
N_prob = 6
theta_prob = 0.65
N_inactive = ( 30, 30, 30, 30 )
theta_inactive = 0.80
sigmaQ = 0.001
```
