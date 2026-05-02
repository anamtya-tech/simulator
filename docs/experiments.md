# Experiment Plan — ODAS + YAMNet Improvement

**Purpose:** Systematic experiments to improve YAMNet classification accuracy and reduce false positives when ODAS is deployed live on a Raspberry Pi.  
**Maintained by:** Simulator pipeline team  
**Last updated:** 2026-05-02

---

## Background & Problem Statement

The full pipeline is:

```
Scene Configurator → Audio Renderer → ODAS Simulator → Results Analyzer
     ↓ GT dataset                        ↓ ODAS detections
GT Dataset Builder                   YAMNet Curator
     ↓                                   ↓
     └──────── Fine-Tune YAMNet ─────────┘
                      ↓
              TFLite → deployed to ODAS on Raspberry Pi
```

Two training-data sources exist and differ significantly in distribution:

| Source | How produced | Model sees at train time | Model sees at deploy time |
|---|---|---|---|
| **GT dataset** | Clip extracted directly from rendered audio sidecar (`.f32`); clean RIR, no beamforming | Pre-beamformed, clean | Post-beamformed, ODAS-filtered |
| **Post-ODAS dataset** | Griffin-Lim reconstruction of ODAS `.bin` spectra; output of the actual beamformer | Post-beamformed, ODAS-filtered | Post-beamformed, ODAS-filtered ✅ |

**Core hypothesis:** A model trained on post-ODAS data should transfer better to live deployment because the training distribution matches the inference distribution exactly.

**Key problems observed:**

1. **False positives (~0.63/s even after SST tuning):** ODAS structural hotspot azimuths (−90°, 0°, −120°) produce spurious tracks during ambient noise. YAMNet has never seen the spectral signature of these ambient-driven ghost tracks during training.
2. **Per-label miss rate varies widely:** `drone_bebop` ≈3% miss vs `bear` ≈62% miss. Short/quiet events are disproportionately hurt.
3. **Domain gap:** GT-trained model sees pre-beamformed audio; deployed model sees post-beamformed ODAS spectra — different spectral shape.
4. **Real-capture ambient has directional components:** Raw file used as ambient source may contain directional noise that bleeds into the SSL hotspot directions, increasing FPs. Whether this matters systematically has not been tested.

---

## Fixed Protocol (all experiments)

To ensure results are comparable, the following are **fixed** across all experiments in Phase 1:

| Parameter | Fixed value |
|---|---|
| ODAS SST config preset | **Balanced** (`N_prob=6, theta_prob=0.65, Pnew=0.06`) |
| Mic array | ReSpeaker 4-mic square, 64mm |
| Scene labels | Elephant, Bear, Frog, Lion, drone_bebop, drone_binary |
| Source distances | 10–150m range |
| Scene duration | 600s per render |
| Render room | 250×250×20m, absorption=0.7, max_order=3 |
| Training recipe | Phase-1: 20 epochs (head only); Phase-2: 20 epochs (top-4 backbone unfreeze); batch=32 |
| Evaluation | Clip-level test accuracy + event-level precision/recall + FP/min on dedicated hold-out render |

**Reproducibility:** Every scene uses a saved scene JSON. Every training run records the dataset path, preset name, and experiment tag in the run/checkpoint metadata.

---

## Phase 1 — Ambient-Type Ablation

**Question:** Does the type of ambient background determine how many false positives ODAS generates, and does training on matched ambient improve accuracy?

### EXP-A1 — Baseline: No Ambient

| Field | Value |
|---|---|
| **Experiment tag** | `exp_a1_no_ambient` |
| **Scene type** | Directional sources only, no ambient sources, no capture |
| **Training data** | Post-ODAS curator dataset from this scene |
| **Label strategy** | GT-only |
| **Expected outcome** | Low FP rate (structural hotspots only from sensor noise); provides noise floor baseline |
| **Acceptance criterion** | FP/min ≤ 2 |

**Steps:**
1. Create scene: Easy mode, 6 labels, 600s, max_radius=150m, **ambient mode = none** (remove all ambient sources)
2. Render → ODAS Simulate (tag: `exp_a1_no_ambient`)
3. Analyze → GT-only label strategy → curate to dataset `odas_a1_no_ambient`
4. Train: GT dataset `gt_a1_no_ambient` only → checkpoint `model_a1_gt`
5. Train: Post-ODAS dataset `odas_a1_no_ambient` only → checkpoint `model_a1_odas`
6. Evaluate both on a **separate** 300s hold-out render (same scene family, different RNG seed)
7. Record: clip acc, event P/R, FP/min

---

### EXP-A2 — Single Synthetic Ambient: Rain

| Field | Value |
|---|---|
| **Experiment tag** | `exp_a2_rain` |
| **Scene type** | Directional sources + synthetic `Rain` ambient (volume 0.3–0.5) |
| **Training data** | Post-ODAS curator dataset |
| **Label strategy** | GT-only |
| **Expected outcome** | Rain is spectrally broadband — should create moderate FP rate. Structural hotspots appear more clearly against flat noise. |

**Steps:** Same as EXP-A1 but add `Rain` as synthetic ambient. Use same fixed SST preset.

---

### EXP-A3 — Single Synthetic Ambient: Wind

Same as EXP-A2 but with `Wind` ambient. Wind is low-frequency heavy — may activate GCC-PHAT at different azimuths than rain.

**Experiment tag:** `exp_a3_wind`

---

### EXP-A4 — Mixed Synthetic Ambient (Bird + Rain + Wind)

| Field | Value |
|---|---|
| **Experiment tag** | `exp_a4_mixed_synth` |
| **Scene type** | Directional sources + 3 synthetic ambients (Bird, Rain, Wind), all volume 0.3 |
| **Rationale** | Closer to natural outdoor environment without directional contamination |

---

### EXP-A5 — Real Capture Ambient (Raw File)

| Field | Value |
|---|---|
| **Experiment tag** | `exp_a5_capture_ambient` |
| **Scene type** | Directional sources + real 6-channel capture ambient |
| **Rationale** | Tests the suspected problem: real captures contain directional components that bleed into the SSL hotspot directions |
| **Key comparison** | EXP-A4 vs EXP-A5 with equivalent ambient SPL: if FP rate is higher in A5, directional contamination in captures is confirmed |

---

### Phase 1 Comparison Matrix

| Experiment | Ambient type | Expected FP/min | GT model clip acc | ODAS model clip acc |
|---|---|---|---|---|
| A1 | None | ~2 | ? | ? |
| A2 | Rain (synth) | ~5 | ? | ? |
| A3 | Wind (synth) | ~5 | ? | ? |
| A4 | Mixed synth | ~8 | ? | ? |
| A5 | Real capture | ~15 | ? | ? |

> Fill in results as experiments complete.

---

## Phase 2 — Training-Data Strategy Ablation

**Question:** How does the choice of training data (GT vs post-ODAS vs hybrid) affect deployment accuracy?

Use a **fixed scene**: EXP-A4 (mixed synthetic ambient) as the canonical scene family.

### EXP-B1 — GT-Only Model

| Field | Value |
|---|---|
| **Experiment tag** | `exp_b1_gt_only` |
| **Training data** | GT Dataset Builder clips (pre-beamformed, clean RIR) |
| **Expected weakness** | Domain gap — training distribution ≠ inference distribution |

### EXP-B2 — Post-ODAS-Only Model

| Field | Value |
|---|---|
| **Experiment tag** | `exp_b2_odas_only` |
| **Training data** | Post-ODAS curator dataset (Griffin-Lim from `.bin` spectral buffers) |
| **Expected strength** | Matches deployment distribution; YAMNet sees exactly what it will see live |
| **Expected weakness** | May have fewer clips (ODAS misses ~39% of events per Phase 1 tuning) |

### EXP-B3 — Hybrid: GT Pretrain + ODAS Fine-tune

| Field | Value |
|---|---|
| **Experiment tag** | `exp_b3_hybrid` |
| **Training recipe** | Warm-start from `model_b1_gt` checkpoint → fine-tune on post-ODAS dataset |
| **Rationale** | GT gives more diverse coverage; ODAS fine-tune corrects the domain gap |

### EXP-B4 — Post-ODAS + Hard Negatives

| Field | Value |
|---|---|
| **Experiment tag** | `exp_b4_hard_negatives` |
| **Training data** | Post-ODAS dataset (B2) + ambient-only ODAS peaks labeled `background` |
| **Ambient-only pipeline** | Create scene with **zero directional sources**, render, simulate, use "Label as background" curator mode in Analyzer |
| **Rationale** | Teach the model that ODAS ghost tracks from ambient noise look different from real animal sounds |

**Ambient-only scene setup:**
1. Create scene: 0 directional sources, ambient `Rain+Wind+Bird` at volume 0.5, duration 600s
2. Render → ODAS Simulate (tag: `exp_b4_ambient_only`)
3. In Analyzer → Dataset Curation Settings → enable **"Label all ODAS peaks as background"**
4. Analyze → all detections saved as `background` class to dataset `odas_hard_negatives`
5. Merge `odas_hard_negatives` with `odas_b2_main` → train → checkpoint `model_b4`

### EXP-B5 — Multi-Background Pooled Model

| Field | Value |
|---|---|
| **Experiment tag** | `exp_b5_pooled` |
| **Training data** | Post-ODAS datasets from EXP-A1 through EXP-A5 merged + hard negatives from B4 |
| **Rationale** | Exposure to diverse ambient conditions during training → robust deployment |
| **Acceptance criterion** | Better test acc on unseen ambient condition than any single-ambient model |

---

### Phase 2 Comparison Matrix

| Model | Training data | Clip acc (synth ambient) | Clip acc (capture ambient) | Event P/R | FP/min |
|---|---|---|---|---|---|
| B1 — GT only | GT clips | ? | ? | ? | ? |
| B2 — ODAS only | Post-ODAS | ? | ? | ? | ? |
| B3 — Hybrid | GT pretrain + ODAS FT | ? | ? | ? | ? |
| B4 — +Hard neg | Post-ODAS + ambient BG | ? | ? | ? | ? |
| B5 — Pooled | All conditions merged | ? | ? | ? | ? |

---

## Phase 3 — Confidence Threshold & Deployment Tuning

Once the best model from Phase 2 is identified:

### EXP-C1 — Threshold Sweep

Use the Deployment Evaluation tab in the Analyzer to sweep the class-confidence accept threshold (0.3 → 0.95 in steps of 0.05) and record:
- Precision and Recall at each threshold
- FP/min at each threshold
- Choose operating point: max F1 while FP/min ≤ 0.1

### EXP-C2 — Per-Class Threshold Tuning

Some classes (Elephant, drone_bebop) are easy; others (Bear, Frog) are hard. Set per-class thresholds based on Phase 2 confusion matrices.

### EXP-C3 — ODAS Config Preset Comparison

Compare **High-Recall** vs **Balanced** vs **Low-FP** presets (available in the Simulator page) on the B4 model. This separates ODAS config effects from model quality effects.

| Preset | N_prob | theta_prob | Pnew | Expected trade-off |
|---|---|---|---|---|
| High-Recall (dataset collection) | 3 | 0.60 | 0.15 | More events detected, more FPs; use only for building datasets |
| Balanced (default) | 6 | 0.65 | 0.06 | Current best balance |
| Low-FP (deployment) | 8 | 0.75 | 0.03 | Fewer FPs, more misses; only use with a model trained on post-ODAS data |

---

## Phase 4 — Real-World Validation

After Phase 3 identifies the best model + threshold + ODAS preset:

1. Export best TFLite model from Fine-Tune YAMNet → Deploy tab
2. Deploy to Raspberry Pi with `odaslive` using the Low-FP preset config
3. Record 10 minutes of each class from ≥30m distance against natural ambient
4. Label detections manually
5. Compute: event precision, event recall, mean azimuth error, FP/min
6. Compare against Phase 3 simulator predictions to validate simulation fidelity

---

## Appendix A — Metrics Definitions

| Metric | Definition |
|---|---|
| **Clip accuracy** | % of held-out test clips classified with correct label (standard multi-class accuracy) |
| **Event precision** | GT events with ≥1 matching detection / total detection groups (direction + time aligned) |
| **Event recall** | GT events with ≥1 matching detection / total GT events |
| **FP/min** | Detections during periods with zero active GT sources, normalised by quiet-time duration |
| **Correct-class + correct-direction** | Detections where YAMNet class = GT label AND azimuth error ≤ 15° |

---

## Appendix B — App Knobs Reference

### Scene Configurator
- **Ambient mode:** Synthetic vs Real Capture — use Synthetic for controlled ambient ablations
- **Source volume:** Per-source gain — use to control SNR
- **Source distance:** Directly controls arrival SNR at array

### ODAS Simulator
- **SST preset:** High-Recall / Balanced / Low-FP — must be fixed within an experiment batch
- **Experiment tag:** Propagates to run JSON and downstream datasets/models for traceability

### Results Analyzer — Dataset Curation Settings
- **Label strategy:** `GT-only` (Phase 1/2 training), `ODAS event voting` (iterative improvement), `Fine-tuned model` (iterative improvement after first model)
- **Label all ODAS peaks as background:** Enabled only for ambient-only runs (EXP-B4 hard negatives)

### Fine-Tune YAMNet — Dataset Tab
- **Max clips per class:** Cap imbalanced classes (e.g., cap Elephant at 200 so rare Bear clips are not swamped)
- **Background injection:** Inject GT background clips when post-ODAS dataset lacks a background class

---

## Appendix C — Known Issues / Constraints

| Issue | Impact | Mitigation |
|---|---|---|
| Flat array → all sources reported at elevation ≈ 0° | Cannot separate elevated sources from ground-level | Use only horizontal sources in scenes |
| GCC-PHAT hotspots at −90°, 0°, −120° | Structural FP floor ~0.6/s irreducible by SST tuning alone | Hard-negative training (EXP-B4) + confidence filtering |
| Griffin-Lim reconstruction artefacts in `.bin`→WAV | Post-ODAS WAVs sound metallic/choppy | Use `.bin` directly for training; WAV is for human verification only |
| Short events (<2s) disproportionately missed at `N_prob=6` | Bear, Frog high miss rate | Use High-Recall preset for dataset collection only |
| Source files repeat across scenes | May cause clip-level train/test leakage | `max_clips_per_class` cap + source-file grouped split in finetuner |
