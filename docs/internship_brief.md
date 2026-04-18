# Internship Brief — Audio DSP / ML Research (1–2 months)

## What we're building

A wildlife/drone detection system using a 4-microphone array. The array captures outdoor audio, a C-based spatial audio processor (ODAS) finds sound source directions in real-time, and a neural classifier (YAMNet fine-tuned) identifies what each source is. The goal is reliable detection of species and drones in noisy outdoor environments — forests, fields, coastal sites.

The system works end-to-end today. The open problem is that it misses too many real events and generates too many false alarms, and we want to understand *why* at the signal level before writing more C code.

---

## What the intern will do

All work is Python-based. No C required.

**Week 1–2 — Understand the pipeline**
- Run the existing simulation + ODAS pipeline on recorded scenes
- Analyse false positive and missed event logs (data is already collected)
- Read two short internal guides (~5 pages each) on what we've already tried

**Week 3–5 — Phase coherence experiments**
- Extend a Jupyter notebook (already scaffolded) that measures inter-microphone phase differences between directional sources and ambient background
- Test whether spatial coherence metrics (MSC, phase-slope R², circular variance) can separate target events from background
- Run ODAS on ambient-only audio, then combined audio, and build a three-way track classifier (GT event / background directional source / geometry artifact)

**Week 6–7 — Data strategy**
- Design a synthetic scene generation pipeline where ambient is built from N controllable point sources at known azimuths (not a mono wav — properly spatialised)
- Sweep: number of background sources, their azimuths, SNR of target vs background
- Measure recall and FP rate across the sweep

**Week 8 — Write-up**
- Short technical note (~2 pages): what works, what doesn't, concrete parameter recommendations for ODAS config and/or a pre-filter design

---

## Required background

- **Solid Python + NumPy.** All the audio work is array operations — FFTs, cross-spectra, matrix slicing. If vectorised NumPy feels natural, you're fine.
- **Basic DSP concepts.** Fourier transform, what a phase difference between two signals means, what a cross-correlation is. Undergraduate signals course or equivalent self-study.
- **Comfortable in Jupyter.** The experiments live in notebooks. You'll be adding cells, plotting, iterating.

---

## Nice to have (not required)

- Some exposure to microphone arrays or beamforming — even just having read about it
- Familiarity with `scipy.signal`, `librosa`, or `pyroomacoustics`
- Any experience with audio ML (doesn't need to be spatial)

---

## What it is not

- Not a software engineering role. There's no frontend, no API to build, no deployment sprint.
- Not a pure ML role. We're not training large models. The classification model exists; the open questions are all in the signal processing before it.
- Not a C/embedded role. ODAS is in C but the intern works entirely in the Python simulation layer.

---

## Practical details

- Remote or on-site (flexible)
- 1–2 months, full-time preferred, part-time negotiable
- All data and code provided on day one — no environment setup time wasted
- Weekly sync, otherwise async and self-directed
- Output: a short technical write-up + notebook(s) that stay in the repo

---

## How to apply

Send a short note (half a page is fine) covering:
1. What signals/audio/DSP work you've done, even if small or coursework
2. One piece of Python code you wrote that you're happy with — a notebook, a script, anything
3. When you're available and for how long
