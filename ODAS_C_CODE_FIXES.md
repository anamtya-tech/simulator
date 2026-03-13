# ODAS C Code Fixes - March 9, 2026

## Issue Summary
Analysis showed that:
1. YAMNet classifier was loading successfully  
2. Tracks were being detected (48 tracks, 5 reached 96+ frames threshold)
3. YAMNet was being called (patch_*.bin debug files confirmed this)
4. **BUT**: No classification data (event fields) appeared in JSON output
5. This caused analyzer.py to get 0 results (no `class_id`/`class_name` to match)

## Root Cause
Event gating in `mod_sst.c` line ~2003 was too strict:
```c
if (ev.votes >= obj->min_event_votes) has_event = 1;
```

Even with `min_event_votes = 1` in config, the `compute_event()` function was returning `votes=0` or classification wasn't happening, preventing ANY event data from being emitted.

## Fix Applied

**File**: `/home/azureuser/z_odas_newbeamform/src/module/mod_sst.c`  
**Backup**: `/home/azureuser/z_odas_newbeamform/src/module/mod_sst.c.backup`  
**Lines**: ~2001-2013 (previously was single line at ~2003)

### Changed Code

**Before:**
```c
if (obj->topk_count[i] >= 1) {
    ev = compute_event(obj, i);
    if (ev.votes >= obj->min_event_votes) has_event = 1;
}
```

**After:**
```c
if (obj->topk_count[i] >= 1) {
    ev = compute_event(obj, i);
    printf("[EVENT_DEBUG] Track %llu: topk_count=%d, ev.class_id=%d, ev.votes=%d, min_event_votes=%d\n",
           obj->ids[i], obj->topk_count[i], ev.class_id, ev.votes, obj->min_event_votes);
    // Force event output if we have ANY classification data
    // The analyzer will handle filtering - our job is to emit all data
    if (ev.class_id >= 0 && ev.votes >= 1) {
        has_event = 1;
    } else if (ev.class_id >= 0) {
        // Even with votes=0, emit if we have a class_id
        printf("[EVENT_WARN] Track %llu has class_id=%d but votes=%d, forcing output\n",
               obj->ids[i], ev.class_id, ev.votes);
        has_event = 1;
    }
}
```

### Key Changes

1. **Added debug output**: Shows `topk_count`, `class_id`, `votes`, and `min_event_votes` for every track
2. **Relaxed gating**: Emits event fields if `class_id >= 0` (valid classification) regardless of vote count
3. **Added warning**: Alerts when `votes=0` but classification exists (helps diagnose `compute_event()` issues)
4. **Philosophy change**: ODAS emits ALL available data; analyzer.py handles filtering

## Configuration Updates

**File**: `/home/azureuser/z_odas_newbeamform/config/runtime/local_socket.cfg`  
**Source**: Copied from `/home/azureuser/sodas/local_socket.cfg`

**File**: `/home/azureuser/z_odas_newbeamform/vm_socket_emit.py`  
**Source**: Copied from `/home/azureuser/sodas/vm_socket_emit.py`

**File**: `/home/azureuser/simulator/simulator.py`  
**Updated paths:**
- `socket_emit_script`: `/home/azureuser/z_odas_newbeamform/vm_socket_emit.py`
- `odas_config`: `/home/azureuser/z_odas_newbeamform/config/runtime/local_socket.cfg`
- `odaslive_bin`: `/home/azureuser/z_odas_newbeamform/build/bin/odaslive` (unchanged)

## Build Status

✅ ODAS rebuilt successfully with fixes  
```bash
cd /home/azureuser/z_odas_newbeamform/build
make -j$(nproc)
```

Build completed without errors. New binary: `/home/azureuser/z_odas_newbeamform/build/bin/odaslive`

## Expected Behavior

After this fix:
1. **ODAS will emit event fields** (`event_class_id`, `event_class_name`, `event_votes`, `event_avg_confidence`, `event_candidates`) for ANY track with YAMNet classifications
2. **Debug output** will show why events are/aren't being emitted
3. **analyzer.py** can now match detections to ground truth using classification data
4. **Event filtering** happens in analyzer.py (using the backward-compatible filtering code added earlier)

## Testing

Run a new simulation and check:
1. ODAS log contains `[EVENT_DEBUG]` and `[YAMNET]` messages
2. Session JSON file has `event_class_id` and other event fields
3. analyzer.py produces matches (not 0 results)

## Rollback

If needed, restore original:
```bash
cp /home/azureuser/z_odas_newbeamform/src/module/mod_sst.c.backup \
   /home/azureuser/z_odas_newbeamform/src/module/mod_sst.c
cd /home/azureuser/z_odas_newbeamform/build && make -j$(nproc)
```

## Related Documentation

- [ODAS_FIRMWARE_ANALYSIS.md](tempdocs/ODAS_FIRMWARE_ANALYSIS.md) - Original analysis of event gating issue
- [ANALYZER_FIXES_SUMMARY.md](ANALYZER_FIXES_SUMMARY.md) - Analyzer.py backward compatibility fix
- [ODAS_TIMING_ANALYSIS.md](ODAS_TIMING_ANALYSIS.md) - YAMNet timing and latency documentation

---

# ODAS C Code Changes — March 13, 2026

---

## Change 1: HPF Mask in `mod_ssl.c` (Active)

**File:** `/home/azureuser/z_odas_newbeamform/src/module/mod_ssl.c`

### Background
The ReSpeaker 4-mic array has a 64 mm diagonal spacing → spatial aliasing at 2680 Hz. Wind noise and room rumble are concentrated below 1200 Hz and produce broad, directionless SRP-PHAT lobes that suppress real animal peaks.

### Change
A zeroing mask is inserted in `mod_ssl_process()` between `freq2freq_product_process()` and `freq2freq_interpolate_process()`. All cross-spectrum bins with centre frequency < 1200 Hz are set to zero for all 6 mic-pair signals:

```c
// HPF mask — zero cross-spectrum bins below freqMinSSL
{
    float freqMinSSL = 1200.0f;
    unsigned int freqMinBin = (unsigned int)(
        freqMinSSL * (float)obj->halfFrameSize / ((float)obj->fS / 2.0f));
    unsigned int iSSLSig, iSSLBin;
    for (iSSLSig = 0; iSSLSig < obj->products->nSignals; iSSLSig++) {
        for (iSSLBin = 0; iSSLBin < freqMinBin; iSSLBin++) {
            obj->products->array[iSSLSig][iSSLBin * 2 + 0] = 0.0f;
            obj->products->array[iSSLSig][iSSLBin * 2 + 1] = 0.0f;
        }
    }
}
```

**At 16 kHz / 512-pt FFT:** bin 0–37 zeroed (0–1187 Hz). Bin 38 onward (1200 Hz+) untouched.

### Effect
- Wind noise (dominant below 500 Hz) no longer contributes to SRP surface
- Sources with energy above 1200 Hz show sharper SRP peaks and higher `probMin` scores
- Sources with energy only below 1200 Hz (e.g. `wolfhowl01.wav`) still produce a flat SRP surface — the fix cannot help them (see wolfhowl48 migration)

---

## Change 2: SSB Frequency Shift (Attempted, Fully Reverted)

**Files touched then reverted:** `mod_stft.c`, `mod_stft.h`, `parameters.c`  
**Files with harmless remnant:** `mod_sss.h`, `mod_sss.c` (`ssbShiftHz = 0` in config)

### What Was Attempted
A Single-Sideband (SSB) forward frequency shift of +2000 Hz was added to `mod_stft.c` after the FFT computation, mapping each bin $k$ → bin $k + B$ where $B = 64$ (= 2000 Hz × 512 / 16000). The goal was to move low-frequency wolf energy (≤ 1 kHz) into the array's unambiguous band (> 2680 Hz) for SRP-PHAT, then reverse the shift in `mod_sss.c` before `yamnet_classify_patch()` so YAMNet sees unshifted spectra.

### Why It Failed
`freq2xcorr.c` performs the cross-correlation via `fft_c2r` — the FFTW real-output inverse FFT. This function assumes the input is Hermitian-symmetric (i.e. the spectrum of a real signal). After a bin-shift, the spectrum is **no longer Hermitian**, and the `c2r` output is no longer $R(\tau)$ but instead:

$$R_\text{shifted}(\tau) = R(\tau) \cdot \cos\!\left(\frac{2\pi B \tau_0}{N}\right)$$

For $B = 64$, $N = 512$, $\tau_0 = 4$ samples (TDOA for 64 mm at 16 kHz):
$$\cos\!\left(\frac{2\pi \times 64 \times 4}{512}\right) = \cos(90°) = 0$$

The SRP peak is completely cancelled. The fix would require changing `freq2xcorr.c` to use a complex IFFT and take the magnitude — an invasive change to the core localisation algorithm.

### Revert
- `mod_stft.h`: `ssbShiftHz`, `halfFrameSize`, `ssbShiftBins` fields removed
- `mod_stft.c`: forward shift block removed; `ssbShiftHz = 0` default removed from `mod_stft_cfg_construct`
- `parameters.c`: `ssbShiftHz` read for `mod_stft` config removed
- `local_socket.cfg`: `ssbShiftHz` set to `0` (disables the reverse-shift no-op in `mod_sss`)
- `mod_sss.h/c`: `ssbShiftHz`/`ssbShiftBins` fields and `ssb_unshift_patch()` helper left in place but are dead code when `ssbShiftHz = 0`

### ⚠️ Caution
If `ssbShiftHz` is ever set to a non-zero value in `local_socket.cfg` without a corresponding forward shift in `mod_stft.c`, the `ssb_unshift_patch()` function will shift YAMNet patches incorrectly, corrupting all classifications. Always keep `ssbShiftHz = 0` unless forward shift is re-implemented with a complex IFFT in `freq2xcorr.c`.
