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
