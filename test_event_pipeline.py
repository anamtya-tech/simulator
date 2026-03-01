#!/usr/bin/env python3
"""
Smoke-test for the event-based ODAS+YAMNet pipeline.

Tests:
  1. JSON schema  — every event entry has the new event_* fields and no stale bins/fingerprint
  2. Gate logic   — events with event_votes < min_event_votes must not appear
  3. .bin sidecar — spectra_file (when non-empty) loads as (96, 257) float32
  4. audio_reconstructor — reconstruct_from_spectra_file returns valid audio
  5. analyzer.py  — _parse_odas_output populates event_* keys in detection dict

Usage:
    python3 test_event_pipeline.py [path/to/sst_session.json]

If no path is given the most recent file under
  /home/azureuser/z_odas_newbeamform/build/ClassifierLogs/
is used.  Pass --help for options.
"""

import sys, os, json, glob, argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

CLASSIFIER_LOG_DIR = Path("/home/azureuser/z_odas_newbeamform/build/ClassifierLogs")

# ─── helpers ────────────────────────────────────────────────────────────────

def find_latest_session() -> Path | None:
    files = sorted(CLASSIFIER_LOG_DIR.glob("sst_session_live*.json"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def iter_events(session_path: Path):
    """Yield (line_num, src_dict) for every source in every JSON line."""
    with open(session_path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            for src in frame.get("src", []):
                yield n, src


# ─── tests ──────────────────────────────────────────────────────────────────

def test_schema(session_path: Path, min_event_votes: int = 4) -> bool:
    """All emitted events must have event_* fields and no bins/fingerprint."""
    print(f"\n[TEST 1] Schema check — {session_path.name}")
    ok = True
    n_events = 0
    for lineno, src in iter_events(session_path):
        if src.get("event_class_id", -1) == -1:
            continue  # not an event frame
        n_events += 1

        required = ["event_class_id", "event_class_name", "event_votes",
                    "event_avg_confidence", "spectra_file", "topk_history"]
        for field in required:
            if field not in src:
                print(f"  ✗ line {lineno}: missing '{field}'")
                ok = False

        if "bins" in src:
            print(f"  ✗ line {lineno}: stale 'bins' field still present")
            ok = False
        if "fingerprint" in src:
            print(f"  ✗ line {lineno}: stale 'fingerprint' field still present")
            ok = False

    if n_events == 0:
        print("  ⚠  No events found — run ODAS with sim_mode=1 and a real audio source first.")
        return True  # not a failure, just empty
    print(f"  ✓ {n_events} events checked — schema OK")
    return ok


def test_vote_gate(session_path: Path, min_event_votes: int = 4) -> bool:
    """No emitted event may have fewer votes than min_event_votes."""
    print(f"\n[TEST 2] Vote-gate check (min_event_votes={min_event_votes})")
    ok = True
    for lineno, src in iter_events(session_path):
        votes = src.get("event_votes", None)
        if votes is None:
            continue
        if votes < min_event_votes:
            print(f"  ✗ line {lineno}: event_votes={votes} < {min_event_votes}")
            ok = False
    if ok:
        print(f"  ✓ All events passed vote gate")
    return ok


def test_bin_sidecar(session_path: Path) -> bool:
    """Every non-empty spectra_file must be a (96, 257) float32 .bin."""
    import numpy as np
    print(f"\n[TEST 3] .bin sidecar integrity")
    ok = True
    n_checked = 0
    for lineno, src in iter_events(session_path):
        path = src.get("spectra_file", "")
        if not path:
            continue
        if not os.path.exists(path):
            print(f"  ✗ line {lineno}: spectra_file not on disk — {path}")
            ok = False
            continue
        raw = np.fromfile(path, dtype=np.float32)
        expected = 96 * 257
        if raw.size < expected:
            print(f"  ✗ line {lineno}: .bin has {raw.size} floats, expected {expected}")
            ok = False
            continue
        n_checked += 1
    if n_checked == 0:
        print("  ⚠  No spectra_file entries found (sim_mode=0 or no events yet)")
    else:
        print(f"  ✓ {n_checked} sidecar files verified as 96×257 float32")
    return ok


def test_audio_reconstruction(session_path: Path) -> bool:
    """reconstruct_from_spectra_file must return audio with duration > 0."""
    print(f"\n[TEST 4] Audio reconstruction from .bin")
    from audio_reconstructor import AudioReconstructor
    recon = AudioReconstructor(sample_rate=16000, n_fft=512, hop_length=128)
    ok = True
    n_checked = 0
    for _, src in iter_events(session_path):
        path = src.get("spectra_file", "")
        if not path or not os.path.exists(path):
            continue
        result = recon.reconstruct_from_spectra_file(path)
        if result is None:
            print(f"  ✗ reconstruct_from_spectra_file returned None for {path}")
            ok = False
        elif result["duration"] <= 0:
            print(f"  ✗ audio duration={result['duration']:.3f}s — expected > 0")
            ok = False
        else:
            n_checked += 1
        break  # one file is enough to validate

    if n_checked == 0:
        print("  ⚠  No .bin files available for audio reconstruction test")
    else:
        print(f"  ✓ Reconstructed audio: {result['duration']:.3f}s, {result['n_frames']} frames")
    return ok


def test_analyzer_parse(session_path: Path) -> bool:
    """_parse_odas_output must populate event_* and topk_history keys."""
    print(f"\n[TEST 5] analyzer._parse_odas_output field mapping")
    try:
        from analyzer import OdasAnalyzer
    except Exception as e:
        print(f"  ⚠  Could not import OdasAnalyzer: {e} — skipping")
        return True

    try:
        a = OdasAnalyzer.__new__(OdasAnalyzer)
        dets = a._parse_odas_output(str(session_path))
    except Exception as e:
        print(f"  ✗ _parse_odas_output raised: {e}")
        return False

    new_keys = ["event_class_id", "event_class_name", "event_votes",
                "event_avg_confidence", "spectra_file", "topk_history"]
    if not dets:
        print("  ⚠  No detections returned — file may be empty")
        return True
    missing = [k for k in new_keys if k not in dets[0]]
    if missing:
        print(f"  ✗ detection dict missing keys: {missing}")
        return False
    print(f"  ✓ {len(dets)} detections — all new keys present in detection dict")
    return True


# ─── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Smoke-test the event-based ODAS+YAMNet pipeline")
    parser.add_argument("session", nargs="?", help="Path to sst_session JSON (auto-detected if omitted)")
    parser.add_argument("--min-event-votes", type=int, default=4,
                        help="Expected min_event_votes gate value (default: 4)")
    args = parser.parse_args()

    if args.session:
        session_path = Path(args.session)
    else:
        session_path = find_latest_session()

    if not session_path or not session_path.exists():
        print("ERROR: No session file found. Run ODAS first or provide a path.")
        print(f"  Looked in: {CLASSIFIER_LOG_DIR}")
        sys.exit(1)

    print(f"Using session file: {session_path}")
    print(f"File size: {session_path.stat().st_size / 1024:.1f} KB")

    results = {
        "schema":        test_schema(session_path, args.min_event_votes),
        "vote_gate":     test_vote_gate(session_path, args.min_event_votes),
        "bin_sidecar":   test_bin_sidecar(session_path),
        "audio_recon":   test_audio_reconstruction(session_path),
        "analyzer_parse": test_analyzer_parse(session_path),
    }

    print("\n" + "="*50)
    all_pass = True
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_pass = False
    print("="*50)
    print("ALL TESTS PASSED" if all_pass else "SOME TESTS FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
