#!/usr/bin/env python3
"""
Compare ODAS session outputs between two runs and estimate a consistent 2D rotation
between detected source directions.

Usage examples:
  python3 compare_odas_rotation.py \
    --run-a outputs/runs/<run_a>.json \
    --run-b outputs/runs/<run_b>.json

  python3 compare_odas_rotation.py \
    --session-a ClassifierLogs/sst_session_live.json_YYYYMMDD_HHMMSS.json \
    --session-b ClassifierLogs/sst_session_live.json_YYYYMMDD_HHMMSS.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi)."""
    while angle >= math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def iter_json_objects(stream_text: str) -> Iterable[dict]:
    """Yield JSON objects from newline-delimited or concatenated JSON streams."""
    decoder = json.JSONDecoder()
    idx = 0
    n = len(stream_text)

    while idx < n:
        while idx < n and stream_text[idx].isspace():
            idx += 1
        if idx >= n:
            break

        try:
            obj, next_idx = decoder.raw_decode(stream_text, idx)
            if isinstance(obj, dict):
                yield obj
            idx = next_idx
            continue
        except json.JSONDecodeError:
            pass

        if stream_text[idx] != "{":
            idx += 1
            continue

        start = idx
        brace_count = 0
        in_string = False
        escape = False
        found = False

        for j in range(idx, n):
            ch = stream_text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                brace_count += 1
            elif ch == "}":
                brace_count -= 1
                if brace_count == 0:
                    chunk = stream_text[start : j + 1]
                    try:
                        parsed = json.loads(chunk)
                        if isinstance(parsed, dict):
                            yield parsed
                    except json.JSONDecodeError:
                        pass
                    idx = j + 1
                    found = True
                    break

        if not found:
            break


def load_run_metadata(run_json_path: Path) -> Dict:
    with run_json_path.open("r", encoding="utf-8", errors="replace") as f:
        return json.load(f)


def load_session_detections(session_file: Path, warmup_seconds: float = 0.0) -> List[Dict]:
    text = session_file.read_text(encoding="utf-8", errors="replace")
    detections: List[Dict] = []

    for frame in iter_json_objects(text):
        ts_hops = int(frame.get("timeStamp", 0))
        ts_seconds = ts_hops * 0.008 - warmup_seconds

        for src in frame.get("src", []) or []:
            x = float(src.get("x", 0.0))
            y = float(src.get("y", 0.0))
            z = float(src.get("z", 0.0))
            norm_xy = math.hypot(x, y)
            if norm_xy < 1e-9:
                continue

            # Use event class first, then fallback to legacy class name.
            cls = src.get("event_class_name") or src.get("class_name") or "unclassified"
            conf = float(
                src.get("event_max_confidence", src.get("class_confidence", 0.0)) or 0.0
            )

            detections.append(
                {
                    "time_hops": ts_hops,
                    "time_seconds": ts_seconds,
                    "x": x,
                    "y": y,
                    "z": z,
                    "ux": x / norm_xy,
                    "uy": y / norm_xy,
                    "azimuth": math.atan2(y, x),
                    "class_name": str(cls),
                    "confidence": conf,
                    "track_id": int(src.get("id", -1) or -1),
                    "activity": float(src.get("activity", 0.0) or 0.0),
                }
            )

    return detections


def group_by_hops(dets: List[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = {}
    for d in dets:
        grouped.setdefault(int(d["time_hops"]), []).append(d)
    return grouped


def angular_distance(a: float, b: float) -> float:
    return abs(wrap_pi(a - b))


def greedy_match_angles(a_rows: List[Dict], b_rows: List[Dict], max_pair_angle_deg: float) -> List[Tuple[Dict, Dict]]:
    """Greedy bipartite matching by minimum angular distance within one hop frame."""
    if not a_rows or not b_rows:
        return []

    max_pair_angle = math.radians(max_pair_angle_deg)
    candidates: List[Tuple[float, int, int]] = []

    for i, a in enumerate(a_rows):
        for j, b in enumerate(b_rows):
            # Keep loose class gate: if both are confidently classified and differ, skip.
            a_cls = a.get("class_name", "unclassified")
            b_cls = b.get("class_name", "unclassified")
            if a_cls != "unclassified" and b_cls != "unclassified" and a_cls != b_cls:
                continue
            d = angular_distance(a["azimuth"], b["azimuth"])
            if d <= max_pair_angle:
                candidates.append((d, i, j))

    candidates.sort(key=lambda x: x[0])

    used_a = set()
    used_b = set()
    matches: List[Tuple[Dict, Dict]] = []

    for _, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((a_rows[i], b_rows[j]))

    return matches


def pair_detections(
    det_a: List[Dict], det_b: List[Dict], max_hop_diff: int = 1, max_pair_angle_deg: float = 60.0
) -> List[Tuple[Dict, Dict]]:
    """Pair detections by nearest hop index and per-hop angle matching."""
    by_a = group_by_hops(det_a)
    by_b = group_by_hops(det_b)

    matched: List[Tuple[Dict, Dict]] = []

    for hop in sorted(by_a.keys()):
        # Pick B hop with largest overlap quality in local neighborhood.
        best_pairs: List[Tuple[Dict, Dict]] = []
        for b_hop in range(hop - max_hop_diff, hop + max_hop_diff + 1):
            if b_hop not in by_b:
                continue
            pairs = greedy_match_angles(by_a[hop], by_b[b_hop], max_pair_angle_deg=max_pair_angle_deg)
            if len(pairs) > len(best_pairs):
                best_pairs = pairs
        matched.extend(best_pairs)

    return matched


def estimate_rotation(pairs: List[Tuple[Dict, Dict]]) -> Dict[str, float]:
    """Estimate R(theta) that maps A -> B for unit XY vectors."""
    if len(pairs) < 3:
        return {
            "n_pairs": len(pairs),
            "theta_rad": float("nan"),
            "theta_deg": float("nan"),
            "circ_std_deg": float("nan"),
            "mean_abs_error_deg": float("nan"),
            "median_abs_error_deg": float("nan"),
            "p90_abs_error_deg": float("nan"),
            "consistency": 0.0,
        }

    cross_sum = 0.0
    dot_sum = 0.0

    for a, b in pairs:
        ax, ay = a["ux"], a["uy"]
        bx, by = b["ux"], b["uy"]
        cross_sum += ax * by - ay * bx
        dot_sum += ax * bx + ay * by

    theta = math.atan2(cross_sum, dot_sum)

    residuals = []
    deltas = []
    for a, b in pairs:
        da = a["azimuth"]
        db = b["azimuth"]
        delta = wrap_pi(db - da)
        deltas.append(delta)
        residuals.append(abs(wrap_pi(delta - theta)))

    residuals_deg = sorted(math.degrees(r) for r in residuals)
    n = len(residuals_deg)
    mean_abs = sum(residuals_deg) / n
    median_abs = residuals_deg[n // 2]
    p90_abs = residuals_deg[min(n - 1, int(round(0.9 * (n - 1))))]

    # Circular concentration R in [0,1], then circular std in degrees.
    c = sum(math.cos(d - theta) for d in deltas) / n
    s = sum(math.sin(d - theta) for d in deltas) / n
    R = math.hypot(c, s)
    circ_std = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(R, 1e-12)))))

    return {
        "n_pairs": n,
        "theta_rad": theta,
        "theta_deg": math.degrees(theta),
        "circ_std_deg": circ_std,
        "mean_abs_error_deg": mean_abs,
        "median_abs_error_deg": median_abs,
        "p90_abs_error_deg": p90_abs,
        "consistency": R,
    }


def describe_rotation(theta_deg: float) -> str:
    if math.isnan(theta_deg):
        return "insufficient matched pairs"
    direction = "counter-clockwise" if theta_deg >= 0 else "clockwise"
    return f"{abs(theta_deg):.2f} deg {direction} (B relative to A)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ODAS run/session rotation.")
    parser.add_argument("--run-a", type=Path, help="Path to run A JSON metadata")
    parser.add_argument("--run-b", type=Path, help="Path to run B JSON metadata")
    parser.add_argument("--session-a", type=Path, help="Path to session A JSON")
    parser.add_argument("--session-b", type=Path, help="Path to session B JSON")
    parser.add_argument(
        "--max-hop-diff",
        type=int,
        default=1,
        help="Allowed hop mismatch when pairing detections (default: 1)",
    )
    parser.add_argument(
        "--max-pair-angle-deg",
        type=float,
        default=60.0,
        help="Per-pair angle gate in degrees (default: 60)",
    )

    args = parser.parse_args()

    if (args.run_a is None) ^ (args.run_b is None):
        parser.error("provide both --run-a and --run-b, or neither")
    if (args.session_a is None) ^ (args.session_b is None):
        parser.error("provide both --session-a and --session-b, or neither")

    if args.run_a and args.run_b:
        run_a = load_run_metadata(args.run_a)
        run_b = load_run_metadata(args.run_b)

        sess_a = Path(run_a["session_live_file"])
        sess_b = Path(run_b["session_live_file"])
        warmup_a = float(run_a.get("warmup_seconds", 0.0) or 0.0)
        warmup_b = float(run_b.get("warmup_seconds", 0.0) or 0.0)

        label_a = run_a.get("run_id", str(args.run_a))
        label_b = run_b.get("run_id", str(args.run_b))
        cfg_a = run_a.get("selected_odas_config_name", run_a.get("odas_config", "N/A"))
        cfg_b = run_b.get("selected_odas_config_name", run_b.get("odas_config", "N/A"))
    elif args.session_a and args.session_b:
        sess_a = args.session_a
        sess_b = args.session_b
        warmup_a = 0.0
        warmup_b = 0.0
        label_a = str(sess_a)
        label_b = str(sess_b)
        cfg_a = "(session mode)"
        cfg_b = "(session mode)"
    else:
        parser.error("provide run paths or session paths")

    if not sess_a.exists() or not sess_b.exists():
        missing = [str(p) for p in [sess_a, sess_b] if not p.exists()]
        raise FileNotFoundError(f"Session file(s) missing: {missing}")

    det_a = load_session_detections(sess_a, warmup_seconds=warmup_a)
    det_b = load_session_detections(sess_b, warmup_seconds=warmup_b)

    pairs = pair_detections(
        det_a,
        det_b,
        max_hop_diff=args.max_hop_diff,
        max_pair_angle_deg=args.max_pair_angle_deg,
    )
    stats = estimate_rotation(pairs)

    print("=== ODAS Rotation Comparison ===")
    print(f"A: {label_a}")
    print(f"   cfg: {cfg_a}")
    print(f"   session: {sess_a}")
    print(f"   detections: {len(det_a)}")
    print(f"B: {label_b}")
    print(f"   cfg: {cfg_b}")
    print(f"   session: {sess_b}")
    print(f"   detections: {len(det_b)}")
    print("-")
    print(f"matched pairs: {stats['n_pairs']}")
    print(f"estimated rotation: {describe_rotation(stats['theta_deg'])}")
    print(f"consistency (R): {stats['consistency']:.4f} (1.0 = perfect)")
    print(f"circular std around rotation: {stats['circ_std_deg']:.2f} deg")
    print(f"residual mean/median/p90: {stats['mean_abs_error_deg']:.2f} / "
          f"{stats['median_abs_error_deg']:.2f} / {stats['p90_abs_error_deg']:.2f} deg")

    if stats["n_pairs"] >= 10 and stats["consistency"] >= 0.85 and stats["median_abs_error_deg"] <= 8.0:
        print("interpretation: strong evidence of a consistent rigid rotation")
    elif stats["n_pairs"] >= 10 and stats["consistency"] >= 0.65:
        print("interpretation: partial/weak rotation pattern (mixed with jitter or non-rigid effects)")
    else:
        print("interpretation: no strong consistent rotation detected")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
