#!/usr/bin/env python3
"""
Parser for ODAS classifier output JSON files.

This module provides utilities to read and parse the JSON output from the
ODAS SST module with YAMNet classification enabled.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class TrackClassification:
    """Represents a classified sound source track."""
    id: int
    tag: str
    x: float
    y: float
    z: float
    activity: float
    type: str
    frame_count: int
    class_id: int
    class_name: str
    class_confidence: float
    class_timestamp: int
    bins: Optional[List[float]] = None
    fingerprint: Optional[List[float]] = None


class OdasClassifierParser:
    """Parser for ODAS classifier JSON output."""
    
    def __init__(self, log_dir: Optional[str] = None):
        if log_dir:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path(__file__).resolve().parent / "ClassifierLogs"

    def _iter_json_objects(self, stream_text: str):
        """Yield JSON objects from mixed/concatenated JSON text.

        Supports:
        - Newline-delimited JSON objects
        - Multiple JSON objects concatenated without separators
        - Pretty-printed multi-line JSON objects
        """
        decoder = json.JSONDecoder()
        idx = 0
        n = len(stream_text)

        while idx < n:
            while idx < n and stream_text[idx].isspace():
                idx += 1
            if idx >= n:
                break

            # Fast path: use the built-in raw decoder when possible.
            try:
                obj, next_idx = decoder.raw_decode(stream_text, idx)
                yield obj
                idx = next_idx
                continue
            except json.JSONDecodeError:
                pass

            # Fallback path: brace-balanced extraction for malformed streams.
            if stream_text[idx] != '{':
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
                    elif ch == '\\':
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == '{':
                    brace_count += 1
                elif ch == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        candidate = stream_text[start:j + 1]
                        try:
                            yield json.loads(candidate)
                        except json.JSONDecodeError:
                            # Skip invalid chunk and continue scanning.
                            pass
                        idx = j + 1
                        found = True
                        break

            if not found:
                # Trailing incomplete fragment.
                break
    
    def parse_session_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Parse a session live JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            List of frame dictionaries with timestamp and track data
        """
        frames = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            stream_text = f.read()

        for frame_data in self._iter_json_objects(stream_text):
            if isinstance(frame_data, dict):
                frames.append(frame_data)
        return frames
    
    def extract_tracks(self, frame_data: Dict[str, Any]) -> List[TrackClassification]:
        """
        Extract track objects from a frame.
        
        Args:
            frame_data: Single frame dictionary from JSON
            
        Returns:
            List of TrackClassification objects
        """
        tracks = []
        for src in frame_data.get('src', []):
            track = TrackClassification(
                id=src.get('id'),
                tag=src.get('tag'),
                x=src.get('x'),
                y=src.get('y'),
                z=src.get('z'),
                activity=src.get('activity'),
                type=src.get('type'),
                frame_count=src.get('frame_count'),
                class_id=src.get('class_id', -1),
                class_name=src.get('class_name', 'unclassified'),
                class_confidence=src.get('class_confidence', 0.0),
                class_timestamp=src.get('class_timestamp', 0),
                bins=src.get('bins'),
                fingerprint=src.get('fingerprint')
            )
            tracks.append(track)
        return tracks
    
    def get_latest_session_file(self, pattern: str = "sst_session_live.json_*.json") -> Optional[Path]:
        """
        Find the most recent session file.
        
        Args:
            pattern: Glob pattern for session files
            
        Returns:
            Path to the latest file or None
        """
        files = sorted(self.log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        return files[0] if files else None
    
    def filter_tracks_by_class(self, tracks: List[TrackClassification], 
                               class_names: List[str]) -> List[TrackClassification]:
        """
        Filter tracks by classification name.
        
        Args:
            tracks: List of tracks
            class_names: List of class names to filter (e.g., ['Speech', 'Music'])
            
        Returns:
            Filtered list of tracks
        """
        return [t for t in tracks if t.class_name in class_names]
    
    def filter_tracks_by_confidence(self, tracks: List[TrackClassification], 
                                    min_confidence: float = 0.5) -> List[TrackClassification]:
        """
        Filter tracks by minimum classification confidence.
        
        Args:
            tracks: List of tracks
            min_confidence: Minimum confidence threshold (0.0 to 1.0)
            
        Returns:
            Filtered list of tracks
        """
        return [t for t in tracks if t.class_confidence >= min_confidence]
    
    def summarize_session(self, filepath: str) -> Dict[str, Any]:
        """
        Generate a summary of a session file.
        
        Args:
            filepath: Path to the session JSON file
            
        Returns:
            Summary dictionary with statistics
        """
        frames = self.parse_session_file(filepath)
        
        total_frames = len(frames)
        total_tracks = 0
        class_counts = {}
        unique_track_ids = set()
        
        for frame in frames:
            tracks = self.extract_tracks(frame)
            total_tracks += len(tracks)
            
            for track in tracks:
                unique_track_ids.add(track.id)
                if track.class_name != 'unclassified':
                    class_counts[track.class_name] = class_counts.get(track.class_name, 0) + 1
        
        return {
            'filepath': str(filepath),
            'total_frames': total_frames,
            'total_track_detections': total_tracks,
            'unique_tracks': len(unique_track_ids),
            'class_distribution': class_counts
        }


def example_usage():
    """Example usage of the parser."""
    parser = OdasClassifierParser()
    
    # Find latest session file
    latest_file = parser.get_latest_session_file()
    if not latest_file:
        print("No session files found")
        return
    
    print(f"Parsing: {latest_file}")
    
    # Parse all frames
    frames = parser.parse_session_file(str(latest_file))
    print(f"Found {len(frames)} frames")
    
    # Process first frame
    if frames:
        first_frame = frames[0]
        print(f"\nFrame timestamp: {first_frame.get('timeStamp')}")
        
        tracks = parser.extract_tracks(first_frame)
        print(f"Tracks in first frame: {len(tracks)}")
        
        for track in tracks:
            print(f"  Track {track.id}: {track.class_name} "
                  f"(confidence: {track.class_confidence:.2f}, "
                  f"direction: ({track.x:.2f}, {track.y:.2f}, {track.z:.2f}))")
    
    # Generate summary
    summary = parser.summarize_session(str(latest_file))
    print(f"\nSession Summary:")
    print(f"  Total frames: {summary['total_frames']}")
    print(f"  Unique tracks: {summary['unique_tracks']}")
    print(f"  Class distribution: {summary['class_distribution']}")


if __name__ == "__main__":
    example_usage()
