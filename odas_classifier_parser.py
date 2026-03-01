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
    
    def __init__(self, log_dir: str = "/home/azureuser/sodas/ClassifierLogs"):
        self.log_dir = Path(log_dir)
    
    def parse_session_file(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Parse a session live JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            List of frame dictionaries with timestamp and track data
        """
        frames = []
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    frame_data = json.loads(line)
                    frames.append(frame_data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
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
