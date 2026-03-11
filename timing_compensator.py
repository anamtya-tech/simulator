"""
ODAS Timing Compensator

Utility module to handle timing discrepancies between:
- ODAS timeStamp (hop counter)
- YAMNet classification latency (96 frame accumulation + 48 frame hop)
- Rolling 6-hop voting delay
- Kalman filter startup/persistence
- Ground truth annotations

Usage:
    compensator = TimingCompensator()
    
    for detection in odas_detections:
        timing = compensator.analyze_detection_timing(detection)
        print(f"Sound estimated at: {timing.estimated_sound_start:.3f}s")
        print(f"Uncertainty: ±{timing.confidence_interval:.3f}s")
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json


@dataclass
class DetectionTiming:
    """
    Comprehensive timing information for an ODAS detection
    """
    # Raw timestamps
    odas_timestamp: int              # Raw ODAS timeStamp (hop counter)
    emission_time: float             # When ODAS emitted this line (seconds)
    
    # YAMNet timing
    frame_count: int                 # YAMNet frames accumulated
    frames_since_last: int           # Frames since last classification
    yamnet_accumulation_start: float # When YAMNet started accumulating (seconds)
    yamnet_accumulation_end: float   # When YAMNet finished = emission_time (seconds)
    yamnet_latency: float            # Total YAMNet latency (seconds)
    
    # Event voting timing
    event_votes: int                 # Number of hops that voted
    voting_window_duration: float    # Duration of voting window (seconds)
    first_vote_time: float           # Estimated time of first vote (seconds)
    
    # Combined estimates
    estimated_sound_start: float     # Best estimate when sound started (seconds)
    estimated_sound_end: float       # Best estimate when sound ended (seconds)
    confidence_interval: float       # Uncertainty ±seconds
    total_system_latency: float      # End-to-end latency (seconds)
    
    # Kalman state (if available)
    kalman_converged: bool           # Whether Kalman has converged
    kalman_startup_delay: float      # Estimated Kalman startup time (seconds)
    kalman_observations: int         # Number of Kalman observations
    
    # Metadata
    is_first_classification: bool    # True if this is first YAMNet classification
    classification_number: int       # Sequential classification count
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'odas_timestamp': self.odas_timestamp,
            'emission_time': self.emission_time,
            'frame_count': self.frame_count,
            'frames_since_last': self.frames_since_last,
            'yamnet': {
                'accumulation_start': self.yamnet_accumulation_start,
                'accumulation_end': self.yamnet_accumulation_end,
                'latency': self.yamnet_latency,
            },
            'event': {
                'votes': self.event_votes,
                'window_duration': self.voting_window_duration,
                'first_vote_time': self.first_vote_time,
            },
            'estimates': {
                'sound_start': self.estimated_sound_start,
                'sound_end': self.estimated_sound_end,
                'confidence_interval': self.confidence_interval,
                'total_latency': self.total_system_latency,
            },
            'kalman': {
                'converged': self.kalman_converged,
                'startup_delay': self.kalman_startup_delay,
                'observations': self.kalman_observations,
            },
            'is_first_classification': self.is_first_classification,
            'classification_number': self.classification_number,
        }


class TimingCompensator:
    """
    Compensate for timing discrepancies in ODAS pipeline
    """
    
    # Constants from ODAS/YAMNet configuration
    ODAS_HOP_DURATION = 0.008        # 8ms per hop (128 samples @ 16kHz)
    YAMNET_FRAME_DURATION = 0.010    # 10ms per frame (160 samples @ 16kHz)
    YAMNET_PATCH_SIZE = 96           # Frames per classification
    YAMNET_PATCH_HOP = 48            # Frames between classifications
    ROLLING_HOPS = 6                 # Rolling window size
    HOP_INTERVAL = 0.048             # ~48ms between rolling hops
    
    def __init__(self, audio_start_offset: float = 0.0):
        """
        Initialize timing compensator
        
        Args:
            audio_start_offset: Offset to add to all times (seconds)
        """
        self.audio_start_offset = audio_start_offset
        self.first_odas_timestamp = None
        self.classification_counts = {}  # Per-track classification counter
        
    def normalize_odas_timestamp(self, odas_timestamp: int) -> float:
        """
        Convert ODAS timeStamp (hop counter) to absolute audio time (seconds)
        
        Args:
            odas_timestamp: Raw ODAS timeStamp value
            
        Returns:
            Absolute time in seconds
        """
        if self.first_odas_timestamp is None:
            self.first_odas_timestamp = odas_timestamp
            
        # Time since ODAS started
        relative_time = (odas_timestamp - self.first_odas_timestamp) * self.ODAS_HOP_DURATION
        
        # Add any offset (e.g., if audio didn't start at t=0)
        return self.audio_start_offset + relative_time
    
    def analyze_detection_timing(self, detection: Dict) -> DetectionTiming:
        """
        Analyze complete timing for an ODAS detection
        
        Args:
            detection: ODAS detection dictionary with keys:
                - odas_timestamp or timeStamp (int)
                - timestamp (float, optional)
                - frame_count (int)
                - event_votes (int, optional)
                - track_id (int, optional)
                - kalman_state (dict, optional)
                
        Returns:
            DetectionTiming object with complete timing analysis
        """
        # Get raw timestamp
        odas_ts = detection.get('odas_timestamp', detection.get('timeStamp', 0))
        emission_time = detection.get('timestamp')
        if emission_time is None:
            emission_time = self.normalize_odas_timestamp(odas_ts)
        
        # YAMNet parameters
        frame_count = detection.get('frame_count', 0)
        event_votes = detection.get('event_votes', 0)
        track_id = detection.get('track_id', detection.get('id', 0))
        
        # Determine if first classification for this track
        if track_id not in self.classification_counts:
            self.classification_counts[track_id] = 0
            is_first = True
            frames_since_last = 0
        else:
            is_first = False
            frames_since_last = self.YAMNET_PATCH_HOP
        
        self.classification_counts[track_id] += 1
        classification_number = self.classification_counts[track_id]
        
        # Calculate YAMNet timing
        if is_first:
            # First classification: needed full 96 frames
            yamnet_latency = self.YAMNET_PATCH_SIZE * self.YAMNET_FRAME_DURATION  # 960ms
            confidence_interval = 0.200  # ±200ms for first classification
        else:
            # Subsequent: only 48 frames since last
            yamnet_latency = self.YAMNET_PATCH_HOP * self.YAMNET_FRAME_DURATION  # 480ms
            confidence_interval = 0.100  # ±100ms for subsequent
        
        yamnet_accumulation_end = emission_time
        yamnet_accumulation_start = emission_time - yamnet_latency
        
        # Calculate event voting timing
        if event_votes > 0:
            # Each vote represents one rolling hop (~48ms)
            voting_window_duration = event_votes * self.HOP_INTERVAL
            first_vote_time = emission_time - voting_window_duration
        else:
            voting_window_duration = 0.0
            first_vote_time = emission_time
        
        # Total system latency
        total_latency = yamnet_latency + voting_window_duration
        
        # Estimate when sound actually started
        # Conservative: assume sound started at beginning of YAMNet accumulation window
        estimated_sound_start = yamnet_accumulation_start
        
        # For sound end: if Kalman is persistent, detection continues after sound ends
        # We can't reliably estimate end time from a single detection
        estimated_sound_end = emission_time  # Placeholder
        
        # Kalman state analysis
        kalman_state = detection.get('kalman_state', {})
        if kalman_state:
            kalman_converged = self._check_kalman_convergence(kalman_state)
            kalman_obs = kalman_state.get('observations', kalman_state.get('observations_count', 0))
            kalman_startup_delay = kalman_obs * self.ODAS_HOP_DURATION
        else:
            # Estimate from frame_count and track history
            kalman_converged = frame_count > 0 or classification_number > 2
            kalman_obs = max(int(yamnet_latency / self.ODAS_HOP_DURATION), 10)
            kalman_startup_delay = 0.0  # Unknown
        
        return DetectionTiming(
            odas_timestamp=odas_ts,
            emission_time=emission_time,
            frame_count=frame_count,
            frames_since_last=frames_since_last,
            yamnet_accumulation_start=yamnet_accumulation_start,
            yamnet_accumulation_end=yamnet_accumulation_end,
            yamnet_latency=yamnet_latency,
            event_votes=event_votes,
            voting_window_duration=voting_window_duration,
            first_vote_time=first_vote_time,
            estimated_sound_start=estimated_sound_start,
            estimated_sound_end=estimated_sound_end,
            confidence_interval=confidence_interval,
            total_system_latency=total_latency,
            kalman_converged=kalman_converged,
            kalman_startup_delay=kalman_startup_delay,
            kalman_observations=kalman_obs,
            is_first_classification=is_first,
            classification_number=classification_number,
        )
    
    def check_temporal_overlap(
        self, 
        gt_start: float, 
        gt_end: float, 
        detection: Dict
    ) -> Tuple[bool, float, Dict]:
        """
        Check if detection temporally overlaps with ground truth interval
        
        Args:
            gt_start: Ground truth start time (seconds)
            gt_end: Ground truth end time (seconds)
            detection: ODAS detection dictionary
            
        Returns:
            Tuple of (has_overlap, confidence, timing_info)
            - has_overlap: True if detection overlaps with GT interval
            - confidence: Overlap confidence (0.0-1.0)
            - timing_info: Dictionary with detailed timing breakdown
        """
        timing = self.analyze_detection_timing(detection)
        
        # Detection interval (with confidence interval)
        det_start = timing.yamnet_accumulation_start - timing.confidence_interval
        det_end = timing.emission_time + timing.confidence_interval
        
        # Check for overlap
        overlap_start = max(gt_start, det_start)
        overlap_end = min(gt_end, det_end)
        
        if overlap_start <= overlap_end:
            # Overlap exists
            overlap_duration = overlap_end - overlap_start
            gt_duration = gt_end - gt_start
            det_duration = det_end - det_start
            
            # Confidence based on proportion of overlap
            confidence = min(
                overlap_duration / gt_duration,   # Proportion of GT covered
                overlap_duration / det_duration   # Proportion of detection in GT
            )
            
            timing_info = {
                'gt_interval': (gt_start, gt_end),
                'detection_interval': (det_start, det_end),
                'overlap_interval': (overlap_start, overlap_end),
                'overlap_duration': overlap_duration,
                'confidence': confidence,
                'timing': timing.to_dict(),
            }
            
            return True, confidence, timing_info
        else:
            # No overlap
            # Calculate how far apart they are
            if det_start > gt_end:
                gap = det_start - gt_end
                direction = 'detection_after_gt'
            else:
                gap = gt_start - det_end
                direction = 'detection_before_gt'
            
            timing_info = {
                'gt_interval': (gt_start, gt_end),
                'detection_interval': (det_start, det_end),
                'gap': gap,
                'direction': direction,
                'timing': timing.to_dict(),
            }
            
            return False, 0.0, timing_info
    
    def _check_kalman_convergence(self, kalman_state: Dict) -> bool:
        """
        Check if Kalman filter has converged based on state metrics
        
        Args:
            kalman_state: Dictionary with Kalman state metrics
            
        Returns:
            True if converged, False otherwise
        """
        # Criteria for convergence:
        # 1. Low position uncertainty (< 5cm)
        # 2. Multiple observations (>= 10)
        # 3. High convergence score (>= 0.9) if available
        
        position_uncertainty = kalman_state.get('position_uncertainty', 999.0)
        observations = kalman_state.get('observations', kalman_state.get('observations_count', 0))
        convergence = kalman_state.get('convergence', kalman_state.get('convergence_score', 0.0))
        
        return (
            position_uncertainty < 0.05 and
            observations >= 10 and
            convergence >= 0.9
        )
    
    def estimate_track_lifetime(self, detections: List[Dict], 
                               activity_threshold: float = 0.5,
                               confidence_threshold: float = 0.6) -> Dict:
        """
        Estimate the actual lifetime of a track (sound source) from multiple detections
        
        IMPORTANT: Track lifetime != Sound duration
        - Track lifetime includes Kalman startup delay and persistence tail
        - Sound duration is when sound was actually present (based on activity/confidence)
        
        Args:
            detections: List of detections for the same track
            activity_threshold: Minimum activity level to consider sound present (default 0.5)
            confidence_threshold: Minimum YAMNet confidence to consider sound present (default 0.6)
            
        Returns:
            Dictionary with:
                - track_start: First detection time (includes startup delay)
                - track_end: Last detection time (includes persistence)
                - sound_start: Estimated when sound actually started
                - sound_end: Estimated when sound actually ended
                - active_detections: Number of detections with actual sound
                - startup_delay: Kalman startup delay
                - persistence_tail: Kalman persistence after sound ended
        """
        if not detections:
            return {}
        
        # Sort by emission time
        sorted_dets = sorted(detections, key=lambda d: d.get('timestamp', 0))
        
        # Analyze first and last detections
        first_timing = self.analyze_detection_timing(sorted_dets[0])
        last_timing = self.analyze_detection_timing(sorted_dets[-1])
        
        # Track lifetime (includes startup + persistence)
        track_start = first_timing.emission_time
        track_end = last_timing.emission_time
        
        # Filter detections that likely have actual sound
        active_detections = []
        for det in sorted_dets:
            activity = det.get('activity', 0.0)
            confidence = det.get('class_confidence', 
                               det.get('event_max_confidence', 
                               det.get('event_avg_confidence', 0.0)))
            
            # Consider detection "active" if it has sufficient energy AND classification confidence
            if activity >= activity_threshold and confidence >= confidence_threshold:
                active_detections.append(det)
        
        # Estimate sound duration from active detections only
        if active_detections:
            first_active = self.analyze_detection_timing(active_detections[0])
            last_active = self.analyze_detection_timing(active_detections[-1])
            
            # Sound likely started at beginning of first active detection's YAMNet window
            sound_start = first_active.yamnet_accumulation_start
            
            # Sound likely ended at emission of last active detection
            # (or slightly before, but this is conservative)
            sound_end = last_active.emission_time
            
            # Calculate delays
            startup_delay = sound_start - track_start
            persistence_tail = track_end - sound_end
        else:
            # No active detections - entire track may be noise/persistence
            sound_start = track_start
            sound_end = track_start  # Duration = 0
            startup_delay = 0.0
            persistence_tail = track_end - track_start
        
        return {
            # Track lifetime (first to last detection, includes delays)
            'track_start': track_start,
            'track_end': track_end,
            'track_duration': track_end - track_start,
            
            # Sound duration (when sound actually present)
            'sound_start': sound_start,
            'sound_end': sound_end,
            'sound_duration': sound_end - sound_start,
            
            # Detection counts
            'total_detections': len(detections),
            'active_detections': len(active_detections),
            'inactive_detections': len(detections) - len(active_detections),
            
            # Timing delays
            'startup_delay': startup_delay,
            'persistence_tail': persistence_tail,
            
            # Kalman state
            'first_is_converged': first_timing.kalman_converged,
            'last_is_converged': last_timing.kalman_converged,
            
            # Filtering criteria used
            'activity_threshold': activity_threshold,
            'confidence_threshold': confidence_threshold,
        }
    
    def generate_timing_report(self, detections: List[Dict]) -> str:
        """
        Generate a human-readable timing report for detections
        
        Args:
            detections: List of ODAS detections
            
        Returns:
            Formatted string report
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ODAS TIMING ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        for i, det in enumerate(detections[:10]):  # First 10 detections
            timing = self.analyze_detection_timing(det)
            
            lines.append(f"Detection #{i+1}")
            lines.append(f"  ODAS timeStamp: {timing.odas_timestamp}")
            lines.append(f"  Emission time:  {timing.emission_time:.3f}s")
            lines.append(f"")
            lines.append(f"  YAMNet:")
            lines.append(f"    Accumulation: [{timing.yamnet_accumulation_start:.3f}s - {timing.yamnet_accumulation_end:.3f}s]")
            lines.append(f"    Latency:      {timing.yamnet_latency:.3f}s")
            lines.append(f"    Frame count:  {timing.frame_count}")
            lines.append(f"    First class:  {timing.is_first_classification}")
            lines.append(f"")
            lines.append(f"  Event Voting:")
            lines.append(f"    Votes:        {timing.event_votes}")
            lines.append(f"    Window:       {timing.voting_window_duration:.3f}s")
            lines.append(f"    First vote:   {timing.first_vote_time:.3f}s")
            lines.append(f"")
            lines.append(f"  Estimates:")
            lines.append(f"    Sound start:  {timing.estimated_sound_start:.3f}s (±{timing.confidence_interval:.3f}s)")
            lines.append(f"    Total latency:{timing.total_system_latency:.3f}s")
            lines.append(f"")
            lines.append(f"  Kalman:")
            lines.append(f"    Converged:    {timing.kalman_converged}")
            lines.append(f"    Observations: {timing.kalman_observations}")
            lines.append("-" * 80)
        
        if len(detections) > 10:
            lines.append(f"... and {len(detections) - 10} more detections")
        
        return "\n".join(lines)


def visualize_timing_comparison(
    gt_sources: List[Dict],
    detections: List[Dict],
    output_file: Optional[str] = None
):
    """
    Create a timeline visualization comparing GT intervals with detection intervals
    
    Args:
        gt_sources: List of ground truth sources with start_time, end_time, label
        detections: List of ODAS detections
        output_file: Optional path to save plot
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("Matplotlib required for visualization")
        return
    
    compensator = TimingCompensator()
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Plot ground truth intervals
    for i, src in enumerate(gt_sources):
        start = src.get('start_time', 0)
        end = src.get('end_time', 0)
        label = src.get('label', 'unknown')
        
        ax.barh(i * 2, end - start, left=start, height=0.8, 
                color='green', alpha=0.5, label='Ground Truth' if i == 0 else "")
        ax.text(start + (end - start) / 2, i * 2, label, 
                ha='center', va='center', fontweight='bold')
    
    # Plot detection intervals with timing compensation
    for i, det in enumerate(detections):
        timing = compensator.analyze_detection_timing(det)
        
        # YAMNet accumulation window
        y_pos = len(gt_sources) * 2 + i * 0.3
        ax.barh(y_pos, timing.yamnet_latency, 
                left=timing.yamnet_accumulation_start, 
                height=0.2, color='blue', alpha=0.6,
                label='YAMNet Window' if i == 0 else "")
        
        # Emission point
        ax.plot(timing.emission_time, y_pos, 'ro', markersize=4,
                label='Detection Emission' if i == 0 else "")
        
        # Confidence interval
        ax.plot([timing.estimated_sound_start - timing.confidence_interval,
                timing.estimated_sound_start + timing.confidence_interval],
               [y_pos, y_pos], 'k-', linewidth=2, alpha=0.3)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Sources / Detections', fontsize=12)
    ax.set_title('ODAS Timeline: Ground Truth vs Detections (with timing compensation)', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Timeline saved to: {output_file}")
    else:
        plt.show()


# Example usage
if __name__ == "__main__":
    # Example detection from ODAS
    example_detection = {
        'timeStamp': 1425,
        'timestamp': 11.400,
        'frame_count': 96,
        'event_votes': 5,
        'track_id': 1,
        'x': 0.5, 'y': 0.3, 'z': 0.1,
        'class_name': 'Wolf',
        'class_confidence': 0.87,
    }
    
    compensator = TimingCompensator()
    
    # Analyze timing
    timing = compensator.analyze_detection_timing(example_detection)
    
    print("Detection Timing Analysis:")
    print(f"  Emission time:      {timing.emission_time:.3f}s")
    print(f"  YAMNet window:      [{timing.yamnet_accumulation_start:.3f}s - {timing.yamnet_accumulation_end:.3f}s]")
    print(f"  Est. sound start:   {timing.estimated_sound_start:.3f}s (±{timing.confidence_interval:.3f}s)")
    print(f"  Total latency:      {timing.total_system_latency:.3f}s")
    print(f"  Event votes:        {timing.event_votes}")
    print(f"  First vote time:    {timing.first_vote_time:.3f}s")
    
    # Check overlap with ground truth
    has_overlap, confidence, timing_info = compensator.check_temporal_overlap(
        gt_start=10.0,
        gt_end=15.0,
        detection=example_detection
    )
    
    print(f"\nGround Truth Overlap:")
    print(f"  Has overlap:        {has_overlap}")
    print(f"  Confidence:         {confidence:.2%}")
    if has_overlap:
        print(f"  Overlap interval:   {timing_info['overlap_interval']}")
        print(f"  Overlap duration:   {timing_info['overlap_duration']:.3f}s")
