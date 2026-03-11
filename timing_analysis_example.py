"""
Example: Using Timing Compensator with ODAS Analyzer

This script demonstrates how to integrate the timing compensator
with the analyzer to get more accurate temporal matching.
"""

import json
import sys
from pathlib import Path
from timing_compensator import TimingCompensator, visualize_timing_comparison


def analyze_session_with_timing_compensation(session_file: str, scene_config: dict):
    """
    Analyze an ODAS session with timing compensation
    
    Args:
        session_file: Path to ODAS session_live JSON file
        scene_config: Scene configuration with ground truth sources
    """
    print(f"Analyzing: {session_file}")
    print("=" * 80)
    
    # Initialize compensator
    compensator = TimingCompensator()
    
    # Parse detections from session file
    detections = []
    with open(session_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                time_stamp = data.get('timeStamp', 0)
                
                for src in data.get('src', []):
                    detection = {
                        'line_number': line_num,
                        'timeStamp': time_stamp,
                        'timestamp': time_stamp * 0.008,  # Convert to seconds
                        'frame_count': src.get('frame_count', 0),
                        'event_votes': src.get('event_votes', 0),
                        'track_id': src.get('id', 0),
                        'x': src.get('x', 0),
                        'y': src.get('y', 0),
                        'z': src.get('z', 0),
                        'class_name': src.get('event_class_name', src.get('class_name', 'unclassified')),
                        'class_confidence': src.get('event_max_confidence', src.get('class_confidence', 0.0)),
                    }
                    detections.append(detection)
            except json.JSONDecodeError:
                continue
    
    print(f"Total detections: {len(detections)}")
    print()
    
    # Analyze timing for each detection
    print("Timing Analysis (first 5 detections):")
    print("-" * 80)
    
    for i, det in enumerate(detections[:5]):
        timing = compensator.analyze_detection_timing(det)
        
        print(f"Detection #{i+1}:")
        print(f"  Track ID:           {det['track_id']}")
        print(f"  Classification:     {det['class_name']} ({det['class_confidence']:.2%})")
        print(f"  ODAS timeStamp:     {timing.odas_timestamp}")
        print(f"  Emission time:      {timing.emission_time:.3f}s")
        print(f"  YAMNet window:      [{timing.yamnet_accumulation_start:.3f}s - {timing.yamnet_accumulation_end:.3f}s]")
        print(f"  Est. sound start:   {timing.estimated_sound_start:.3f}s (±{timing.confidence_interval:.3f}s)")
        print(f"  Total latency:      {timing.total_system_latency:.3f}s")
        print(f"  Event votes:        {timing.event_votes}")
        print(f"  Classification #:   {timing.classification_number}")
        print()
    
    # Match detections to ground truth sources with timing compensation
    print("\nGround Truth Matching (with timing compensation):")
    print("=" * 80)
    
    sources = scene_config.get('directional_sources', [])
    
    for src_idx, src in enumerate(sources):
        src_label = src.get('label', 'unknown')
        src_start = src.get('start_time', 0)
        src_end = src.get('end_time', 0)
        
        print(f"\nSource: {src_label}")
        print(f"  GT interval: [{src_start:.3f}s - {src_end:.3f}s]")
        print(f"  Duration:    {src_end - src_start:.3f}s")
        print()
        
        # Find matching detections using timing compensation
        matches = []
        for det in detections:
            has_overlap, confidence, timing_info = compensator.check_temporal_overlap(
                gt_start=src_start,
                gt_end=src_end,
                detection=det
            )
            
            if has_overlap and confidence >= 0.1:  # At least 10% overlap
                matches.append({
                    'detection': det,
                    'confidence': confidence,
                    'timing_info': timing_info
                })
        
        print(f"  Matched detections: {len(matches)}")
        
        if matches:
            # Sort by confidence
            matches.sort(key=lambda m: m['confidence'], reverse=True)
            
            # Show top 3 matches
            print(f"  Top matches:")
            for i, match in enumerate(matches[:3]):
                det = match['detection']
                conf = match['confidence']
                timing_info = match['timing_info']
                
                det_interval = timing_info['detection_interval']
                overlap_interval = timing_info['overlap_interval']
                overlap_duration = timing_info['overlap_duration']
                
                print(f"    #{i+1}: {det['class_name']} ({det['class_confidence']:.2%})")
                print(f"        Track ID:         {det['track_id']}")
                print(f"        Det. interval:    [{det_interval[0]:.3f}s - {det_interval[1]:.3f}s]")
                print(f"        Overlap:          [{overlap_interval[0]:.3f}s - {overlap_interval[1]:.3f}s]")
                print(f"        Overlap duration: {overlap_duration:.3f}s")
                print(f"        Confidence:       {conf:.2%}")
        else:
            print(f"  No matches found!")
            # Find closest detection
            closest_det = None
            min_gap = float('inf')
            
            for det in detections:
                _, _, timing_info = compensator.check_temporal_overlap(
                    gt_start=src_start,
                    gt_end=src_end,
                    detection=det
                )
                
                if 'gap' in timing_info:
                    if timing_info['gap'] < min_gap:
                        min_gap = timing_info['gap']
                        closest_det = (det, timing_info)
            
            if closest_det:
                det, timing_info = closest_det
                print(f"  Closest detection:")
                print(f"    {det['class_name']} ({det['class_confidence']:.2%})")
                print(f"    Track ID:     {det['track_id']}")
                print(f"    Gap:          {timing_info['gap']:.3f}s ({timing_info['direction']})")
    
    # Generate visualization
    print("\n" + "=" * 80)
    print("Generating timeline visualization...")
    
    output_file = Path(session_file).parent / f"{Path(session_file).stem}_timeline.png"
    visualize_timing_comparison(sources, detections[:50], str(output_file))  # First 50 detections
    
    # Generate detailed timing report
    print("\nGenerating detailed timing report...")
    report = compensator.generate_timing_report(detections)
    
    report_file = Path(session_file).parent / f"{Path(session_file).stem}_timing_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_file}")
    
    print("\nAnalysis complete!")


def compare_with_without_compensation(session_file: str, scene_config: dict):
    """
    Compare matching results with and without timing compensation
    """
    compensator = TimingCompensator()
    
    # Parse detections
    detections = []
    with open(session_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                time_stamp = data.get('timeStamp', 0)
                for src in data.get('src', []):
                    detection = {
                        'timeStamp': time_stamp,
                        'timestamp': time_stamp * 0.008,
                        'frame_count': src.get('frame_count', 0),
                        'event_votes': src.get('event_votes', 0),
                        'track_id': src.get('id', 0),
                        'x': src.get('x', 0),
                        'y': src.get('y', 0),
                        'z': src.get('z', 0),
                    }
                    detections.append(detection)
            except json.JSONDecodeError:
                continue
    
    sources = scene_config.get('directional_sources', [])
    
    print("Comparison: With vs Without Timing Compensation")
    print("=" * 80)
    
    for src in sources:
        src_label = src.get('label', 'unknown')
        src_start = src.get('start_time', 0)
        src_end = src.get('end_time', 0)
        
        print(f"\nSource: {src_label} [{src_start:.1f}s - {src_end:.1f}s]")
        
        # Method 1: Without compensation (simple timestamp comparison)
        simple_matches = 0
        window_pre = 5.0
        window_post = 14.0
        
        for det in detections:
            det_time = det['timestamp']
            if (src_start - window_pre) <= det_time <= (src_end + window_post):
                simple_matches += 1
        
        # Method 2: With compensation (interval overlap)
        compensated_matches = 0
        for det in detections:
            has_overlap, confidence, _ = compensator.check_temporal_overlap(
                gt_start=src_start,
                gt_end=src_end,
                detection=det
            )
            if has_overlap:
                compensated_matches += 1
        
        print(f"  Without compensation: {simple_matches} matches")
        print(f"  With compensation:    {compensated_matches} matches")
        print(f"  Difference:           {compensated_matches - simple_matches:+d} matches")


if __name__ == "__main__":
    # Example usage with wolf_frog_ele scene
    
    # Example scene configuration
    example_scene = {
        'directional_sources': [
            {
                'label': 'wolf',
                'start_time': 1.0,
                'end_time': 6.0,
                'position': [1.0, 0.0, 0.0],
            },
            {
                'label': 'frog',
                'start_time': 15.0,
                'end_time': 20.0,
                'position': [0.707, 0.707, 0.0],
            },
            {
                'label': 'elephant',
                'start_time': 25.0,
                'end_time': 30.0,
                'position': [0.0, 1.0, 0.0],
            },
        ]
    }
    
    # Find a session file
    classifier_logs_dir = Path("/home/azureuser/simulator/ClassifierLogs")
    session_files = list(classifier_logs_dir.glob("sst_session_live*.json"))
    
    if session_files:
        print(f"Found {len(session_files)} session files")
        print(f"Analyzing: {session_files[0].name}\n")
        
        # Analyze with timing compensation
        analyze_session_with_timing_compensation(
            str(session_files[0]),
            example_scene
        )
        
        print("\n" + "=" * 80)
        print()
        
        # Compare methods
        compare_with_without_compensation(
            str(session_files[0]),
            example_scene
        )
    else:
        print("No session files found in ClassifierLogs/")
        print("\nUsage:")
        print("  python timing_analysis_example.py <session_file.json>")
        print("\nOr modify the example_scene dictionary in this script")
