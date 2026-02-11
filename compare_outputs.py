#!/usr/bin/env python3
"""
Compare Custom DOA output with SODAS output side-by-side.

Usage:
    python compare_outputs.py <custom_doa.json> <sodas_sst_classify_events.json>
"""

import sys
import json
from pathlib import Path
import pandas as pd


def load_custom_results(filepath):
    """Load custom DOA results"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_sodas_results(filepath):
    """Load SODAS classify events results (JSONL format)"""
    events = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def print_comparison(custom_file, sodas_file):
    """Print side-by-side comparison"""
    
    print("="*80)
    print("CUSTOM DOA vs SODAS OUTPUT COMPARISON")
    print("="*80)
    print(f"\nCustom DOA: {custom_file}")
    print(f"SODAS:      {sodas_file}")
    print()
    
    # Load data
    custom = load_custom_results(custom_file)
    sodas = load_sodas_results(sodas_file)
    
    # High-level comparison
    print("="*80)
    print("HIGH-LEVEL METRICS")
    print("="*80)
    
    print(f"\n{'Metric':<30} | {'Custom DOA':<20} | {'SODAS':<20}")
    print("-"*80)
    
    # Custom metrics
    custom_duration = custom['metadata']['duration']
    custom_sources = custom['summary']['unique_sources']
    custom_detections = custom['metadata']['total_peaks_validated']
    custom_frames = custom['metadata']['frames_processed']
    
    print(f"{'Duration (s)':<30} | {custom_duration:<20.2f} | {'N/A':<20}")
    print(f"{'Unique Sources/Events':<30} | {custom_sources:<20} | {len(sodas):<20}")
    print(f"{'Total Detections':<30} | {custom_detections:<20} | {'N/A':<20}")
    print(f"{'Frames Processed':<30} | {custom_frames:<20} | {'N/A':<20}")
    
    # Custom-specific metrics
    print(f"\n{'Custom DOA Specific Metrics':<30}")
    print("-"*80)
    print(f"Peaks Detected:        {custom['metadata']['total_peaks_detected']}")
    print(f"Peaks Validated:       {custom['metadata']['total_peaks_validated']}")
    print(f"Validation Rate:       {custom['metadata']['validation_rate']*100:.1f}%")
    print(f"Avg Confidence:        {custom['summary'].get('avg_confidence', 0):.3f}")
    
    # Sources breakdown
    print("\n" + "="*80)
    print("DETECTED SOURCES/EVENTS")
    print("="*80)
    
    print(f"\n{'Custom DOA Top Sources':<40} | {'SODAS Events':<35}")
    print("-"*80)
    
    # Custom sources
    custom_sources_list = custom['summary'].get('sources', [])[:10]
    
    # SODAS events
    sodas_events = sodas if isinstance(sodas, list) else []
    
    max_rows = max(len(custom_sources_list), len(sodas_events))
    
    for i in range(max_rows):
        # Custom side
        if i < len(custom_sources_list):
            src = custom_sources_list[i]
            custom_str = f"{src['frequency']:6.1f}Hz @ {src['azimuth']:5.1f}° (n={src['occurrence_count']})"
        else:
            custom_str = ""
        
        # SODAS side
        if i < len(sodas_events):
            event = sodas_events[i]
            # Extract relevant info from SODAS event
            if isinstance(event, dict):
                # Try to extract meaningful fields
                event_type = event.get('event_type', event.get('type', 'unknown'))
                timestamp = event.get('timestamp', event.get('time', 'N/A'))
                sodas_str = f"{event_type} @ t={timestamp}"
            else:
                sodas_str = str(event)[:35]
        else:
            sodas_str = ""
        
        print(f"{custom_str:<40} | {sodas_str:<35}")
    
    # Frequency distribution comparison
    if custom_sources_list:
        print("\n" + "="*80)
        print("FREQUENCY ANALYSIS (Custom DOA)")
        print("="*80)
        
        freq_ranges = [
            (0, 500, "Very Low (0-500 Hz)"),
            (500, 1000, "Low (500-1000 Hz)"),
            (1000, 2000, "Mid-Low (1000-2000 Hz)"),
            (2000, 4000, "Mid-High (2000-4000 Hz)"),
            (4000, 8000, "High (4000-8000 Hz)")
        ]
        
        print(f"\n{'Range':<25} | {'Count':<6} | Distribution")
        print("-"*80)
        
        max_count = 0
        for freq_min, freq_max, label in freq_ranges:
            count = sum(1 for s in custom['summary']['sources'] 
                       if freq_min <= s['frequency'] < freq_max)
            max_count = max(max_count, count)
        
        for freq_min, freq_max, label in freq_ranges:
            count = sum(1 for s in custom['summary']['sources'] 
                       if freq_min <= s['frequency'] < freq_max)
            if max_count > 0:
                bar_length = int(40 * count / max_count)
                bar = "█" * bar_length
            else:
                bar = ""
            print(f"{label:<25} | {count:<6} | {bar}")
        
        # Direction analysis
        print("\n" + "="*80)
        print("DIRECTION ANALYSIS (Custom DOA)")
        print("="*80)
        
        direction_ranges = [
            (315, 45, "Front (±45°)"),
            (45, 135, "Right (45-135°)"),
            (135, 225, "Back (135-225°)"),
            (225, 315, "Left (225-315°)")
        ]
        
        print(f"\n{'Direction':<25} | {'Count':<6} | Distribution")
        print("-"*80)
        
        max_count = 0
        for az_min, az_max, label in direction_ranges:
            if az_min > az_max:
                count = sum(1 for s in custom['summary']['sources'] 
                           if s['azimuth'] >= az_min or s['azimuth'] < az_max)
            else:
                count = sum(1 for s in custom['summary']['sources'] 
                           if az_min <= s['azimuth'] < az_max)
            max_count = max(max_count, count)
        
        for az_min, az_max, label in direction_ranges:
            if az_min > az_max:
                count = sum(1 for s in custom['summary']['sources'] 
                           if s['azimuth'] >= az_min or s['azimuth'] < az_max)
            else:
                count = sum(1 for s in custom['summary']['sources'] 
                           if az_min <= s['azimuth'] < az_max)
            if max_count > 0:
                bar_length = int(40 * count / max_count)
                bar = "█" * bar_length
            else:
                bar = ""
            print(f"{label:<25} | {count:<6} | {bar}")
    
    # SODAS event analysis
    if sodas_events:
        print("\n" + "="*80)
        print("SODAS EVENT DETAILS")
        print("="*80)
        
        print(f"\nTotal events: {len(sodas_events)}")
        
        # Try to categorize SODAS events
        if sodas_events and isinstance(sodas_events[0], dict):
            # Get all keys from first event
            sample_keys = list(sodas_events[0].keys())
            print(f"Event structure keys: {', '.join(sample_keys)}")
            
            # Show first few events in detail
            print(f"\nFirst {min(3, len(sodas_events))} events (detailed):")
            for i, event in enumerate(sodas_events[:3], 1):
                print(f"\nEvent {i}:")
                for key, value in event.items():
                    print(f"  {key}: {value}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\n✓ Custom DOA detected {custom_sources} unique sources")
    print(f"✓ SODAS generated {len(sodas_events)} events")
    
    if custom_sources > len(sodas_events):
        print(f"\n→ Custom DOA detected MORE sources ({custom_sources - len(sodas_events)} additional)")
        print("  This could indicate:")
        print("  - Custom DOA is more sensitive to transient sounds")
        print("  - SODAS may be filtering out short-duration events")
        print("  - Different detection thresholds")
    elif custom_sources < len(sodas_events):
        print(f"\n→ SODAS detected MORE events ({len(sodas_events) - custom_sources} additional)")
        print("  This could indicate:")
        print("  - SODAS tracking is splitting sources over time")
        print("  - Custom DOA may be clustering similar detections")
        print("  - Different temporal windowing")
    else:
        print(f"\n→ Both systems detected the SAME number of sources/events")
        print("  This suggests good agreement between methods")
    
    print()


def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_outputs.py <custom_doa.json> <sodas_output.json>")
        print("\nExample:")
        print("  python compare_outputs.py \\")
        print("    outputs/renders/test_20251116_022813_custom_doa.json \\")
        print("    ../z_odas/ClassifierLogs/sst_classify_events_1763260938.json")
        sys.exit(1)
    
    custom_file = sys.argv[1]
    sodas_file = sys.argv[2]
    
    if not Path(custom_file).exists():
        print(f"Error: Custom DOA file not found: {custom_file}")
        sys.exit(1)
    
    if not Path(sodas_file).exists():
        print(f"Error: SODAS file not found: {sodas_file}")
        sys.exit(1)
    
    print_comparison(custom_file, sodas_file)


if __name__ == "__main__":
    main()
