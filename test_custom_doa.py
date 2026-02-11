#!/usr/bin/env python3
"""
Quick test script for custom DOA processor.

Usage:
    python test_custom_doa.py <path_to_raw_file>
"""

import sys
from pathlib import Path
from custom_doa_processor import process_audio_file, ProcessingConfig

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_custom_doa.py <path_to_raw_file>")
        print("\nExample:")
        print("  python test_custom_doa.py outputs/renders/test_20251116_022813.raw")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print("="*60)
    print("Custom DOA Processor - Quick Test")
    print("="*60)
    print(f"Input: {input_file}\n")
    
    # Create custom config for testing
    config = ProcessingConfig(
        sample_rate=16000,
        frame_size=128,  # 8ms
        hop_size=64,     # 4ms overlap
        fft_size=256,
        min_peak_snr=12.0,
        min_frequency=200.0,
        max_frequency=6000.0,
        confidence_window_size=8,
        confidence_threshold=0.625,
        direction_tolerance=20.0,
        frequency_tolerance=100.0
    )
    
    print("Configuration:")
    print(f"  Frame size: {config.frame_size} samples ({config.frame_size/config.sample_rate*1000:.1f}ms)")
    print(f"  Hop size: {config.hop_size} samples ({config.hop_size/config.sample_rate*1000:.1f}ms)")
    print(f"  FFT size: {config.fft_size}")
    print(f"  Frequency range: {config.min_frequency}-{config.max_frequency} Hz")
    print(f"  Min SNR: {config.min_peak_snr} dB")
    print(f"  Confidence window: {config.confidence_window_size} frames")
    print(f"  Confidence threshold: {config.confidence_threshold*100:.1f}%")
    print()
    
    # Generate output filename
    output_file = str(Path(input_file).with_suffix('')) + '_custom_doa.json'
    
    # Process
    results = process_audio_file(input_file, output_file, config)
    
    # Additional analysis
    print("\n" + "="*60)
    print("DETAILED ANALYSIS")
    print("="*60)
    
    if results['summary']['unique_sources'] > 0:
        print(f"\n🎯 Top {min(10, len(results['summary']['sources']))} Sources by Occurrence:")
        print("-" * 60)
        for i, source in enumerate(results['summary']['sources'][:10], 1):
            print(f"{i:2d}. {source['frequency']:7.1f} Hz | "
                  f"Azimuth: {source['azimuth']:6.1f}° | "
                  f"Count: {source['occurrence_count']:4d} | "
                  f"Confidence: {source['avg_confidence']:.3f}")
        
        # Track information
        if 'tracks' in results and results['tracks']:
            print(f"\n🔗 Track Information:")
            print("-" * 60)
            print(f"Total tracks: {len(results['tracks'])}")
            
            # Show top tracks by duration
            tracks_sorted = sorted(results['tracks'], key=lambda x: x['duration_seconds'], reverse=True)
            print(f"\nTop {min(10, len(tracks_sorted))} Longest Tracks:")
            print("-" * 60)
            for i, track in enumerate(tracks_sorted[:10], 1):
                print(f"{i:2d}. Track ID {track['track_id']:3d} | "
                      f"{track['avg_frequency']:7.1f} Hz @ {track['avg_azimuth']:5.1f}° | "
                      f"Duration: {track['duration_seconds']:5.2f}s | "
                      f"Detections: {track['detection_count']:3d}")
        
        # Frequency distribution
        print(f"\n📊 Frequency Distribution:")
        print("-" * 60)
        freq_ranges = [
            (0, 500, "Very Low"),
            (500, 1000, "Low"),
            (1000, 2000, "Mid-Low"),
            (2000, 4000, "Mid-High"),
            (4000, 8000, "High")
        ]
        
        for freq_min, freq_max, label in freq_ranges:
            count = sum(1 for s in results['summary']['sources'] 
                       if freq_min <= s['frequency'] < freq_max)
            if count > 0:
                bar = "█" * min(count, 40)
                print(f"{label:12s} ({freq_min:4d}-{freq_max:4d} Hz): {bar} {count}")
        
        # Direction distribution
        print(f"\n🧭 Direction Distribution:")
        print("-" * 60)
        direction_ranges = [
            (315, 45, "Front (±45°)"),
            (45, 135, "Right (45-135°)"),
            (135, 225, "Back (135-225°)"),
            (225, 315, "Left (225-315°)")
        ]
        
        for az_min, az_max, label in direction_ranges:
            if az_min > az_max:  # Handle wrap-around
                count = sum(1 for s in results['summary']['sources'] 
                           if s['azimuth'] >= az_min or s['azimuth'] < az_max)
            else:
                count = sum(1 for s in results['summary']['sources'] 
                           if az_min <= s['azimuth'] < az_max)
            if count > 0:
                bar = "█" * min(count, 40)
                print(f"{label:20s}: {bar} {count}")
    
    print("\n✅ Processing complete!")
    print(f"📄 Results saved to: {output_file}")

if __name__ == "__main__":
    main()
