"""
Diagnostic tool to debug why ODAS isn't detecting sources

This will process a few frames and show what's happening in the SSL stage
"""

import numpy as np
from pathlib import Path
import json
import sys

from odas_optimized import ODASProcessorOptimized, MicArray

def diagnose_audio(raw_file_path):
    """Diagnose why audio isn't being detected"""
    
    print("="*60)
    print("ODAS Detection Diagnostic")
    print("="*60)
    
    # Load audio
    print(f"\n1. Loading audio: {raw_file_path}")
    audio_int16 = np.fromfile(raw_file_path, dtype=np.int16)
    n_samples = len(audio_int16) // 6
    audio_6ch = audio_int16.reshape(n_samples, 6).T
    # Extract mic channels 1-4 (indices 1,2,3,4) to match renderer output
    mic_audio = audio_6ch[1:5, :].astype(np.float32) / 32767.0
    
    duration = n_samples / 16000
    print(f"   Duration: {duration:.2f}s")
    print(f"   Samples: {n_samples}")
    
    # Check signal levels
    rms_levels = np.sqrt(np.mean(mic_audio**2, axis=1))
    print(f"\n2. Signal levels (RMS):")
    for i, rms in enumerate(rms_levels):
        db = 20 * np.log10(rms + 1e-10)
        print(f"   Mic {i}: {rms:.6f} ({db:.1f} dB)")
    
    if np.max(rms_levels) < 0.001:
        print("   ⚠️  WARNING: Signal levels very low!")
    
    # Setup processor with RELAXED settings
    mic_positions = np.array([
        [-0.032, 0.000, 0.000],
        [0.000, -0.032, 0.000],
        [0.032, 0.000, 0.000],
        [0.000, 0.032, 0.000]
    ])
    mic_array = MicArray(positions=mic_positions)
    
    # Try different coherence thresholds
    thresholds = [0.3, 0.5, 0.65]
    
    print(f"\n3. Testing different coherence thresholds:")
    
    for threshold in thresholds:
        config = {
            'ssl_n_pots': 5,
            'ssl_n_grid_points': 256,
            'ssl_freq_min': 100.0,  # Lower to catch more
            'ssl_freq_max': 8000.0,
            'ssl_min_coherence': threshold,
        }
        
        processor = ODASProcessorOptimized(mic_array, config=config)
        
        # Process middle frames (where audio should be)
        frame_size = 512
        hop_size = 128
        test_frames = [
            n_samples // 4,      # 25% through
            n_samples // 2,      # 50% through
            3 * n_samples // 4,  # 75% through
        ]
        
        total_pots = 0
        best_coherence = 0
        
        for test_pos in test_frames:
            if test_pos + frame_size > mic_audio.shape[1]:
                continue
            
            frame = mic_audio[:, test_pos:test_pos+frame_size]
            frame_idx = test_pos // hop_size
            
            result = processor.process_frame(frame, frame_idx)
            pots = result.get('pots', [])
            total_pots += len(pots)
            
            if pots:
                for pot in pots:
                    if pot['coherence'] > best_coherence:
                        best_coherence = pot['coherence']
        
        print(f"   Threshold {threshold:.2f}: {total_pots} pots detected, "
              f"best coherence: {best_coherence:.3f}")
    
    # Detailed analysis with lowest threshold
    print(f"\n4. Detailed analysis (threshold=0.3):")
    config = {
        'ssl_n_pots': 5,
        'ssl_n_grid_points': 256,
        'ssl_freq_min': 100.0,
        'ssl_freq_max': 8000.0,
        'ssl_min_coherence': 0.3,
    }
    
    processor = ODASProcessorOptimized(mic_array, config=config)
    
    # Process 10 frames in the middle
    start_pos = n_samples // 2
    detections = []
    
    for i in range(10):
        pos = start_pos + i * hop_size
        if pos + frame_size > mic_audio.shape[1]:
            break
        
        frame = mic_audio[:, pos:pos+frame_size]
        frame_idx = pos // hop_size
        
        result = processor.process_frame(frame, frame_idx)
        pots = result.get('pots', [])
        
        if pots:
            for pot in pots:
                detections.append({
                    'time': pos / 16000,
                    'azimuth': pot['azimuth'],
                    'coherence': pot['coherence'],
                    'energy': pot['energy']
                })
    
    if detections:
        print(f"   Found {len(detections)} detections:")
        for det in detections[:5]:  # Show first 5
            print(f"   Time {det['time']:.2f}s: Az={det['azimuth']:6.1f}°, "
                  f"Coh={det['coherence']:.3f}, Energy={det['energy']:.1f}dB")
    else:
        print("   ❌ No detections found even with low threshold!")
        print("   Possible issues:")
        print("      - Audio file is silent or very quiet")
        print("      - Wrong channel mapping")
        print("      - Audio is pure tone (no modulation)")
        print("      - Sample rate mismatch")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS:")
    print("="*60)
    
    if np.max(rms_levels) < 0.001:
        print("📢 INCREASE AUDIO LEVELS")
        print("   Your audio is very quiet. Increase gain when rendering.")
    elif total_pots == 0:
        print("📉 LOWER COHERENCE THRESHOLD")
        print("   Set ssl_min_coherence to 0.3 or lower")
        print("   Expand frequency range: 100-8000 Hz")
    elif best_coherence < 0.65:
        print("🎯 ADJUST THRESHOLD")
        print(f"   Best coherence was {best_coherence:.3f}")
        print(f"   Set ssl_min_coherence to {max(0.3, best_coherence - 0.1):.2f}")
    else:
        print("✅ Detection should work with default settings")
    
    print("="*60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_odas.py <path_to_raw_file>")
        print("\nExample:")
        print("  python diagnose_odas.py outputs/renders/scene_123.raw")
        sys.exit(1)
    
    diagnose_audio(sys.argv[1])
