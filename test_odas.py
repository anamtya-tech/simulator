"""
Test Suite for ODAS Improved Processor

Tests various scenarios from simple to complex to ensure the processor
handles both cases accurately.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import List, Dict, Tuple
from odas import (
    ODASProcessor, MicArray, SSLPot, SSTTrack
)


class SyntheticSceneGenerator:
    """Generate synthetic audio scenes for testing"""
    
    def __init__(self, mic_array: MicArray, duration: float = 60.0):
        self.mic_array = mic_array
        self.duration = duration
        self.sample_rate = mic_array.sample_rate
        self.n_samples = int(duration * self.sample_rate)
        
    def generate_simple_scene(self, sources: List[Dict]) -> np.ndarray:
        """
        Generate simple scene with non-overlapping sources
        
        sources: List of dicts with keys:
            - frequency: Hz
            - azimuth: degrees (-180 to 180)
            - time_range: (start, end) in seconds
            - amplitude: 0-1
        """
        # 6 channels: 2 unused + 4 mics
        audio = np.zeros((6, self.n_samples))
        t = np.arange(self.n_samples) / self.sample_rate
        
        for source in sources:
            freq = source['frequency']
            azimuth = source['azimuth']
            t_start, t_end = source['time_range']
            amplitude = source.get('amplitude', 0.5)
            
            # Generate signal
            signal = amplitude * np.sin(2 * np.pi * freq * t)
            signal[t < t_start] = 0
            signal[t > t_end] = 0
            
            # Apply delays based on azimuth
            az_rad = np.radians(azimuth)
            direction = np.array([np.cos(az_rad), np.sin(az_rad), 0])
            
            for mic_idx, pos in enumerate(self.mic_array.positions):
                # Calculate time delay
                delay_distance = np.dot(pos, direction)
                delay_time = delay_distance / self.mic_array.speed_of_sound
                delay_samples = int(delay_time * self.sample_rate)
                
                # Apply delay
                if delay_samples != 0:
                    shifted = np.roll(signal, delay_samples)
                    if delay_samples > 0:
                        shifted[:delay_samples] = 0
                    else:
                        shifted[delay_samples:] = 0
                    audio[mic_idx + 2] += shifted
                else:
                    audio[mic_idx + 2] += signal
        
        return audio
    
    def generate_complex_scene(self, sources: List[Dict], 
                             snr_db: float = 20,
                             reverb_level: float = 0.1) -> np.ndarray:
        """
        Generate complex scene with noise, overlapping sources, and reverb
        """
        # Start with simple scene
        audio = self.generate_simple_scene(sources)
        
        # Add white noise
        if snr_db < 100:
            signal_power = np.mean(audio[2:6]**2)
            noise_power = signal_power / (10**(snr_db/10))
            noise = np.random.randn(4, self.n_samples) * np.sqrt(noise_power)
            audio[2:6] += noise
        
        # Simple reverb simulation (delay + attenuation)
        if reverb_level > 0:
            delay_ms = 50  # 50ms reverb tail
            delay_samples = int(delay_ms * self.sample_rate / 1000)
            
            for ch in range(2, 6):
                reverb = np.roll(audio[ch], delay_samples) * reverb_level
                reverb[:delay_samples] = 0
                audio[ch] += reverb
        
        return audio
    
    def generate_moving_source(self, frequency: float,
                              start_azimuth: float,
                              end_azimuth: float,
                              time_range: Tuple[float, float]) -> np.ndarray:
        """Generate a moving source"""
        audio = np.zeros((6, self.n_samples))
        t = np.arange(self.n_samples) / self.sample_rate
        t_start, t_end = time_range
        
        for sample_idx in range(self.n_samples):
            time = sample_idx / self.sample_rate
            
            if t_start <= time <= t_end:
                # Interpolate azimuth
                progress = (time - t_start) / (t_end - t_start)
                azimuth = start_azimuth + progress * (end_azimuth - start_azimuth)
                
                # Generate sample
                signal_value = 0.5 * np.sin(2 * np.pi * frequency * time)
                
                # Apply to each mic with appropriate delay
                az_rad = np.radians(azimuth)
                direction = np.array([np.cos(az_rad), np.sin(az_rad), 0])
                
                for mic_idx, pos in enumerate(self.mic_array.positions):
                    delay_distance = np.dot(pos, direction)
                    delay_time = delay_distance / self.mic_array.speed_of_sound
                    phase_shift = 2 * np.pi * frequency * delay_time
                    
                    audio[mic_idx + 2, sample_idx] = signal_value * np.cos(phase_shift)
        
        return audio


def test_simple_static_sources():
    """Test 1: Simple static sources (non-overlapping)"""
    print("\n" + "="*60)
    print("TEST 1: Simple Static Sources")
    print("="*60)
    
    # Setup
    mic_positions = np.array([
        [-0.032, 0.000, 0.000],  # Left (-X)
        [0.000, -0.032, 0.000],  # Back (-Y)
        [0.032, 0.000, 0.000],   # Right (+X)
        [0.000, 0.032, 0.000]    # Front (+Y)
    ])
    mic_array = MicArray(positions=mic_positions)
    
    # Generate scene
    generator = SyntheticSceneGenerator(mic_array, duration=30)
    sources = [
        {'frequency': 300, 'azimuth': 0, 'time_range': (5, 10)},    # +X axis
        {'frequency': 500, 'azimuth': 90, 'time_range': (15, 20)},  # +Y axis
        {'frequency': 700, 'azimuth': -90, 'time_range': (25, 30)}, # -Y axis
    ]
    
    audio = generator.generate_simple_scene(sources)
    
    # Save and process
    audio_int16 = (audio.T * 32767).astype(np.int16)
    test_file = './test1_simple.raw'
    audio_int16.tofile(test_file)
    
    processor = ODASProcessor(mic_array)
    results = processor.process_file(test_file)
    
    # Analyze results
    detections = analyze_detections(results, sources)
    
    print("\nExpected vs Detected:")
    for i, source in enumerate(sources):
        print(f"\nSource {i+1}:")
        print(f"  Expected: {source['azimuth']}° at {source['frequency']}Hz "
              f"from {source['time_range'][0]}-{source['time_range'][1]}s")
        
        if i < len(detections):
            det = detections[i]
            print(f"  Detected: {det['mean_azimuth']:.1f}° ± {det['std_azimuth']:.1f}° "
                  f"from {det['start_time']:.1f}-{det['end_time']:.1f}s")
            print(f"  Error: {det['azimuth_error']:.1f}°, "
                  f"Timing error: {det['timing_error']:.1f}s")
        else:
            print("  NOT DETECTED!")
    
    return results


def test_overlapping_sources():
    """Test 2: Overlapping sources"""
    print("\n" + "="*60)
    print("TEST 2: Overlapping Sources")
    print("="*60)
    
    mic_positions = np.array([
        [-0.032, 0.000, 0.000],
        [0.000, -0.032, 0.000],
        [0.032, 0.000, 0.000],
        [0.000, 0.032, 0.000]
    ])
    mic_array = MicArray(positions=mic_positions)
    
    generator = SyntheticSceneGenerator(mic_array, duration=20)
    sources = [
        {'frequency': 400, 'azimuth': 45, 'time_range': (5, 15)},   # Long source
        {'frequency': 600, 'azimuth': -45, 'time_range': (10, 20)}, # Overlapping
    ]
    
    audio = generator.generate_simple_scene(sources)
    
    audio_int16 = (audio.T * 32767).astype(np.int16)
    test_file = './test2_overlapping.raw'
    audio_int16.tofile(test_file)
    
    processor = ODASProcessor(mic_array)
    results = processor.process_file(test_file)
    
    # Check if both sources detected during overlap
    overlap_start, overlap_end = 10, 15
    overlap_frames = [f for f in results['frames'] 
                     if overlap_start <= f['timestamp'] <= overlap_end]
    
    simultaneous_detections = []
    for frame in overlap_frames:
        if len(frame['tracks']) >= 2:
            simultaneous_detections.append(frame)
    
    print(f"\nOverlap period: {overlap_start}-{overlap_end}s")
    print(f"Frames with 2+ sources: {len(simultaneous_detections)}/{len(overlap_frames)}")
    
    if simultaneous_detections:
        frame = simultaneous_detections[len(simultaneous_detections)//2]
        print(f"\nMid-overlap detection at {frame['timestamp']:.1f}s:")
        for track in frame['tracks']:
            print(f"  Track {track['track_id']}: {track['azimuth']:.1f}°")
    
    return results


def test_moving_source():
    """Test 3: Moving source"""
    print("\n" + "="*60)
    print("TEST 3: Moving Source")
    print("="*60)
    
    mic_positions = np.array([
        [-0.032, 0.000, 0.000],
        [0.000, -0.032, 0.000],
        [0.032, 0.000, 0.000],
        [0.000, 0.032, 0.000]
    ])
    mic_array = MicArray(positions=mic_positions)
    
    generator = SyntheticSceneGenerator(mic_array, duration=20)
    
    # Moving source from -90° to +90° over 10 seconds
    audio = generator.generate_moving_source(
        frequency=500,
        start_azimuth=-90,
        end_azimuth=90,
        time_range=(5, 15)
    )
    
    audio_int16 = (audio.T * 32767).astype(np.int16)
    test_file = './test3_moving.raw'
    audio_int16.tofile(test_file)
    
    processor = ODASProcessor(mic_array)
    results = processor.process_file(test_file)
    
    # Track motion
    motion_frames = [f for f in results['frames'] if 5 <= f['timestamp'] <= 15]
    
    if motion_frames:
        # Get the main track
        track_ids = []
        for frame in motion_frames:
            for track in frame['tracks']:
                track_ids.append(track['track_id'])
        
        # Most common track ID
        from collections import Counter
        main_track_id = Counter(track_ids).most_common(1)[0][0] if track_ids else None
        
        if main_track_id:
            # Extract trajectory
            trajectory = []
            for frame in motion_frames:
                for track in frame['tracks']:
                    if track['track_id'] == main_track_id:
                        trajectory.append({
                            'time': frame['timestamp'],
                            'azimuth': track['azimuth'],
                            'is_static': track['is_static']
                        })
            
            if len(trajectory) > 2:
                times = [t['time'] for t in trajectory]
                azimuths = [t['azimuth'] for t in trajectory]
                
                # Fit line to check linear motion
                from scipy import stats
                slope, intercept, r_value, _, _ = stats.linregress(times, azimuths)
                
                print(f"\nMotion Analysis:")
                print(f"  Track ID: {main_track_id}")
                print(f"  Start: {azimuths[0]:.1f}° at {times[0]:.1f}s")
                print(f"  End: {azimuths[-1]:.1f}° at {times[-1]:.1f}s")
                print(f"  Motion rate: {slope:.1f}°/s (expected: 18°/s)")
                print(f"  Linearity (R²): {r_value**2:.3f}")
                
                # Check if detected as moving
                static_count = sum(1 for t in trajectory if t['is_static'])
                print(f"  Detected as static: {static_count}/{len(trajectory)} frames")
    
    return results


def test_noisy_scene():
    """Test 4: Noisy scene"""
    print("\n" + "="*60)
    print("TEST 4: Noisy Scene (10dB SNR)")
    print("="*60)
    
    mic_positions = np.array([
        [-0.032, 0.000, 0.000],
        [0.000, -0.032, 0.000],
        [0.032, 0.000, 0.000],
        [0.000, 0.032, 0.000]
    ])
    mic_array = MicArray(positions=mic_positions)
    
    generator = SyntheticSceneGenerator(mic_array, duration=20)
    sources = [
        {'frequency': 500, 'azimuth': 0, 'time_range': (5, 15)},
    ]
    
    # Generate with noise
    audio = generator.generate_complex_scene(sources, snr_db=10, reverb_level=0)
    
    audio_int16 = (audio.T * 32767).astype(np.int16)
    test_file = './test4_noisy.raw'
    audio_int16.tofile(test_file)
    
    # Process with default settings
    processor = ODASProcessor(mic_array)
    results = processor.process_file(test_file)
    
    # Count detections
    detection_frames = [f for f in results['frames'] 
                       if 5 <= f['timestamp'] <= 15 and f['tracks']]
    
    expected_frames = len([f for f in results['frames'] 
                          if 5 <= f['timestamp'] <= 15])
    
    print(f"\nDetection rate: {len(detection_frames)}/{expected_frames} "
          f"({100*len(detection_frames)/expected_frames:.1f}%)")
    
    if detection_frames:
        azimuths = []
        for frame in detection_frames:
            for track in frame['tracks']:
                azimuths.append(track['azimuth'])
        
        print(f"Average azimuth: {np.mean(azimuths):.1f}° ± {np.std(azimuths):.1f}°")
        print(f"Expected: 0°")
    
    # Count false positives
    false_positive_frames = [f for f in results['frames']
                            if (f['timestamp'] < 5 or f['timestamp'] > 15) and f['tracks']]
    
    print(f"False positives: {len(false_positive_frames)} frames")
    
    return results


def analyze_detections(results: Dict, expected_sources: List[Dict]) -> List[Dict]:
    """Analyze detection accuracy"""
    detections = []
    
    for source in expected_sources:
        t_start, t_end = source['time_range']
        expected_az = source['azimuth']
        
        # Find matching detections
        matching_frames = []
        for frame in results['frames']:
            if t_start <= frame['timestamp'] <= t_end:
                for track in frame['tracks']:
                    az_error = abs(track['azimuth'] - expected_az)
                    # Handle wrap-around
                    if az_error > 180:
                        az_error = 360 - az_error
                    
                    if az_error < 30:  # Within 30 degrees
                        matching_frames.append({
                            'time': frame['timestamp'],
                            'azimuth': track['azimuth'],
                            'track_id': track['track_id']
                        })
        
        if matching_frames:
            times = [f['time'] for f in matching_frames]
            azimuths = [f['azimuth'] for f in matching_frames]
            
            detection = {
                'mean_azimuth': np.mean(azimuths),
                'std_azimuth': np.std(azimuths),
                'start_time': min(times),
                'end_time': max(times),
                'azimuth_error': abs(np.mean(azimuths) - expected_az),
                'timing_error': abs(min(times) - t_start) + abs(max(times) - t_end),
                'detection_rate': len(matching_frames) / ((t_end - t_start) * 
                                 results['metadata']['sample_rate'] / 
                                 results['config']['hop_size'])
            }
            detections.append(detection)
    
    return detections


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "="*70)
    print(" ODAS IMPROVED PROCESSOR - COMPLETE TEST SUITE")
    print("="*70)
    
    test_results = {}
    
    # Test 1: Simple static sources
    results1 = test_simple_static_sources()
    test_results['simple'] = evaluate_test(results1, "Simple Static Sources")
    
    # Test 2: Overlapping sources
    results2 = test_overlapping_sources()
    test_results['overlapping'] = evaluate_test(results2, "Overlapping Sources")
    
    # Test 3: Moving source
    results3 = test_moving_source()
    test_results['moving'] = evaluate_test(results3, "Moving Source")
    
    # Test 4: Noisy scene
    results4 = test_noisy_scene()
    test_results['noisy'] = evaluate_test(results4, "Noisy Scene")
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    for test_name, score in test_results.items():
        status = "PASS" if score > 0.7 else "FAIL"
        print(f"{test_name.capitalize():15} Score: {score:.2f} [{status}]")
    
    overall_score = np.mean(list(test_results.values()))
    print(f"\nOverall Score: {overall_score:.2f}")
    
    if overall_score > 0.8:
        print("✓ System performing well for both simple and complex scenes")
    else:
        print("✗ System needs improvement")
    
    return test_results


def evaluate_test(results: Dict, test_name: str) -> float:
    """Evaluate test results and return score 0-1"""
    score = 1.0
    
    # Check for tracks created
    if results['metadata']['tracks_created'] == 0:
        print(f"  ✗ {test_name}: No tracks created")
        return 0.0
    
    # Check detection rate (simplified)
    frames_with_tracks = sum(1 for f in results['frames'] if f['tracks'])
    total_frames = len(results['frames'])
    
    if frames_with_tracks < total_frames * 0.1:
        score *= 0.5
        print(f"  ⚠ {test_name}: Low detection rate ({frames_with_tracks}/{total_frames})")
    
    return score


if __name__ == "__main__":
    # Run all tests
    test_results = run_all_tests()
    
    # Save summary
    with open('./test_summary.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print("\nTest results saved to: ./test_summary.json")