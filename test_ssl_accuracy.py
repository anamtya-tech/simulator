"""
Test the optimized ODAS SSL stage to verify it detects correct directions
"""

import numpy as np
from odas_optimized import ODASProcessorOptimized, MicArray

def test_ssl_accuracy():
    """Test if SSL correctly identifies source direction"""
    
    print("="*60)
    print("SSL Accuracy Test")
    print("="*60)
    
    # Setup mic array (ReSpeaker)
    mic_positions = np.array([
        [-0.032, 0.000, 0.000],  # Left
        [0.000, -0.032, 0.000],  # Back
        [0.032, 0.000, 0.000],   # Right
        [0.000, 0.032, 0.000]    # Front
    ])
    mic_array = MicArray(positions=mic_positions, sample_rate=16000)
    
    # Test directions
    test_cases = [
        ("Front (0°)", 0.0, 0.0),
        ("Left (90°)", 90.0, 0.0),
        ("Back (180°)", 180.0, 0.0),
        ("Right (-90°)", -90.0, 0.0),
        ("Front-Left (45°)", 45.0, 0.0),
        ("Back-Right (-135°)", -135.0, 0.0),
        ("Back-Left (135°)", 135.0, 0.0),
    ]
    
    sample_rate = 16000
    frame_size = 512
    frequency = 1000  # Hz test tone
    
    for name, true_az, true_el in test_cases:
        print(f"\n{name}")
        print(f"  True azimuth: {true_az:.1f}°")
        
        # Convert to unit vector
        az_rad = np.radians(true_az)
        el_rad = np.radians(true_el)
        direction = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])
        
        # Generate test signal with correct TDOAs
        t = np.arange(frame_size) / sample_rate
        base_signal = np.sin(2 * np.pi * frequency * t)
        
        # Create 4-channel signal with delays
        mic_signals = np.zeros((4, frame_size))
        for mic_idx in range(4):
            # Calculate TDOA for this mic
            tdoa = np.dot(mic_positions[mic_idx], direction) / 343.0
            delay_samples = int(tdoa * sample_rate)
            
            # Apply delay
            if delay_samples > 0:
                mic_signals[mic_idx, delay_samples:] = base_signal[:-delay_samples]
            elif delay_samples < 0:
                mic_signals[mic_idx, :delay_samples] = base_signal[-delay_samples:]
            else:
                mic_signals[mic_idx] = base_signal
        
        # Process with ODAS
        config = {
            'ssl_n_pots': 3,
            'ssl_n_grid_points': 256,
            'ssl_min_coherence': 0.3,
        }
        processor = ODASProcessorOptimized(mic_array, config=config)
        result = processor.process_frame(mic_signals, 0)
        
        pots = result.get('pots', [])
        if pots:
            best_pot = pots[0]  # Highest coherence
            detected_az = best_pot['azimuth']
            detected_coh = best_pot['coherence']
            
            # Calculate error
            az_error = abs(detected_az - true_az)
            if az_error > 180:
                az_error = 360 - az_error
            
            status = "✓" if az_error < 15 else "✗"
            print(f"  Detected: {detected_az:.1f}° (coherence: {detected_coh:.3f})")
            print(f"  Error: {az_error:.1f}° {status}")
        else:
            print(f"  ✗ No detection!")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_ssl_accuracy()
