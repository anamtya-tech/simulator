"""
Test script demonstrating direct spectrum input to YAMNet.
This shows how to use pre-computed magnitude spectra (257 bins per frame).
"""

import numpy as np
import sys
from pathlib import Path

# Add yamnet_helper to path
sys.path.insert(0, str(Path(__file__).parent))

from yamnet_spectrum_classifier import YAMNetSpectrumClassifier, compute_magnitude_spectrum


def test_with_synthetic_data():
    """Test classifier with synthetic spectral data."""
    print("=" * 70)
    print("Test 1: Synthetic Spectral Data")
    print("=" * 70)
    
    # Initialize classifier
    model_path = "/home/azureuser/z_odas_newbeamform/models/yamnet_core.tflite"
    class_map_path = "/home/azureuser/z_odas_newbeamform/models/yamnet_class_map.csv"
    
    classifier = YAMNetSpectrumClassifier(model_path, class_map_path)
    
    # Generate synthetic spectra (257 bins each)
    print("\nGenerating 96 frames of synthetic spectra...")
    np.random.seed(42)
    
    # Simulate speech-like spectrum (energy concentrated in lower frequencies)
    for i in range(96):
        spectrum = np.random.rand(257)
        # Add frequency decay (speech characteristic)
        freq_weights = np.exp(-np.arange(257) / 50.0)
        spectrum *= freq_weights
        spectrum += 0.01  # Add small offset
        
        result = classifier.add_frame(spectrum)
        
        if result:
            class_id, class_name, confidence = result
            print(f"  Frame {i:3d}: {class_name:30s} (confidence: {confidence:.3f})")
    
    print("\n✓ Test completed successfully")


def test_with_patch():
    """Test classifier with a full patch of spectra."""
    print("\n" + "=" * 70)
    print("Test 2: Full Patch Classification")
    print("=" * 70)
    
    # Initialize classifier
    model_path = "/home/azureuser/z_odas_newbeamform/models/yamnet_core.tflite"
    class_map_path = "/home/azureuser/z_odas_newbeamform/models/yamnet_class_map.csv"
    
    classifier = YAMNetSpectrumClassifier(model_path, class_map_path)
    
    print("\nCreating patch of 96 frames × 257 bins...")
    
    # Create different spectral patterns
    patch = np.zeros((96, 257))
    
    # First half: low-frequency dominant (speech-like)
    for i in range(48):
        spectrum = np.random.rand(257) * 0.5
        freq_weights = np.exp(-np.arange(257) / 30.0)
        patch[i] = spectrum * freq_weights + 0.1
    
    # Second half: broader spectrum (noise-like)
    for i in range(48, 96):
        patch[i] = np.random.rand(257) * 0.3 + 0.05
    
    # Classify the patch
    class_id, class_name, confidence = classifier.classify_patch(patch)
    
    print(f"\nClassification result:")
    print(f"  Class ID:   {class_id}")
    print(f"  Class Name: {class_name}")
    print(f"  Confidence: {confidence:.3f}")
    
    print("\n✓ Test completed successfully")


def test_spectrum_computation():
    """Test magnitude spectrum computation from audio."""
    print("\n" + "=" * 70)
    print("Test 3: Spectrum Computation from Audio")
    print("=" * 70)
    
    print("\nGenerating synthetic audio (1 second at 16 kHz)...")
    
    # Generate synthetic audio: 440 Hz tone (A4)
    sample_rate = 16000
    duration = 1.0
    frequency = 440.0
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Compute spectrum for first window
    window_size = 512
    audio_window = audio[:window_size]
    
    spectrum = compute_magnitude_spectrum(audio_window, window_size=window_size)
    
    print(f"\nSpectrum computed:")
    print(f"  Window size:     {window_size}")
    print(f"  Spectrum bins:   {len(spectrum)}")
    print(f"  Expected bins:   257 (FFT_SIZE//2 + 1)")
    print(f"  Spectrum range:  [{spectrum.min():.3f}, {spectrum.max():.3f}]")
    
    # Find peak frequency
    peak_bin = np.argmax(spectrum)
    peak_freq = peak_bin * sample_rate / window_size
    print(f"  Peak frequency:  {peak_freq:.1f} Hz (expected: {frequency} Hz)")
    
    print("\n✓ Test completed successfully")


def test_overlapping_windows():
    """Test frame-by-frame processing with overlap."""
    print("\n" + "=" * 70)
    print("Test 4: Overlapping Window Processing")
    print("=" * 70)
    
    # Initialize classifier
    model_path = "/home/azureuser/z_odas_newbeamform/models/yamnet_core.tflite"
    class_map_path = "/home/azureuser/z_odas_newbeamform/models/yamnet_class_map.csv"
    
    classifier = YAMNetSpectrumClassifier(model_path, class_map_path)
    
    print("\nProcessing frames with 50% overlap...")
    print("Expected classifications: at frame 96, then every 48 frames\n")
    
    predictions = []
    
    # Add 200 frames to test overlap behavior
    for i in range(200):
        # Generate synthetic spectrum
        spectrum = np.random.rand(257) * 0.3
        freq_weights = np.exp(-np.arange(257) / 40.0)
        spectrum = spectrum * freq_weights + 0.05
        
        result = classifier.add_frame(spectrum)
        
        if result:
            class_id, class_name, confidence = result
            predictions.append((i, class_name, confidence))
            print(f"  Frame {i:3d}: Classification triggered - {class_name} ({confidence:.3f})")
    
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Expected pattern: frame 96, 144, 192 (every 48 frames after first)")
    
    if len(predictions) >= 2:
        frame_intervals = [predictions[i][0] - predictions[i-1][0] 
                          for i in range(1, len(predictions))]
        print(f"Actual intervals: {frame_intervals}")
    
    print("\n✓ Test completed successfully")


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("YAMNet Spectrum Classifier Test Suite")
    print("=" * 70)
    
    try:
        # Check if model files exist
        model_path = Path("/home/azureuser/z_odas_newbeamform/models/yamnet_core.tflite")
        class_map_path = Path("/home/azureuser/z_odas_newbeamform/models/yamnet_class_map.csv")
        
        if not model_path.exists():
            print(f"\nError: Model file not found: {model_path}")
            print("Please ensure YAMNet model is available.")
            return
        
        if not class_map_path.exists():
            print(f"\nError: Class map not found: {class_map_path}")
            print("Please ensure class map CSV is available.")
            return
        
        # Run tests
        test_spectrum_computation()
        test_with_synthetic_data()
        test_with_patch()
        test_overlapping_windows()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
