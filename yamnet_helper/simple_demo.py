"""
Simple Blackbox YAMNet Classifier Demo

This script demonstrates using YAMNet as a simple blackbox:
- Input: magnitude spectra (257 bins per frame)
- Output: audio class predictions

Works with both Python TFLite and C++ implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Try to use C++ wrapper first, fall back to Python
try:
    from yamnet_c_wrapper import create_yamnet_classifier
    print("Using automatic implementation selection...")
except:
    print("Using Python-only implementation...")
    from yamnet_spectrum_classifier import YAMNetSpectrumClassifier as create_yamnet_classifier


class SimpleYAMNetClassifier:
    """
    Simple blackbox wrapper for YAMNet classification.
    
    Usage:
        classifier = SimpleYAMNetClassifier()
        
        # Feed spectra one at a time
        for spectrum in my_spectra:
            result = classifier.predict(spectrum)
            if result:
                print(f"Detected: {result['class']}")
    """
    
    def __init__(self, 
                 model_path='/home/azureuser/z_odas_newbeamform/models/yamnet_core.tflite',
                 class_map_path='/home/azureuser/z_odas_newbeamform/models/yamnet_class_map.csv'):
        """
        Initialize the classifier.
        
        Args:
            model_path: Path to TFLite model
            class_map_path: Path to class names CSV
        """
        self.classifier = create_yamnet_classifier(model_path, class_map_path)
        print("✓ Classifier ready")
    
    def predict(self, spectrum: np.ndarray) -> dict:
        """
        Add a spectrum frame and get prediction when ready.
        
        Args:
            spectrum: numpy array of 257 float values (magnitude spectrum)
            
        Returns:
            Dictionary with 'class_id', 'class', 'confidence' if ready,
            None otherwise
        """
        result = self.classifier.add_frame(spectrum)
        
        if result:
            class_id, class_name, confidence = result
            return {
                'class_id': class_id,
                'class': class_name,
                'confidence': confidence
            }
        
        return None
    
    def predict_batch(self, spectra: np.ndarray) -> dict:
        """
        Classify a full batch of 96 spectra immediately.
        
        Args:
            spectra: numpy array of shape (96, 257)
            
        Returns:
            Dictionary with 'class_id', 'class', 'confidence'
        """
        class_id, class_name, confidence = self.classifier.classify_patch(spectra)
        
        return {
            'class_id': class_id,
            'class': class_name,
            'confidence': confidence
        }
    
    def reset(self):
        """Clear internal buffer."""
        self.classifier.reset()


def demo_frame_by_frame():
    """Demo: Process spectra frame-by-frame."""
    print("\n" + "="*70)
    print("DEMO 1: Frame-by-Frame Classification")
    print("="*70)
    
    classifier = SimpleYAMNetClassifier()
    
    print("\nGenerating 150 random spectra...")
    predictions = []
    
    for i in range(150):
        # Generate synthetic spectrum (random for demo)
        spectrum = np.random.rand(257).astype(np.float32) * 0.5
        
        # Feed to classifier
        result = classifier.predict(spectrum)
        
        if result:
            predictions.append(result)
            print(f"Frame {i:3d}: {result['class']:30s} (confidence: {result['confidence']:.3f})")
    
    print(f"\nTotal predictions: {len(predictions)}")


def demo_batch():
    """Demo: Classify a full batch."""
    print("\n" + "="*70)
    print("DEMO 2: Batch Classification")
    print("="*70)
    
    classifier = SimpleYAMNetClassifier()
    
    print("\nGenerating batch of 96 × 257 spectra...")
    
    # Create a batch
    spectra_batch = np.random.rand(96, 257).astype(np.float32) * 0.5
    
    # Classify
    result = classifier.predict_batch(spectra_batch)
    
    print(f"\nPrediction:")
    print(f"  Class:      {result['class']}")
    print(f"  ID:         {result['class_id']}")
    print(f"  Confidence: {result['confidence']:.3f}")


def demo_with_audio():
    """Demo: Process real audio file."""
    print("\n" + "="*70)
    print("DEMO 3: Real Audio File Processing")
    print("="*70)
    
    # Check for sample audio files
    sample_paths = [
        "/home/azureuser/audio_cache/test",
        "/home/azureuser/audio_cache/birds_sample",
        "/home/azureuser/audio_cache/elephants_sample"
    ]
    
    audio_file = None
    for path in sample_paths:
        path_obj = Path(path)
        if path_obj.exists():
            audio_files = list(path_obj.glob("*.wav")) + list(path_obj.glob("*.mp3"))
            if audio_files:
                audio_file = audio_files[0]
                break
    
    if not audio_file:
        print("\n⚠ No audio files found in sample directories")
        print("Skipping real audio demo")
        return
    
    print(f"\nProcessing: {audio_file.name}")
    
    try:
        import librosa
        
        # Load audio
        audio, sr = librosa.load(str(audio_file), sr=16000, duration=10.0)
        print(f"Loaded: {len(audio)} samples ({len(audio)/sr:.2f}s)")
        
        # Initialize classifier
        classifier = SimpleYAMNetClassifier()
        
        # Process in windows
        window_size = 512
        hop_size = 160
        
        predictions = []
        num_windows = (len(audio) - window_size) // hop_size + 1
        
        print(f"Processing {num_windows} windows...")
        
        for i in range(min(num_windows, 200)):  # Limit to 200 frames
            start = i * hop_size
            end = start + window_size
            
            if end > len(audio):
                break
            
            # Extract window and compute FFT
            window = audio[start:end] * np.hanning(window_size)
            spectrum = np.abs(np.fft.rfft(window, n=window_size)).astype(np.float32)
            
            # Classify
            result = classifier.predict(spectrum)
            
            if result:
                timestamp = start / sr
                predictions.append((timestamp, result))
                print(f"  [{timestamp:6.2f}s] {result['class']:30s} ({result['confidence']:.3f})")
        
        # Summary
        if predictions:
            print(f"\nDetected {len(predictions)} classifications")
            
            # Count classes
            class_counts = {}
            for _, pred in predictions:
                cls = pred['class']
                class_counts[cls] = class_counts.get(cls, 0) + 1
            
            print("\nTop classes:")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {cls:30s}: {count}")
    
    except ImportError:
        print("\n⚠ librosa not installed, skipping audio demo")
        print("Install with: pip install librosa")
    except Exception as e:
        print(f"\n⚠ Error processing audio: {e}")


def main():
    """Run all demos."""
    print("="*70)
    print("Simple YAMNet Classifier - Blackbox Demo")
    print("="*70)
    
    try:
        demo_frame_by_frame()
        demo_batch()
        demo_with_audio()
        
        print("\n" + "="*70)
        print("✓ All demos completed")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
