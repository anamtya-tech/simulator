"""
YAMNet Spectrum Classifier
Parallel Python implementation of the C++ YAMNet integration in ODAS.

This module accepts pre-computed magnitude spectra (257 bins) and performs
classification using YAMNet, matching the behavior of the C++ implementation.
"""

import numpy as np
import tensorflow as tf
import csv
from pathlib import Path
from typing import Tuple, Optional, List


class YAMNetSpectrumClassifier:
    """
    YAMNet classifier that accepts magnitude spectra as input.
    Mirrors the C++ implementation in z_odas_newbeamform/src/yamnet/
    """
    
    # Constants matching C++ implementation
    SAMPLE_RATE = 16000
    FRAME_LENGTH = 400
    FRAME_STEP = 160
    FFT_SIZE = 512
    SPECTRUM_BINS = 257  # (FFT_SIZE // 2) + 1
    MEL_BINS = 64
    PATCH_FRAMES = 96
    PATCH_HOP = 48  # 50% overlap
    NUM_CLASSES = 521
    MEL_MIN_HZ = 125.0
    MEL_MAX_HZ = 7500.0
    LOG_OFFSET = 0.001
    
    def __init__(self, model_path: str, class_map_path: str):
        """
        Initialize YAMNet classifier.
        
        Args:
            model_path: Path to yamnet_core.tflite model
            class_map_path: Path to yamnet_class_map.csv
        """
        self.model_path = model_path
        self.class_map_path = class_map_path
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Load class names
        self.class_names = self._load_class_names(class_map_path)
        
        # Initialize mel filterbank
        self.mel_filterbank = self._create_mel_filterbank()
        
        # Frame buffer for accumulating spectra
        self.frame_buffer = []
        self.frames_since_last_classification = 0
        
        print(f"YAMNet initialized:")
        print(f"  Model: {model_path}")
        print(f"  Classes: {len(self.class_names)}")
        print(f"  Input shape: {self.input_details[0]['shape']}")
        print(f"  Output shape: {self.output_details[0]['shape']}")
    
    def _load_class_names(self, csv_path: str) -> List[str]:
        """Load class names from CSV file."""
        class_names = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row['display_name'])
        
        assert len(class_names) == self.NUM_CLASSES, \
            f"Expected {self.NUM_CLASSES} classes, got {len(class_names)}"
        
        return class_names
    
    def _hz_to_mel(self, hz: float) -> float:
        """Convert Hz to mel scale."""
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    
    def _mel_to_hz(self, mel: float) -> float:
        """Convert mel to Hz scale."""
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """
        Create mel filterbank matrix.
        Converts linear frequency bins to mel-frequency bins.
        """
        filterbank = np.zeros((self.MEL_BINS, self.SPECTRUM_BINS))
        
        # Create mel scale
        mel_min = self._hz_to_mel(self.MEL_MIN_HZ)
        mel_max = self._hz_to_mel(self.MEL_MAX_HZ)
        mel_points = np.linspace(mel_min, mel_max, self.MEL_BINS + 2)
        
        # Convert mel points to frequency bins
        hz_points = self._mel_to_hz(mel_points)
        bin_points = (self.FFT_SIZE + 1) * hz_points / self.SAMPLE_RATE
        
        # Create triangular filters
        for i in range(self.MEL_BINS):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            
            for j in range(self.SPECTRUM_BINS):
                if left <= j <= center:
                    if center > left:
                        filterbank[i, j] = (j - left) / (center - left)
                elif center < j <= right:
                    if right > center:
                        filterbank[i, j] = (right - j) / (right - center)
        
        return filterbank
    
    def spectrum_to_mel(self, magnitude_spectrum: np.ndarray) -> np.ndarray:
        """
        Convert magnitude spectrum to mel-frequency representation.
        
        Args:
            magnitude_spectrum: Array of shape (257,) containing magnitude values
            
        Returns:
            mel_spectrum: Array of shape (64,) containing log mel values
        """
        assert magnitude_spectrum.shape == (self.SPECTRUM_BINS,), \
            f"Expected shape ({self.SPECTRUM_BINS},), got {magnitude_spectrum.shape}"
        
        # Apply mel filterbank
        mel_spectrum = np.dot(self.mel_filterbank, magnitude_spectrum)
        
        # Apply log with offset
        mel_spectrum = np.log(mel_spectrum + self.LOG_OFFSET)
        
        return mel_spectrum
    
    def classify_patch(self, patch: np.ndarray) -> Tuple[int, str, float]:
        """
        Classify a full patch of 96 frames × 257 bins.
        
        Args:
            patch: Array of shape (96, 257) containing magnitude spectra
            
        Returns:
            (class_id, class_name, confidence)
        """
        assert patch.shape == (self.PATCH_FRAMES, self.SPECTRUM_BINS), \
            f"Expected shape ({self.PATCH_FRAMES}, {self.SPECTRUM_BINS}), got {patch.shape}"
        
        # Convert each spectrum to mel
        mel_patch = np.zeros((self.PATCH_FRAMES, self.MEL_BINS), dtype=np.float32)
        for i in range(self.PATCH_FRAMES):
            mel_patch[i] = self.spectrum_to_mel(patch[i])
        
        # Run inference - add channel dimension for 4D input [batch, frames, mel_bins, channels]
        input_tensor = mel_patch[np.newaxis, :, :, np.newaxis]  # Shape: (1, 96, 64, 1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        
        # Get predictions
        scores = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        # Find top class
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        class_name = self.class_names[class_id] if class_id < len(self.class_names) else "Unknown"
        
        return int(class_id), class_name, float(confidence)
    
    def add_frame(self, magnitude_spectrum: np.ndarray) -> Optional[Tuple[int, str, float]]:
        """
        Add one frame of spectrum and return classification when ready.
        Mimics the frame-by-frame API from C++ implementation.
        
        Args:
            magnitude_spectrum: Array of shape (257,) containing magnitude values
            
        Returns:
            (class_id, class_name, confidence) if classification ready, None otherwise
        """
        assert magnitude_spectrum.shape == (self.SPECTRUM_BINS,), \
            f"Expected shape ({self.SPECTRUM_BINS},), got {magnitude_spectrum.shape}"
        
        # Add frame to buffer
        self.frame_buffer.append(magnitude_spectrum.copy())
        
        # Keep only necessary frames (96 for current + 48 for next overlap)
        if len(self.frame_buffer) > self.PATCH_FRAMES + self.PATCH_HOP:
            self.frame_buffer = self.frame_buffer[self.PATCH_HOP:]
        
        self.frames_since_last_classification += 1
        
        # First classification when we have 96 frames
        if len(self.frame_buffer) == self.PATCH_FRAMES:
            self.frames_since_last_classification = 0
            patch = np.array(self.frame_buffer[-self.PATCH_FRAMES:])
            return self.classify_patch(patch)
        
        # Subsequent classifications every 48 frames (50% overlap)
        if (len(self.frame_buffer) > self.PATCH_FRAMES and 
            self.frames_since_last_classification >= self.PATCH_HOP):
            self.frames_since_last_classification = 0
            patch = np.array(self.frame_buffer[-self.PATCH_FRAMES:])
            return self.classify_patch(patch)
        
        return None
    
    def reset(self):
        """Clear the frame buffer."""
        self.frame_buffer = []
        self.frames_since_last_classification = 0
    
    def get_class_name(self, class_id: int) -> str:
        """Get class name from ID."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return "Unknown"


def compute_magnitude_spectrum(audio_segment: np.ndarray, 
                               window_size: int = 512,
                               hop_size: int = 160) -> np.ndarray:
    """
    Compute magnitude spectrum from audio segment.
    
    Args:
        audio_segment: Audio samples (should be window_size samples)
        window_size: FFT window size (default: 512)
        hop_size: Not used here, but included for API compatibility
        
    Returns:
        magnitude_spectrum: Array of shape (257,) containing magnitude values
    """
    # Ensure we have exactly window_size samples (pad if needed)
    if len(audio_segment) < window_size:
        audio_segment = np.pad(audio_segment, (0, window_size - len(audio_segment)), mode='constant')
    elif len(audio_segment) > window_size:
        audio_segment = audio_segment[:window_size]
    
    # Apply Hann window
    window = np.hanning(window_size)
    windowed = audio_segment * window
    
    # Compute FFT
    fft_result = np.fft.rfft(windowed, n=window_size)
    
    # Compute magnitude
    magnitude = np.abs(fft_result)
    
    return magnitude
