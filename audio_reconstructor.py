"""
Audio Reconstruction Utilities

Advanced audio reconstruction from ODAS frequency bins using:
1. Griffin-Lim algorithm for phase reconstruction
2. Overlap-add for temporal continuity
3. Window functions for smooth transitions

This module provides high-quality audio reconstruction from magnitude spectra.
"""

import numpy as np
import os
from scipy import signal
from scipy.io import wavfile
import warnings


class AudioReconstructor:
    """Reconstruct audio from ODAS frequency bins"""
    
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=128, 
                 frame_duration=0.008, use_griffin_lim=True):
        """
        Initialize audio reconstructor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            n_fft: FFT size
            hop_length: Hop length between frames
            frame_duration: Duration of each frame (seconds)
            use_griffin_lim: Use Griffin-Lim vs random phase
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.frame_duration = frame_duration
        self.use_griffin_lim = use_griffin_lim
        
        # Window function for STFT
        self.window = signal.get_window('hann', n_fft)
    
    def reconstruct_single_frame(self, bins, method='griffin_lim', n_iter=50):
        """
        Reconstruct audio from a single frame of frequency bins.
        
        Args:
            bins: Magnitude spectrum (1024 bins)
            method: 'griffin_lim' or 'random_phase'
            n_iter: Number of Griffin-Lim iterations
        
        Returns:
            np.ndarray: Audio waveform
        """
        bins = np.array(bins, dtype=np.float32)
        
        if method == 'griffin_lim':
            return self._griffin_lim_single(bins, n_iter)
        else:
            return self._random_phase_single(bins)
    
    def reconstruct_multi_frame(self, frames_bins, overlap_frames=3):
        """
        Reconstruct audio from multiple consecutive frames using overlap-add.
        
        Args:
            frames_bins: List of magnitude spectra, one per frame
            overlap_frames: Number of frames to overlap for smoothing
        
        Returns:
            np.ndarray: Reconstructed audio waveform
        """
        if len(frames_bins) == 0:
            return np.array([], dtype=np.float32)
        
        if len(frames_bins) == 1:
            return self.reconstruct_single_frame(frames_bins[0])
        
        # Convert to numpy array
        magnitude_specs = np.array(frames_bins, dtype=np.float32)
        
        # Use STFT inversion with Griffin-Lim
        audio = self._griffin_lim_multi_frame(magnitude_specs)
        
        return audio
    
    def _random_phase_single(self, magnitude):
        """Simple reconstruction with random phase"""
        phase = np.random.uniform(-np.pi, np.pi, len(magnitude))
        complex_spec = magnitude * np.exp(1j * phase)
        audio = np.fft.irfft(complex_spec, n=self.n_fft)
        
        # Apply window
        audio = audio * self.window
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio.astype(np.float32)
    
    def _griffin_lim_single(self, magnitude, n_iter=50):
        """Griffin-Lim algorithm for single frame"""
        magnitude = np.array(magnitude, dtype=np.float32)
        
        # Initialize with random phase
        phase = np.random.uniform(-np.pi, np.pi, len(magnitude))
        complex_spec = magnitude * np.exp(1j * phase)
        
        # Griffin-Lim iterations
        for _ in range(n_iter):
            # Inverse FFT
            audio = np.fft.irfft(complex_spec, n=self.n_fft)
            
            # Apply window
            audio_windowed = audio * self.window
            
            # Forward FFT
            complex_spec = np.fft.rfft(audio_windowed, n=self.n_fft)
            
            # Replace magnitude, keep phase
            phase = np.angle(complex_spec)
            complex_spec = magnitude * np.exp(1j * phase)
        
        # Final inverse FFT
        audio = np.fft.irfft(complex_spec, n=self.n_fft)
        audio = audio * self.window
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio.astype(np.float32)
    
    def _griffin_lim_multi_frame(self, magnitude_specs, n_iter=50):
        """
        Griffin-Lim for multiple frames with overlap-add.
        
        Args:
            magnitude_specs: (n_frames, n_fft//2 + 1) magnitude spectrogram
            n_iter: Number of iterations
        
        Returns:
            np.ndarray: Reconstructed audio
        """
        n_frames = magnitude_specs.shape[0]
        
        # Estimate output length
        audio_length = (n_frames - 1) * self.hop_length + self.n_fft
        
        # Initialize with random phase
        phase = np.random.uniform(-np.pi, np.pi, magnitude_specs.shape)
        complex_spec = magnitude_specs * np.exp(1j * phase)
        
        # Griffin-Lim iterations
        for iteration in range(n_iter):
            # Inverse STFT (overlap-add)
            audio = self._istft(complex_spec)
            
            # Forward STFT
            complex_spec = self._stft(audio)
            
            # Replace magnitude, keep phase
            phase = np.angle(complex_spec)
            
            # Ensure we have the right shape
            min_frames = min(complex_spec.shape[0], magnitude_specs.shape[0])
            complex_spec[:min_frames] = magnitude_specs[:min_frames] * np.exp(1j * phase[:min_frames])
        
        # Final inverse STFT
        audio = self._istft(complex_spec)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio.astype(np.float32)
    
    def _stft(self, audio):
        """Compute STFT"""
        # Pad audio
        audio = np.pad(audio, (self.n_fft // 2, self.n_fft // 2), mode='reflect')
        
        # Calculate number of frames
        n_frames = 1 + (len(audio) - self.n_fft) // self.hop_length
        
        # Allocate output
        spec = np.zeros((n_frames, self.n_fft // 2 + 1), dtype=np.complex64)
        
        # Compute STFT
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            frame = audio[start:end] * self.window
            spec[i] = np.fft.rfft(frame, n=self.n_fft)
        
        return spec
    
    def _istft(self, complex_spec):
        """Compute inverse STFT with overlap-add"""
        n_frames = complex_spec.shape[0]
        audio_length = (n_frames - 1) * self.hop_length + self.n_fft
        
        # Initialize output
        audio = np.zeros(audio_length, dtype=np.float32)
        window_sum = np.zeros(audio_length, dtype=np.float32)
        
        # Overlap-add
        for i in range(n_frames):
            # Inverse FFT
            frame = np.fft.irfft(complex_spec[i], n=self.n_fft)
            
            # Apply window
            frame = frame * self.window
            
            # Add to output
            start = i * self.hop_length
            end = start + self.n_fft
            audio[start:end] += frame
            window_sum[start:end] += self.window
        
        # Normalize by window overlap
        nonzero = window_sum > 1e-10
        audio[nonzero] /= window_sum[nonzero]
        
        # Remove padding
        audio = audio[self.n_fft // 2: -self.n_fft // 2]
        
        return audio
    
    def reconstruct_from_spectra_file(self, spectra_file_path):
        """
        Reconstruct audio from a 96×257 float32 .bin sidecar file.

        These files are written by ODAS when sim_mode=1 — one per YAMNet hop
        (~480ms of spectrogram context at 50% overlap).  Six consecutive hops
        cover ~3s of audio.

        Args:
            spectra_file_path: Path to the .bin file (str or Path).
                               Written by mod_sst.c as:
                               fwrite(patch, sizeof(float), 96*257, fp)

        Returns:
            dict:
                'audio'        – np.ndarray float32 waveform
                'sample_rate'  – int (always self.sample_rate)
                'duration'     – float seconds
                'n_frames'     – int (96 for a full hop)
            or None if the file is missing / unreadable.
        """
        path = str(spectra_file_path)
        if not path or not os.path.exists(path):
            return None

        try:
            raw = np.fromfile(path, dtype=np.float32)
            n_frames_in_file = raw.size // 257
            if n_frames_in_file == 0:
                return None
            frames = raw[: n_frames_in_file * 257].reshape(n_frames_in_file, 257)
        except Exception:
            return None

        audio = self.reconstruct_multi_frame(frames)
        return {
            'audio':       audio,
            'sample_rate': self.sample_rate,
            'duration':    len(audio) / self.sample_rate,
            'n_frames':    n_frames_in_file
        }

    def reconstruct_from_spectra_files(self, spectra_file_paths):
        """
        Stitch audio reconstructed from multiple .bin sidecar files.

        Designed for the 6-hop rolling window: pass the last 6
        ``spectra_file`` paths (oldest → newest) to get ~3 s of audio.

        Args:
            spectra_file_paths: Iterable of .bin file paths (str / Path).
                                Missing or empty-string entries are silently
                                skipped so callers can pass the raw JSON list.

        Returns:
            dict (same schema as reconstruct_from_spectra_file) or None.
        """
        all_frames = []
        for p in spectra_file_paths:
            if not p:
                continue
            try:
                raw = np.fromfile(str(p), dtype=np.float32)
                n = raw.size // 257
                if n > 0:
                    all_frames.append(raw[: n * 257].reshape(n, 257))
            except Exception:
                continue

        if not all_frames:
            return None

        frames = np.vstack(all_frames)          # (total_frames × 257)
        audio = self.reconstruct_multi_frame(frames)
        return {
            'audio':       audio,
            'sample_rate': self.sample_rate,
            'duration':    len(audio) / self.sample_rate,
            'n_frames':    len(frames)
        }

    def reconstruct_from_detections(self, detections, target_duration=1.0):
        """
        Reconstruct audio from a sequence of ODAS detections.
        
        Args:
            detections: List of detection dicts with 'bins' and 'timestamp'
            target_duration: Target duration in seconds
        
        Returns:
            dict: {
                'audio': reconstructed waveform,
                'sample_rate': sample rate,
                'duration': actual duration,
                'n_frames': number of frames used
            }
        """
        # Sort by timestamp
        detections_sorted = sorted(detections, key=lambda x: x.get('timestamp', 0))
        
        # Extract bins
        bins_list = [d['bins'] for d in detections_sorted if len(d.get('bins', [])) > 0]
        
        if not bins_list:
            return {
                'audio': np.array([], dtype=np.float32),
                'sample_rate': self.sample_rate,
                'duration': 0,
                'n_frames': 0
            }
        
        # Reconstruct
        audio = self.reconstruct_multi_frame(bins_list)
        
        # Adjust to target duration if needed
        target_samples = int(target_duration * self.sample_rate)
        
        if len(audio) < target_samples:
            # Pad with silence
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
        elif len(audio) > target_samples:
            # Trim
            audio = audio[:target_samples]
        
        return {
            'audio': audio,
            'sample_rate': self.sample_rate,
            'duration': len(audio) / self.sample_rate,
            'n_frames': len(bins_list)
        }
    
    def save_audio(self, audio, filepath, normalize=True):
        """
        Save audio waveform to WAV file.
        
        Args:
            audio: Audio waveform (float32, range [-1, 1])
            filepath: Output path
            normalize: Normalize to [-1, 1] before saving
        """
        if normalize:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.9
        
        # Ensure in valid range
        audio = np.clip(audio, -1.0, 1.0)
        
        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save
        wavfile.write(filepath, self.sample_rate, audio_int16)


def reconstruct_context_window(detections, target_timestamp, window_duration=1.0):
    """
    Extract and reconstruct audio around a specific timestamp.
    
    Args:
        detections: All detections from ODAS
        target_timestamp: Center timestamp
        window_duration: Total window duration (seconds)
    
    Returns:
        dict: Audio reconstruction result
    """
    half_window = window_duration / 2
    
    # Filter detections within window
    window_detections = [
        d for d in detections
        if abs(d.get('timestamp', 0) - target_timestamp) <= half_window
    ]
    
    if not window_detections:
        return None
    
    # Reconstruct
    reconstructor = AudioReconstructor()
    return reconstructor.reconstruct_from_detections(window_detections, window_duration)


def batch_reconstruct_audio(samples, output_dir, use_griffin_lim=True, progress_callback=None):
    """
    Batch reconstruct audio from multiple samples.
    
    Args:
        samples: List of sample dicts with 'bins' or detections
        output_dir: Directory to save audio files
        use_griffin_lim: Use Griffin-Lim algorithm
        progress_callback: Optional callback(current, total)
    
    Returns:
        list: Paths to saved audio files
    """
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    reconstructor = AudioReconstructor(use_griffin_lim=use_griffin_lim)
    saved_files = []
    
    for i, sample in enumerate(samples):
        try:
            # Extract bins
            if 'bins' in sample:
                audio = reconstructor.reconstruct_single_frame(sample['bins'])
            elif 'detection' in sample and 'bins' in sample['detection']:
                audio = reconstructor.reconstruct_single_frame(sample['detection']['bins'])
            else:
                continue
            
            # Generate filename
            filename = f"sample_{i:05d}.wav"
            if 'label' in sample:
                filename = f"sample_{i:05d}_{sample['label']}.wav"
            
            filepath = output_dir / filename
            
            # Save
            reconstructor.save_audio(audio, filepath)
            saved_files.append(str(filepath))
            
            if progress_callback:
                progress_callback(i + 1, len(samples))
                
        except Exception as e:
            print(f"WARNING: Failed to reconstruct sample {i}: {e}")
            continue
    
    return saved_files
