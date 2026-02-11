"""
Custom Direction-of-Arrival (DOA) Processor

Processes raw 6-channel audio files to detect sound source directions.
Uses phase differences between microphones for DOA estimation with
temporal validation through confidence windows.

Algorithm:
1. 8ms frames (128 samples @ 16kHz)
2. FFT-based peak detection
3. Phase difference → Direction (GCC-PHAT)
4. Confidence window validation (n frames)
5. Output averaged peaks with direction

Microphone Array Geometry (ReSpeaker USB 4 Mic Array):
- Mic 1 (Ch 2): [-0.032, 0.000, 0.000]  # Left
- Mic 2 (Ch 3): [0.000, -0.032, 0.000]  # Back
- Mic 3 (Ch 4): [0.032, 0.000, 0.000]   # Right
- Mic 4 (Ch 5): [0.000, 0.032, 0.000]   # Front
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, rfftfreq
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
from pathlib import Path
from collections import deque
from datetime import datetime


@dataclass
class Peak:
    """Represents a detected frequency peak"""
    frequency: float  # Hz
    bin_index: int
    energy: float  # dB
    snr: float  # dB above noise floor


@dataclass
class Detection:
    """Represents a sound source detection"""
    frequency: float  # Hz
    azimuth: float  # degrees (0=front, 90=right, 180=back, 270=left)
    elevation: float  # degrees (0=horizontal, 90=up, -90=down)
    x: float  # Unit vector X
    y: float  # Unit vector Y
    z: float  # Unit vector Z
    energy: float  # dB
    confidence: float  # 0-1
    frame_count: int  # Number of frames this detection persisted
    source_id: Optional[int] = None  # Unique ID for tracking same source over time


@dataclass
class Track:
    """Represents a tracked source over time"""
    track_id: int
    first_frame: int
    last_frame: int
    frequency: float  # Average frequency
    azimuth: float  # Average azimuth
    detections: List[Detection]
    

@dataclass
class ProcessingConfig:
    """Configuration for DOA processing"""
    sample_rate: int = 16000
    frame_size: int = 128  # 8ms @ 16kHz
    hop_size: int = 64  # 50% overlap (4ms)
    fft_size: int = 256  # Zero-padding for better frequency resolution
    
    # Peak detection
    min_peak_snr: float = 12.0  # dB above noise floor
    min_frequency: float = 200.0  # Hz
    max_frequency: float = 6000.0  # Hz
    peak_prominence: float = 3.0  # dB
    
    # Confidence window
    confidence_window_size: int = 8  # frames (64ms)
    confidence_threshold: float = 0.625  # 62.5% of frames (5/8)
    direction_tolerance: float = 20.0  # degrees
    frequency_tolerance: float = 100.0  # Hz
    
    # Tracking
    max_frames_gap: int = 10  # Maximum frames gap to maintain track identity
    track_similarity_tolerance: float = 1.5  # Multiplier for matching (looser than validation)
    
    # Mic array geometry (meters)
    mic_positions: np.ndarray = None
    
    def __post_init__(self):
        if self.mic_positions is None:
            self.mic_positions = np.array([
                [-0.032, 0.000, 0.000],  # Mic 1: Left
                [0.000, -0.032, 0.000],  # Mic 2: Back
                [0.032, 0.000, 0.000],   # Mic 3: Right
                [0.000, 0.032, 0.000]    # Mic 4: Front
            ])


class CustomDOAProcessor:
    """
    Custom Direction-of-Arrival processor for microphone array audio.
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        
        # Speed of sound (m/s)
        self.c = 343.0
        
        # Microphone pair distances
        self.d_LR = np.linalg.norm(self.config.mic_positions[2] - self.config.mic_positions[0])  # Left-Right
        self.d_BF = np.linalg.norm(self.config.mic_positions[3] - self.config.mic_positions[1])  # Back-Front
        
        # Window function for FFT
        self.window = signal.windows.hann(self.config.frame_size)
        
        # Confidence tracking
        self.detection_history = deque(maxlen=self.config.confidence_window_size)
        
        # Source tracking
        self.active_tracks: Dict[int, Track] = {}  # track_id -> Track
        self.next_track_id = 1
        self.all_tracks: List[Track] = []  # Historical record of all tracks
        
        # Statistics
        self.frames_processed = 0
        self.total_peaks_detected = 0
        self.total_peaks_validated = 0
        
    def process_file(self, raw_file_path: str) -> Dict:
        """
        Process a complete raw audio file.
        
        Args:
            raw_file_path: Path to 6-channel raw PCM file (S16_LE)
            
        Returns:
            Dictionary with processing results and detected sources
        """
        print(f"Loading audio file: {raw_file_path}")
        
        # Load raw file
        audio_int16 = np.fromfile(raw_file_path, dtype=np.int16)
        
        # Reshape to 6 channels
        n_samples = len(audio_int16) // 6
        audio_6ch = audio_int16.reshape(n_samples, 6).T
        
        # Extract 4 microphone channels (channels 2-5, indices 1-4)
        mic_signals = audio_6ch[1:5, :].astype(np.float32) / 32767.0
        
        print(f"Audio loaded: {n_samples} samples ({n_samples/self.config.sample_rate:.2f}s)")
        print(f"Processing with {self.config.frame_size} sample frames, {self.config.hop_size} hop size")
        
        # Process frames
        all_detections = []
        frame_results = []
        
        n_frames = (n_samples - self.config.frame_size) // self.config.hop_size + 1
        
        for frame_idx in range(n_frames):
            start_sample = frame_idx * self.config.hop_size
            end_sample = start_sample + self.config.frame_size
            
            if end_sample > n_samples:
                break
            
            # Extract frame from all mics
            frame_data = mic_signals[:, start_sample:end_sample]
            
            # Process frame
            detections = self.process_frame(frame_data, frame_idx)
            
            frame_results.append({
                'frame': frame_idx,
                'time': start_sample / self.config.sample_rate,
                'detections': [self._detection_to_dict(d) for d in detections]
            })
            
            all_detections.extend(detections)
            
            if (frame_idx + 1) % 100 == 0:
                print(f"Processed {frame_idx + 1}/{n_frames} frames "
                      f"({100*(frame_idx+1)/n_frames:.1f}%)")
        
        # Finalize all remaining tracks
        for track in self.active_tracks.values():
            self.all_tracks.append(track)
        
        # Aggregate results
        results = {
            'metadata': {
                'file': raw_file_path,
                'duration': n_samples / self.config.sample_rate,
                'sample_rate': self.config.sample_rate,
                'frames_processed': self.frames_processed,
                'total_peaks_detected': self.total_peaks_detected,
                'total_peaks_validated': self.total_peaks_validated,
                'validation_rate': self.total_peaks_validated / max(self.total_peaks_detected, 1),
                'total_tracks': len(self.all_tracks),
                'timestamp': datetime.now().isoformat()
            },
            'config': {
                'frame_size': self.config.frame_size,
                'hop_size': self.config.hop_size,
                'fft_size': self.config.fft_size,
                'confidence_window': self.config.confidence_window_size,
                'confidence_threshold': self.config.confidence_threshold,
                'min_peak_snr': self.config.min_peak_snr,
                'frequency_range': [self.config.min_frequency, self.config.max_frequency]
            },
            'frames': frame_results,
            'summary': self._generate_summary(all_detections),
            'tracks': self._generate_track_summary()
        }
        
        return results
    
    def process_frame(self, frame_data: np.ndarray, frame_idx: int) -> List[Detection]:
        """
        Process a single frame of 4-channel audio.
        
        Args:
            frame_data: (4, frame_size) array of mic signals
            frame_idx: Frame index for tracking
            
        Returns:
            List of validated Detection objects
        """
        self.frames_processed += 1
        
        # 1. Compute FFT for all channels
        spectra = []
        for mic_idx in range(4):
            windowed = frame_data[mic_idx] * self.window
            spectrum = rfft(windowed, n=self.config.fft_size)
            spectra.append(spectrum)
        
        freqs = rfftfreq(self.config.fft_size, 1/self.config.sample_rate)
        
        # 2. Find peaks in average spectrum
        avg_magnitude = np.mean([np.abs(s) for s in spectra], axis=0)
        avg_magnitude_db = 20 * np.log10(avg_magnitude + 1e-10)
        
        # Estimate noise floor
        noise_floor = np.percentile(avg_magnitude_db, 25)
        
        peaks = self._find_peaks(freqs, avg_magnitude_db, noise_floor)
        self.total_peaks_detected += len(peaks)
        
        # 3. Estimate direction for each peak
        raw_detections = []
        for peak in peaks:
            direction = self._estimate_direction(peak, spectra, freqs)
            if direction is not None:
                detection = Detection(
                    frequency=peak.frequency,
                    azimuth=direction['azimuth'],
                    elevation=direction['elevation'],
                    x=direction['x'],
                    y=direction['y'],
                    z=direction['z'],
                    energy=peak.energy,
                    confidence=0.0,  # Will be set by validation
                    frame_count=1
                )
                raw_detections.append(detection)
        
        # 4. Validate with confidence window
        validated_detections = self._validate_detections(raw_detections)
        self.total_peaks_validated += len(validated_detections)
        
        # 5. Assign track IDs
        tracked_detections = self._assign_track_ids(validated_detections, frame_idx)
        
        return tracked_detections
    
    def _find_peaks(self, freqs: np.ndarray, magnitude_db: np.ndarray, 
                    noise_floor: float) -> List[Peak]:
        """Find frequency peaks in spectrum."""
        # Frequency range mask
        freq_mask = (freqs >= self.config.min_frequency) & (freqs <= self.config.max_frequency)
        
        # Find peaks using scipy
        peak_indices, properties = signal.find_peaks(
            magnitude_db,
            prominence=self.config.peak_prominence,
            distance=2  # Minimum 2 bins apart
        )
        
        # Filter by frequency range and SNR
        peaks = []
        for idx in peak_indices:
            if not freq_mask[idx]:
                continue
            
            snr = magnitude_db[idx] - noise_floor
            if snr >= self.config.min_peak_snr:
                peaks.append(Peak(
                    frequency=freqs[idx],
                    bin_index=idx,
                    energy=magnitude_db[idx],
                    snr=snr
                ))
        
        return peaks
    
    def _estimate_direction(self, peak: Peak, spectra: List[np.ndarray], 
                           freqs: np.ndarray) -> Optional[Dict]:
        """
        Estimate direction using phase differences (GCC-PHAT approach).
        
        Uses Left-Right and Back-Front mic pairs to estimate azimuth.
        """
        bin_idx = peak.bin_index
        
        # Get complex values at peak frequency
        mic_left = spectra[0][bin_idx]    # Mic 1
        mic_back = spectra[1][bin_idx]    # Mic 2
        mic_right = spectra[2][bin_idx]   # Mic 3
        mic_front = spectra[3][bin_idx]   # Mic 4
        
        # Calculate phase differences (GCC-PHAT)
        # Left-Right pair (X-axis)
        cross_LR = mic_right * np.conj(mic_left)
        phi_LR = np.angle(cross_LR)  # radians
        
        # Back-Front pair (Y-axis)
        cross_BF = mic_front * np.conj(mic_back)
        phi_BF = np.angle(cross_BF)  # radians
        
        # Convert phase difference to time delay
        wavelength = self.c / peak.frequency
        
        # TDOA (Time Difference of Arrival)
        tau_LR = phi_LR / (2 * np.pi * peak.frequency)
        tau_BF = phi_BF / (2 * np.pi * peak.frequency)
        
        # Physical constraint: |tau| <= d/c
        max_tau_LR = self.d_LR / self.c
        max_tau_BF = self.d_BF / self.c
        
        if abs(tau_LR) > max_tau_LR or abs(tau_BF) > max_tau_BF:
            # Phase wrapping issue or invalid detection
            return None
        
        # Estimate direction
        # Azimuth: angle in XY plane (0=front/+Y, 90=right/+X)
        azimuth_rad = np.arctan2(tau_LR * self.c / self.d_LR, 
                                  tau_BF * self.c / self.d_BF)
        azimuth_deg = np.degrees(azimuth_rad)
        
        # Normalize azimuth to [0, 360)
        azimuth_deg = azimuth_deg % 360
        
        # For elevation, we'd need vertical mic pairs (not available in planar array)
        # Assume sources are roughly on horizontal plane
        elevation_deg = 0.0
        
        # Convert to unit vector
        az_rad = np.radians(azimuth_deg)
        el_rad = np.radians(elevation_deg)
        
        x = np.cos(el_rad) * np.sin(az_rad)
        y = np.cos(el_rad) * np.cos(az_rad)
        z = np.sin(el_rad)
        
        return {
            'azimuth': azimuth_deg,
            'elevation': elevation_deg,
            'x': x,
            'y': y,
            'z': z
        }
    
    def _validate_detections(self, current_detections: List[Detection]) -> List[Detection]:
        """
        Validate detections using confidence window.
        A detection is valid if similar detections appear in enough previous frames.
        """
        # Add current detections to history
        self.detection_history.append(current_detections)
        
        # Need enough history to validate
        if len(self.detection_history) < self.config.confidence_window_size:
            return []
        
        validated = []
        
        for detection in current_detections:
            # Count how many times similar detection appeared in history
            match_count = 0
            matched_detections = []
            
            for hist_frame in self.detection_history:
                for hist_detection in hist_frame:
                    if self._is_similar_detection(detection, hist_detection):
                        match_count += 1
                        matched_detections.append(hist_detection)
                        break  # Only count once per frame
            
            # Calculate confidence
            confidence = match_count / len(self.detection_history)
            
            if confidence >= self.config.confidence_threshold:
                # Average properties over matched detections
                avg_detection = self._average_detections([detection] + matched_detections)
                avg_detection.confidence = confidence
                avg_detection.frame_count = match_count
                validated.append(avg_detection)
        
        return validated
    
    def _is_similar_detection(self, det1: Detection, det2: Detection) -> bool:
        """Check if two detections are similar enough to be considered the same source."""
        # Frequency similarity
        freq_diff = abs(det1.frequency - det2.frequency)
        if freq_diff > self.config.frequency_tolerance:
            return False
        
        # Direction similarity (using azimuth)
        # Handle wrap-around at 0/360 degrees
        az_diff = abs(det1.azimuth - det2.azimuth)
        az_diff = min(az_diff, 360 - az_diff)
        
        if az_diff > self.config.direction_tolerance:
            return False
        
        return True
    
    def _average_detections(self, detections: List[Detection]) -> Detection:
        """Average properties of similar detections."""
        if len(detections) == 1:
            return detections[0]
        
        avg_freq = np.mean([d.frequency for d in detections])
        avg_energy = np.mean([d.energy for d in detections])
        
        # Average angles (handle wrap-around)
        azimuths = np.array([d.azimuth for d in detections])
        # Convert to unit vectors for proper averaging
        x_avg = np.mean([np.cos(np.radians(az)) for az in azimuths])
        y_avg = np.mean([np.sin(np.radians(az)) for az in azimuths])
        avg_azimuth = np.degrees(np.arctan2(y_avg, x_avg)) % 360
        
        avg_elevation = np.mean([d.elevation for d in detections])
        
        # Recalculate unit vector
        az_rad = np.radians(avg_azimuth)
        el_rad = np.radians(avg_elevation)
        x = np.cos(el_rad) * np.sin(az_rad)
        y = np.cos(el_rad) * np.cos(az_rad)
        z = np.sin(el_rad)
        
        return Detection(
            frequency=avg_freq,
            azimuth=avg_azimuth,
            elevation=avg_elevation,
            x=x,
            y=y,
            z=z,
            energy=avg_energy,
            confidence=0.0,  # Will be set by caller
            frame_count=len(detections)
        )
    
    def _assign_track_ids(self, detections: List[Detection], frame_idx: int) -> List[Detection]:
        """Assign track IDs to detections, maintaining identity across frames."""
        if not detections:
            # Clean up inactive tracks
            self._cleanup_tracks(frame_idx)
            return []
        
        # Try to match each detection to existing tracks
        assigned_detections = []
        unmatched_detections = []
        matched_tracks = set()
        
        for detection in detections:
            best_match_id = None
            best_match_score = float('inf')
            
            # Find best matching active track
            for track_id, track in self.active_tracks.items():
                if track_id in matched_tracks:
                    continue  # Already matched to another detection
                
                # Check if detection matches track
                freq_diff = abs(detection.frequency - track.frequency)
                az_diff = abs(detection.azimuth - track.azimuth)
                az_diff = min(az_diff, 360 - az_diff)  # Handle wrap-around
                
                freq_tol = self.config.frequency_tolerance * self.config.track_similarity_tolerance
                dir_tol = self.config.direction_tolerance * self.config.track_similarity_tolerance
                
                if freq_diff < freq_tol and az_diff < dir_tol:
                    # Calculate match score (lower is better)
                    score = (freq_diff / freq_tol) + (az_diff / dir_tol)
                    
                    if score < best_match_score:
                        best_match_score = score
                        best_match_id = track_id
            
            if best_match_id is not None:
                # Match found - update existing track
                detection.source_id = best_match_id
                track = self.active_tracks[best_match_id]
                track.last_frame = frame_idx
                track.detections.append(detection)
                
                # Update running averages
                n = len(track.detections)
                track.frequency = (track.frequency * (n-1) + detection.frequency) / n
                
                # Average azimuth (handle wrap-around)
                az_sum_x = sum(np.cos(np.radians(d.azimuth)) for d in track.detections)
                az_sum_y = sum(np.sin(np.radians(d.azimuth)) for d in track.detections)
                track.azimuth = np.degrees(np.arctan2(az_sum_y, az_sum_x)) % 360
                
                matched_tracks.add(best_match_id)
                assigned_detections.append(detection)
            else:
                # No match - will create new track
                unmatched_detections.append(detection)
        
        # Create new tracks for unmatched detections
        for detection in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            
            detection.source_id = track_id
            
            new_track = Track(
                track_id=track_id,
                first_frame=frame_idx,
                last_frame=frame_idx,
                frequency=detection.frequency,
                azimuth=detection.azimuth,
                detections=[detection]
            )
            
            self.active_tracks[track_id] = new_track
            assigned_detections.append(detection)
        
        # Clean up old tracks
        self._cleanup_tracks(frame_idx)
        
        return assigned_detections
    
    def _cleanup_tracks(self, current_frame: int):
        """Remove tracks that haven't been detected recently."""
        tracks_to_remove = []
        
        for track_id, track in self.active_tracks.items():
            frames_since_last = current_frame - track.last_frame
            
            if frames_since_last > self.config.max_frames_gap:
                # Track has ended - move to historical record
                self.all_tracks.append(track)
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.active_tracks[track_id]
    
    def _detection_to_dict(self, detection: Detection) -> Dict:
        """Convert Detection to dictionary for JSON serialization."""
        return {
            'frequency': float(detection.frequency),
            'azimuth': float(detection.azimuth),
            'elevation': float(detection.elevation),
            'x': float(detection.x),
            'y': float(detection.y),
            'z': float(detection.z),
            'energy': float(detection.energy),
            'confidence': float(detection.confidence),
            'frame_count': int(detection.frame_count),
            'source_id': int(detection.source_id) if detection.source_id is not None else None
        }
    
    def _generate_summary(self, all_detections: List[Detection]) -> Dict:
        """Generate summary statistics of all detections."""
        if not all_detections:
            return {
                'total_detections': 0,
                'unique_sources': 0,
                'frequency_range': [0, 0],
                'avg_confidence': 0.0
            }
        
        # Cluster detections by frequency/direction to count unique sources
        unique_sources = self._cluster_detections(all_detections)
        
        frequencies = [d.frequency for d in all_detections]
        confidences = [d.confidence for d in all_detections]
        
        return {
            'total_detections': len(all_detections),
            'unique_sources': len(unique_sources),
            'frequency_range': [min(frequencies), max(frequencies)],
            'avg_confidence': np.mean(confidences),
            'sources': [
                {
                    'frequency': s['frequency'],
                    'azimuth': s['azimuth'],
                    'occurrence_count': s['count'],
                    'avg_confidence': s['avg_confidence']
                }
                for s in unique_sources
            ]
        }
    
    def _cluster_detections(self, detections: List[Detection]) -> List[Dict]:
        """Cluster detections to identify unique sources."""
        if not detections:
            return []
        
        clusters = []
        
        for detection in detections:
            # Find matching cluster
            matched = False
            for cluster in clusters:
                if (abs(detection.frequency - cluster['frequency']) < self.config.frequency_tolerance * 2 and
                    abs(detection.azimuth - cluster['azimuth']) < self.config.direction_tolerance * 2):
                    # Add to cluster
                    cluster['count'] += 1
                    cluster['frequencies'].append(detection.frequency)
                    cluster['azimuths'].append(detection.azimuth)
                    cluster['confidences'].append(detection.confidence)
                    matched = True
                    break
            
            if not matched:
                # Create new cluster
                clusters.append({
                    'frequency': detection.frequency,
                    'azimuth': detection.azimuth,
                    'count': 1,
                    'frequencies': [detection.frequency],
                    'azimuths': [detection.azimuth],
                    'confidences': [detection.confidence]
                })
        
        # Average cluster properties
        for cluster in clusters:
            cluster['frequency'] = np.mean(cluster['frequencies'])
            cluster['azimuth'] = np.mean(cluster['azimuths'])
            cluster['avg_confidence'] = np.mean(cluster['confidences'])
            del cluster['frequencies']
            del cluster['azimuths']
            del cluster['confidences']
        
        # Sort by occurrence count
        clusters.sort(key=lambda x: x['count'], reverse=True)
        
        return clusters
    
    def _generate_track_summary(self) -> List[Dict]:
        """Generate summary of all tracks."""
        track_summaries = []
        
        for track in self.all_tracks:
            duration_frames = track.last_frame - track.first_frame + 1
            duration_seconds = duration_frames * self.config.hop_size / self.config.sample_rate
            
            # Calculate statistics
            frequencies = [d.frequency for d in track.detections]
            energies = [d.energy for d in track.detections]
            confidences = [d.confidence for d in track.detections]
            
            track_summaries.append({
                'track_id': track.track_id,
                'first_frame': track.first_frame,
                'last_frame': track.last_frame,
                'duration_frames': duration_frames,
                'duration_seconds': float(duration_seconds),
                'detection_count': len(track.detections),
                'avg_frequency': float(track.frequency),
                'freq_std': float(np.std(frequencies)),
                'avg_azimuth': float(track.azimuth),
                'avg_energy': float(np.mean(energies)),
                'avg_confidence': float(np.mean(confidences)),
                'start_time': float(track.first_frame * self.config.hop_size / self.config.sample_rate),
                'end_time': float(track.last_frame * self.config.hop_size / self.config.sample_rate)
            })
        
        # Sort by track_id
        track_summaries.sort(key=lambda x: x['track_id'])
        
        return track_summaries


def process_audio_file(input_file: str, output_file: Optional[str] = None,
                      config: Optional[ProcessingConfig] = None) -> Dict:
    """
    Convenience function to process an audio file and save results.
    
    Args:
        input_file: Path to raw audio file
        output_file: Path to save JSON results (optional)
        config: Processing configuration (optional)
        
    Returns:
        Processing results dictionary
    """
    processor = CustomDOAProcessor(config)
    results = processor.process_file(input_file)
    
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Duration: {results['metadata']['duration']:.2f}s")
    print(f"Frames processed: {results['metadata']['frames_processed']}")
    print(f"Peaks detected: {results['metadata']['total_peaks_detected']}")
    print(f"Peaks validated: {results['metadata']['total_peaks_validated']}")
    print(f"Validation rate: {results['metadata']['validation_rate']*100:.1f}%")
    print(f"\nUnique sources identified: {results['summary']['unique_sources']}")
    print(f"Total tracks: {results['metadata']['total_tracks']}")
    
    if results['summary']['unique_sources'] > 0:
        print("\nTop sources:")
        for i, source in enumerate(results['summary']['sources'][:5], 1):
            print(f"  {i}. {source['frequency']:.1f} Hz at {source['azimuth']:.1f}° "
                  f"(count: {source['occurrence_count']}, confidence: {source['avg_confidence']:.2f})")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python custom_doa_processor.py <input.raw> [output.json]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if output_file is None:
        # Auto-generate output filename
        output_file = str(Path(input_file).with_suffix('')) + '_custom_doa.json'
    
    results = process_audio_file(input_file, output_file)
