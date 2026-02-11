"""
ODAS-Style DOA Processor - OPTIMIZED FOR REAL-TIME

Performance optimizations:
1. Reduced scan grid (256 directions instead of 4352)
2. Vectorized SRP computation using NumPy
3. Cached computations
4. Simplified tracking
5. Optional features that can be disabled

Target: <10ms per frame for real-time processing
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, irfft, rfftfreq
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MicArray:
    """Microphone array configuration"""
    positions: np.ndarray  # (n_mics, 3) array of positions in meters
    sample_rate: int = 16000
    speed_of_sound: float = 343.0
    
    def __post_init__(self):
        self.n_mics = len(self.positions)
        self.pairs = [(i, j) for i in range(self.n_mics) 
                      for j in range(i+1, self.n_mics)]
        self.n_pairs = len(self.pairs)


@dataclass
class SSLPot:
    """Sound Source Localization Potential"""
    azimuth: float
    elevation: float
    x: float
    y: float
    z: float
    energy: float
    coherence: float
    frame_idx: int
    timestamp: float


@dataclass
class Track:
    """Simplified tracking"""
    track_id: int
    azimuth: float
    elevation: float
    position: np.ndarray  # [x, y, z]
    velocity: np.ndarray  # [vx, vy, vz]
    first_frame: int
    last_frame: int
    status: str
    confidence: float = 0.9


# ============================================================================
# Optimized ODAS Processor
# ============================================================================

class ODASProcessorOptimized:
    """
    Real-time optimized ODAS processor
    
    Key optimizations:
    - Reduced scan grid (256 directions)
    - Vectorized SRP computation
    - Simple EKF tracking instead of full Kalman
    - Cached precomputations
    """
    
    def __init__(self, mic_array: MicArray, config: Optional[Dict] = None):
        self.mic_array = mic_array
        self.config = self._default_config()
        if config:
            self.config.update(config)
        
        # Initialize components
        self._init_stft()
        self._init_ssl_vectorized()
        self._init_tracking()
        
        # Statistics
        self.frames_processed = 0
        self.total_processing_time = 0.0
        
    def _default_config(self) -> Dict:
        """Optimized configuration for real-time performance"""
        return {
            # STFT
            'frame_size': 512,
            'hop_size': 128,
            'window': 'hann',
            
            # SSL - OPTIMIZED
            'ssl_n_pots': 3,
            'ssl_n_grid_points': 256,  # Reduced from 4352!
            'ssl_freq_min': 100.0,     # Lowered to catch more sounds
            'ssl_freq_max': 8000.0,    # Increased range
            'ssl_min_coherence': 0.45,  # Lowered from 0.65 for better detection
            
            # SST - SIMPLIFIED
            'sst_max_distance': 20.0,  # degrees
            'sst_inactive_frames': 50,
            'sst_min_confidence': 0.3,
            
            # Performance
            'enable_spectral_fingerprint': False,  # Disable for speed
            'enable_adaptive_kalman': False,       # Use simple tracking
        }
    
    def _init_stft(self):
        """Initialize STFT components"""
        if self.config['window'] == 'hann':
            self.window = signal.windows.hann(self.config['frame_size'])
        else:
            self.window = np.ones(self.config['frame_size'])
        
        # FFT frequencies
        self.freqs = rfftfreq(self.config['frame_size'], 
                              1/self.mic_array.sample_rate)
        
        # Frequency mask
        self.freq_mask = (self.freqs >= self.config['ssl_freq_min']) & \
                        (self.freqs <= self.config['ssl_freq_max'])
        
        # Noise floor
        self.noise_floor = np.ones(len(self.freqs)) * 1e-6
        self.noise_alpha = 0.95
        
    def _init_ssl_vectorized(self):
        """Initialize SSL with vectorized computations"""
        # Generate REDUCED scan grid (Fibonacci sphere)
        n_points = self.config['ssl_n_grid_points']
        
        indices = np.arange(n_points)
        theta = np.pi * (3 - np.sqrt(5)) * indices  # Golden angle
        phi = np.arccos(1 - 2 * indices / n_points)
        
        # Convert to azimuth/elevation
        self.scan_azimuths = (np.degrees(theta) % 360 - 180)  # [-180, 180]
        self.scan_elevations = 90 - np.degrees(phi)  # [-90, 90]
        
        # Convert to unit vectors (n_points, 3)
        az_rad = np.radians(self.scan_azimuths)
        el_rad = np.radians(self.scan_elevations)
        
        self.scan_directions = np.stack([
            np.cos(el_rad) * np.cos(az_rad),  # X
            np.cos(el_rad) * np.sin(az_rad),  # Y
            np.sin(el_rad)                     # Z
        ], axis=1)
        
        # Precompute TDOAs for ALL directions and pairs (vectorized)
        # Shape: (n_points, n_pairs)
        self.tdoa_table = np.zeros((n_points, self.mic_array.n_pairs))
        
        for pair_idx, (i, j) in enumerate(self.mic_array.pairs):
            # TDOA: positive when source is closer to mic i (arrives earlier at i)
            # Standard GCC-PHAT convention: cross_spec = spec[j] * conj(spec[i])
            # This gives positive delay when j lags i, so we use (pos[i] - pos[j])
            mic_vec = self.mic_array.positions[i] - self.mic_array.positions[j]
            # Dot product for all directions at once
            self.tdoa_table[:, pair_idx] = \
                (self.scan_directions @ mic_vec) / self.mic_array.speed_of_sound
        
        # Convert TDOAs to sample delays
        self.delay_table = self.tdoa_table * self.mic_array.sample_rate
        
        # Precompute delay indices for correlation lookup
        self.delay_indices = np.round(self.delay_table).astype(np.int32) + \
                            self.config['frame_size'] // 2
        
        # Valid delay mask
        self.valid_delays = (self.delay_indices >= 0) & \
                           (self.delay_indices < self.config['frame_size'])
        
        print(f"SSL initialized with {n_points} scan directions (vectorized)")
        
    def _init_tracking(self):
        """Initialize simple tracking"""
        self.tracks: List[Track] = []
        self.next_track_id = 1
        
    # ========================================================================
    # STFT Module (Same as before but optimized)
    # ========================================================================
    
    def stft_process(self, multi_channel_frame: np.ndarray) -> np.ndarray:
        """
        Vectorized STFT processing
        
        Returns:
            spectra: (n_mics, n_freqs) complex array
        """
        # Vectorized windowing and FFT
        windowed = multi_channel_frame * self.window[np.newaxis, :]
        spectra = rfft(windowed, n=self.config['frame_size'], axis=1)
        
        # Update noise floor
        avg_magnitude = np.mean(np.abs(spectra), axis=0)
        self.noise_floor = self.noise_alpha * self.noise_floor + \
                          (1 - self.noise_alpha) * avg_magnitude
        
        return spectra
    
    # ========================================================================
    # SSL Module - VECTORIZED FOR SPEED
    # ========================================================================
    
    def ssl_process_vectorized(self, spectra: np.ndarray, frame_idx: int) -> List[SSLPot]:
        """
        VECTORIZED Sound Source Localization
        
        This is the KEY optimization - all SRP computation in one NumPy operation
        """
        # 1. Compute cross-spectra for all pairs (vectorized)
        cross_spectra = np.zeros((self.mic_array.n_pairs, len(self.freqs)), 
                                 dtype=np.complex64)
        
        for pair_idx, (i, j) in enumerate(self.mic_array.pairs):
            cross_spectra[pair_idx] = spectra[j] * np.conj(spectra[i])
        
        # 2. GCC-PHAT with coherence weighting
        coherence = np.abs(cross_spectra) / (np.abs(cross_spectra) + 
                                             self.noise_floor + 1e-10)
        
        # PHAT normalization
        phat_spectra = cross_spectra / (np.abs(cross_spectra) + 1e-10)
        phat_spectra *= coherence
        
        # Apply frequency mask
        phat_spectra[:, ~self.freq_mask] = 0
        
        # 3. IFFT to get correlations (vectorized for all pairs)
        correlations = np.fft.fftshift(
            irfft(phat_spectra, n=self.config['frame_size'], axis=1),
            axes=1
        )
        # Shape: (n_pairs, frame_size)
        
        # 4. VECTORIZED SRP computation
        # For each scan direction, sum correlation values at expected delays
        srp_powers = np.zeros(len(self.scan_directions))
        srp_coherence = np.zeros(len(self.scan_directions))
        
        for pair_idx in range(self.mic_array.n_pairs):
            # Get delay indices for this pair across all directions
            delays = self.delay_indices[:, pair_idx]
            valid = self.valid_delays[:, pair_idx]
            
            # Vectorized lookup
            corr_values = correlations[pair_idx, delays]
            
            # Mask invalid delays
            corr_values[~valid] = 0
            
            # Accumulate
            srp_powers += corr_values
            
            # Coherence tracking
            max_corr = np.max(np.abs(correlations[pair_idx]))
            if max_corr > 0:
                srp_coherence += np.abs(corr_values) / max_corr
        
        # Normalize
        srp_powers /= self.mic_array.n_pairs
        srp_coherence /= self.mic_array.n_pairs
        
        # 5. Combined score
        srp_powers_norm = srp_powers / (np.max(np.abs(srp_powers)) + 1e-10)
        scores = srp_powers_norm * srp_coherence
        
        # 6. Find peaks (vectorized)
        # Simple approach: find local maxima
        n_pots = self.config['ssl_n_pots']
        top_indices = np.argsort(scores)[-n_pots*3:][::-1]  # Get more candidates
        
        # Filter by threshold
        min_coherence = self.config['ssl_min_coherence']
        pots = []
        timestamp = frame_idx * self.config['hop_size'] / self.mic_array.sample_rate
        
        for idx in top_indices:
            if srp_coherence[idx] < min_coherence:
                continue
            
            if len(pots) >= n_pots:
                break
            
            azimuth = self.scan_azimuths[idx]
            elevation = self.scan_elevations[idx]
            
            pot = SSLPot(
                azimuth=azimuth,
                elevation=elevation,
                x=self.scan_directions[idx, 0],
                y=self.scan_directions[idx, 1],
                z=self.scan_directions[idx, 2],
                energy=20 * np.log10(np.abs(srp_powers_norm[idx]) + 1e-10),
                coherence=srp_coherence[idx],
                frame_idx=frame_idx,
                timestamp=timestamp
            )
            
            pots.append(pot)
        
        return pots
    
    # ========================================================================
    # SST Module - SIMPLIFIED
    # ========================================================================
    
    def sst_process_simple(self, pots: List[SSLPot], frame_idx: int) -> List[Track]:
        """
        Simplified tracking - just associate pots to tracks by distance
        Much faster than full Kalman filtering
        """
        dt = self.config['hop_size'] / self.mic_array.sample_rate
        max_distance = self.config['sst_max_distance']
        
        # Update existing tracks (simple prediction)
        for track in self.tracks:
            # Simple linear prediction
            track.position += track.velocity * dt
            
            # Normalize to unit sphere
            norm = np.linalg.norm(track.position)
            if norm > 0:
                track.position /= norm
            
            # Update azimuth/elevation
            track.azimuth = np.degrees(np.arctan2(track.position[1], 
                                                   track.position[0]))
            track.elevation = np.degrees(np.arcsin(track.position[2]))
            
            # Decay confidence
            track.confidence *= 0.95
        
        # Associate pots to tracks
        if self.tracks and pots:
            # Build distance matrix
            n_tracks = len(self.tracks)
            n_pots = len(pots)
            
            distance_matrix = np.zeros((n_tracks, n_pots))
            
            for t_idx, track in enumerate(self.tracks):
                for p_idx, pot in enumerate(pots):
                    # Angular distance
                    dist = self._angular_distance_fast(
                        track.azimuth, track.elevation,
                        pot.azimuth, pot.elevation
                    )
                    distance_matrix[t_idx, p_idx] = dist
            
            # Simple greedy association (faster than Hungarian)
            associated_pots = set()
            
            for t_idx, track in enumerate(self.tracks):
                best_pot_idx = np.argmin(distance_matrix[t_idx])
                best_distance = distance_matrix[t_idx, best_pot_idx]
                
                if best_distance < max_distance and best_pot_idx not in associated_pots:
                    # Update track
                    pot = pots[best_pot_idx]
                    
                    new_position = np.array([pot.x, pot.y, pot.z])
                    track.velocity = (new_position - track.position) / (dt + 1e-6)
                    track.position = new_position
                    
                    track.azimuth = pot.azimuth
                    track.elevation = pot.elevation
                    track.last_frame = frame_idx
                    track.status = 'active'
                    track.confidence = min(1.0, track.confidence + 0.1)
                    
                    associated_pots.add(best_pot_idx)
            
            # Create new tracks for unassociated pots
            for p_idx, pot in enumerate(pots):
                if p_idx not in associated_pots:
                    track = Track(
                        track_id=self.next_track_id,
                        azimuth=pot.azimuth,
                        elevation=pot.elevation,
                        position=np.array([pot.x, pot.y, pot.z]),
                        velocity=np.zeros(3),
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                        status='new'
                    )
                    self.tracks.append(track)
                    self.next_track_id += 1
        
        elif pots:
            # No tracks yet, create from pots
            for pot in pots:
                track = Track(
                    track_id=self.next_track_id,
                    azimuth=pot.azimuth,
                    elevation=pot.elevation,
                    position=np.array([pot.x, pot.y, pot.z]),
                    velocity=np.zeros(3),
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    status='new'
                )
                self.tracks.append(track)
                self.next_track_id += 1
        
        # Remove inactive tracks
        inactive_threshold = self.config['sst_inactive_frames']
        min_confidence = self.config['sst_min_confidence']
        
        self.tracks = [t for t in self.tracks 
                      if (frame_idx - t.last_frame) < inactive_threshold
                      and t.confidence > min_confidence]
        
        return self.tracks
    
    def _angular_distance_fast(self, az1: float, el1: float, 
                               az2: float, el2: float) -> float:
        """Fast angular distance (great circle)"""
        # Convert to radians
        az1_rad, el1_rad = np.radians(az1), np.radians(el1)
        az2_rad, el2_rad = np.radians(az2), np.radians(el2)
        
        # Haversine formula (faster than full 3D)
        dlat = el2_rad - el1_rad
        dlon = az2_rad - az1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(el1_rad) * np.cos(el2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return np.degrees(c)
    
    # ========================================================================
    # Main Processing Pipeline
    # ========================================================================
    
    def process_frame(self, multi_channel_frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Process one frame - OPTIMIZED
        """
        import time
        start_time = time.perf_counter()
        
        # 1. STFT
        spectra = self.stft_process(multi_channel_frame)
        
        # 2. SSL - Vectorized
        pots = self.ssl_process_vectorized(spectra, frame_idx)
        
        # Filter pots: only track the strongest ones to reduce spurious tracks
        # Sort by coherence and keep top candidates
        max_pots_per_frame = 5  # Reasonable limit for simultaneous sources
        if len(pots) > max_pots_per_frame:
            pots_sorted = sorted(pots, key=lambda p: p.coherence, reverse=True)
            pots = pots_sorted[:max_pots_per_frame]
        
        # 3. SST - Simplified
        tracks = self.sst_process_simple(pots, frame_idx)
        
        # Filter to only active tracks (updated in this or recent frame)
        # This prevents outputting stale tracks
        active_tracks = [t for t in tracks if (frame_idx - t.last_frame) <= 2]
        
        # Timing
        elapsed = time.perf_counter() - start_time
        self.total_processing_time += elapsed
        self.frames_processed += 1
        
        return {
            'frame_idx': frame_idx,
            'timestamp': frame_idx * self.config['hop_size'] / self.mic_array.sample_rate,
            'processing_time_ms': elapsed * 1000,
            'pots': [self._pot_to_dict(p) for p in pots],
            'tracks': [self._track_to_dict(t) for t in active_tracks]
        }
    
    def process_file(self, audio_file: str, output_file: Optional[str] = None) -> Dict:
        """Process complete audio file with performance monitoring"""
        print(f"Processing: {audio_file}")
        
        # Load audio
        audio_int16 = np.fromfile(audio_file, dtype=np.int16)
        n_samples = len(audio_int16) // 6
        audio_6ch = audio_int16.reshape(n_samples, 6).T
        
        # Extract mic channels (2-5)
        mic_audio = audio_6ch[2:6, :].astype(np.float32) / 32767.0
        
        # Process frames
        frame_size = self.config['frame_size']
        hop_size = self.config['hop_size']
        n_frames = (n_samples - frame_size) // hop_size + 1
        
        results = []
        processing_times = []
        
        import time
        overall_start = time.perf_counter()
        
        for frame_idx in range(n_frames):
            start = frame_idx * hop_size
            end = start + frame_size
            
            if end > n_samples:
                break
            
            frame = mic_audio[:, start:end]
            frame_results = self.process_frame(frame, frame_idx)
            results.append(frame_results)
            processing_times.append(frame_results['processing_time_ms'])
            
            if (frame_idx + 1) % 100 == 0:
                avg_time = np.mean(processing_times[-100:])
                print(f"  Frame {frame_idx+1}/{n_frames} "
                      f"({100*(frame_idx+1)/n_frames:.1f}%) - "
                      f"Avg: {avg_time:.2f}ms/frame")
        
        overall_elapsed = time.perf_counter() - overall_start
        
        # Performance summary
        processing_times = np.array(processing_times)
        audio_duration = n_samples / self.mic_array.sample_rate
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Audio duration: {audio_duration:.2f}s")
        print(f"Processing time: {overall_elapsed:.2f}s")
        print(f"Real-time factor: {audio_duration/overall_elapsed:.2f}x")
        print(f"Frames processed: {n_frames}")
        print(f"Avg time per frame: {np.mean(processing_times):.2f}ms")
        print(f"Min time per frame: {np.min(processing_times):.2f}ms")
        print(f"Max time per frame: {np.max(processing_times):.2f}ms")
        print(f"Std dev: {np.std(processing_times):.2f}ms")
        
        real_time_constraint = (hop_size / self.mic_array.sample_rate) * 1000
        print(f"\nReal-time constraint: {real_time_constraint:.2f}ms per frame")
        if np.mean(processing_times) < real_time_constraint:
            print("✓ CAN RUN IN REAL-TIME")
        else:
            print("✗ TOO SLOW FOR REAL-TIME")
        print("="*60 + "\n")
        
        # Compile results
        output = {
            'metadata': {
                'file': audio_file,
                'duration': audio_duration,
                'sample_rate': self.mic_array.sample_rate,
                'frames_processed': n_frames,
                'processing_time': overall_elapsed,
                'real_time_factor': audio_duration / overall_elapsed,
                'avg_frame_time_ms': float(np.mean(processing_times)),
                'timestamp': datetime.now().isoformat()
            },
            'config': self.config,
            'frames': results
        }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Results saved to: {output_file}")
        
        return output
    
    def _pot_to_dict(self, pot: SSLPot) -> Dict:
        """Convert SSLPot to dictionary"""
        return {
            'azimuth': pot.azimuth,
            'elevation': pot.elevation,
            'x': pot.x,
            'y': pot.y,
            'z': pot.z,
            'energy': pot.energy,
            'coherence': pot.coherence,
            'timestamp': pot.timestamp
        }
    
    def _track_to_dict(self, track: Track) -> Dict:
        """Convert Track to dictionary"""
        return {
            'track_id': track.track_id,
            'azimuth': track.azimuth,
            'elevation': track.elevation,
            'position': track.position.tolist(),
            'first_frame': track.first_frame,
            'last_frame': track.last_frame,
            'status': track.status,
            'confidence': track.confidence
        }


# ============================================================================
# Performance Testing
# ============================================================================

def benchmark_comparison():
    """Compare optimized vs original performance"""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    # Setup
    mic_positions = np.array([
        [-0.032, 0.000, 0.000],
        [0.000, -0.032, 0.000],
        [0.032, 0.000, 0.000],
        [0.000, 0.032, 0.000]
    ])
    mic_array = MicArray(positions=mic_positions)
    
    # Create test frame
    test_frame = np.random.randn(4, 512).astype(np.float32)
    
    # Optimized processor
    processor_opt = ODASProcessorOptimized(mic_array)
    
    # Warm-up
    for _ in range(10):
        processor_opt.process_frame(test_frame, 0)
    
    # Benchmark
    import time
    n_iterations = 100
    
    start = time.perf_counter()
    for i in range(n_iterations):
        processor_opt.process_frame(test_frame, i)
    elapsed = time.perf_counter() - start
    
    avg_time_ms = (elapsed / n_iterations) * 1000
    
    print(f"\nOptimized processor:")
    print(f"  Average time: {avg_time_ms:.2f}ms per frame")
    print(f"  Throughput: {n_iterations/elapsed:.1f} frames/sec")
    
    # Real-time check
    hop_size = 128
    sample_rate = 16000
    real_time_constraint = (hop_size / sample_rate) * 1000
    
    print(f"\nReal-time constraint: {real_time_constraint:.2f}ms")
    print(f"Speedup needed: {avg_time_ms/real_time_constraint:.2f}x")
    
    if avg_time_ms < real_time_constraint:
        print("✓ CAN RUN IN REAL-TIME!")
    else:
        print("✗ Still too slow - further optimization needed")
        print(f"  Need to be {real_time_constraint/avg_time_ms:.2f}x faster")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'benchmark':
        benchmark_comparison()
    elif len(sys.argv) > 1:
        # Process file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'odas_optimized_output.json'
        
        mic_positions = np.array([
            [-0.032, 0.000, 0.000],
            [0.000, -0.032, 0.000],
            [0.032, 0.000, 0.000],
            [0.000, 0.032, 0.000]
        ])
        mic_array = MicArray(positions=mic_positions)
        
        processor = ODASProcessorOptimized(mic_array)
        results = processor.process_file(input_file, output_file)
    else:
        print("Usage:")
        print("  python odas_optimized.py benchmark              - Run performance benchmark")
        print("  python odas_optimized.py <audio_file> [output]  - Process audio file")
