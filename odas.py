"""
ODAS-Style DOA Processor

Complete implementation following ODAS pipeline architecture with fixes for:
1. Proper coordinate system (-180 to 180, with 0=+X, 90=+Y)
2. Adaptive Kalman filtering for static/moving sources
3. Better phase unwrapping and SSL
4. Improved tracking with spectral fingerprinting

Coordinate System:
- 0° = +X axis (front)
- 90° = +Y axis (left)
- 180/-180° = -X axis (back)
- -90° = -Y axis (right)
"""

import numpy as np
from scipy import signal
from scipy.fft import rfft, irfft, rfftfreq
from scipy.spatial.distance import cdist
from scipy.cluster import hierarchy
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
import json
from pathlib import Path
from collections import deque
from datetime import datetime
from enum import Enum


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
        # Precompute mic pairs and distances
        self.pairs = [(i, j) for i in range(self.n_mics) 
                      for j in range(i+1, self.n_mics)]
        self.pair_distances = {}
        self.pair_vectors = {}
        for i, j in self.pairs:
            vec = self.positions[j] - self.positions[i]
            self.pair_distances[(i, j)] = np.linalg.norm(vec)
            self.pair_vectors[(i, j)] = vec / (np.linalg.norm(vec) + 1e-10)


@dataclass
class SSLPot:
    """Sound Source Localization Potential (ODAS terminology)"""
    azimuth: float  # Degrees (-180 to 180)
    elevation: float  # Degrees (-90 to 90)
    x: float  # Unit vector X component
    y: float  # Unit vector Y component
    z: float  # Unit vector Z component
    energy: float  # dB
    coherence: float  # 0-1
    frame_idx: int
    timestamp: float  # seconds
    spectrum: Optional[np.ndarray] = None
    

@dataclass
class KalmanState:
    """Kalman filter state for tracking"""
    x: np.ndarray  # State vector [x, y, z, vx, vy, vz]
    P: np.ndarray  # Covariance matrix (6x6)
    Q: np.ndarray  # Process noise covariance
    R: np.ndarray  # Measurement noise covariance
    F: np.ndarray  # State transition matrix
    H: np.ndarray  # Measurement matrix
    
    @classmethod
    def initialize(cls, position: np.ndarray, sigmaQ: float = 0.00001):
        """Initialize Kalman filter for a new track"""
        x = np.zeros(6)
        x[:3] = position
        
        P = np.eye(6)
        P[:3, :3] *= 0.01  # Position uncertainty
        P[3:, 3:] *= 1.0   # Velocity uncertainty (higher initially)
        
        Q = np.eye(6) * sigmaQ
        R = np.eye(3) * 0.001  # Measurement uncertainty
        
        # State transition (will be updated with dt)
        F = np.eye(6)
        
        # Measurement matrix (observe position only)
        H = np.zeros((3, 6))
        H[:3, :3] = np.eye(3)
        
        return cls(x, P, Q, R, F, H)
    
    def predict(self, dt: float, is_static: bool = False):
        """Predict next state"""
        # Update state transition matrix with time step
        self.F[:3, 3:] = np.eye(3) * dt
        
        if is_static:
            # Decay velocity for static sources
            self.F[3:, 3:] = np.eye(3) * 0.95
        
        # Predict
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, measurement: np.ndarray):
        """Update with measurement"""
        y = measurement - self.H @ self.x  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return np.linalg.norm(y)  # Return innovation magnitude


@dataclass
class SSTTrack:
    """Sound Source Tracking track"""
    track_id: int
    kalman: KalmanState
    first_frame: int
    last_frame: int
    status: str  # 'active', 'inactive', 'ended'
    
    # Track properties
    azimuth_history: List[float] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)
    spectrum_history: List[np.ndarray] = field(default_factory=list)
    
    # Motion detection
    position_variance: float = 0.0
    is_static: bool = True
    
    # Probabilities (ODAS style)
    prob_active: float = 0.5
    prob_exist: float = 0.9
    
    # Identity preservation
    spectral_fingerprint: Optional[np.ndarray] = None
    
    def get_position(self) -> np.ndarray:
        """Get current 3D position"""
        return self.kalman.x[:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity"""
        return self.kalman.x[3:].copy()
    
    def to_azimuth_elevation(self) -> Tuple[float, float]:
        """Convert position to azimuth/elevation"""
        x, y, z = self.get_position()
        r = np.sqrt(x**2 + y**2 + z**2) + 1e-10
        
        # Azimuth: angle in XY plane
        # 0° = +X, 90° = +Y, ±180° = -X, -90° = -Y
        azimuth = np.degrees(np.arctan2(y, x))
        
        # Elevation: angle from XY plane
        elevation = np.degrees(np.arcsin(z / r))
        
        return azimuth, elevation


class TrackStatus(Enum):
    """Track lifecycle states"""
    NEW = "new"
    ACTIVE = "active" 
    INACTIVE = "inactive"
    ENDED = "ended"


# ============================================================================
# ODAS Pipeline Implementation
# ============================================================================

class ODASProcessor:
    """
    Complete ODAS-style pipeline with improvements for simple and complex scenes
    """
    
    def __init__(self, mic_array: MicArray, config: Optional[Dict] = None):
        self.mic_array = mic_array
        self.config = self._default_config()
        if config:
            self.config.update(config)
        
        # Initialize components
        self._init_stft()
        self._init_ssl()
        self._init_sst()
        
        # Statistics
        self.frames_processed = 0
        self.total_pots_detected = 0
        self.total_tracks_created = 0
        
    def _default_config(self) -> Dict:
        """Default configuration matching ODAS settings"""
        return {
            # STFT
            'frame_size': 512,
            'hop_size': 128,
            'window': 'hann',
            
            # SSL
            'ssl_n_pots': 3,
            'ssl_prob_min': 0.65,
            'ssl_n_matches': 7,
            'ssl_scan_levels': [4, 6],
            'ssl_freq_min': 200.0,
            'ssl_freq_max': 6000.0,
            
            # SST
            'sst_mode': 'kalman',
            'sst_sigma_q': 0.00001,  # Very small for static sources
            'sst_sigma_r2_prob': 0.0001,
            'sst_sigma_r2_active': 0.001,
            'sst_p_false': 0.02,
            'sst_p_new': 0.2,
            'sst_p_track': 0.9,
            'sst_theta_new': 0.8,
            'sst_n_prob': 3,
            'sst_n_inactive': 250,
            'sst_theta_inactive': 0.9,
            
            # Motion detection
            'motion_threshold': 0.01,
            'static_variance_threshold': 0.001,
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
        
        # Noise floor estimation
        self.noise_floor = np.zeros(len(self.freqs))
        self.noise_alpha = 0.95  # MCRA smoothing
        
    def _init_ssl(self):
        """Initialize SSL components"""
        # Generate scanning grid (spherical coordinates)
        self.scan_grid = self._generate_scan_grid()
        
        # Precompute TDOAs for each direction
        self.tdoa_table = self._precompute_tdoas()
        
        # Spatial aliasing frequencies for each mic pair
        self.aliasing_freqs = {}
        for pair in self.mic_array.pairs:
            distance = self.mic_array.pair_distances[pair]
            self.aliasing_freqs[pair] = self.mic_array.speed_of_sound / (2 * distance)
    
    def _init_sst(self):
        """Initialize SST components"""
        self.tracks: List[SSTTrack] = []
        self.next_track_id = 1
        self.track_history: List[SSTTrack] = []
        
    def _generate_scan_grid(self) -> List[Tuple[float, float]]:
        """Generate spherical scanning grid (ODAS-style)"""
        grid = []
        
        for level in self.config['ssl_scan_levels']:
            # Fibonacci sphere for uniform distribution
            n_points = 4 ** level
            
            for i in range(n_points):
                # Fibonacci lattice
                theta = np.pi * (3 - np.sqrt(5)) * i  # Golden angle
                phi = np.arccos(1 - 2 * i / n_points)
                
                # Convert to azimuth/elevation
                azimuth = np.degrees(theta) % 360 - 180  # [-180, 180]
                elevation = 90 - np.degrees(phi)  # [-90, 90]
                
                grid.append((azimuth, elevation))
        
        return grid
    
    def _precompute_tdoas(self) -> Dict:
        """Precompute TDOAs for all directions and mic pairs"""
        tdoas = {}
        
        for azimuth, elevation in self.scan_grid:
            # Convert to unit vector
            az_rad = np.radians(azimuth)
            el_rad = np.radians(elevation)
            
            direction = np.array([
                np.cos(el_rad) * np.cos(az_rad),  # X
                np.cos(el_rad) * np.sin(az_rad),  # Y
                np.sin(el_rad)                     # Z
            ])
            
            # TDOA for each mic pair
            pair_tdoas = {}
            for pair in self.mic_array.pairs:
                mic_vec = self.mic_array.positions[pair[1]] - \
                         self.mic_array.positions[pair[0]]
                tdoa = np.dot(mic_vec, direction) / self.mic_array.speed_of_sound
                pair_tdoas[pair] = tdoa
            
            tdoas[(azimuth, elevation)] = pair_tdoas
        
        return tdoas
    
    # ========================================================================
    # STFT Module
    # ========================================================================
    
    def stft_process(self, multi_channel_frame: np.ndarray) -> List[np.ndarray]:
        """
        STFT processing (mod_stft.c equivalent)
        
        Args:
            multi_channel_frame: (n_mics, frame_size) array
            
        Returns:
            List of complex spectra
        """
        spectra = []
        
        for mic_idx in range(self.mic_array.n_mics):
            # Apply window
            windowed = multi_channel_frame[mic_idx] * self.window
            
            # FFT
            spectrum = rfft(windowed, n=self.config['frame_size'])
            spectra.append(spectrum)
        
        # Update noise floor (MCRA)
        avg_magnitude = np.mean([np.abs(s) for s in spectra], axis=0)
        self.noise_floor = self.noise_alpha * self.noise_floor + \
                          (1 - self.noise_alpha) * avg_magnitude
        
        return spectra
    
    # ========================================================================
    # SSL Module
    # ========================================================================
    
    def ssl_process(self, spectra: List[np.ndarray], frame_idx: int) -> List[SSLPot]:
        """
        Sound Source Localization (mod_ssl.c equivalent)
        
        Returns list of SSL "pots" (potential sources)
        """
        # 1. Cross-spectral processing
        cross_spectra = self._compute_cross_spectra(spectra)
        
        # 2. GCC-PHAT
        correlations = self._gcc_phat(cross_spectra)
        
        # 3. SRP-PHAT (Steered Response Power)
        srp_map = self._compute_srp(correlations)
        
        # 4. Peak detection in SRP map
        peaks = self._find_srp_peaks(srp_map)
        
        # 5. Refine and validate peaks
        pots = []
        timestamp = frame_idx * self.config['hop_size'] / self.mic_array.sample_rate
        
        for peak in peaks[:self.config['ssl_n_pots']]:
            if peak['probability'] < self.config['ssl_prob_min']:
                continue
            
            azimuth, elevation = peak['direction']
            
            # Convert to unit vector
            az_rad = np.radians(azimuth)
            el_rad = np.radians(elevation)
            
            pot = SSLPot(
                azimuth=azimuth,
                elevation=elevation,
                x=np.cos(el_rad) * np.cos(az_rad),
                y=np.cos(el_rad) * np.sin(az_rad),
                z=np.sin(el_rad),
                energy=peak['energy'],
                coherence=peak['coherence'],
                frame_idx=frame_idx,
                timestamp=timestamp,
                spectrum=self._extract_directional_spectrum(spectra, azimuth, elevation)
            )
            
            pots.append(pot)
            self.total_pots_detected += 1
        
        return pots
    
    def _compute_cross_spectra(self, spectra: List[np.ndarray]) -> Dict:
        """Compute cross-spectra for all mic pairs"""
        cross_spectra = {}
        
        for pair in self.mic_array.pairs:
            i, j = pair
            cross_spectra[pair] = spectra[j] * np.conj(spectra[i])
        
        return cross_spectra
    
    def _gcc_phat(self, cross_spectra: Dict) -> Dict:
        """
        Generalized Cross-Correlation with Phase Transform
        Includes frequency weighting to avoid spatial aliasing
        """
        correlations = {}
        
        for pair, cross_spectrum in cross_spectra.items():
            # Frequency weighting to avoid aliasing
            f_alias = self.aliasing_freqs[pair]
            weights = np.ones_like(self.freqs)
            
            # Smooth transition around aliasing frequency
            transition_band = (self.freqs > f_alias/2) & (self.freqs < f_alias)
            weights[transition_band] = 0.5 * (1 + np.cos(
                np.pi * (self.freqs[transition_band] - f_alias/2) / (f_alias/2)
            ))
            weights[self.freqs >= f_alias] = 0.1
            
            # Coherence weighting
            i, j = pair
            coherence = np.abs(cross_spectrum) / (
                np.abs(cross_spectrum) + self.noise_floor + 1e-10
            )
            weights *= coherence
            
            # PHAT normalization
            phat_spectrum = cross_spectrum / (np.abs(cross_spectrum) + 1e-10)
            phat_spectrum *= weights
            
            # IFFT to get correlation
            correlation = irfft(phat_spectrum, n=self.config['frame_size'])
            correlations[pair] = np.fft.fftshift(correlation)
        
        return correlations
    
    def _compute_srp(self, correlations: Dict) -> Dict:
        """Compute Steered Response Power for all scan directions"""
        srp_map = {}
        
        for direction in self.scan_grid:
            # Get precomputed TDOAs
            tdoas = self.tdoa_table[direction]
            
            # Sum correlations at expected delays
            power = 0.0
            coherence_sum = 0.0
            
            for pair, tdoa in tdoas.items():
                # Convert TDOA to samples
                delay_samples = int(tdoa * self.mic_array.sample_rate)
                delay_idx = delay_samples + self.config['frame_size'] // 2
                
                if 0 <= delay_idx < self.config['frame_size']:
                    # Get correlation value at delay
                    corr_value = correlations[pair][delay_idx]
                    
                    # Interpolate for sub-sample accuracy
                    if 0 < delay_idx < self.config['frame_size'] - 1:
                        frac = (tdoa * self.mic_array.sample_rate) - delay_samples
                        corr_value = (1 - frac) * correlations[pair][delay_idx] + \
                                    frac * correlations[pair][delay_idx + 1]
                    
                    power += corr_value
                    
                    # Track coherence
                    max_corr = np.max(np.abs(correlations[pair]))
                    if max_corr > 0:
                        coherence_sum += np.abs(corr_value) / max_corr
            
            n_pairs = len(self.mic_array.pairs)
            srp_map[direction] = {
                'power': power / n_pairs,
                'coherence': coherence_sum / n_pairs
            }
        
        return srp_map
    
    def _find_srp_peaks(self, srp_map: Dict) -> List[Dict]:
        """Find peaks in SRP map"""
        # Convert to arrays for processing
        directions = list(srp_map.keys())
        powers = np.array([srp_map[d]['power'] for d in directions])
        coherences = np.array([srp_map[d]['coherence'] for d in directions])
        
        # Normalize powers
        if np.max(powers) > 0:
            powers = powers / np.max(powers)
        
        # Combined score
        scores = powers * coherences
        
        # Find peaks
        peaks = []
        peak_indices = signal.find_peaks(scores, height=0.3, distance=5)[0]
        
        for idx in peak_indices:
            peaks.append({
                'direction': directions[idx],
                'power': powers[idx],
                'coherence': coherences[idx],
                'probability': scores[idx],
                'energy': 20 * np.log10(powers[idx] + 1e-10)
            })
        
        # Sort by probability
        peaks.sort(key=lambda x: x['probability'], reverse=True)
        
        return peaks
    
    def _extract_directional_spectrum(self, spectra: List[np.ndarray], 
                                     azimuth: float, elevation: float) -> np.ndarray:
        """Extract spectrum from direction using beamforming"""
        # Simple delay-and-sum beamforming
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        direction = np.array([
            np.cos(el_rad) * np.cos(az_rad),
            np.cos(el_rad) * np.sin(az_rad),
            np.sin(el_rad)
        ])
        
        beamformed = np.zeros_like(spectra[0])
        
        for mic_idx, spectrum in enumerate(spectra):
            # Compute delay for this mic
            delay = np.dot(self.mic_array.positions[mic_idx], direction) / \
                   self.mic_array.speed_of_sound
            
            # Phase shift
            phase_shift = np.exp(-1j * 2 * np.pi * self.freqs * delay)
            
            # Apply and sum
            beamformed += spectrum * phase_shift
        
        return beamformed / len(spectra)
    
    # ========================================================================
    # SST Module
    # ========================================================================
    
    def sst_process(self, pots: List[SSLPot], frame_idx: int) -> List[SSTTrack]:
        """
        Sound Source Tracking (mod_sst.c equivalent)
        
        Implements Kalman filter tracking with adaptive parameters
        """
        dt = self.config['hop_size'] / self.mic_array.sample_rate
        
        # 1. Predict existing tracks
        for track in self.tracks:
            # Detect if track is static or moving
            track.is_static = self._is_track_static(track)
            
            # Adaptive process noise
            if track.is_static:
                track.kalman.Q = np.eye(6) * self.config['sst_sigma_q']
            else:
                track.kalman.Q = np.eye(6) * self.config['sst_sigma_q'] * 100
            
            # Predict
            track.kalman.predict(dt, track.is_static)
            
            # Update probabilities
            track.prob_exist *= self.config['sst_theta_inactive']
        
        # 2. Associate pots with tracks
        associations = self._associate_pots_to_tracks(pots)
        
        # 3. Update tracks with associated pots
        for track_idx, pot_idx in associations:
            if pot_idx is not None:
                track = self.tracks[track_idx]
                pot = pots[pot_idx]
                
                # Measurement update
                measurement = np.array([pot.x, pot.y, pot.z])
                innovation = track.kalman.update(measurement)
                
                # Update track properties
                track.last_frame = frame_idx
                track.status = TrackStatus.ACTIVE.value
                track.prob_exist = min(1.0, track.prob_exist / self.config['sst_theta_inactive'] + 0.1)
                track.prob_active = self.config['sst_p_track']
                
                # Store history
                track.azimuth_history.append(pot.azimuth)
                track.energy_history.append(pot.energy)
                if pot.spectrum is not None:
                    track.spectrum_history.append(pot.spectrum)
                    track.spectral_fingerprint = self._compute_spectral_fingerprint(
                        track.spectrum_history
                    )
        
        # 4. Create new tracks for unassociated pots
        unassociated_pots = [pots[i] for i in range(len(pots)) 
                            if i not in [a[1] for a in associations if a[1] is not None]]
        
        for pot in unassociated_pots:
            if np.random.random() < self.config['sst_p_new']:
                # Create new track
                position = np.array([pot.x, pot.y, pot.z])
                kalman = KalmanState.initialize(position, self.config['sst_sigma_q'])
                
                track = SSTTrack(
                    track_id=self.next_track_id,
                    kalman=kalman,
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    status=TrackStatus.NEW.value,
                    azimuth_history=[pot.azimuth],
                    energy_history=[pot.energy],
                    spectrum_history=[pot.spectrum] if pot.spectrum is not None else []
                )
                
                self.tracks.append(track)
                self.next_track_id += 1
                self.total_tracks_created += 1
        
        # 5. Remove inactive tracks
        active_tracks = []
        for track in self.tracks:
            frames_inactive = frame_idx - track.last_frame
            
            if track.prob_exist > self.config['sst_p_false'] and \
               frames_inactive < self.config['sst_n_inactive']:
                active_tracks.append(track)
            else:
                # Move to history
                track.status = TrackStatus.ENDED.value
                self.track_history.append(track)
        
        self.tracks = active_tracks
        
        return self.tracks
    
    def _is_track_static(self, track: SSTTrack) -> bool:
        """Determine if track is static or moving"""
        if len(track.azimuth_history) < 5:
            return True  # Assume static initially
        
        # Check position variance
        recent_positions = track.azimuth_history[-10:]
        position_variance = np.var(recent_positions)
        
        # Check velocity magnitude
        velocity_magnitude = np.linalg.norm(track.get_velocity())
        
        return (position_variance < self.config['static_variance_threshold'] and
                velocity_magnitude < self.config['motion_threshold'])
    
    def _associate_pots_to_tracks(self, pots: List[SSLPot]) -> List[Tuple[int, Optional[int]]]:
        """
        Associate SSL pots to existing tracks using Hungarian algorithm
        """
        if not self.tracks or not pots:
            return [(i, None) for i in range(len(self.tracks))]
        
        # Build cost matrix
        n_tracks = len(self.tracks)
        n_pots = len(pots)
        cost_matrix = np.full((n_tracks, n_pots), 1000.0)  # Large cost for no association
        
        for t_idx, track in enumerate(self.tracks):
            track_az, track_el = track.to_azimuth_elevation()
            
            for p_idx, pot in enumerate(pots):
                # Angular distance
                angular_dist = self._angular_distance(
                    track_az, track_el, pot.azimuth, pot.elevation
                )
                
                # Spectral similarity (if available)
                spectral_sim = 0.0
                if track.spectral_fingerprint is not None and pot.spectrum is not None:
                    spectral_sim = self._spectral_similarity(
                        track.spectral_fingerprint, pot.spectrum
                    )
                
                # Combined cost (lower is better)
                cost = angular_dist - spectral_sim * 10  # Weight spectral similarity
                
                # Apply threshold
                if angular_dist < 20.0:  # Within 20 degrees
                    cost_matrix[t_idx, p_idx] = cost
        
        # Solve assignment problem
        from scipy.optimize import linear_sum_assignment
        track_indices, pot_indices = linear_sum_assignment(cost_matrix)
        
        # Build associations
        associations = []
        for t_idx in range(n_tracks):
            if t_idx in track_indices:
                idx = np.where(track_indices == t_idx)[0][0]
                p_idx = pot_indices[idx]
                if cost_matrix[t_idx, p_idx] < 100:  # Valid association
                    associations.append((t_idx, p_idx))
                else:
                    associations.append((t_idx, None))
            else:
                associations.append((t_idx, None))
        
        return associations
    
    def _angular_distance(self, az1: float, el1: float, az2: float, el2: float) -> float:
        """Compute angular distance between two directions"""
        # Convert to radians
        az1_rad, el1_rad = np.radians(az1), np.radians(el1)
        az2_rad, el2_rad = np.radians(az2), np.radians(el2)
        
        # Convert to unit vectors
        v1 = np.array([
            np.cos(el1_rad) * np.cos(az1_rad),
            np.cos(el1_rad) * np.sin(az1_rad),
            np.sin(el1_rad)
        ])
        
        v2 = np.array([
            np.cos(el2_rad) * np.cos(az2_rad),
            np.cos(el2_rad) * np.sin(az2_rad),
            np.sin(el2_rad)
        ])
        
        # Angular distance
        cos_angle = np.clip(np.dot(v1, v2), -1, 1)
        return np.degrees(np.arccos(cos_angle))
    
    def _spectral_similarity(self, spectrum1: np.ndarray, spectrum2: np.ndarray) -> float:
        """Compute spectral similarity (0-1)"""
        # Normalize
        s1 = np.abs(spectrum1) / (np.linalg.norm(spectrum1) + 1e-10)
        s2 = np.abs(spectrum2) / (np.linalg.norm(spectrum2) + 1e-10)
        
        # Correlation coefficient
        return np.corrcoef(s1, s2)[0, 1]
    
    def _compute_spectral_fingerprint(self, spectra_history: List[np.ndarray]) -> np.ndarray:
        """Compute average spectral fingerprint"""
        if not spectra_history:
            return None
        
        # Average magnitude spectrum
        avg_spectrum = np.mean([np.abs(s) for s in spectra_history[-10:]], axis=0)
        
        return avg_spectrum
    
    # ========================================================================
    # Main Processing Pipeline
    # ========================================================================
    
    def process_frame(self, multi_channel_frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Process one frame through complete ODAS pipeline
        
        Args:
            multi_channel_frame: (n_mics, frame_size) array
            frame_idx: Frame index
            
        Returns:
            Dictionary with SSL pots and SST tracks
        """
        self.frames_processed += 1
        
        # 1. STFT
        spectra = self.stft_process(multi_channel_frame)
        
        # 2. SSL - Sound Source Localization
        pots = self.ssl_process(spectra, frame_idx)
        
        # 3. SST - Sound Source Tracking
        tracks = self.sst_process(pots, frame_idx)
        
        return {
            'frame_idx': frame_idx,
            'timestamp': frame_idx * self.config['hop_size'] / self.mic_array.sample_rate,
            'pots': [self._pot_to_dict(p) for p in pots],
            'tracks': [self._track_to_dict(t) for t in tracks]
        }
    
    def process_file(self, audio_file: str, output_file: Optional[str] = None) -> Dict:
        """
        Process complete audio file
        
        Args:
            audio_file: Path to raw audio file (6 channels, S16_LE)
            output_file: Optional output JSON file
            
        Returns:
            Complete processing results
        """
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
        
        for frame_idx in range(n_frames):
            start = frame_idx * hop_size
            end = start + frame_size
            
            if end > n_samples:
                break
            
            frame = mic_audio[:, start:end]
            frame_results = self.process_frame(frame, frame_idx)
            results.append(frame_results)
            
            if (frame_idx + 1) % 100 == 0:
                print(f"  Processed {frame_idx+1}/{n_frames} frames "
                      f"({100*(frame_idx+1)/n_frames:.1f}%)")
        
        # Compile final results
        output = {
            'metadata': {
                'file': audio_file,
                'duration': n_samples / self.mic_array.sample_rate,
                'sample_rate': self.mic_array.sample_rate,
                'frames_processed': self.frames_processed,
                'pots_detected': self.total_pots_detected,
                'tracks_created': self.total_tracks_created,
                'timestamp': datetime.now().isoformat()
            },
            'config': self.config,
            'frames': results,
            'track_summary': [self._track_to_dict(t) for t in self.track_history]
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
    
    def _track_to_dict(self, track: SSTTrack) -> Dict:
        """Convert SSTTrack to dictionary"""
        az, el = track.to_azimuth_elevation()
        return {
            'track_id': track.track_id,
            'azimuth': az,
            'elevation': el,
            'position': track.get_position().tolist(),
            'velocity': track.get_velocity().tolist(),
            'status': track.status,
            'prob_exist': track.prob_exist,
            'is_static': track.is_static,
            'first_frame': track.first_frame,
            'last_frame': track.last_frame
        }


# ============================================================================
# Testing and Validation
# ============================================================================

def test_with_synthetic_scene():
    """
    Test with simple synthetic scene
    Elephant: 45° from 5-10s
    Lion: 135° from 15-20s
    """
    print("\nTesting ODAS Processor")
    print("=" * 60)
    
    # Setup mic array (ReSpeaker USB 4 Mic Array)
    mic_positions = np.array([
        [-0.032, 0.000, 0.000],  # Left
        [0.000, -0.032, 0.000],  # Back
        [0.032, 0.000, 0.000],   # Right
        [0.000, 0.032, 0.000]    # Front
    ])
    
    mic_array = MicArray(positions=mic_positions)
    
    # Create processor
    processor = ODASProcessor(mic_array)
    
    # Generate synthetic test signal
    print("\nGenerating synthetic test scene...")
    sample_rate = 16000
    duration = 30  # seconds
    n_samples = duration * sample_rate
    
    # Create 6-channel audio (2 unused + 4 mics)
    audio = np.zeros((6, n_samples))
    
    # Add elephant sound at 45° from 5-10s
    t1_start, t1_end = 5, 10
    f1 = 250  # Hz - elephant rumble
    t = np.arange(n_samples) / sample_rate
    
    # Direction: 45° azimuth (between front and left)
    # Delays for 45°: positive delay for right/back mics
    elephant_signal = np.sin(2 * np.pi * f1 * t)
    elephant_signal[t < t1_start] = 0
    elephant_signal[t > t1_end] = 0
    
    # Apply delays based on 45° direction
    # For 45°: x=cos(45°)=0.707, y=sin(45°)=0.707
    for mic_idx, pos in enumerate(mic_positions):
        delay_distance = (0.707 * pos[0] + 0.707 * pos[1])
        delay_samples = int(delay_distance / 343.0 * sample_rate)
        if delay_samples != 0:
            shifted = np.roll(elephant_signal, delay_samples)
            if delay_samples > 0:
                shifted[:delay_samples] = 0
            else:
                shifted[delay_samples:] = 0
            audio[mic_idx + 2] += shifted
        else:
            audio[mic_idx + 2] += elephant_signal
    
    # Add lion sound at 135° from 15-20s
    t2_start, t2_end = 15, 20
    f2 = 500  # Hz - lion roar
    
    lion_signal = np.sin(2 * np.pi * f2 * t)
    lion_signal[t < t2_start] = 0
    lion_signal[t > t2_end] = 0
    
    # For 135°: x=cos(135°)=-0.707, y=sin(135°)=0.707
    for mic_idx, pos in enumerate(mic_positions):
        delay_distance = (-0.707 * pos[0] + 0.707 * pos[1])
        delay_samples = int(delay_distance / 343.0 * sample_rate)
        if delay_samples != 0:
            shifted = np.roll(lion_signal, delay_samples)
            if delay_samples > 0:
                shifted[:delay_samples] = 0
            else:
                shifted[delay_samples:] = 0
            audio[mic_idx + 2] += shifted
        else:
            audio[mic_idx + 2] += lion_signal
    
    # Convert to int16 and save
    audio_int16 = (audio.T * 32767).astype(np.int16)
    test_file = '/home/azureuser/test_scene.raw'
    audio_int16.tofile(test_file)
    
    print(f"Test file created: {test_file}")
    print(f"  Duration: {duration}s")
    print(f"  Elephant: 45° from 5-10s (250 Hz)")
    print(f"  Lion: 135° from 15-20s (500 Hz)")
    
    # Process the file
    results = processor.process_file(test_file, '/home/azureuser/test_results.json')
    
    # Analyze results
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)
    
    # Check detection accuracy
    elephant_frames = []
    lion_frames = []
    
    for frame in results['frames']:
        time = frame['timestamp']
        
        for track in frame['tracks']:
            az = track['azimuth']
            
            # Check elephant detection (45° ± 10°)
            if 5 <= time <= 10 and 35 <= az <= 55:
                elephant_frames.append({
                    'time': time,
                    'azimuth': az,
                    'track_id': track['track_id']
                })
            
            # Check lion detection (135° ± 10°)
            elif 15 <= time <= 20 and 125 <= az <= 145:
                lion_frames.append({
                    'time': time,
                    'azimuth': az,
                    'track_id': track['track_id']
                })
    
    print(f"\nElephant detection:")
    if elephant_frames:
        azimuths = [f['azimuth'] for f in elephant_frames]
        track_ids = set(f['track_id'] for f in elephant_frames)
        print(f"  Detected in {len(elephant_frames)} frames")
        print(f"  Average azimuth: {np.mean(azimuths):.1f}° (expected: 45°)")
        print(f"  Azimuth std: {np.std(azimuths):.2f}°")
        print(f"  Track IDs: {track_ids}")
    else:
        print("  NOT DETECTED!")
    
    print(f"\nLion detection:")
    if lion_frames:
        azimuths = [f['azimuth'] for f in lion_frames]
        track_ids = set(f['track_id'] for f in lion_frames)
        print(f"  Detected in {len(lion_frames)} frames")
        print(f"  Average azimuth: {np.mean(azimuths):.1f}° (expected: 135°)")
        print(f"  Azimuth std: {np.std(azimuths):.2f}°")
        print(f"  Track IDs: {track_ids}")
    else:
        print("  NOT DETECTED!")
    
    # Check for false positives
    false_positives = []
    for frame in results['frames']:
        time = frame['timestamp']
        if not ((5 <= time <= 10) or (15 <= time <= 20)):
            if frame['tracks']:
                false_positives.append(frame)
    
    print(f"\nFalse positives: {len(false_positives)} frames")
    if false_positives:
        print("  Times:", [f['timestamp'] for f in false_positives[:5]], "...")
    
    print("\n" + "=" * 60)
    print("Test complete!")
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process user-provided file
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'odas_output.json'
        
        # Setup mic array
        mic_positions = np.array([
            [-0.032, 0.000, 0.000],
            [0.000, -0.032, 0.000],
            [0.032, 0.000, 0.000],
            [0.000, 0.032, 0.000]
        ])
        mic_array = MicArray(positions=mic_positions)
        
        # Process
        processor = ODASProcessor(mic_array)
        results = processor.process_file(input_file, output_file)
        
    else:
        # Run test
        test_results = test_with_synthetic_scene()