"""
YAMNet Dataset Curator

This module curates training datasets for fine-tuning YAMNet based on ODAS output analysis.

Key features:
1. Extract samples needing fine-tuning (mismatches, low confidence, unclassified)
2. Store in TensorFlow-compatible format (WAV audio + CSV labels)
3. Reconstruct audio from frequency bins using inverse FFT
4. Maintain detailed logs for dataset provenance
5. Support multi-run dataset aggregation

YAMNet training format (from TensorFlow Hub):
- Audio: 16kHz mono WAV files
- Labels: CSV with filename, label, fold (for train/val/test split)
- YAMNet accepts variable-length audio (minimum ~1 second recommended)

Data flow:
ODAS bins (1024) -> Inverse FFT -> Audio waveform -> Save WAV + metadata
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import scipy.io.wavfile as wavfile
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf


class YAMNetDatasetCurator:
    """Curates datasets for YAMNet fine-tuning from ODAS analysis results"""
    
    def __init__(self, output_dir='outputs/yamnet_datasets'):
        """
        Initialize curator.
        
        Args:
            output_dir: Root directory for YAMNet training datasets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create config
        self.config_path = self.output_dir / 'curator_config.json'
        self.config = self._load_or_create_config()
        
        # Audio parameters (must match ODAS configuration)
        # ODAS uses frameSize=512 → halfFrameSize=257 bins, hopSize=128 @ 16kHz
        self.sample_rate = 16000  # Hz - YAMNet requires 16kHz
        self.frame_duration = 0.008  # 8ms per hop (128/16000)
        self.n_fft = 512           # Must match ODAS frameSize (NOT 1024)
        self.hop_length = 128      # hopSize from ODAS config

        # AudioReconstructor for multi-frame .bin sidecar reconstruction
        from audio_reconstructor import AudioReconstructor
        self._recon = AudioReconstructor(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
    def _load_or_create_config(self):
        """Load existing config or create default, merging any new default keys"""
        default_criteria = {
            'include_mismatches': True,
            'include_low_confidence': True,
            'confidence_threshold': 0.75,
            'include_unclassified': True,
            'include_direction_mismatch': True,
            'direction_threshold_deg': 15.0,
            'min_activity': 0.01,
            'save_unknown': True,
        }
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            # Merge any missing/updated keys into curation_criteria
            criteria = config.setdefault('curation_criteria', {})
            changed = False
            for k, v in default_criteria.items():
                if k not in criteria:
                    criteria[k] = v
                    changed = True
            # Force min_activity down if old config has a value >= 0.3
            if criteria.get('min_activity', 0) >= 0.3:
                criteria['min_activity'] = 0.01
                changed = True
            if changed:
                self._save_config(config)
            return config

        config = {
            'created_at': datetime.now().isoformat(),
            'active_dataset': 'yamnet_train_001',
            'unknown_dataset': 'yamnet_unknown_001',  # For manual labeling
            'datasets': {},
            'curation_criteria': {
                'include_mismatches': True,  # YAMNet != ground truth
                'include_low_confidence': True,  # YAMNet confidence < threshold
                'confidence_threshold': 0.75,  # Save if < this value (75%)
                'include_unclassified': True,  # YAMNet didn't classify
                'include_direction_mismatch': True,  # Direction doesn't match ground truth
                'direction_threshold_deg': 15.0,  # Max angular error (degrees)
                'min_activity': 0.01,  # Minimum activity level (near-zero filters only dead tracks)
                'save_unknown': True  # Save non-matching samples for manual labeling
            },
            'audio_params': {
                'sample_rate': 16000,
                'target_duration': 1.0,  # Target 1 second clips
                'overlap_frames': 5  # Overlap adjacent frames for temporal context
            }
        }
        self._save_config(config)
        return config
    
    def _save_config(self, config=None):
        """Save configuration"""
        if config is None:
            config = self.config
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def get_active_dataset(self):
        """Get active dataset name"""
        return self.config['active_dataset']
    
    def get_unknown_dataset(self):
        """Get unknown dataset name for manual labeling"""
        return self.config.get('unknown_dataset', 'yamnet_unknown_001')
    
    def set_unknown_dataset(self, dataset_name):
        """Set unknown dataset for manual labeling"""
        self.config['unknown_dataset'] = dataset_name
        if dataset_name not in self.config['datasets']:
            self._create_dataset(dataset_name)
        self._save_config()
    
    def set_active_dataset(self, dataset_name):
        """Set active dataset and create if needed"""
        self.config['active_dataset'] = dataset_name
        
        if dataset_name not in self.config['datasets']:
            self._create_dataset(dataset_name)
        
        self._save_config()
        return self.get_dataset_path(dataset_name)
    
    def _create_dataset(self, dataset_name):
        """Create new dataset structure"""
        dataset_path = self.output_dir / dataset_name
        dataset_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (dataset_path / 'audio').mkdir(exist_ok=True)
        (dataset_path / 'spectrograms').mkdir(exist_ok=True)  # For visualization
        (dataset_path / 'metadata').mkdir(exist_ok=True)
        
        # Initialize dataset metadata
        self.config['datasets'][dataset_name] = {
            'created_at': datetime.now().isoformat(),
            'sample_count': 0,
            'samples_by_label': {},
            'runs_processed': [],
            'curation_log': []
        }
        self._save_config()
    
    def get_dataset_path(self, dataset_name=None):
        """Get path to dataset"""
        if dataset_name is None:
            dataset_name = self.get_active_dataset()
        return self.output_dir / dataset_name
    
    def list_datasets(self):
        """List all available datasets"""
        return list(self.config['datasets'].keys())
    
    def curate_from_analysis(self, analysis_results, run_id):
        """
        Extract samples for training from analysis results.
        
        Clean auto-collection strategy:
        1. First verify spatial/temporal alignment with ground truth
        2. If aligned AND has issues (low conf/wrong label) → Training dataset
        3. If NOT aligned OR no ground truth → Unknown dataset for manual verification
        
        Args:
            analysis_results: Dict from analyzer with matches and yamnet_stats
            run_id: Unique identifier for this run
        
        Returns:
            dict: Statistics about curation
        """
        criteria = self.config['curation_criteria']
        matches = analysis_results.get('matches', [])
        
        # Filter matches based on criteria
        samples_for_training = []  # Clean samples with issues - for training
        samples_unknown = []  # Need manual verification
        
        for match in matches:
            # Get data
            yamnet_class = match.get('yamnet_class', 'unclassified')
            yamnet_conf = match.get('yamnet_confidence', 0.0)
            ground_truth = match.get('label', 'unknown')
            activity = match['detection'].get('activity', 0)
            angular_error = match.get('angular_error', None)
            confidence = match.get('confidence', 0)  # Spatial matching confidence
            match_type = match.get('match_type', 'unmatched')
            
            # Skip low activity samples (too quiet) — UNLESS the firmware
            # already saved a spectra .bin for this detection.  The .bin is
            # written only when activity was high during the 6-hop window;
            # the per-frame activity value may legitimately be near-zero on
            # the frame where the event was *reported* (the buffer is flushed
            # at the 6th hop, which might follow the peak activity frames).
            spectra_file = match['detection'].get('spectra_file', '')
            has_spectra = bool(spectra_file and os.path.exists(spectra_file))
            if activity < criteria['min_activity'] and not has_spectra:
                continue
            
            # STEP 1: Check if this is spatially and temporally aligned with ground truth
            direction_threshold = criteria.get('direction_threshold_deg', 15.0)
            
            is_spatially_aligned = False
            if angular_error is not None and angular_error <= direction_threshold:
                is_spatially_aligned = True
            
            # Check if matched to a ground truth source (implies temporal alignment)
            is_temporally_aligned = (match_type == 'ground_truth')
            
            # Must be both spatially AND temporally aligned for clean training data
            is_clean_match = is_spatially_aligned and is_temporally_aligned and ground_truth != 'unknown'
            
            if not is_clean_match:
                # NOT aligned with ground truth → Manual verification needed
                if criteria.get('save_unknown', True):
                    reason_parts = []
                    if not is_spatially_aligned and angular_error is not None:
                        reason_parts.append(f'direction_error_{angular_error:.1f}deg')
                    if not is_temporally_aligned:
                        reason_parts.append('temporal_mismatch')
                    if ground_truth == 'unknown':
                        reason_parts.append('no_ground_truth')
                    
                    match['curation_reason'] = ','.join(reason_parts) if reason_parts else 'needs_manual_verification'
                    match['dataset_type'] = 'unknown'
                    match['manual_verification_needed'] = True
                    samples_unknown.append(match)
                continue
            
            # STEP 2: Sample is clean (aligned with ground truth)
            # Now check if it has issues that need training
            has_issues = False
            reason = []
            
            # Issue 1: Unclassified
            if criteria.get('include_unclassified', True) and yamnet_class == 'unclassified':
                has_issues = True
                reason.append('unclassified')
            
            # Issue 2: Low confidence (< 75% default)
            if criteria.get('include_low_confidence', True) and yamnet_conf < criteria.get('confidence_threshold', 0.75):
                has_issues = True
                reason.append(f'low_confidence_{yamnet_conf:.2f}')
            
            # Issue 3: Label mismatch
            if criteria.get('include_mismatches', True):
                yamnet_lower = yamnet_class.lower()
                gt_lower = ground_truth.lower()
                
                is_label_match = (yamnet_lower == gt_lower or 
                                 yamnet_lower in gt_lower or 
                                 gt_lower in yamnet_lower)
                
                if not is_label_match:
                    has_issues = True
                    reason.append(f'mismatch_yamnet:{yamnet_class}_gt:{ground_truth}')
            
            # STEP 3: Categorize
            if has_issues:
                # Ambiguous top-K (two classes tied on hop-votes): the firmware
                # already resolves the tie by avg_confidence, so event_class_name
                # is the best available YAMNet guess.  Since training is supervised
                # by the GT label (not YAMNet), an ambiguous detection is actually
                # a valuable training sample — keep it in training, just flag it.
                match['curation_reason'] = ','.join(reason) + (',ambiguous_topk' if match.get('ambiguous') else '')
                match['dataset_type'] = 'training'
                match['clean_match'] = True
                match['angular_error_deg'] = angular_error
                samples_for_training.append(match)
            # else: Clean sample with no issues → Skip (already working well)
        
        # Save training samples to active dataset
        stats = {'training': {}, 'unknown': {}}
        
        if samples_for_training:
            training_stats = self._save_samples(
                samples_for_training, 
                run_id, 
                analysis_results,
                dataset_name=self.get_active_dataset()
            )
            stats['training'] = training_stats
        else:
            stats['training'] = {
                'saved': 0,
                'skipped': 0,
                'dataset': self.get_active_dataset()
            }
        
        # Save unknown samples to unknown dataset
        if samples_unknown and criteria.get('save_unknown', True):
            unknown_stats = self._save_samples(
                samples_unknown,
                run_id,
                analysis_results,
                dataset_name=self.get_unknown_dataset()
            )
            stats['unknown'] = unknown_stats
        else:
            stats['unknown'] = {
                'saved': 0,
                'skipped': len(samples_unknown),
                'dataset': self.get_unknown_dataset()
            }
        
        # Return combined stats
        return {
            'saved': stats['training']['saved'],
            'unknown_saved': stats['unknown']['saved'],
            'total_processed': len(matches),
            'dataset': self.get_active_dataset(),
            'unknown_dataset': self.get_unknown_dataset(),
            'label_distribution': stats['training'].get('label_distribution', {}),
            'unknown_count': stats['unknown']['saved']
        }
    
    def _save_samples(self, samples, run_id, analysis_results, dataset_name=None):
        """Save samples to specified dataset"""
        if dataset_name is None:
            dataset_name = self.get_active_dataset()
        
        # Ensure dataset exists
        if dataset_name not in self.config['datasets']:
            self._create_dataset(dataset_name)
        
        dataset_path = self.get_dataset_path(dataset_name)
        audio_dir = dataset_path / 'audio'
        spec_dir = dataset_path / 'spectrograms'
        
        saved_count = 0
        label_counts = {}
        sample_metadata = []
        csv_path = None  # Initialize to None
        
        # ── Group samples by track_id and stitch consecutive .bin files ────────
        # Each ODAS track can fire multiple ROLLING_HOPS evaluation events,
        # each producing one .bin sidecar (96×257 float32, ~0.76 s).
        # Stitching all .bin files for the same track yields 1.5–4.6 s WAVs,
        # which is much better for YAMNet (ideal input ≥ 0.96 s).
        from collections import defaultdict as _dd
        track_groups = _dd(list)
        for sample in samples:
            tid = sample['detection'].get('track_id', id(sample))
            track_groups[tid].append(sample)

        # Build one stitched event per track
        stitched_events = []
        for tid, group_samples in track_groups.items():
            # Collect unique spectra_files in timestamp order (oldest→newest)
            seen_sf: set = set()
            sf_ordered = []
            fallback_bins = []
            for s in sorted(group_samples, key=lambda x: x['detection'].get('timestamp', 0)):
                sf = s['detection'].get('spectra_file', '')
                if sf and sf not in seen_sf and os.path.exists(sf):
                    seen_sf.add(sf)
                    sf_ordered.append(sf)
                if not sf and s['detection'].get('bins'):
                    fallback_bins = s['detection'].get('bins', [])

            # Representative sample → highest activity in the group
            rep = max(group_samples, key=lambda s: s['detection'].get('activity', 0))

            stitched_events.append({
                'rep': rep,
                'spectra_files': sf_ordered,
                'fallback_bins': fallback_bins,
                'n_bins': len(sf_ordered),
            })

        for idx, event in enumerate(stitched_events):
            try:
                rep = event['rep']
                det = rep['detection']
                sf_ordered = event['spectra_files']
                fallback_bins = event['fallback_bins']
                n_bins = event['n_bins']

                # ── Audio reconstruction ───────────────────────────────────
                # Priority: stitch all .bin → single .bin → legacy 257-float bins
                audio_waveform = None
                used_spectra_file = sf_ordered[0] if sf_ordered else None

                if n_bins > 1:
                    result = self._recon.reconstruct_from_spectra_files(sf_ordered)
                    if result is not None:
                        audio_waveform = result['audio']
                elif n_bins == 1:
                    result = self._recon.reconstruct_from_spectra_file(sf_ordered[0])
                    if result is not None:
                        audio_waveform = result['audio']
                elif fallback_bins:
                    audio_waveform = self._reconstruct_audio_from_bins(fallback_bins)

                if audio_waveform is None or len(audio_waveform) == 0:
                    continue

                # Get label (prefer ground truth over YAMNet)
                label = rep.get('label', 'unknown')
                if label == 'unknown':
                    label = rep.get('yamnet_class', rep.get('event_class_name', 'unknown'))

                # Generate unique filename — include number of stitched bins
                timestamp = det.get('timestamp', 0)
                timestamp_str = f"{timestamp:.3f}".replace('.', '_')
                bins_tag = f"{n_bins}bins" if n_bins > 0 else "legacy"

                angular_error = rep.get('angular_error_deg', rep.get('angular_error'))
                if angular_error is not None and angular_error >= 0:
                    filename_base = f"{run_id}_{idx:04d}_t{timestamp_str}_{bins_tag}_dir{angular_error:.0f}deg_{label}"
                else:
                    filename_base = f"{run_id}_{idx:04d}_t{timestamp_str}_{bins_tag}_{label}"

                # Save audio as WAV
                wav_path = audio_dir / f"{filename_base}.wav"
                self._save_audio(audio_waveform, wav_path)

                # Save spectrogram visualization using first (or only) .bin
                spec_path = spec_dir / f"{filename_base}.png"
                if used_spectra_file and os.path.exists(used_spectra_file):
                    patch = np.fromfile(used_spectra_file, dtype=np.float32)
                    n = patch.size // 257
                    self._save_spectrogram_plot(patch[:n*257].reshape(n, 257), spec_path, label)
                elif fallback_bins:
                    self._save_spectrogram_plot(np.array(fallback_bins).reshape(1, -1), spec_path, label)

                # Record metadata (enhanced for manual verification)
                top_k = rep.get('top_k_candidates', [])
                top_k_summary = '|'.join(
                    f"{c.get('class_name','?')}({c.get('hop_votes',0)}v,{c.get('avg_confidence',0):.2f})"
                    for c in top_k[:5]  # top-5 at most
                )
                metadata = {
                    'filename': f"{filename_base}.wav",
                    'run_id': run_id,
                    'timestamp': timestamp,
                    'label': label,
                    'yamnet_class': rep.get('yamnet_class', 'unclassified'),
                    'yamnet_confidence': rep.get('yamnet_confidence', 0.0),
                    'yamnet_votes': rep.get('yamnet_votes', 0),
                    'yamnet_ambiguous': rep.get('ambiguous', False),
                    'top_k_candidates': top_k_summary,
                    'ground_truth': rep.get('label', 'unknown'),
                    'curation_reason': rep.get('curation_reason', ''),
                    'activity': det.get('activity', 0),
                    'n_stitched_bins': n_bins,
                    'stitched_duration_s': round(len(audio_waveform) / self._recon.sample_rate, 3),
                    'position': {
                        'x': det['x'],
                        'y': det['y'],
                        'z': det['z']
                    },
                    'confidence': rep.get('confidence', 0),
                    'angular_error': rep.get('angular_error', -1),
                    'dataset_type': rep.get('dataset_type', 'unknown'),
                    'clean_match': rep.get('clean_match', False),
                    'manual_verification_needed': rep.get('manual_verification_needed', False)
                }
                sample_metadata.append(metadata)

                # Update counts
                saved_count += 1
                label_counts[label] = label_counts.get(label, 0) + 1

            except Exception as e:
                print(f"WARNING: Failed to save sample {idx}: {e}")
                continue
        
        # Save metadata CSV (YAMNet format: filename, label, fold)
        if sample_metadata:
            df = pd.DataFrame(sample_metadata)
            
            # Add fold for train/val/test split (can be modified later)
            df['fold'] = 'train'  # Default to train, can be reassigned
            
            # Save to CSV
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = dataset_path / 'metadata' / f'{run_id}_{timestamp_str}.csv'
            df.to_csv(csv_path, index=False)
            
            # Also append to master labels file (deduplicate by filename)
            master_csv = dataset_path / 'labels.csv'
            if master_csv.exists():
                df_master = pd.read_csv(master_csv)
                # Drop any rows whose filename already exists in the master
                # (prevents duplicate rows when the same run_id is re-analysed)
                df_new = df[~df['filename'].isin(df_master['filename'])]
                if not df_new.empty:
                    df_master = pd.concat([df_master, df_new], ignore_index=True)
                    df_master.to_csv(master_csv, index=False)
            else:
                df.to_csv(master_csv, index=False)
        
        # Update dataset config — skip if this run_id was already recorded
        dataset_meta = self.config['datasets'][dataset_name]
        already_processed = any(
            r.get('run_id') == run_id for r in dataset_meta.get('runs_processed', [])
        )
        if not already_processed:
            dataset_meta['sample_count'] += saved_count

            for label, count in label_counts.items():
                dataset_meta['samples_by_label'][label] = \
                    dataset_meta['samples_by_label'].get(label, 0) + count

            dataset_meta['runs_processed'].append({
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'samples_added': saved_count,
                'label_distribution': label_counts
            })

            dataset_meta['curation_log'].append({
                'timestamp': datetime.now().isoformat(),
                'run_id': run_id,
                'action': 'curate_from_analysis',
                'samples_added': saved_count
            })
        
        dataset_meta['last_updated'] = datetime.now().isoformat()
        self._save_config()
        
        return {
            'saved': saved_count,
            'skipped': len(samples) - saved_count,
            'dataset': dataset_name,
            'label_distribution': label_counts,
            'csv_path': str(csv_path) if csv_path else 'N/A'
        }
    
    def _reconstruct_audio_from_bins(self, bins):
        """
        Reconstruct audio waveform from magnitude spectrum bins using inverse FFT.
        
        Args:
            bins: List/array of magnitude spectrum values (1024 bins from ODAS)
        
        Returns:
            np.ndarray: Reconstructed audio waveform
        
        Note: This is a simplified reconstruction. ODAS provides magnitude spectrum only,
        so we use Griffin-Lim algorithm or random phase initialization.
        """
        bins = np.array(bins, dtype=np.float32)
        
        # ODAS provides magnitude spectrum
        # For inverse FFT, we need complex spectrum (magnitude + phase)
        # Since we don't have phase, use Griffin-Lim algorithm for better quality
        
        # Simplified version: random phase initialization
        # For production, implement full Griffin-Lim
        magnitude = bins
        
        # Random phase
        phase = np.random.uniform(-np.pi, np.pi, len(magnitude))
        
        # Create complex spectrum
        complex_spec = magnitude * np.exp(1j * phase)
        
        # Inverse FFT
        audio = np.fft.irfft(complex_spec, n=self.n_fft)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio.astype(np.float32)
    
    def _reconstruct_audio_griffin_lim(self, magnitude_spec, n_iter=50):
        """
        Reconstruct audio using Griffin-Lim algorithm.
        
        Better quality than random phase, but slower.
        
        Args:
            magnitude_spec: Magnitude spectrum (1024 bins)
            n_iter: Number of Griffin-Lim iterations
        
        Returns:
            np.ndarray: Reconstructed audio
        """
        magnitude = np.array(magnitude_spec, dtype=np.float32)
        
        # Initialize with random phase
        phase = np.random.uniform(-np.pi, np.pi, len(magnitude))
        complex_spec = magnitude * np.exp(1j * phase)
        
        # Griffin-Lim iterations
        for _ in range(n_iter):
            # Inverse FFT
            audio = np.fft.irfft(complex_spec, n=self.n_fft)
            
            # Forward FFT
            complex_spec = np.fft.rfft(audio, n=self.n_fft)
            
            # Replace magnitude, keep phase
            phase = np.angle(complex_spec)
            complex_spec = magnitude * np.exp(1j * phase)
        
        # Final inverse FFT
        audio = np.fft.irfft(complex_spec, n=self.n_fft)
        
        # Normalize
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.9
        
        return audio.astype(np.float32)
    
    def _save_audio(self, waveform, filepath):
        """Save audio waveform as WAV file"""
        # Ensure waveform is in correct range [-1, 1]
        waveform = np.clip(waveform, -1.0, 1.0)
        
        # Convert to int16 for WAV
        waveform_int16 = (waveform * 32767).astype(np.int16)
        
        # Save using scipy
        wavfile.write(filepath, self.sample_rate, waveform_int16)
    
    def _save_spectrogram_plot(self, data, filepath, label):
        """Save spectrogram visualization.

        Args:
            data: 2-D array (frames × bins) or 1-D flat bins array.
        """
        data = np.array(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        plt.figure(figsize=(10, 4))
        if data.shape[0] > 1:
            # Multi-frame patch — show as heatmap (time × freq)
            plt.imshow(data.T, aspect='auto', origin='lower',
                       interpolation='nearest', cmap='magma')
            plt.colorbar(label='Magnitude')
            plt.xlabel('Frame')
            plt.ylabel('Frequency Bin')
        else:
            plt.plot(data[0])
            plt.xlabel('Frequency Bin')
            plt.ylabel('Magnitude')
            plt.grid(True, alpha=0.3)
        plt.title(f'Spectrogram — {label}')
        plt.tight_layout()
        plt.savefig(filepath, dpi=100, bbox_inches='tight')
        plt.close()
    
    def get_dataset_stats(self, dataset_name=None):
        """Get statistics for a dataset"""
        if dataset_name is None:
            dataset_name = self.get_active_dataset()
        
        if dataset_name not in self.config['datasets']:
            return None
        
        dataset_meta = self.config['datasets'][dataset_name]
        dataset_path = self.get_dataset_path(dataset_name)
        
        # Count actual files
        audio_dir = dataset_path / 'audio'
        audio_files = list(audio_dir.glob('*.wav')) if audio_dir.exists() else []
        
        return {
            'name': dataset_name,
            'path': str(dataset_path),
            'sample_count': dataset_meta['sample_count'],
            'actual_audio_files': len(audio_files),
            'samples_by_label': dataset_meta['samples_by_label'],
            'runs_processed': len(dataset_meta['runs_processed']),
            'created_at': dataset_meta.get('created_at'),
            'last_updated': dataset_meta.get('last_updated'),
            'recent_runs': dataset_meta['runs_processed'][-5:]  # Last 5 runs
        }
    
    def create_tensorflow_dataset(self, dataset_name=None, train_val_test_split=(0.7, 0.15, 0.15)):
        """
        Prepare dataset in TensorFlow Hub YAMNet format with train/val/test split.
        
        Args:
            dataset_name: Name of dataset to prepare
            train_val_test_split: Tuple of (train, val, test) proportions
        
        Returns:
            dict: Paths to train/val/test splits
        """
        if dataset_name is None:
            dataset_name = self.get_active_dataset()
        
        dataset_path = self.get_dataset_path(dataset_name)
        labels_csv = dataset_path / 'labels.csv'
        
        if not labels_csv.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_csv}")
        
        # Load labels
        df = pd.read_csv(labels_csv)
        
        # Assign folds (stratified by label)
        from sklearn.model_selection import train_test_split
        
        # Split into train, val, test
        train_ratio, val_ratio, test_ratio = train_val_test_split
        
        # First split: train vs (val+test)
        train_df, temp_df = train_test_split(
            df, 
            test_size=(1 - train_ratio),
            stratify=df['label'],
            random_state=42
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['label'],
            random_state=42
        )
        
        # Assign folds
        df.loc[train_df.index, 'fold'] = 'train'
        df.loc[val_df.index, 'fold'] = 'val'
        df.loc[test_df.index, 'fold'] = 'test'
        
        # Save updated labels
        df.to_csv(labels_csv, index=False)
        
        # Create split CSVs for convenience
        train_csv = dataset_path / 'train_labels.csv'
        val_csv = dataset_path / 'val_labels.csv'
        test_csv = dataset_path / 'test_labels.csv'
        
        train_df.to_csv(train_csv, index=False)
        val_df.to_csv(val_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        
        return {
            'dataset_path': str(dataset_path),
            'labels_csv': str(labels_csv),
            'train_csv': str(train_csv),
            'val_csv': str(val_csv),
            'test_csv': str(test_csv),
            'splits': {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df)
            }
        }
    
    def merge_datasets(self, dataset_names, new_dataset_name):
        """
        Merge multiple datasets into a new one.
        
        Args:
            dataset_names: List of dataset names to merge
            new_dataset_name: Name for the merged dataset
        
        Returns:
            dict: Statistics about merged dataset
        """
        # Create new dataset
        self._create_dataset(new_dataset_name)
        new_dataset_path = self.get_dataset_path(new_dataset_name)
        
        all_metadata = []
        
        # Copy files from each dataset
        for dataset_name in dataset_names:
            if dataset_name not in self.config['datasets']:
                print(f"WARNING: Dataset '{dataset_name}' not found, skipping")
                continue
            
            src_path = self.get_dataset_path(dataset_name)
            src_labels = src_path / 'labels.csv'
            
            if not src_labels.exists():
                continue
            
            # Load metadata
            df = pd.read_csv(src_labels)
            all_metadata.append(df)
            
            # Copy audio files
            for _, row in df.iterrows():
                src_audio = src_path / 'audio' / row['filename']
                dst_audio = new_dataset_path / 'audio' / row['filename']
                
                if src_audio.exists():
                    import shutil
                    shutil.copy2(src_audio, dst_audio)
        
        # Merge metadata
        if all_metadata:
            merged_df = pd.concat(all_metadata, ignore_index=True)
            merged_df.to_csv(new_dataset_path / 'labels.csv', index=False)
            
            # Update config
            dataset_meta = self.config['datasets'][new_dataset_name]
            dataset_meta['sample_count'] = len(merged_df)
            dataset_meta['merged_from'] = dataset_names
            dataset_meta['samples_by_label'] = merged_df['label'].value_counts().to_dict()
            self._save_config()
            
            return {
                'dataset': new_dataset_name,
                'total_samples': len(merged_df),
                'source_datasets': dataset_names
            }
        
        return None
