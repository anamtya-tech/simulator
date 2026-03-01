"""
Analyzer module to process ODAS output and create training datasets.

This module:
1. Loads scene configuration (ground truth source locations and timing)
2. Parses ODAS output (detected peaks with x,y,z and frequency bins)
3. Matches detected peaks to known sources using angle/distance threshold
4. Creates labeled dataset: [1024 frequency bins, label] for ML training
5. Generates interactive HTML visualization report
6. Saves analysis results as JSON

File outputs:
- outputs/analysis/{run_id}_analysis.json: Complete analysis results
- outputs/analysis/{run_id}_report.html: Interactive Plotly visualization
- outputs/analysis/{run_id}_dataset.csv: Training dataset [bins, label]
"""

import streamlit as st
import numpy as np
import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dataset_manager import DatasetManager
from yamnet_dataset_curator import YAMNetDatasetCurator

# Global configuration
CONFIG = {
    'angle_threshold_deg': 15.0,  # Max angular difference to match (widened from 10→15 to catch ODAS localisation jitter)
    # Asymmetric time windows around each GT source.
    # ODAS Kalman filter has a variable startup delay (observed: 0–6s) and a
    # track-persistence tail after the sound ends (observed: 2.5–13s).
    # Using a single symmetric offset (old: ±2.5s) caused temporal_mismatch on
    # every sample that ODAS started tracking late or kept alive after GT end.
    # pre:  how far BEFORE the GT start we accept a detection (Frog early-start: 3.5s)
    # post: how far AFTER  the GT end   we accept a detection (Wolfhowl persistence: 12.75s)
    'time_window_pre_s':  5.0,   # seconds before GT start  (was: 2.5s symmetric)
    'time_window_post_s': 14.0,  # seconds after  GT end    (was: 2.5s symmetric)
    'distance_weight': 0.1,  # Weight for distance in matching (vs angle)
    # Planar microphone arrays (all mics at z=0) cannot reliably estimate
    # source elevation.  When True, spatial matching uses azimuth-only
    # (horizontal-plane) angle difference instead of full 3D angular distance.
    # This prevents the large elevation error from blocking correct matches.
    'use_azimuth_only_matching': True,
}

class ResultAnalyzer:
    def __init__(self, output_dir, odas_logs_dir):
        self.base_output_dir = Path(output_dir)
        self.runs_dir = self.base_output_dir / 'runs'
        self.analysis_dir = self.base_output_dir / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.odas_logs_dir = odas_logs_dir
        
        # Initialize dataset manager
        self.dataset_manager = DatasetManager(output_dir)
        # Initialize YAMNet curator (writes audio/spectrograms to yamnet_datasets/)
        self.yamnet_curator = YAMNetDatasetCurator(
            output_dir=str(self.base_output_dir / 'yamnet_datasets')
        )
    
    def render(self):
        """Render the analyzer interface"""
        st.subheader("Results Analysis")
        st.markdown("Analyze ODAS output and generate training datasets with interactive visualization")
        st.info("🎯 Using YAMNet classifications from ODAS")
        
        # Dataset curation settings
        with st.expander("💾 Dataset Curation Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                save_to_dataset = st.checkbox(
                    "Save to dataset",
                    value=True,
                    help="Curate GT-matched detections into the YAMNet training dataset"
                )
                active_dataset = self.dataset_manager.get_active_dataset_name()
                st.info(f"Active dataset: **{active_dataset}**")
            
            with col2:
                # Legacy spatial-confidence threshold (kept for DatasetManager CSV path)
                confidence_threshold = self.dataset_manager._load_config()['confidence_threshold']

                # YAMNet classification confidence threshold for the curator.
                # Samples are saved when YAMNet confidence is *below* this value
                # (wrong / unsure = needs training). Correct high-confidence
                # detections are skipped — they don't need more training data.
                cur_criteria = self.yamnet_curator.config.get('curation_criteria', {})
                yamnet_conf_threshold = st.slider(
                    "YAMNet confidence threshold",
                    min_value=0.0, max_value=1.0,
                    value=float(cur_criteria.get('confidence_threshold', 0.75)),
                    step=0.05,
                    help=(
                        "Save sample when YAMNet confidence is **below** this value. "
                        "Lower = only save very wrong predictions. "
                        "Higher = save anything YAMNet isn't fully sure about. "
                        "Default 0.75 (save if < 75% confident)."
                    )
                )
                # Persist change to curator config immediately
                if yamnet_conf_threshold != cur_criteria.get('confidence_threshold', 0.75):
                    self.yamnet_curator.config['curation_criteria']['confidence_threshold'] = yamnet_conf_threshold
                    self.yamnet_curator._save_config()

            st.markdown("**🏷️ Label Strategy**")
            LABEL_STRATEGIES = [
                "ODAS event voting",
                "Python YAMNet (re-classify .bin)",
                "Ground truth only",
                "Fine-tuned model",
            ]
            label_strategy = st.selectbox(
                "Label source",
                LABEL_STRATEGIES,
                index=0,
                help=(
                    "**ODAS event voting** — use top-K × N-hop vote winner from firmware (default).\n\n"
                    "**Python YAMNet (re-classify .bin)** — ignore firmware labels, re-run Python YAMNet "
                    "on the saved .bin sidecar patches. Useful after updating the model without re-running ODAS.\n\n"
                    "**Ground truth only** — label = scene ground truth from spatial alignment. "
                    "Ignores YAMNet entirely; unmatched detections are skipped.\n\n"
                    "**Fine-tuned model** — re-classify .bin patches using the active fine-tuned .pth model "
                    "(if available)."
                )
            )
            st.session_state['label_strategy'] = label_strategy
        
        # Load run selection
        run_files = sorted(
            self.runs_dir.glob("*.json"),
            key=os.path.getmtime,
            reverse=True
        )
        
        if not run_files:
            st.warning("No simulation runs found. Please run a simulation first.")
            return
        
        selected_run_file = st.selectbox(
            "Select Run",
            run_files,
            format_func=lambda x: x.stem
        )
        
        # Load run data
        with open(selected_run_file, 'r') as f:
            run_data = json.load(f)
        
        run_id = run_data.get('run_id', run_data.get('run_name', selected_run_file.stem))
        
        # Display run info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Run ID", run_id)
        with col2:
            st.metric("Scene", run_data.get('scene_name', 'Unknown'))
        with col3:
            st.metric("Render ID", run_data.get('render_id', 'N/A'))
        with col4:
            st.metric("Duration", f"{run_data.get('scene_metadata', {}).get('duration', 0)}s")
        
        # Configuration
        with st.expander("⚙️ Analysis Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                angle_threshold = st.slider(
                    "Angle Threshold (degrees)",
                    1.0, 45.0, CONFIG['angle_threshold_deg'], 1.0,
                    help="Max angular difference to match detection to source"
                )
            with col2:
                save_unmatched = st.checkbox(
                    "Include unmatched in dataset",
                    value=False,
                    help="Save unmatched detections with label 'unknown'"
                )
            st.caption(
                "🕒 **ODAS Kalman timing offsets** — ODAS takes time to converge on a source "
                "(startup delay) and keeps tracking after it ends (persistence tail). "
                "Widen the post-window to catch late starters; the pre-window catches "
                "early ghost tracks."
            )
            col3, col4 = st.columns(2)
            with col3:
                time_pre = st.slider(
                    "Pre-window (s before GT start)",
                    0.0, 10.0, CONFIG['time_window_pre_s'], 0.5,
                    help="Accept ODAS detections up to this many seconds BEFORE the GT source starts. "
                         "Handles cases where Kalman locks on early (observed: up to 3.5s)."
                )
            with col4:
                time_post = st.slider(
                    "Post-window (s after GT end)",
                    0.0, 20.0, CONFIG['time_window_post_s'], 0.5,
                    help="Accept ODAS detections up to this many seconds AFTER the GT source ends. "
                         "Handles Kalman persistence tail (observed: up to 12.75s for Wolfhowl)."
                )
        
        # Check if analysis exists
        analysis_path = self._get_analysis_path(run_id)
        report_path = self._get_report_path(run_id)
        dataset_path = self._get_dataset_path(run_id)
        
        analysis_exists = analysis_path.exists()
        
        # Analyze button
        col1, col2 = st.columns([3, 1])
        with col1:
            analyze_button = st.button(
                "🔍 Analyze Run" if not analysis_exists else "🔄 Regenerate Analysis",
                type="primary",
                use_container_width=True
            )
        with col2:
            if analysis_exists:
                if st.button("🗑️ Delete", use_container_width=True):
                    self._delete_analysis(run_id)
                    st.rerun()
        
        # Run analysis
        if analyze_button:
            with st.spinner("Analyzing..."):
                results = self._analyze_run(run_data, angle_threshold, save_unmatched,
                                            time_pre=time_pre, time_post=time_post)
                
                if results:
                    # Use YAMNet classifications instead of custom model
                    strategy = st.session_state.get('label_strategy', 'ODAS event voting')
                    results = self._apply_yamnet_classifications(results, label_strategy=strategy)
                    
                    # Save analysis JSON
                    self._save_analysis(run_id, results, angle_threshold)
                    
                    # Generate HTML report
                    self._generate_html_report(run_id, results)
                    
                    # Create dataset CSV
                    self._create_dataset(results, run_id, save_unmatched)
                    
                    # Save to training dataset if enabled
                    if save_to_dataset:
                        dataset_stats = self.dataset_manager.save_matches_to_dataset(
                            results['matches'],
                            run_id,
                            confidence_threshold=confidence_threshold
                        )
                        
                        st.success(f"💾 Dataset curation: {dataset_stats['saved']} samples saved to {dataset_stats['dataset']}")
                        if dataset_stats['skipped_low_confidence'] > 0:
                            st.info(f"ℹ️ Skipped {dataset_stats['skipped_low_confidence']} low-confidence samples")
                        if dataset_stats['skipped_unknown'] > 0:
                            st.info(f"ℹ️ Skipped {dataset_stats['skipped_unknown']} unknown samples")

                        # Also curate into YAMNet audio/spectrogram dataset
                        try:
                            # Apply the UI threshold before curating
                            self.yamnet_curator.config['curation_criteria']['confidence_threshold'] = yamnet_conf_threshold
                            yamnet_stats = self.yamnet_curator.curate_from_analysis(results, run_id)
                            saved_t = yamnet_stats.get('saved', 0)
                            saved_u = yamnet_stats.get('unknown_saved', 0)
                            if saved_t or saved_u:
                                st.info(f"🎵 YAMNet dataset: {saved_t} training + {saved_u} unknown samples saved")
                        except Exception as e:
                            st.warning(f"⚠️ YAMNet curation skipped: {e}")
                    
                    st.success("✅ Analysis complete!")
                    st.session_state['analysis_just_completed'] = True
            # st.rerun() must be OUTSIDE the spinner context — calling it inside
            # keeps the spinner open forever on the next render.
            if st.session_state.pop('analysis_just_completed', False):
                st.rerun()
        
        # Display results if analysis exists
        if analysis_exists:
            try:
                with open(analysis_path, 'r') as f:
                    analysis_data = json.load(f)
            except json.JSONDecodeError as e:
                st.error(f"❌ Analysis file is corrupted: {e}")
                st.info("The file may have been corrupted due to an interrupted save. Try deleting and regenerating the analysis.")
                if st.button("🗑️ Delete Corrupted Analysis", key="delete_corrupted"):
                    self._delete_analysis(run_id)
                    st.rerun()
                return
            
            self._display_summary(analysis_data)
            
            # Action buttons
            st.markdown("---")
            
            if report_path.exists():
                st.success("📊 Interactive Report Generated!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    view_report = st.button("🔍 Open Report (Full Page)", key=f"open_{run_id}", width='stretch', type="primary")
                
                with col2:
                    with open(report_path, 'rb') as f:
                        st.download_button(
                            "📥 Download HTML",
                            f,
                            file_name=report_path.name,
                            mime="text/html",
                            width='stretch'
                        )
                
                with col3:
                    st.text_input(
                        "File Path",
                        value=str(report_path),
                        key=f"path_{run_id}",
                        label_visibility="collapsed"
                    )
                
                # Show report in full page when button clicked
                if view_report:
                    with open(report_path, 'r') as f:
                        html_content = f.read()
                    import streamlit.components.v1 as components
                    
                    st.markdown("---")
                    st.markdown("### 📊 Interactive Report Viewer")
                    st.info("⚡ Fully interactive - rotate 3D plots, zoom timeline, hover for details")
                    
                    # Use full width and large height for better viewing
                    components.html(html_content, width=None, height=2000, scrolling=True)
                    
                    st.markdown("---")
                    if st.button("⬆️ Back to Top", key=f"back_{run_id}"):
                        st.rerun()
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if dataset_path.exists():
                    with open(dataset_path, 'rb') as f:
                        st.download_button(
                            "📥 Download Dataset CSV",
                            f,
                            file_name=dataset_path.name,
                            mime="text/csv",
                            width='stretch'
                        )
            
            with col2:
                with st.expander("📄 View Analysis JSON"):
                    st.json(analysis_data)
        
        # Show recent analyses
        st.markdown("---")
        self._show_recent_analyses()
    
    def _get_analysis_path(self, run_id):
        """Get path to analysis JSON file"""
        return self.analysis_dir / f"{run_id}_analysis.json"
    
    def _get_report_path(self, run_id):
        """Get path to HTML report file"""
        return self.analysis_dir / f"{run_id}_report.html"
    
    def _get_dataset_path(self, run_id):
        """Get path to dataset CSV file"""
        return self.analysis_dir / f"{run_id}_dataset.csv"
    
    def _delete_analysis(self, run_id):
        """Delete all analysis files for a run"""
        for path in [self._get_analysis_path(run_id), 
                     self._get_report_path(run_id), 
                     self._get_dataset_path(run_id)]:
            if path.exists():
                path.unlink()
        st.success(f"Deleted analysis for {run_id}")
    
    def _analyze_run(self, run_data, angle_threshold, save_unmatched,
                     time_pre=None, time_post=None):
        """Analyze a simulation run"""
        try:
            # Get session_live file
            session_live_file = run_data.get('session_live_file')
            if not session_live_file or not os.path.exists(session_live_file):
                st.error(f"Session live file not found: {session_live_file}")
                return None
            
            # Get scene file
            scene_file = run_data.get('scene_file')
            if not scene_file or not os.path.exists(scene_file):
                st.error(f"Scene file not found: {scene_file}")
                return None
            
            # Load scene
            with open(scene_file, 'r') as f:
                scene_data = json.load(f)
            
            # Parse ODAS output
            detections = self._parse_odas_output(session_live_file)
            st.info(f"Parsed {len(detections)} detections from ODAS output")
            
            if not detections:
                st.warning("No detections found in ODAS output")
                return None
            
            # Match detections to sources
            matches, unmatched = self._match_detections_to_sources(
                detections, scene_data, angle_threshold,
                                time_pre=time_pre, time_post=time_post
            )
            
            st.info(f"Matched: {len(matches)}, Unmatched: {len(unmatched)}")
            
            # Calculate statistics
            stats = self._calculate_statistics(matches, unmatched, scene_data)
            
            # Compile results
            results = {
                'run_id': run_data.get('run_id', run_data.get('run_name', 'unknown')),
                'render_id': run_data.get('render_id', 'N/A'),
                'scene_name': run_data.get('scene_name', 'unknown'),
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'angular_threshold': angle_threshold,
                    'save_unmatched': save_unmatched
                },
                'summary': stats['summary'],
                'by_source': stats['by_source'],
                'matches': matches,
                'unmatched': unmatched,
                'scene': scene_data,
                'run_metadata': run_data
            }
            
            return results
            
        except Exception as e:
            st.error(f"Error analyzing run: {e}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    def _parse_odas_output(self, session_live_file):
        """Parse ODAS session_live JSON file"""
        detections = []
        # Base dir for resolving relative spectra_file paths written by ODAS
        session_base_dir = os.path.dirname(os.path.abspath(session_live_file))
        
        with open(session_live_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    time_stamp = data.get('timeStamp', 0)
                    
                    # timeStamp is the cumulative ODAS hop count (each hop = 8ms).
                    # With ROLLING_HOPS=6 the JSON is gated at 48ms so line_num
                    # no longer maps 1:1 with 8ms steps — using line_num would
                    # compress a 33s session into ~5.5s, making Frog/Elephant GT
                    # windows (15-30s) completely unreachable.
                    # Correct conversion: actual_seconds = timeStamp * hop_duration
                    line_timestamp = time_stamp * 0.008  # hop_count × 8ms/hop
                    
                    for src in data.get('src', []):
                        frame_count = src.get('frame_count', 0)
                        
                        detection = {
                            'timestamp': line_timestamp,
                            'frame_count': frame_count,
                            'line_number': line_num,
                            'odas_timestamp': time_stamp,
                            'x': src.get('x', 0),
                            'y': src.get('y', 0),
                            'z': src.get('z', 0),
                            'activity': src.get('activity', 0),
                            # Legacy single-frame bins (backward compat — empty in new firmware)
                            'bins': src.get('bins', []),
                            # Legacy single-class fields (backward compat)
                            'class_id': src.get('class_id', -1),
                            'class_name': src.get('class_name', 'unclassified'),
                            'class_confidence': src.get('class_confidence', 0.0),
                            'class_timestamp': src.get('class_timestamp', 0),
                            # ── Event fields (6-hop rolling mode, min_event_votes gated) ──
                            # Emitted only when ROLLING_HOPS hops are full and
                            # event_votes >= min_event_votes (default 4/6).
                            'event_class_id':        src.get('event_class_id', -1),
                            'event_class_name':      src.get('event_class_name', 'unclassified'),
                            'event_votes':           src.get('event_votes', 0),
                            'event_avg_confidence':  src.get('event_avg_confidence', 0.0),
                            'event_max_confidence':  src.get('event_max_confidence', 0.0),
                            # ── Full ranked candidate list (top-K × N-hop voting) ──
                            # [{class_id, class_name, hop_votes, avg_confidence}, ...] sorted desc.
                            'event_candidates':      src.get('event_candidates', []),
                            # ── Spectra sidecar (sim_mode=1 only) ──
                            # Path to 96×257 float32 .bin file for this event's last hop.
                            # Load with: np.fromfile(path, dtype=np.float32).reshape(96, 257)
                            # Empty string on Pi (sim_mode=0).
                            'spectra_file': self._resolve_spectra_path(
                                src.get('spectra_file', ''), session_base_dir),
                            # ── Full 6-hop Top-K history ──
                            # List of up to 6 dicts: {timestamp, class_ids[5], class_names[5], confidences[5]}
                            'topk_history': src.get('topk_history', []),
                            'track_id':   src.get('id', 0),
                            'track_tag':  src.get('tag', ''),
                            'track_type': src.get('type', '')
                        }
                        detections.append(detection)
                except json.JSONDecodeError:
                    continue
        
        return detections
    
    def _resolve_spectra_path(self, spectra_file, base_dir):
        """Resolve spectra_file path to an absolute path that actually exists.
        
        Old firmware writes relative paths like ./ClassifierLogs/patch_5_1425.bin
        relative to the ODAS build dir (the *parent* of the ClassifierLogs dir
        where the session JSON lives).  New firmware (after the getcwd() fix)
        writes absolute paths directly.
        """
        if not spectra_file:
            return ''
        if os.path.isabs(spectra_file):
            return spectra_file  # New firmware: absolute already
        # Relative path: try parent of session-file dir (= ODAS build dir) first,
        # then the session-file dir itself, then CWD.
        for candidate_base in [os.path.dirname(base_dir), base_dir, os.getcwd()]:
            p = os.path.normpath(os.path.join(candidate_base, spectra_file))
            if os.path.exists(p):
                return p
        return spectra_file  # return original if nothing matched

    def _cartesian_to_spherical(self, x, y, z):
        """Convert Cartesian coordinates to spherical (azimuth, elevation)"""
        r = np.sqrt(x**2 + y**2 + z**2)
        if r == 0:
            return 0, 0
        elevation = np.arcsin(z / r)
        azimuth = np.arctan2(y, x)
        return azimuth, elevation
    
    def _angular_distance(self, az1, el1, az2, el2):
        """Calculate angular distance between two directions in degrees"""
        # Convert to Cartesian unit vectors
        x1 = np.cos(el1) * np.cos(az1)
        y1 = np.cos(el1) * np.sin(az1)
        z1 = np.sin(el1)
        
        x2 = np.cos(el2) * np.cos(az2)
        y2 = np.cos(el2) * np.sin(az2)
        z2 = np.sin(el2)
        
        # Dot product
        dot = x1*x2 + y1*y2 + z1*z2
        dot = np.clip(dot, -1.0, 1.0)  # Handle floating point errors
        
        # Angular distance
        angle_rad = np.arccos(dot)
        angle_deg = np.degrees(angle_rad)
        
        return angle_deg

    def _azimuth_distance(self, az1, az2):
        """Horizontal-plane (azimuth-only) angular distance in degrees.

        Ignores elevation entirely.  For planar microphone arrays the
        elevation estimate is unreliable (all mics are at z=0), but
        azimuth is well-determined.  Using azimuth-only distance prevents
        large elevation errors (~40°) from blocking valid spatial matches.

        Args:
            az1, az2: azimuths in radians (output of _cartesian_to_spherical)
        Returns:
            Angular difference in degrees, in [0, 180].
        """
        diff = abs(az1 - az2)
        # Wrap to [0, π]
        if diff > np.pi:
            diff = 2 * np.pi - diff
        return float(np.degrees(diff))
    
    def _calculate_confidence(self, angular_error):
        """Calculate confidence score based on angular error using cosine similarity
        
        Returns value between 0 and 1, where:
        - 1.0 = perfect match (0° error)
        - 0.0 = orthogonal or worse (90°+ error)
        """
        # Convert angular error to radians
        angle_rad = np.radians(angular_error)
        # Cosine similarity: cos(0°) = 1, cos(90°) = 0
        confidence = max(0.0, np.cos(angle_rad))
        return float(confidence)
    
    def _match_detections_to_sources(self, detections, scene, angle_threshold,
                                     time_pre=None, time_post=None):
        """Match detected peaks to ground truth sources using asymmetric time windows.

        ODAS Kalman filter has two characteristic timing artefacts:
          - Startup delay:   track first appears 1–6 s AFTER the sound starts
          - Persistence tail: track stays alive 2.5–13 s AFTER the sound ends
        Using a single symmetric window (±2.5s) causes temporal_mismatch on most
        samples.  Asymmetric pre/post windows absorb both effects.

        Two-pass approach:
        1. For each source window [start - pre, end + post], match direction‑matched
           detections to that source (first match wins; prevents double-assigning).
        2. Label remaining unmatched detections as 'unknown'.
        """
        sources = scene.get('directional_sources', [])
        pre  = time_pre  if time_pre  is not None else CONFIG['time_window_pre_s']
        post = time_post if time_post is not None else CONFIG['time_window_post_s']
        
        matched_detection_indices = set()
        matches = []
        
        # PASS 1: Match detections to sources within time windows
        for src in sources:
            src_start = src.get('start_time', 0)
            src_end = src.get('end_time', float('inf'))
            src_label = src.get('label', 'unknown')
            
            # Define asymmetric search window:
            # Pre-window catches ODAS early-starts (e.g. Frog detected 3.5s before GT start).
            # Post-window catches Kalman persistence tails (e.g. Wolfhowl tracked 12.75s after GT end).
            window_start = src_start - pre
            window_end   = src_end   + post
            
            # Get source position
            if 'position' in src:
                src_pos = src['position']
            else:
                src_pos = [src.get('x', 0), src.get('y', 0), src.get('z', 0)]
            src_az, src_el = self._cartesian_to_spherical(src_pos[0], src_pos[1], src_pos[2])
            
            # Find all detections in this time window
            for idx, det in enumerate(detections):
                det_time = det['timestamp']
                
                # Skip if outside time window
                if not (window_start <= det_time <= window_end):
                    continue
                
                # Skip if already matched to another source
                if idx in matched_detection_indices:
                    continue
                
                # Calculate angular distance.
                # For planar arrays (use_azimuth_only_matching=True) we compare
                # only horizontal-plane azimuths because elevation estimates are
                # unreliable when all microphones lie in the same horizontal plane.
                det_az, det_el = self._cartesian_to_spherical(det['x'], det['y'], det['z'])
                if CONFIG.get('use_azimuth_only_matching', True):
                    angle_diff = self._azimuth_distance(det_az, src_az)
                else:
                    angle_diff = self._angular_distance(det_az, det_el, src_az, src_el)
                
                # Match if within angular threshold
                if angle_diff <= angle_threshold:
                    confidence = self._calculate_confidence(angle_diff)
                    
                    matches.append({
                        'detection': det,
                        'source': src,
                        'angular_error': angle_diff,
                        'confidence': confidence,
                        'label': src_label,
                        'match_type': 'ground_truth'  # Indicate this is ground truth matching
                    })
                    
                    matched_detection_indices.add(idx)
        
        # PASS 2: Label unmatched detections as 'unknown'
        unmatched = []
        for idx, det in enumerate(detections):
            if idx not in matched_detection_indices:
                matches.append({
                    'detection': det,
                    'source': None,
                    'angular_error': None,
                    'confidence': 0.0,
                    'label': 'unknown',
                    'match_type': 'unmatched'
                })
                unmatched.append(det)
        
        return matches, unmatched
    
    def _derive_label(self, det, strategy='ODAS event voting'):
        """
        Derive (class_id, class_name, confidence, votes) from a detection dict
        using the chosen label strategy.  Returns a dict with keys:
          class_id, class_name, confidence, votes, strategy_used,
          top_k_candidates   ← full ranked list from event_candidates[]
          ambiguous          ← True when #2 candidate has same hop-votes as #1
        so callers are decoupled from firmware field names.
        """
        # ── Strategy 1: firmware top-K × N-hop vote winner ──────────────────
        if strategy == 'ODAS event voting':
            ev_id    = det.get('event_class_id', -1)
            ev_name  = det.get('event_class_name', 'unclassified')
            ev_conf  = det.get('event_max_confidence') or det.get('event_avg_confidence', 0.0)
            ev_votes = det.get('event_votes', 0)

            # Full ranked candidate list from the 6-hop top-K pool.
            # Each entry: {class_id, class_name, hop_votes, avg_confidence}
            candidates = det.get('event_candidates', [])

            # Ambiguity: #2 candidate has the same hop-vote count as the winner.
            # Ambiguous detections are less trustworthy as training labels.
            ambiguous = (
                len(candidates) >= 2 and
                candidates[0].get('hop_votes', 0) == candidates[1].get('hop_votes', 0)
            )

            if ev_id != -1 and ev_name not in ('unclassified', ''):
                return dict(class_id=ev_id, class_name=ev_name,
                            confidence=ev_conf, votes=ev_votes,
                            top_k_candidates=candidates,
                            ambiguous=ambiguous,
                            strategy_used='odas_voting')
            # Fallback to legacy single-hop fields
            return dict(class_id=det.get('class_id', -1),
                        class_name=det.get('class_name', 'unclassified'),
                        confidence=det.get('class_confidence', 0.0),
                        votes=0, top_k_candidates=[], ambiguous=False,
                        strategy_used='odas_legacy')

        # ── Strategy 2: re-classify .bin sidecar with Python YAMNet ─────────
        if strategy == 'Python YAMNet (re-classify .bin)':
            spectra_file = det.get('spectra_file', '')
            if spectra_file and os.path.exists(spectra_file):
                try:
                    import numpy as np
                    from yamnet_helper.yamnet_spectrum_classifier import YAMNetSpectrumClassifier
                    if not hasattr(self, '_py_yamnet'):
                        MODEL = '/home/azureuser/z_odas_newbeamform/models/yamnet_core.tflite'
                        CSV   = '/home/azureuser/z_odas_newbeamform/models/yamnet_class_map.csv'
                        self._py_yamnet = YAMNetSpectrumClassifier(MODEL, CSV)
                    patch = np.fromfile(spectra_file, dtype=np.float32).reshape(96, 257)
                    cid, cname, conf = self._py_yamnet.classify_patch(patch)
                    return dict(class_id=cid, class_name=cname, confidence=float(conf),
                                votes=1, strategy_used='python_yamnet')
                except Exception:
                    pass
            return dict(class_id=-1, class_name='unclassified', confidence=0.0,
                        votes=0, strategy_used='python_yamnet_missing_bin')

        # ── Strategy 3: ground truth only ───────────────────────────────────
        if strategy == 'Ground truth only':
            # Caller must have set match['label'] from scene config already.
            # Return a sentinel so _apply_yamnet_classifications skips the
            # YAMNet-vs-GT comparison and just trusts the GT label.
            return dict(class_id=-2, class_name='__ground_truth__', confidence=1.0,
                        votes=0, strategy_used='ground_truth')

        # ── Strategy 4: fine-tuned model ─────────────────────────────────────
        if strategy == 'Fine-tuned model':
            spectra_file = det.get('spectra_file', '')
            if spectra_file and os.path.exists(spectra_file):
                try:
                    import numpy as np, torch
                    from model_trainer import load_finetuned_model  # if available
                    if not hasattr(self, '_ft_model'):
                        model_path = str(Path(self.base_output_dir) / 'models' / 'active_model.pth')
                        self._ft_model = load_finetuned_model(model_path)
                    patch = np.fromfile(spectra_file, dtype=np.float32).reshape(96, 257)
                    cid, cname, conf = self._ft_model.predict_patch(patch)
                    return dict(class_id=cid, class_name=cname, confidence=float(conf),
                                votes=1, strategy_used='finetuned_model')
                except Exception:
                    pass
            return dict(class_id=-1, class_name='unclassified', confidence=0.0,
                        votes=0, strategy_used='finetuned_missing')

        # Fallback
        return dict(class_id=-1, class_name='unclassified', confidence=0.0,
                    votes=0, strategy_used='unknown')

    def _apply_yamnet_classifications(self, results, label_strategy='ODAS event voting'):
        """
        Derive labels for all detections using the chosen label_strategy.
        Compares against ground truth, marks samples needing fine-tuning.
        """
        strategy_labels = {
            'ODAS event voting':               '🎯 ODAS top-K voting',
            'Python YAMNet (re-classify .bin)':'🐍 Python YAMNet re-classify',
            'Ground truth only':               '📍 Ground truth labels',
            'Fine-tuned model':                '🧠 Fine-tuned model',
        }
        st.info(f"{strategy_labels.get(label_strategy, label_strategy)} — labeling detections...")
        
        matches_needing_training = []
        yamnet_predicted = 0
        yamnet_correct = 0
        yamnet_incorrect = 0
        unclassified = 0
        
        for match in results['matches']:
            det = match['detection']
            lbl = self._derive_label(det, strategy=label_strategy)
            yamnet_class = lbl['class_name']
            yamnet_conf  = lbl['confidence']
            yamnet_id    = lbl['class_id']
            ev_votes     = lbl['votes']
            match['label_strategy'] = lbl['strategy_used']

            # Store prediction (both as yamnet_* and model_* for compatibility)
            match['yamnet_class'] = yamnet_class
            match['yamnet_confidence'] = yamnet_conf
            match['yamnet_id'] = yamnet_id
            match['yamnet_votes'] = ev_votes
            match['model_prediction'] = yamnet_class
            match['model_confidence'] = yamnet_conf
            # Expose full candidate list to the visualizer
            match['event_candidates'] = det.get('event_candidates', [])
            # Top-K ambiguity flag from _derive_label
            match['top_k_candidates'] = lbl.get('top_k_candidates', [])
            match['ambiguous'] = lbl.get('ambiguous', False)

            # Ground-truth-only strategy: label IS the ground truth — no YAMNet
            # comparison needed; mark as correct if spatially matched.
            if yamnet_id == -2:  # sentinel from _derive_label ground_truth strategy
                if match['match_type'] == 'ground_truth':
                    yamnet_correct += 1
                    match['yamnet_class'] = match.get('label', 'unknown')
                    match['yamnet_match'] = True
                    match['needs_training'] = False
                    match['model_prediction'] = match.get('label', 'unknown')
                    yamnet_predicted += 1
                else:
                    unclassified += 1
                    match['needs_training'] = True
                    match['training_reason'] = 'no_ground_truth'
                    matches_needing_training.append(match)
                continue

            if yamnet_class == 'unclassified' or yamnet_id == -1:
                unclassified += 1
                match['needs_training'] = True
                match['training_reason'] = 'unclassified'
                matches_needing_training.append(match)
                continue
            
            yamnet_predicted += 1
            
            # Compare with ground truth
            if match['match_type'] == 'ground_truth':
                gt_label = match['label']
                yamnet_lower = yamnet_class.lower()
                gt_lower = gt_label.lower()
                
                if yamnet_lower == gt_lower or yamnet_lower in gt_lower or gt_lower in yamnet_lower:
                    yamnet_correct += 1
                    match['yamnet_match'] = True
                    match['needs_training'] = False
                else:
                    yamnet_incorrect += 1
                    match['yamnet_match'] = False
                    match['needs_training'] = True
                    match['training_reason'] = f'mismatch (pred: {yamnet_class}, gt: {gt_label})'
                    matches_needing_training.append(match)
                
                if yamnet_conf < 0.5:
                    match['needs_training'] = True
                    match['training_reason'] = match.get('training_reason', '') + ' low_confidence'
                    if match not in matches_needing_training:
                        matches_needing_training.append(match)
            else:
                match['label'] = yamnet_class
                match['confidence'] = yamnet_conf
                if yamnet_conf < 0.5:
                    match['needs_training'] = True
                    match['training_reason'] = 'low_confidence'
                    matches_needing_training.append(match)
        
        # Update summary stats
        results['yamnet_stats'] = {
            'total_detections': len(results['matches']),
            'yamnet_classified': yamnet_predicted,
            'unclassified': unclassified,
            'correct': yamnet_correct,
            'incorrect': yamnet_incorrect,
            'needs_training': len(matches_needing_training),
            'accuracy': yamnet_correct / max(yamnet_predicted, 1)
        }
        
        st.info(f"✅ YAMNet: {yamnet_predicted} classified | {yamnet_correct} correct | {yamnet_incorrect} incorrect | {len(matches_needing_training)} need training")
        
        return results
    
    def _apply_model_predictions_DEPRECATED(self):
        """
        Apply model predictions to all detections.
        
        Strategy:
        1. For each detection, get model prediction and confidence
        2. If ground truth match exists with high confidence: keep as-is (already correct)
        3. If ground truth match exists but low confidence: update with model prediction
        4. If unknown: use model prediction
        5. Only add to new training data if:
           - Model prediction disagrees with ground truth (potential mislabel)
           - Model confidence is low (uncertain prediction)
        """
        st.info("🤖 Applying model predictions...")
        
        # First pass: determine bin count and valid indices
        bin_count = None
        valid_indices = []
        
        for i, match in enumerate(results['matches']):
            bins = match['detection'].get('bins', [])
            if len(bins) > 0:
                if bin_count is None:
                    bin_count = len(bins)
                if len(bins) == bin_count:
                    valid_indices.append(i)
        
        if not valid_indices:
            return results
        
        st.info(f"📊 Predicting on {len(valid_indices)} detections with {bin_count} bins each")
        
        # Collect only valid detections (memory efficient)
        all_detections = [results['matches'][i]['detection']['bins'] for i in valid_indices]
        
        # Predict with model (with batching to prevent OOM)
        X = np.array(all_detections, dtype=np.float32)
        predicted_labels, model_confidences = self.model_trainer.predict(X, batch_size=256)
        
        # Free memory
        del X, all_detections
        import gc
        gc.collect()
        
        # Update matches with model predictions
        matches_needing_training = []
        
        for pred_idx, match_idx in enumerate(valid_indices):
            match = results['matches'][match_idx]
            
            model_label = predicted_labels[pred_idx]
            model_conf = model_confidences[pred_idx]
            
            # Store model prediction
            match['model_prediction'] = model_label
            match['model_confidence'] = model_conf
            
            # Decision logic
            if match['match_type'] == 'ground_truth':
                # Has ground truth match
                gt_label = match['label']
                gt_confidence = match['confidence']
                
                if gt_confidence >= 0.85:
                    # High confidence ground truth - trust it
                    if model_label != gt_label:
                        # Model disagrees - potential for model improvement
                        match['needs_training'] = True
                        match['training_reason'] = f'model_mismatch (model: {model_label}, gt: {gt_label})'
                        matches_needing_training.append(match)
                    else:
                        match['needs_training'] = False
                else:
                    # Low confidence ground truth - use model if confident
                    if model_conf >= 0.85:
                        match['label'] = model_label
                        match['confidence'] = model_conf
                        match['match_type'] = 'model_prediction'
                        match['needs_training'] = False
                    else:
                        # Both uncertain - needs training
                        match['needs_training'] = True
                        match['training_reason'] = 'both_low_confidence'
                        matches_needing_training.append(match)
            
            else:
                # Unknown detection - use model prediction
                if model_conf >= 0.85:
                    match['label'] = model_label
                    match['confidence'] = model_conf
                    match['match_type'] = 'model_prediction'
                    match['needs_training'] = False
                else:
                    # Low confidence prediction - needs training
                    match['label'] = model_label
                    match['confidence'] = model_conf
                    match['match_type'] = 'model_prediction_uncertain'
                    match['needs_training'] = True
                    match['training_reason'] = 'low_model_confidence'
                    matches_needing_training.append(match)
        
        # Update summary stats
        model_predicted = len([m for m in results['matches'] if m.get('match_type') == 'model_prediction'])
        needs_training_count = len(matches_needing_training)
        
        results['model_stats'] = {
            'total_predictions': len(valid_indices),
            'model_predicted': model_predicted,
            'needs_training': needs_training_count,
            'avg_model_confidence': float(np.mean(model_confidences))
        }
        
        st.info(f"✅ Model predictions: {model_predicted} samples | {needs_training_count} need training")
        
        # Clean up memory
        del predicted_labels, model_confidences
        import gc
        gc.collect()
        
        return results
    
    def _calculate_statistics(self, matches, unmatched, scene):
        """Calculate detection statistics"""
        # Separate matched from unknown
        matched_to_sources = [m for m in matches if m['label'] != 'unknown']
        unknown_count = len([m for m in matches if m['label'] == 'unknown'])
        
        total_detections = len(matches)
        match_rate = len(matched_to_sources) / total_detections if total_detections > 0 else 0
        
        # Calculate per-source stats
        by_source = {}
        for match in matched_to_sources:
            label = match['label']
            if label not in by_source:
                by_source[label] = {
                    'detections': 0,
                    'errors': [],
                    'confidences': []
                }
            by_source[label]['detections'] += 1
            by_source[label]['errors'].append(match['angular_error'])
            by_source[label]['confidences'].append(match['confidence'])
        
        # Calculate averages
        for label in by_source:
            errors = by_source[label]['errors']
            confidences = by_source[label]['confidences']
            by_source[label]['avg_error'] = np.mean(errors)
            by_source[label]['min_error'] = np.min(errors)
            by_source[label]['max_error'] = np.max(errors)
            by_source[label]['std_error'] = np.std(errors)
            by_source[label]['avg_confidence'] = np.mean(confidences)
            by_source[label]['min_confidence'] = np.min(confidences)
            by_source[label]['max_confidence'] = np.max(confidences)
            del by_source[label]['errors']  # Remove raw errors from summary
            del by_source[label]['confidences']  # Remove raw confidences from summary
        
        # Calculate time span
        all_times = [m['detection']['timestamp'] for m in matches]
        time_span = max(all_times) - min(all_times) if all_times else 0
        
        # Overall stats for matched sources only
        all_errors = [m['angular_error'] for m in matched_to_sources]
        all_confidences = [m['confidence'] for m in matched_to_sources]
        avg_error = np.mean(all_errors) if all_errors else 0
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
        
        summary = {
            'total_detections': total_detections,
            'matched': len(matched_to_sources),
            'unmatched': unknown_count,
            'match_rate': match_rate,
            'avg_angular_error': float(avg_error),
            'avg_confidence': float(avg_confidence),
            'time_span_seconds': float(time_span),
            'unique_sources': len(by_source)
        }
        
        return {
            'summary': summary,
            'by_source': by_source
        }
    
    def _convert_to_native(self, obj):
        """Recursively convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_native(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native(item) for item in obj]
        else:
            return obj
    
    def _save_analysis(self, run_id, results, angle_threshold):
        """Save analysis results to JSON"""
        analysis_path = self._get_analysis_path(run_id)
        
        # Create a saveable version (without large bin arrays)
        save_data = {
            'analysis_id': run_id,
            'render_id': results['render_id'],
            'run_id': results['run_id'],
            'scene_name': results['scene_name'],
            'created_at': results['timestamp'],
            'config': results['config'],
            'summary': self._convert_to_native(results['summary']),
            'by_source': self._convert_to_native(results['by_source']),
            'model_stats': self._convert_to_native(results.get('model_stats', {})),
            'matches': [
                {
                    'timestamp': float(m['detection']['timestamp']),
                    'frame_count': int(m['detection']['frame_count']),
                    'position': [float(m['detection']['x']), float(m['detection']['y']), float(m['detection']['z'])],
                    'activity': float(m['detection']['activity']),
                    'source_label': str(m['label']),
                    'source_position': [float(x) for x in m['source'].get('position', [m['source'].get('x', 0), m['source'].get('y', 0), m['source'].get('z', 0)])] if m['source'] else None,
                    'angular_error': float(m['angular_error']) if m['angular_error'] is not None else None,
                    'confidence': float(m['confidence']),
                    'bins_count': (1 if m['detection'].get('spectra_file') and os.path.exists(m['detection'].get('spectra_file','')) else len(m['detection'].get('bins', []))),
                    # Add model prediction fields if available
                    'model_prediction': str(m['model_prediction']) if 'model_prediction' in m else None,
                    'model_confidence': float(m['model_confidence']) if 'model_confidence' in m else None,
                    'match_type': str(m.get('match_type', 'ground_truth'))
                }
                for m in results['matches']
            ],
            'unmatched': [
                {
                    'timestamp': float(u['timestamp']),
                    'frame_count': int(u['frame_count']),
                    'position': [float(u['x']), float(u['y']), float(u['z'])],
                    'activity': float(u['activity']),
                    'bins_count': (1 if u.get('spectra_file') and os.path.exists(u.get('spectra_file','')) else len(u.get('bins', [])))
                }
                for u in results['unmatched']
            ]
        }
        
        # Write to temporary file first, then atomically rename
        # This prevents corruption if the process is killed during write
        temp_path = analysis_path.with_suffix('.json.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8', errors='replace') as f:
                # ensure_ascii=True converts any surrogates/high codepoints to
                # \uXXXX escapes so the JSON is always valid UTF-8
                json.dump(save_data, f, indent=2, ensure_ascii=True)
            # Atomic rename - if this succeeds, the file is complete
            temp_path.rename(analysis_path)
        except Exception as e:
            # Clean up temp file if something went wrong
            if temp_path.exists():
                temp_path.unlink()
            raise e
        
        st.success(f"💾 Saved analysis to {analysis_path.name}")
    
    def _create_dataset(self, results, run_id, save_unmatched):
        """Create training dataset CSV with confidence scores"""
        dataset_path = self._get_dataset_path(run_id)
        
        rows = []
        
        # Add all matches (includes both matched sources and unknown)
        for match in results['matches']:
            bins = match['detection']['bins']
            if len(bins) == 1024:
                # Include confidence and angular error in dataset
                row = bins + [
                    match['label'],
                    match['confidence'],
                    match['angular_error'] if match['angular_error'] is not None else -1.0,
                    match['detection']['timestamp']
                ]
                rows.append(row)
        
        # Note: save_unmatched parameter is now ignored since unknown are already in matches
        # This preserves backward compatibility
        
        # Create DataFrame
        columns = [f'bin_{i}' for i in range(1024)] + ['label', 'confidence', 'angular_error', 'timestamp']
        df = pd.DataFrame(rows, columns=columns)
        
        # Save
        df.to_csv(dataset_path, index=False)
        st.success(f"💾 Saved dataset with {len(df)} samples to {dataset_path.name}")
        
        return dataset_path
    
    def _add_audio_waveform_section(self, html_parts, results):
        """Add audio waveform visualization and player"""
        import base64
        import struct
        import wave
        import io
        
        # Get audio file path from run metadata
        run_metadata = results.get('run_metadata', {})
        raw_audio_file = run_metadata.get('raw_audio_file')
        scene_metadata = run_metadata.get('scene_metadata', {})
        
        if not raw_audio_file or not os.path.exists(raw_audio_file):
            return
        
        # Get audio parameters
        sample_rate = scene_metadata.get('sample_rate', 16000)
        n_channels = scene_metadata.get('n_channels', 6)
        duration = scene_metadata.get('duration', 10.0)
        
        try:
            # Read raw audio file (S16_LE format - signed 16-bit little-endian)
            with open(raw_audio_file, 'rb') as f:
                raw_data = f.read()
            
            # Calculate total samples and samples per channel
            bytes_per_sample = 2  # 16-bit = 2 bytes
            total_samples = len(raw_data) // bytes_per_sample
            samples_per_channel = total_samples // n_channels
            
            # Unpack all samples
            samples = struct.unpack(f'<{total_samples}h', raw_data)
            
            # Extract channel 3 (index 2) - can be made configurable
            channel_to_plot = 2  # Channel 3 (0-indexed)
            channel_samples = [samples[i] for i in range(channel_to_plot, total_samples, n_channels)]
            
            # Normalize to -1 to 1 range
            max_val = 32768.0
            normalized_samples = [s / max_val for s in channel_samples]
            
            # Create time axis
            time_axis = [i / sample_rate for i in range(len(normalized_samples))]
            
            # Downsample for plotting if too many samples
            max_plot_points = 10000
            if len(time_axis) > max_plot_points:
                step = len(time_axis) // max_plot_points
                time_axis = time_axis[::step]
                normalized_samples = normalized_samples[::step]
            
            # Create WAV file in memory for audio player
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                # Convert back to 16-bit integers
                wav_data = struct.pack(f'<{len(channel_samples)}h', *channel_samples)
                wav_file.writeframes(wav_data)
            
            # Encode WAV to base64
            wav_buffer.seek(0)
            wav_base64 = base64.b64encode(wav_buffer.read()).decode('utf-8')
            
            # Add HTML section
            html_parts.append(f"""
    <div class="section">
        <h2>🎵 Audio Waveform (Channel {channel_to_plot + 1})</h2>
        <p style="color: #666; margin-bottom: 15px;">
            Waveform of the recorded audio aligned with the timeline above. Use the audio player to listen while viewing the timeline.
        </p>
        <div style="margin-bottom: 20px;">
            <audio controls style="width: 100%;">
                <source src="data:audio/wav;base64,{wav_base64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
        </div>
        <div id="waveform"></div>
    </div>
""")
            
            # Add waveform plot
            html_parts.append(f"""
    <script>
        var waveformData = [{{
            x: {json.dumps(time_axis)},
            y: {json.dumps(normalized_samples)},
            type: 'scatter',
            mode: 'lines',
            line: {{ color: '#667eea', width: 0.5 }},
            name: 'Channel {channel_to_plot + 1}',
            hovertemplate: 'Time: %{{x:.3f}}s<br>Amplitude: %{{y:.3f}}<extra></extra>'
        }}];
        var waveformLayout = {{
            title: 'Audio Waveform',
            xaxis: {{ 
                title: 'Time (seconds)',
                showgrid: true,
                gridcolor: '#e0e0e0'
            }},
            yaxis: {{ 
                title: 'Amplitude',
                range: [-1, 1],
                showgrid: true,
                gridcolor: '#e0e0e0'
            }},
            hovermode: 'closest',
            height: 300,
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            margin: {{ l: 60, r: 50, t: 60, b: 60 }}
        }};
        Plotly.newPlot('waveform', waveformData, waveformLayout);
    </script>
""")
        
        except Exception as e:
            html_parts.append(f"""
    <div class="section">
        <h2>🎵 Audio Waveform</h2>
        <p style="color: red;">Error loading audio: {str(e)}</p>
    </div>
""")
    
    def _generate_html_report(self, run_id, results):
        """Generate interactive HTML report with Plotly"""
        report_path = self._get_report_path(run_id)
        
        # Create the report
        html_content = self._create_plotly_report(results)
        
        # Strip surrogate characters that json-c may embed in malformed UTF-8
        # class names — Python's strict UTF-8 codec rejects them on write.
        html_content = html_content.encode('utf-8', errors='surrogatepass') \
                                   .decode('utf-8', errors='replace')
        
        # Save to file
        with open(report_path, 'w', encoding='utf-8', errors='replace') as f:
            f.write(html_content)
        
        st.success(f"📊 Generated interactive report: {report_path.name}")
    
    def _create_plotly_report(self, results):
        """Create comprehensive Plotly-based HTML report"""
        
        # Extract data
        matches = results['matches']
        unmatched = results['unmatched']
        summary = results['summary']
        by_source = results['by_source']
        scene = results['scene']
        sources = scene.get('directional_sources', [])
        
        # Create HTML structure
        html_parts = []
        
        # Header
        html_parts.append(f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ODAS Analysis Report - {results['run_id']}</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{ 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{ margin: 0 0 10px 0; }}
        .header .info {{ opacity: 0.9; font-size: 14px; }}
        .section {{
            background: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{ font-size: 32px; font-weight: bold; margin: 10px 0; }}
        .stat-label {{ font-size: 14px; opacity: 0.9; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #667eea;
            color: white;
        }}
        tr:hover {{ background: #f5f5f5; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 ODAS Detection Analysis Report</h1>
        <div class="info">
            <strong>Run ID:</strong> {results['run_id']}<br>
            <strong>Scene:</strong> {results['scene_name']}<br>
            <strong>Render ID:</strong> {results['render_id']}<br>
            <strong>Analysis Time:</strong> {results['timestamp']}<br>
            <strong>Angular Threshold:</strong> {results['config']['angular_threshold']}°
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Total Detections</div>
                <div class="stat-value">{summary['total_detections']}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Match Rate</div>
                <div class="stat-value">{summary['match_rate']*100:.1f}%</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Angular Error</div>
                <div class="stat-value">{summary['avg_angular_error']:.2f}°</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">{summary.get('avg_confidence', 0):.3f}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Time Span</div>
                <div class="stat-value">{summary['time_span_seconds']:.1f}s</div>
            </div>
        </div>
    </div>
""")
        
        # Model prediction statistics if available
        if 'model_stats' in results and results['model_stats']:
            model_stats = results['model_stats']
            html_parts.append(f"""
    <div class="section">
        <h2>🤖 Model Prediction Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <div class="stat-label">Total Predictions</div>
                <div class="stat-value">{model_stats['total_predictions']}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <div class="stat-label">Model Predicted</div>
                <div class="stat-value">{model_stats['model_predicted']}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);">
                <div class="stat-label">Needs Training</div>
                <div class="stat-value">{model_stats['needs_training']}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);">
                <div class="stat-label">Avg Model Confidence</div>
                <div class="stat-value">{model_stats['avg_model_confidence']:.3f}</div>
            </div>
        </div>
        <p style="color: #666; font-size: 14px; margin-top: 10px;">
            <strong>Model Predicted:</strong> High-confidence predictions used directly<br>
            <strong>Needs Training:</strong> Mismatches or low-confidence predictions flagged for retraining
        </p>
    </div>
""")
        
        # Per-source statistics table
        if by_source:
            html_parts.append("""
    <div class="section">
        <h2>📈 Per-Source Statistics</h2>
        <table>
            <tr>
                <th>Source Label</th>
                <th>Detections</th>
                <th>Avg Error (°)</th>
                <th>Min Error (°)</th>
                <th>Max Error (°)</th>
                <th>Std Dev (°)</th>
                <th>Avg Confidence</th>
                <th>Min Confidence</th>
                <th>Max Confidence</th>
            </tr>
""")
            for label, stats in by_source.items():
                html_parts.append(f"""
            <tr>
                <td><strong>{label}</strong></td>
                <td>{stats['detections']}</td>
                <td>{stats['avg_error']:.2f}</td>
                <td>{stats['min_error']:.2f}</td>
                <td>{stats['max_error']:.2f}</td>
                <td>{stats['std_error']:.2f}</td>
                <td>{stats.get('avg_confidence', 0):.3f}</td>
                <td>{stats.get('min_confidence', 0):.3f}</td>
                <td>{stats.get('max_confidence', 0):.3f}</td>
            </tr>
""")
            html_parts.append("        </table>\n    </div>\n")
        
        # YAMNet Classification Statistics
        # Use yamnet_class/yamnet_confidence from match dict — set by
        # _apply_yamnet_classifications(), which already prefers event_* fields.
        classified_matches = [m for m in matches
                              if m.get('yamnet_class', 'unclassified') not in ('unclassified', '', None)
                              and m.get('yamnet_id', -1) != -1]
        if classified_matches:
            # Calculate classification statistics
            class_counts = {}
            class_confidences = {}
            class_votes    = {}   # event_votes per class (0 for legacy firmware)
            for m in classified_matches:
                cname = m.get('yamnet_class', 'unknown')
                votes = m.get('yamnet_votes', 0)
                class_counts[cname] = class_counts.get(cname, 0) + 1
                if cname not in class_confidences:
                    class_confidences[cname] = []
                    class_votes[cname]       = []
                class_confidences[cname].append(m.get('yamnet_confidence', 0))
                class_votes[cname].append(votes)

            avg_class_conf = np.mean([m.get('yamnet_confidence', 0) for m in classified_matches])
            has_votes = any(m.get('yamnet_votes', 0) > 0 for m in classified_matches)

            mode_label = '6-hop Rolling Mode' if has_votes else 'Single-hop'
            html_parts.append(f"""
    <div class="section">
        <h2>🎯 YAMNet Classification Statistics <small style="font-size:13px;color:#888;">({mode_label})</small></h2>
        <div class="stats-grid">
            <div class="stat-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">  
                <div class="stat-label">Classified Events</div>
                <div class="stat-value">{len(classified_matches)}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">  
                <div class="stat-label">Unique Classes</div>
                <div class="stat-value">{len(class_counts)}</div>
            </div>
            <div class="stat-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">  
                <div class="stat-label">Avg Confidence</div>
                <div class="stat-value">{avg_class_conf:.3f}</div>
            </div>
            {f'<div class="stat-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">' if has_votes else ''}
            {f'<div class="stat-label">Avg Votes / Event</div><div class="stat-value">{np.mean([m.get("yamnet_votes",0) for m in classified_matches]):.1f} / 6</div></div>' if has_votes else ''}
        </div>
        
        <h3>Classification Distribution</h3>
        <table>
            <tr>
                <th>Class Name</th>
                <th>Events</th>
                <th>%</th>
                <th>Avg Conf</th>
                <th>Min Conf</th>
                <th>Max Conf</th>
                {'<th>Avg Votes</th>' if has_votes else ''}
            </tr>
""")
            for cname, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(classified_matches)) * 100
                avg_conf = np.mean(class_confidences[cname])
                min_conf = np.min(class_confidences[cname])
                max_conf = np.max(class_confidences[cname])
                avg_v    = np.mean(class_votes[cname]) if has_votes else 0
                votes_cell = f'<td>{avg_v:.1f} / 6</td>' if has_votes else ''
                html_parts.append(f"""
            <tr>
                <td><strong>{cname}</strong></td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
                <td>{avg_conf:.3f}</td>
                <td>{min_conf:.3f}</td>
                <td>{max_conf:.3f}</td>
                {votes_cell}
            </tr>
""")
            html_parts.append("""
        </table>
    </div>
""")
        
        # Time-based comparison table with slider
        html_parts.append("""
    <div class="section">
        <h2>🔍 Detection Comparison by Time</h2>
        <p>Use the slider to select a timestamp and see ground truth sources vs detected peaks</p>
        <div style="margin: 20px 0;">
            <label for="timeSlider" style="font-weight: bold;">Time (seconds): <span id="timeValue">0.00</span></label><br>
            <input type="range" id="timeSlider" min="0" max="10" step="0.1" value="0" style="width: 100%; margin: 10px 0;">
        </div>
        <div id="comparisonTable"></div>
    </div>
""")
        
        # 3D Spatial views
        html_parts.append('    <div class="section">\n        <h2>🌐 3D Spatial Distribution</h2>\n        <div id="spatial3d"></div>\n    </div>\n')
        
        # Error distribution
        html_parts.append('    <div class="section">\n        <h2>📉 Angular Error Distribution</h2>\n        <div id="error_dist"></div>\n    </div>\n')
        
        # JavaScript for plots
        html_parts.append("\n    <script>\n")
        
        # 3D Spatial plot
        spatial_traces = []
        
        # Ground truth sources - handle both position array and x/y/z fields
        # Normalize to unit vectors for comparison with ODAS detections
        src_x = []
        src_y = []
        src_z = []
        src_labels = [s['label'] for s in sources]
        
        for s in sources:
            if 'position' in s:
                pos = s['position']
            else:
                pos = [s.get('x', 0), s.get('y', 0), s.get('z', 0)]
            
            # Normalize to unit vector
            magnitude = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
            if magnitude > 0:
                src_x.append(pos[0] / magnitude)
                src_y.append(pos[1] / magnitude)
                src_z.append(pos[2] / magnitude)
            else:
                src_x.append(0)
                src_y.append(0)
                src_z.append(0)
        
        spatial_traces.append({
            'x': src_x,
            'y': src_y,
            'z': src_z,
            'mode': 'markers+text',
            'name': 'Ground Truth',
            'marker': {'size': 12, 'color': 'gold', 'symbol': 'diamond'},
            'text': src_labels,
            'textposition': 'top center',
            'hovertemplate': '<b>%{text}</b><br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
        })
        
        # Matched detections (colored by source)
        # Group matches by label
        labels_in_matches = set(m['label'] for m in matches)
        for label in labels_in_matches:
            matched_for_label = [m for m in matches if m['label'] == label]
            det_x = [m['detection']['x'] for m in matched_for_label]
            det_y = [m['detection']['y'] for m in matched_for_label]
            det_z = [m['detection']['z'] for m in matched_for_label]
            
            # Different styling for 'unknown'
            if label == 'unknown':
                spatial_traces.append({
                    'x': det_x,
                    'y': det_y,
                    'z': det_z,
                    'mode': 'markers',
                    'name': 'Unknown',
                    'marker': {'size': 4, 'color': 'red', 'symbol': 'x', 'opacity': 0.6},
                    'hovertemplate': '<b>Unknown</b><br>(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                })
            else:
                errors = [m['angular_error'] for m in matched_for_label]
                confidences = [m['confidence'] for m in matched_for_label]
                spatial_traces.append({
                    'x': det_x,
                    'y': det_y,
                    'z': det_z,
                    'mode': 'markers',
                    'name': f'{label} (det)',
                    'marker': {'size': 4, 'opacity': 0.6},
                    'text': [f"Error: {e:.2f}°<br>Conf: {c:.3f}" if e is not None else f"Conf: {c:.3f}" for e, c in zip(errors, confidences)],
                    'hovertemplate': f'<b>{label}</b><br>(%{{x:.2f}}, %{{y:.2f}}, %{{z:.2f}})<br>%{{text}}<extra></extra>'
                })
        
        # Remove old unmatched block since they're now in matches as 'unknown'
        
        # Add mic array at origin
        spatial_traces.append({
            'x': [0],
            'y': [0],
            'z': [0],
            'mode': 'markers',
            'name': 'Mic Array',
            'marker': {'size': 10, 'color': 'black', 'symbol': 'square'},
            'hovertemplate': '<b>Mic Array</b><br>(0, 0, 0)<extra></extra>'
        })
        
        html_parts.append(f"""
        var spatialData = {json.dumps(spatial_traces)};
        var spatialLayout = {{
            title: '3D Spatial Distribution',
            scene: {{
                xaxis: {{ title: 'X (m)' }},
                yaxis: {{ title: 'Y (m)' }},
                zaxis: {{ title: 'Z (m)' }},
                aspectmode: 'cube'
            }},
            height: 700
        }};
        Plotly.newPlot('spatial3d', spatialData, spatialLayout);
""")
        
        # Error distribution histogram (only for matched sources, not unknown)
        matched_to_sources = [m for m in matches if m['label'] != 'unknown']
        if matched_to_sources:
            all_errors = [m['angular_error'] for m in matched_to_sources]
            html_parts.append(f"""
        var errorData = [{{
            x: {json.dumps(all_errors)},
            type: 'histogram',
            name: 'Angular Errors',
            marker: {{ color: '#667eea' }},
            nbinsx: 30
        }}];
        var errorLayout = {{
            title: 'Angular Error Distribution (Matched Sources Only)',
            xaxis: {{ title: 'Angular Error (degrees)' }},
            yaxis: {{ title: 'Frequency' }},
            height: 400
        }};
        Plotly.newPlot('error_dist', errorData, errorLayout);
""")
        
        # Add time slider logic for comparison table
        # Get scene duration
        scene_duration = scene.get('duration', 10.0)
        
        # Create data structure for comparison table - cover full time range
        comparison_data = {}
        
        # Create entries for every 0.1s in the full duration
        for i in range(int(scene_duration * 10) + 1):
            time_key = round(i * 0.1, 1)
            comparison_data[time_key] = {'sources': [], 'detections': []}
        
        # Add ground truth sources active at each time
        for src in sources:
            src_start = src.get('start_time', 0)
            src_end = src.get('end_time', float('inf'))
            src_label = src.get('label', 'unknown')
            
            if 'position' in src:
                src_pos = src['position']
            else:
                src_pos = [src.get('x', 0), src.get('y', 0), src.get('z', 0)]
            
            # Normalize to unit vector for comparison with ODAS output
            magnitude = (src_pos[0]**2 + src_pos[1]**2 + src_pos[2]**2)**0.5
            if magnitude > 0:
                src_x_norm = src_pos[0] / magnitude
                src_y_norm = src_pos[1] / magnitude
                src_z_norm = src_pos[2] / magnitude
            else:
                src_x_norm = src_y_norm = src_z_norm = 0
            
            for time_key in comparison_data.keys():
                if src_start <= time_key <= src_end:
                    comparison_data[time_key]['sources'].append({
                        'label': src_label,
                        'x': src_x_norm,
                        'y': src_y_norm,
                        'z': src_z_norm,
                        'x_orig': src_pos[0],
                        'y_orig': src_pos[1],
                        'z_orig': src_pos[2]
                    })
        
        # Add detections (all are now in matches, whether matched to source or unknown)
        for match in matches:
            # Handle both full results structure and saved JSON structure
            if 'detection' in match:
                # Full results structure (in-memory)
                det = match['detection']
                det_x = det['x']
                det_y = det['y']
                det_z = det['z']
                det_timestamp = det['timestamp']
                det_activity = det.get('activity', 0)
            else:
                # Saved JSON structure (flattened)
                det_x = match['position'][0]
                det_y = match['position'][1]
                det_z = match['position'][2]
                det_timestamp = match['timestamp']
                det_activity = match['activity']
            
            time_key = round(det_timestamp, 1)
            if time_key in comparison_data:
                det_type = 'matched' if match.get('source_label', match.get('label', 'unknown')) != 'unknown' else 'unmatched'
                detection_entry = {
                    'type': det_type,
                    'label': match.get('source_label', match.get('label', 'unknown')),
                    'x': det_x,
                    'y': det_y,
                    'z': det_z,
                    'error': match.get('angular_error'),
                    'confidence': match.get('confidence', 0),
                    'activity': det_activity
                }
                
                # Add YAMNet classification info if available
                if 'yamnet_class' in match and match['yamnet_class']:
                    detection_entry['yamnet_class'] = match['yamnet_class']
                    detection_entry['yamnet_confidence'] = match.get('yamnet_confidence', 0)
                    detection_entry['yamnet_votes'] = match.get('yamnet_votes', 0)
                    detection_entry['match_type'] = match.get('match_type', 'ground_truth')
                
                comparison_data[time_key]['detections'].append(detection_entry)
        
        html_parts.append(f"""
        // Time slider for comparison table
        var comparisonData = {json.dumps(self._convert_to_native(comparison_data))};
        var timeSlider = document.getElementById('timeSlider');
        var timeValue = document.getElementById('timeValue');
        var comparisonTable = document.getElementById('comparisonTable');
        
        // Set slider range based on actual data
        var times = Object.keys(comparisonData).map(parseFloat).sort((a,b) => a-b);
        if (times.length > 0) {{
            timeSlider.min = times[0];
            timeSlider.max = times[times.length - 1];
            timeSlider.value = times[0];
        }}
        
        function updateComparisonTable(time) {{
            timeValue.textContent = parseFloat(time).toFixed(2);
            
            // Find closest time key
            var timeKey = parseFloat(time).toFixed(1);
            var data = comparisonData[timeKey];
            
            if (!data) {{
                comparisonTable.innerHTML = '<p style="color: #666;">No data at this timestamp</p>';
                return;
            }}
            
            var html = '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">';
            
            // Ground truth sources
            html += '<div>';
            html += '<h3 style="color: #667eea;">Ground Truth Sources</h3>';
            if (data.sources.length === 0) {{
                html += '<p style="color: #666;">No active sources at this time</p>';
            }} else {{
                html += '<table style="width: 100%; margin-top: 10px;">';
                html += '<tr style="background: #667eea; color: white;"><th>Label</th><th>X (unit)</th><th>Y (unit)</th><th>Z (unit)</th><th>Position (m)</th></tr>';
                data.sources.forEach(function(src) {{
                    html += '<tr>';
                    html += '<td><strong>' + src.label + '</strong></td>';
                    html += '<td>' + src.x.toFixed(3) + '</td>';
                    html += '<td>' + src.y.toFixed(3) + '</td>';
                    html += '<td>' + src.z.toFixed(3) + '</td>';
                    html += '<td>(' + src.x_orig.toFixed(1) + ', ' + src.y_orig.toFixed(1) + ', ' + src.z_orig.toFixed(1) + ')</td>';
                    html += '</tr>';
                }});
                html += '</table>';
            }}
            html += '</div>';
            
            // Detected peaks
            html += '<div>';
            html += '<h3 style="color: #4CAF50;">Detected Peaks</h3>';
            if (data.detections.length === 0) {{
                html += '<p style="color: #666;">No detections at this time</p>';
            }} else {{
                html += '<table style="width: 100%; margin-top: 10px; font-size: 12px;">';
                html += '<tr style="background: #4CAF50; color: white;">';
                html += '<th>GT Label</th><th>YAMNet</th><th>X</th><th>Y</th><th>Z</th><th>Error</th><th>Conf</th><th>Activity</th><th>Type</th>';
                html += '</tr>';
                data.detections.forEach(function(det) {{
                    var rowColor = det.type === 'matched' ? '' : 'background: #ffebee;';
                    
                    // Highlight mismatch between GT and YAMNet prediction
                    if (det.yamnet_class && det.label !== det.yamnet_class && det.label !== 'unknown') {{
                        rowColor = 'background: #fff3cd; border-left: 4px solid #ff9800;';
                    }}
                    
                    html += '<tr style="' + rowColor + '">';
                    html += '<td><strong>' + det.label + '</strong></td>';
                    
                    // YAMNet prediction column
                    if (det.yamnet_class) {{
                        var yamnetColor = det.label === det.yamnet_class ? '#4CAF50' : '#f44336';
                        html += '<td style="color: ' + yamnetColor + '; font-weight: bold;">' + det.yamnet_class;
                        if (det.yamnet_confidence !== undefined) {{
                            html += '<br><small>conf: ' + det.yamnet_confidence.toFixed(2);
                            if (det.yamnet_votes && det.yamnet_votes > 0) {{
                                html += ' &nbsp;·&nbsp; votes: ' + det.yamnet_votes + '/6';
                            }}
                            html += '</small>';
                        }}
                        html += '</td>';
                    }} else {{
                        html += '<td style="color: #999;">-</td>';
                    }}
                    
                    html += '<td>' + det.x.toFixed(3) + '</td>';
                    html += '<td>' + det.y.toFixed(3) + '</td>';
                    html += '<td>' + det.z.toFixed(3) + '</td>';
                    html += '<td>' + (det.error !== null && det.error !== undefined ? det.error.toFixed(2) + '°' : 'N/A') + '</td>';
                    html += '<td>' + (det.confidence !== undefined ? det.confidence.toFixed(3) : '0.000') + '</td>';
                    html += '<td>' + det.activity.toFixed(3) + '</td>';
                    html += '<td style="font-size: 10px;">' + (det.match_type || det.type) + '</td>';
                    html += '</tr>';
                }});
                html += '</table>';
                
                // Legend
                html += '<div style="margin-top: 10px; font-size: 11px; color: #666;">';
                html += '<strong>Legend:</strong> ';
                html += '<span style="background: #fff3cd; padding: 2px 6px; margin: 0 4px;">⚠️ GT ≠ YAMNet</span> ';
                html += '<span style="background: #ffebee; padding: 2px 6px; margin: 0 4px;">Unmatched</span>';
                html += '</div>';
            }}
            html += '</div>';
            
            html += '</div>';
            comparisonTable.innerHTML = html;
        }}
        
        timeSlider.addEventListener('input', function() {{
            updateComparisonTable(this.value);
        }});
        
        // Initialize
        updateComparisonTable(timeSlider.value);
    </script>
""")
        
        # Collect YAMNet classification data
        # Prefer event fields (6-hop rolling mode); fall back to legacy fields.
        yamnet_data = {}
        for match in matches:
            det = match['detection']
            ev_id   = det.get('event_class_id', -1)
            ev_name = det.get('event_class_name', 'unclassified')
            ev_conf = det.get('event_avg_confidence', 0.0)
            if ev_id != -1 and ev_name not in ('unclassified', ''):
                class_name = ev_name
                confidence = ev_conf
            else:
                class_name = det.get('class_name', 'unclassified')
                confidence = det.get('class_confidence', 0.0)
            timestamp = det['timestamp']
            track_id = det.get('track_id', 0)

            # Skip if no valid classification
            if class_name == 'unclassified' or confidence == 0.0 or class_name == '':
                continue
            
            if class_name not in yamnet_data:
                yamnet_data[class_name] = {
                    'times': [],
                    'confidences': [],
                    'track_ids': [],
                    'labels': []
                }
            
            yamnet_data[class_name]['times'].append(timestamp)
            yamnet_data[class_name]['confidences'].append(confidence)
            yamnet_data[class_name]['track_ids'].append(track_id)
            yamnet_data[class_name]['labels'].append(match.get('label', 'unknown'))
        
        # Only create the plot if we have YAMNet data
        if yamnet_data:
            html_parts.append("""
    <div class="section">
        <h2>🎵 YAMNet Classification Timeline</h2>
        <p style="color: #666; margin-bottom: 15px;">
            Audio classifications from ODAS beamformed signals. 
            Bar height represents confidence level.
        </p>
        <div id="yamnet_timeline"></div>
    </div>
""")
            
            # Create unique class list and assign indices
            unique_classes = sorted(yamnet_data.keys())
            class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
            
            # Create traces for each class
            yamnet_traces = []
            colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
            
            for cls_name in unique_classes:
                data = yamnet_data[cls_name]
                class_idx = class_to_idx[cls_name]
                color = colors[class_idx % len(colors)]
                
                # Create hover text with details
                hover_texts = []
                for i in range(len(data['times'])):
                    hover_text = (
                        f"<b>{cls_name}</b><br>"
                        f"Time: {data['times'][i]:.2f}s<br>"
                        f"Confidence: {data['confidences'][i]:.3f}<br>"
                        f"Track ID: {data['track_ids'][i]}<br>"
                        f"GT Label: {data['labels'][i]}"
                    )
                    hover_texts.append(hover_text)
                
                trace = {
                    'x': data['times'],
                    'y': [class_idx] * len(data['times']),
                    'customdata': data['confidences'],
                    'mode': 'markers',
                    'marker': {
                        'color': color,
                        'size': [c * 15 + 5 for c in data['confidences']],  # Size based on confidence
                        'opacity': [c * 0.7 + 0.3 for c in data['confidences']],  # Opacity based on confidence
                        'line': {'color': 'white', 'width': 1}
                    },
                    'name': cls_name,
                    'text': hover_texts,
                    'hovertemplate': '%{text}<extra></extra>',
                    'showlegend': True
                }
                yamnet_traces.append(trace)
            
            html_parts.append(f"""
    <script>
        var yamnetData = {json.dumps(yamnet_traces)};
        var yamnetLayout = {{
            title: {{
                text: 'YAMNet Audio Classifications Over Time',
                font: {{ size: 16, color: '#333' }}
            }},
            xaxis: {{ 
                title: 'Time (seconds)',
                showgrid: true,
                gridcolor: '#e0e0e0'
            }},
            yaxis: {{ 
                title: 'Detected Audio Class',
                ticktext: {json.dumps(unique_classes)},
                tickvals: {json.dumps(list(range(len(unique_classes))))},
                showgrid: true,
                gridcolor: '#e0e0e0'
            }},
            hovermode: 'closest',
            height: {max(400, len(unique_classes) * 35)},
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            margin: {{ l: 150, r: 50, t: 60, b: 60 }},
            legend: {{
                orientation: 'h',
                yanchor: 'bottom',
                y: -0.2,
                xanchor: 'center',
                x: 0.5
            }}
        }};
        Plotly.newPlot('yamnet_timeline', yamnetData, yamnetLayout);
    </script>
""")

        # ── Event Votes Distribution chart (new firmware only) ────────────────
        votes_data = []
        for match in matches:
            v = match.get('yamnet_votes', 0)
            c = match.get('yamnet_class', '')
            if v > 0 and c and c != 'unclassified':
                votes_data.append({'votes': v, 'class': c,
                                   'ts': match['detection']['timestamp']})

        if votes_data:
            # Histogram of vote counts (1–6) coloured by class
            from collections import defaultdict
            vote_by_class = defaultdict(lambda: [0]*7)  # index = vote count 0-6
            for d in votes_data:
                vote_by_class[d['class']][d['votes']] += 1

            vote_traces = []
            colors = px.colors.qualitative.Dark24
            for ci, (cls, counts) in enumerate(sorted(vote_by_class.items())):
                vote_traces.append({
                    'x': list(range(1, 7)),
                    'y': counts[1:7],
                    'name': cls,
                    'type': 'bar',
                    'marker': {'color': colors[ci % len(colors)]},
                    'hovertemplate': f'<b>{cls}</b><br>Votes: %{{x}}/6<br>Events: %{{y}}<extra></extra>'
                })

            html_parts.append(f"""
    <div class="section">
        <h2>\ud83d\uddf3\ufe0f Event Votes Distribution</h2>
        <p style="color: #666; margin-bottom: 15px;">
            Number of events by vote count (out of 6 rolling hops).
            Higher votes = more consistent classification over time.
        </p>
        <div id="votes_chart"></div>
    </div>
    <script>
        var votesData = {json.dumps(vote_traces)};
        var votesLayout = {{
            barmode: 'stack',
            xaxis: {{ title: 'Hop votes (out of 6)', dtick: 1 }},
            yaxis: {{ title: 'Number of events' }},
            height: 320,
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            margin: {{ l: 60, r: 40, t: 20, b: 60 }},
            legend: {{ orientation: 'h', yanchor: 'bottom', y: -0.4,
                       xanchor: 'center', x: 0.5 }}
        }};
        Plotly.newPlot('votes_chart', votesData, votesLayout);
    </script>
""")
        # ─────────────────────────────────────────────────────────────────────
        # TOP-K HISTORY HEATMAP
        # For every confirmed event that has topk_history, show a heatmap of
        # top-1 confidence over time (one row per YAMNet class seen in top-5,
        # one column per hop 1-6).  Each event gets its own Plotly chart.
        topk_events = [m for m in matches
                       if m['detection'].get('topk_history')
                       and len(m['detection']['topk_history']) > 0]
        if topk_events:
            html_parts.append("""
    <div class="section">
        <h2>📊 Top-K Classification History per Event</h2>
        <p style="color:#666;margin-bottom:15px;">
            Heatmap of top-5 class confidences across the 6-hop rolling window
            for each confirmed event.  Brighter = higher confidence.
        </p>
""")
            for ei, m in enumerate(topk_events[:20]):   # cap at 20 events
                det    = m['detection']
                hops   = det['topk_history']
                ev_cls = det.get('event_class_name', det.get('yamnet_class', '?'))
                ev_ts  = det.get('timestamp', 0)
                votes  = det.get('event_votes', 0)

                # Collect all unique class names across all hops (top-5 each)
                all_classes = []
                for hop in hops:
                    for cn in hop.get('class_names', []):
                        if cn not in all_classes:
                            all_classes.append(cn)

                # Build (class × hop) confidence matrix
                n_cls  = len(all_classes)
                n_hops = len(hops)
                matrix = [[0.0]*n_hops for _ in range(n_cls)]
                for hi, hop in enumerate(hops):
                    for ki, cn in enumerate(hop.get('class_names', [])):
                        ci = all_classes.index(cn)
                        matrix[ci][hi] = hop.get('confidences', [0]*5)[ki]

                # Highlight row of the winning event class
                win_idx = all_classes.index(ev_cls) if ev_cls in all_classes else -1
                row_colors = ['rgba(255,193,7,0.25)' if i == win_idx else 'rgba(0,0,0,0)'
                              for i in range(n_cls)]

                chart_id = f'topk_heatmap_{ei}'
                html_parts.append(f'        <div id="{chart_id}" style="margin-bottom:8px;"></div>\n')
                html_parts.append(f"""    <script>
        (function() {{
            var z    = {json.dumps(matrix)};
            var text = z.map(function(row) {{
                return row.map(function(v) {{ return v.toFixed(3); }});
            }});
            var data = [{{
                z: z, text: text, type: 'heatmap',
                colorscale: 'YlOrRd', zmin: 0, zmax: 1,
                x: {json.dumps([f'hop {h+1}' for h in range(n_hops)])},
                y: {json.dumps(list(reversed(all_classes)))},
                hovertemplate: '<b>%{{y}}</b><br>%{{x}}: conf %{{text}}<extra></extra>',
                texttemplate: '%{{text}}'
            }}];
            var layout = {{
                title: {{ text: 'Event @ {ev_ts:.2f}s &nbsp;·&nbsp; <b>{ev_cls}</b> &nbsp;·&nbsp; votes {votes}/6',
                          font: {{ size: 13 }} }},
                height: {max(180, n_cls * 28 + 80)},
                margin: {{ l: 160, r: 40, t: 40, b: 50 }},
                xaxis: {{ side: 'bottom' }},
                plot_bgcolor: '#fafafa', paper_bgcolor: 'white'
            }};
            Plotly.newPlot('{chart_id}', data, layout, {{responsive: true}});
        }})();
    </script>
""")
            html_parts.append("    </div>\n")  # close section

        # ─────────────────────────────────────────────────────────────────────

        self._add_audio_waveform_section(html_parts, results)
        
        html_parts.append("""
</body>
</html>
""")
        
        return ''.join(html_parts)
    
    def _display_summary(self, analysis_data):
        """Display analysis summary in Streamlit"""
        summary = analysis_data['summary']
        
        st.subheader("📊 Analysis Summary")
        
        # Check if YAMNet stats are available
        has_yamnet_stats = 'yamnet_stats' in analysis_data
        
        if has_yamnet_stats:
            # Show YAMNet stats prominently
            yamnet_stats = analysis_data['yamnet_stats']
            st.info(f"🎯 **Using YAMNet classifications**: {yamnet_stats['yamnet_classified']} classified detections")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("YAMNet Classified", yamnet_stats['yamnet_classified'])
            with col2:
                st.metric("Correct", yamnet_stats['correct'], help="Matches with ground truth")
            with col3:
                st.metric("Incorrect", yamnet_stats['incorrect'], help="Mismatches needing fine-tuning")
            with col4:
                accuracy = yamnet_stats.get('accuracy', 0)
                st.metric("Accuracy", f"{accuracy*100:.1f}%")
            
            # Show samples needing fine-tuning
            needs_training = yamnet_stats.get('needs_training', 0)
            if needs_training > 0:
                st.warning(f"⚠️ **{needs_training} samples marked for YAMNet fine-tuning dataset** (mismatches, low confidence, or unclassified)")
            
            st.markdown("---")
        
        # Check if OLD model stats exist (for backwards compatibility)
        has_model_stats = 'model_stats' in analysis_data
        
        if has_model_stats and isinstance(analysis_data['model_stats'], dict):
            # Show model stats prominently (old format)
            model_stats = analysis_data['model_stats']
            if 'total_predictions' in model_stats:
                st.info(f"🤖 **Model predictions applied**: {model_stats['total_predictions']} detections analyzed")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Model Predicted", model_stats.get('model_predicted', 0))
                with col2:
                    st.metric("Needs Training", model_stats.get('needs_training', 0))
                with col3:
                    st.metric("Model Confidence", f"{model_stats.get('avg_model_confidence', 0):.3f}")
                with col4:
                    st.metric("Total Detections", summary['total_detections'])
                
                st.markdown("---")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Total Detections", summary['total_detections'])
        with col2:
            st.metric("Match Rate", f"{summary['match_rate']*100:.1f}%")
        with col3:
            st.metric("Avg Error", f"{summary['avg_angular_error']:.2f}°")
        with col4:
            st.metric("Avg Confidence", f"{summary.get('avg_confidence', 0):.3f}")
        with col5:
            st.metric("Time Span", f"{summary['time_span_seconds']:.1f}s")
        
        # Per-source breakdown
        if 'by_source' in analysis_data and analysis_data['by_source']:
            st.subheader("📈 Per-Source Statistics")
            source_data = []
            for label, stats in analysis_data['by_source'].items():
                source_data.append({
                    'Source': label,
                    'Detections': stats['detections'],
                    'Avg Error (°)': f"{stats['avg_error']:.2f}",
                    'Min Error (°)': f"{stats['min_error']:.2f}",
                    'Max Error (°)': f"{stats['max_error']:.2f}",
                    'Avg Confidence': f"{stats.get('avg_confidence', 0):.3f}",
                    'Min Confidence': f"{stats.get('min_confidence', 0):.3f}",
                    'Max Confidence': f"{stats.get('max_confidence', 0):.3f}"
                })
            st.dataframe(source_data, width='stretch')
        
        # YAMNet Classification Statistics
        if 'matches' in analysis_data:
            matches = analysis_data['matches']
            classified_matches = [m for m in matches if m.get('class_name', 'unclassified') != 'unclassified']
            
            if classified_matches:
                st.subheader("🎯 YAMNet Classification Statistics")
                
                # Overall classification stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Classified Detections", len(classified_matches))
                with col2:
                    avg_class_conf = np.mean([m.get('class_confidence', 0) for m in classified_matches])
                    st.metric("Avg Classification Confidence", f"{avg_class_conf:.3f}")
                with col3:
                    unique_classes = len(set(m.get('class_name', 'unknown') for m in classified_matches))
                    st.metric("Unique Classes Detected", unique_classes)
                
                # Classification distribution
                class_counts = {}
                class_confidences = {}
                for m in classified_matches:
                    cname = m.get('class_name', 'unknown')
                    class_counts[cname] = class_counts.get(cname, 0) + 1
                    if cname not in class_confidences:
                        class_confidences[cname] = []
                    class_confidences[cname].append(m.get('class_confidence', 0))
                
                # Create classification table
                class_data = []
                for cname, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                    avg_conf = np.mean(class_confidences[cname])
                    class_data.append({
                        'Class': cname,
                        'Count': count,
                        'Avg Confidence': f"{avg_conf:.3f}",
                        'Min Confidence': f"{np.min(class_confidences[cname]):.3f}",
                        'Max Confidence': f"{np.max(class_confidences[cname]):.3f}"
                    })
                
                st.dataframe(class_data, width='stretch')
                
                # Show ground truth vs predicted comparison if available
                with st.expander("🔍 Ground Truth vs YAMNet Predictions"):
                    comparison_data = []
                    for m in classified_matches[:50]:  # Show first 50
                        comparison_data.append({
                            'Time (s)': f"{m.get('timestamp', 0):.2f}",
                            'Ground Truth': m.get('matched_label', 'Unknown'),
                            'YAMNet Prediction': m.get('class_name', 'unclassified'),
                            'Confidence': f"{m.get('class_confidence', 0):.3f}",
                            'Angular Error': f"{m.get('angular_error', 0):.2f}°"
                        })
                    st.dataframe(comparison_data, width='stretch')
                    if len(classified_matches) > 50:
                        st.info(f"Showing first 50 of {len(classified_matches)} classified detections")
    
    def _show_recent_analyses(self):
        """Show table of recent analyses"""
        st.subheader("📁 Recent Analyses")
        
        analysis_files = sorted(
            self.analysis_dir.glob("*_analysis.json"),
            key=os.path.getmtime,
            reverse=True
        )
        
        if not analysis_files:
            st.info("No analyses found")
            return
        
        analyses_data = []
        for analysis_file in analysis_files[:10]:  # Show last 10
            try:
                with open(analysis_file, 'r') as f:
                    data = json.load(f)
                
                run_id = data.get('run_id', analysis_file.stem.replace('_analysis', ''))
                report_exists = self._get_report_path(run_id).exists()
                dataset_exists = self._get_dataset_path(run_id).exists()
                
                analyses_data.append({
                    'Run ID': run_id,
                    'Scene': data.get('scene_name', 'Unknown'),
                    'Detections': data['summary']['total_detections'],
                    'Match %': f"{data['summary']['match_rate']*100:.1f}",
                    'Avg Error': f"{data['summary']['avg_angular_error']:.2f}°",
                    'Report': '✅' if report_exists else '❌',
                    'Dataset': '✅' if dataset_exists else '❌',
                    'Created': data.get('created_at', '')[:19]
                })
            except:
                continue
        
        if analyses_data:
            st.dataframe(analyses_data, width='stretch')
