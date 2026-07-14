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
import re
import zipfile
import pandas as pd
import soundfile as sf
from pathlib import Path
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from yamnet_dataset_curator import YAMNetDatasetCurator
from timing_compensator import TimingCompensator
from audio_reconstructor import AudioReconstructor

# Global configuration
CONFIG = {
    'angle_threshold_deg': 15.0,  # Max angular difference to match (widened from 10→15 to catch ODAS localisation jitter)
    # Asymmetric time windows around each GT source.
    # ODAS Kalman filter has a variable startup delay (observed: 0–6s) and a
    # track-persistence tail after the sound ends (observed: 2.5–13s).
    # Using a single symmetric offset (old: ±2.5s) caused temporal_mismatch on
    # every sample that ODAS started tracking late or kept alive after GT end.
    # YAMNet has 960ms-2400ms latency BEFORE emission (accumulation delay).
    # Kalman has 0-6s startup delay and 2.5-13s persistence tail.
    # pre:  catch YAMNet latency + Kalman startup (detections arrive late)
    # post: catch Kalman persistence tail (tracks linger after sound ends)
    'time_window_pre_s':  2.0,   # seconds before GT start — warmup silence handles cold-start, 2s covers Kalman jitter
    'time_window_post_s': 3.0,   # seconds after  GT end   — tail silence handles flush, 3s catches Kalman persistence
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
        self.project_root = self.base_output_dir.parent
        self.runs_dir = self.base_output_dir / 'runs'
        self.analysis_dir = self.base_output_dir / 'analysis'
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.odas_logs_dir = Path(odas_logs_dir)
        self.mic_array_root = self.project_root / 'Mic_Array'
        self.live_audio_dir = self.mic_array_root / 'Live_Audio'
        self.passive_audio_dir = self.mic_array_root / 'Passive_Audio'
        self.mic_array_cache_dir = self.base_output_dir / 'mic_array_imports'
        self.mic_array_cache_dir.mkdir(parents=True, exist_ok=True)

        models_candidates = [
            Path.home() / 'chatak-odas' / 'models',
            self.project_root.parent / 'chatak-odas' / 'models',
        ]
        default_models_dir = next((p for p in models_candidates if p.exists()), models_candidates[0])
        self.odas_models_dir = Path(os.getenv('ODAS_MODELS_DIR', str(default_models_dir)))
        
        # Initialize YAMNet curator (writes audio/spectrograms to yamnet_datasets/)
        self.yamnet_curator = YAMNetDatasetCurator(
            output_dir=str(self.base_output_dir / 'yamnet_datasets')
        )
        
        # Initialize timing compensator for interval-based matching
        self.timing_compensator = TimingCompensator()

    def _map_legacy_path(self, raw_path):
        """Map legacy absolute paths from older runs to current workspace paths."""
        if not raw_path:
            return ''

        mapped_path = str(raw_path)
        replacements = {
            '/home/azureuser/z_odas_newbeamform/build/ClassifierLogs': str(self.odas_logs_dir),
            '/home/azureuser/sodas/ClassifierLogs': str(self.odas_logs_dir),
            '/home/azureuser/simulator': str(self.project_root),
            '/home/azureuser/config/scenes': str(self.project_root / 'config' / 'scenes'),
        }
        for legacy_prefix, current_prefix in replacements.items():
            if mapped_path.startswith(legacy_prefix):
                return mapped_path.replace(legacy_prefix, current_prefix, 1)
        return mapped_path

    def _resolve_run_path(self, raw_path, search_dirs=None):
        """Resolve stale run metadata path to an existing local file when possible."""
        if not raw_path:
            return ''

        mapped = self._map_legacy_path(raw_path)
        for candidate in [str(raw_path), mapped]:
            if candidate and os.path.exists(candidate):
                return os.path.abspath(candidate)

        filename = os.path.basename(mapped)
        if filename and search_dirs:
            for directory in search_dirs:
                candidate = Path(directory) / filename
                if candidate.exists():
                    return str(candidate.resolve())

        # Return mapped path even when missing so errors point to the new location.
        return os.path.abspath(mapped)

    def _discover_session_live_file(self, run_data):
        """Find the best session_live file when metadata path is missing/stale.

        Strategy:
        1) Search known ClassifierLogs directories.
        2) If run timestamp exists, pick nearest filename timestamp.
        3) Otherwise use newest file.
        """
        search_dirs = [
            self.odas_logs_dir,
            self.project_root / 'ClassifierLogs',
            Path.home() / 'chatak-odas' / 'build' / 'ClassifierLogs',
            self.project_root.parent / 'chatak-odas' / 'build' / 'ClassifierLogs',
            Path.home() / 'simulator' / 'ClassifierLogs',
            Path.home() / 'Git_Dev' / 'simulator' / 'ClassifierLogs',
        ]

        candidates = []
        for d in search_dirs:
            d = Path(d)
            if d.exists():
                candidates.extend(d.glob('sst_session_live.json_*.json'))

        candidates = [c for c in candidates if c.exists()]
        if not candidates:
            return ''

        run_ts = run_data.get('timestamp', '')
        if not run_ts:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            return str(newest.resolve())

        try:
            target = datetime.strptime(run_ts, '%Y%m%d_%H%M%S')
            best = None
            best_diff = None
            for c in candidates:
                m = re.search(r'sst_session_live\.json_(\d{8}_\d{6})\.json$', c.name)
                if not m:
                    continue
                c_ts = datetime.strptime(m.group(1), '%Y%m%d_%H%M%S')
                diff = abs((c_ts - target).total_seconds())
                if best is None or diff < best_diff:
                    best = c
                    best_diff = diff

            if best is not None:
                return str(best.resolve())
        except Exception:
            pass

        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(newest.resolve())
    
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
                ambient_only_mode = st.checkbox(
                    "🌿 Label ALL peaks as background (ambient-only run)",
                    value=False,
                    help=(
                        "Enable this when the scene has ZERO directional sources "
                        "(ambient-only render, e.g. EXP-B4 hard negatives). "
                        "Every ODAS detection will be saved as the 'background' class "
                        "regardless of GT or YAMNet prediction. "
                        "This teaches the model to reject ambient ghost tracks."
                    )
                )
            
            with col2:
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
                    "**Fine-tuned model** — re-classify .bin patches using the active fine-tuned TFLite model "
                    "set in the 🧠 Fine-Tune YAMNet page. Same mel pipeline as standard YAMNet, "
                    "outputs your custom class labels instead of 521 AudioSet classes."
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

        use_mic_array_imports = st.toggle(
            "Use Mic Array Imports (optional)",
            value=False,
            help=(
                "Enable only when analyzing external Live/Passive mic-array sessions. "
                "For normal rendered runs, keep this off."
            ),
            key="use_mic_array_imports",
        )

        if use_mic_array_imports:
            mic_array_context = self._render_mic_array_inputs()
        else:
            mic_array_context = {'active_session': None}
        
        # Load run data
        with open(selected_run_file, 'r') as f:
            run_data = json.load(f)
        
        run_id = run_data.get('run_id', run_data.get('run_name', selected_run_file.stem))
        active_mic_session = mic_array_context.get('active_session')
        analysis_id = mic_array_context.get('analysis_id') if active_mic_session else run_id
        selected_cfg_name, selected_cfg_path, selected_model_name, _ = self._extract_runtime_selection(run_data)
        
        # Display run info
        col1, col2, col3, col4 = st.columns(4)
        if active_mic_session:
            with col1:
                st.metric("Session ID", mic_array_context.get('session_name', 'Unknown'))
            with col2:
                st.metric("Source", mic_array_context.get('session_type', 'mic_array').replace('_', ' ').title())
            with col3:
                gt_status = "Uploaded" if mic_array_context.get('ground_truth_scene') else "None"
                st.metric("Ground Truth", gt_status)
            with col4:
                st.metric("Model", mic_array_context.get('selected_model_name', 'default'))
        else:
            with col1:
                st.metric("Run ID", run_id)
            with col2:
                st.metric("Scene", run_data.get('scene_name', 'Unknown'))
            with col3:
                st.metric("Render ID", run_data.get('render_id', 'N/A'))
            with col4:
                st.metric("Duration", f"{run_data.get('scene_metadata', {}).get('duration', 0)}s")
            prov_col1, prov_col2 = st.columns(2)
            with prov_col1:
                st.caption(f"Config used: {selected_cfg_name}")
                if selected_cfg_path:
                    st.caption(f"Path: {selected_cfg_path}")
            with prov_col2:
                st.caption(f"Model used: {selected_model_name}")

        # Show experiment provenance if tagged
        exp_tag   = run_data.get('experiment_tag', '')
        odas_preset = run_data.get('odas_preset', '')
        if exp_tag or odas_preset:
            tag_parts = []
            if exp_tag:    tag_parts.append(f"🧪 `{exp_tag}`")
            if odas_preset: tag_parts.append(f"⚙️ preset: *{odas_preset}*")
            st.caption("  ·  ".join(tag_parts))

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
        analysis_path = self._get_analysis_path(analysis_id)
        report_path = self._get_report_path(analysis_id)
        dataset_path = self._get_dataset_path(analysis_id)

        analysis_exists = analysis_path.exists()
        generated_analysis_data = None

        # Analyze button
        col1, col2 = st.columns([3, 1])
        with col1:
            disable_analyze = bool(active_mic_session) and not bool(mic_array_context.get('tracks_path'))
            analyze_button = st.button(
                "🔍 Analyze Session" if active_mic_session and not analysis_exists else
                "🔄 Regenerate Session Analysis" if active_mic_session else
                "🔍 Analyze Run" if not analysis_exists else "🔄 Regenerate Analysis",
                type="primary",
                  width="stretch",
                disabled=disable_analyze
            )
        with col2:
            if analysis_exists:
                  if st.button("🗑️ Delete", width="stretch"):
                    self._delete_analysis(analysis_id)
                    st.rerun()

        # Run analysis
        if analyze_button:
            with st.spinner("Analyzing..."):
                if active_mic_session:
                    results = self._analyze_mic_array_session(
                        mic_array_context,
                        angle_threshold,
                        time_pre=time_pre,
                        time_post=time_post,
                    )
                else:
                    results = self._analyze_run(run_data, angle_threshold, save_unmatched,
                                                time_pre=time_pre, time_post=time_post)

                if results:
                    # Use YAMNet classifications instead of custom model
                    strategy = st.session_state.get('label_strategy', 'ODAS event voting')
                    results = self._apply_yamnet_classifications(results, label_strategy=strategy)

                    # Save analysis JSON
                    self._save_analysis(analysis_id, results, angle_threshold)

                    # Generate HTML report
                    self._generate_html_report(analysis_id, results)

                    # Create dataset CSV
                    self._create_dataset(results, analysis_id, save_unmatched)

                    # Save to YAMNet training dataset if enabled
                    if save_to_dataset:
                        try:
                            # Apply the UI threshold before curating
                            self.yamnet_curator.config['curation_criteria']['confidence_threshold'] = yamnet_conf_threshold
                            if ambient_only_mode:
                                # Ambient-only run: label all peaks as 'background' hard negatives
                                bg_stats = self.yamnet_curator.curate_ambient_as_background(
                                    results, analysis_id)
                                saved_bg = bg_stats.get('saved', 0)
                                st.info(f"🌿 Ambient-only mode: {saved_bg} peaks saved as 'background' hard negatives")
                            else:
                                yamnet_stats = self.yamnet_curator.curate_from_analysis(results, analysis_id)
                                saved_t = yamnet_stats.get('saved', 0)
                                saved_u = yamnet_stats.get('unknown_saved', 0)
                                if saved_t or saved_u:
                                    st.info(f"🎵 YAMNet dataset: {saved_t} training + {saved_u} unknown samples saved")
                        except Exception as e:
                            st.warning(f"⚠️ YAMNet curation skipped: {e}")

                    st.success("✅ Analysis complete!")
                    analysis_exists = True
                    generated_analysis_data = self._convert_to_native(results)
                    if active_mic_session:
                        st.session_state[f'auto_open_report_{analysis_id}'] = True

        # Display results if analysis exists
        if analysis_exists:
            if generated_analysis_data is not None:
                analysis_data = generated_analysis_data
            else:
                try:
                    with open(analysis_path, 'r') as f:
                        analysis_data = json.load(f)
                except json.JSONDecodeError as e:
                    st.error(f"❌ Analysis file is corrupted: {e}")
                    st.info("The file may have been corrupted due to an interrupted save. Try deleting and regenerating the analysis.")
                    if st.button("🗑️ Delete Corrupted Analysis", key="delete_corrupted"):
                        self._delete_analysis(analysis_id)
                        st.rerun()
                    return

            # ── Tabs: standard results + deployment evaluation + window explorer ─
            res_tab, deploy_tab, windows_tab = st.tabs([
                "📊 Results",
                "🚀 Deployment Evaluation",
                "🪟 Window Explorer"
            ])

            with res_tab:
                self._display_summary(analysis_data)

                # Action buttons
                st.markdown("---")

                if report_path.exists():
                    st.success("📊 Interactive Report Generated!")
                    auto_open_report = bool(st.session_state.pop(f'auto_open_report_{analysis_id}', False))

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        view_report = st.button("🔍 Open Report (Full Page)", key=f"open_{analysis_id}", width='stretch', type="primary") or auto_open_report

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
                            key=f"path_{analysis_id}",
                            label_visibility="collapsed"
                        )

                    # Show report in full page when button clicked
                    if view_report:
                        st.markdown("---")
                        st.markdown("### 📊 Interactive Report Viewer")
                        try:
                            with open(report_path, 'r') as f:
                                html_content = f.read()
                            html_size_kb = len(html_content) / 1024
                            if html_size_kb < 2048:  # <2MB safe to embed
                                import streamlit.components.v1 as components
                                st.info(f"⚡ Fully interactive ({html_size_kb:.0f} KB) — rotate 3D plots, zoom timeline, hover for details")
                                components.html(html_content, width=None, height=1400, scrolling=True)
                            else:
                                st.warning(
                                    f"📁 Report is **{html_size_kb:.0f} KB** — too large to embed safely.  \n"
                                    f"Use the **Download** button above, then open the file in your browser."
                                )
                        except Exception as _e:
                            st.error(f"Could not load report: {_e}")
                        st.markdown("---")
                        if st.button("⬆️ Back to Top", key=f"back_{analysis_id}"):
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
                        # Show summary only — full matches array can be thousands of
                        # entries and pushing it over Streamlit's WebSocket causes
                        # StreamClosedError.  Offer a download for the full file.
                        summary_view = {
                            'run_id':     analysis_data.get('run_id'),
                            'scene_name': analysis_data.get('scene_name'),
                            'timestamp':  analysis_data.get('timestamp'),
                            'summary':    analysis_data.get('summary'),
                            'by_source':  analysis_data.get('by_source'),
                            'match_count': len(analysis_data.get('matches', [])),
                        }
                        st.json(summary_view)
                        if analysis_path.exists():
                            with open(analysis_path, 'rb') as f:
                                st.download_button(
                                    "📥 Download full analysis JSON",
                                    f,
                                    file_name=analysis_path.name,
                                    mime='application/json',
                                    key=f'dl_json_{analysis_id}'
                                )

            with deploy_tab:
                self._render_deployment_eval(analysis_data, analysis_id)

            with windows_tab:
                self._render_window_explorer(analysis_data, analysis_id)

        # Show recent analyses
        st.markdown("---")
        self._show_recent_analyses()

    def _list_mic_array_sources(self, base_dir):
        """Return zip files and session folders for a Mic Array source directory."""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []

        entries = []
        for path in sorted(base_path.iterdir()):
            if path.is_dir() or path.suffix.lower() == '.zip':
                entries.append(path)
        return entries

    def _list_local_model_dirs(self):
        """Return model directories containing a TFLite + class map pair."""
        model_roots = [
            Path.home() / 'chatak-odas' / 'models',
            self.project_root.parent / 'chatak-odas' / 'models',
        ]
        discovered = []
        for root in model_roots:
            if not root.exists():
                continue
            if (root / 'yamnet_core.tflite').exists() and (root / 'yamnet_class_map.csv').exists():
                discovered.append(root)
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / 'yamnet_core.tflite').exists() and (child / 'yamnet_class_map.csv').exists():
                    discovered.append(child)
        unique = []
        seen = set()
        for path in discovered:
            resolved = str(path.resolve())
            if resolved not in seen:
                unique.append(path)
                seen.add(resolved)
        return unique

    def _extract_model_path_from_cfg(self, cfg_text):
        match = re.search(r'model_path\s*=\s*"([^"]+)"', cfg_text)
        return match.group(1).strip() if match else ''

    def _extract_mic_array_source(self, source_path):
        """Return discovered files for a Mic Array folder/zip, extracting zips into cache."""
        source_path = Path(source_path)
        session_root = source_path
        if source_path.suffix.lower() == '.zip':
            extract_dir = self.mic_array_cache_dir / source_path.stem
            if not extract_dir.exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(source_path, 'r') as zf:
                    zf.extractall(extract_dir)
            children = [p for p in extract_dir.iterdir() if p.is_dir()]
            session_root = children[0] if len(children) == 1 else extract_dir

        tracks_files = sorted(session_root.rglob('*_tracks.json'))
        cfg_files = sorted(session_root.rglob('*.cfg'))
        latlong_files = sorted(session_root.rglob('*_latlong.txt'))
        txt_files = sorted(session_root.rglob('*.txt'))

        return {
            'session_root': session_root,
            'tracks_path': tracks_files[0] if tracks_files else None,
            'cfg_path': cfg_files[0] if cfg_files else None,
            'latlong_path': latlong_files[0] if latlong_files else None,
            'notes_path': next((p for p in txt_files if p not in latlong_files), None),
        }

    def _coerce_ground_truth_scene(self, raw_data, scene_name='uploaded_ground_truth'):
        """Coerce uploaded GT JSON into the scene structure expected by the matcher."""
        if isinstance(raw_data, dict) and 'directional_sources' in raw_data:
            return raw_data

        if isinstance(raw_data, dict):
            items = raw_data.get('sources') or raw_data.get('events') or []
        elif isinstance(raw_data, list):
            items = raw_data
        else:
            items = []

        directional_sources = []
        for item in items:
            if not isinstance(item, dict):
                continue
            label = item.get('label') or item.get('class') or item.get('name') or 'unknown'
            start_time = float(item.get('start_time', item.get('start', 0.0)))
            end_time = float(item.get('end_time', item.get('end', start_time)))
            if 'position' in item and isinstance(item['position'], (list, tuple)) and len(item['position']) >= 3:
                position = [float(item['position'][0]), float(item['position'][1]), float(item['position'][2])]
            elif all(key in item for key in ('x', 'y', 'z')):
                position = [float(item['x']), float(item['y']), float(item['z'])]
            elif 'azimuth_deg' in item:
                az = np.radians(float(item.get('azimuth_deg', 0.0)))
                el = np.radians(float(item.get('elevation_deg', 0.0)))
                position = [float(np.cos(el) * np.cos(az)), float(np.cos(el) * np.sin(az)), float(np.sin(el))]
            else:
                position = [0.0, 0.0, 1.0]
            directional_sources.append({
                'label': label,
                'start_time': start_time,
                'end_time': end_time,
                'position': position,
            })

        return {'scene_name': scene_name, 'directional_sources': directional_sources}

    def _parse_concatenated_json_objects(self, text):
        decoder = json.JSONDecoder()
        idx = 0
        objects = []
        while idx < len(text):
            while idx < len(text) and text[idx].isspace():
                idx += 1
            if idx >= len(text):
                break
            obj, next_idx = decoder.raw_decode(text, idx)
            objects.append(obj)
            idx = next_idx
        return objects

    def _parse_mic_array_tracks(self, tracks_path):
        """Parse Mic Array *_tracks.json into analyzer detection records."""
        text = Path(tracks_path).read_text(encoding='utf-8', errors='replace')
        frames = self._parse_concatenated_json_objects(text)
        detections = []
        first_ts = None
        timestamp_scale = 1.0
        for frame_index, frame in enumerate(frames, 1):
            raw_ts = frame.get('timeStamp', frame_index)
            if first_ts is None:
                first_ts = raw_ts
                timestamp_scale = 0.001 if raw_ts > 1_000_000 else 1.0
            rel_ts = max(0.0, (raw_ts - first_ts) * timestamp_scale)
            for src in frame.get('src', []):
                class_name = src.get('class') or 'unclassified'
                class_conf = float(src.get('class_conf', 0.0))
                event_votes = 1 if class_name and class_name != 'unclassified' else 0
                detections.append({
                    'timestamp': float(rel_ts),
                    'frame_count': 1,
                    'line_number': frame_index,
                    'odas_timestamp': raw_ts,
                    'x': float(src.get('x', 0.0)),
                    'y': float(src.get('y', 0.0)),
                    'z': float(src.get('z', 0.0)),
                    'activity': float(src.get('activity', 0.0)),
                    'bins': [],
                    'class_id': -1,
                    'class_name': class_name,
                    'class_confidence': class_conf,
                    'class_timestamp': raw_ts,
                    'event_class_id': -1,
                    'event_class_name': class_name,
                    'event_votes': event_votes,
                    'event_avg_confidence': class_conf,
                    'event_max_confidence': class_conf,
                    'event_candidates': [],
                    'spectra_file': '',
                    'spectral_count': 0,
                    'topk_history': [],
                    'track_id': int(src.get('id', 0)),
                    'track_tag': src.get('tag', ''),
                    'track_type': src.get('type', src.get('tag', '')),
                })
        return detections

    def _build_unmatched_records(self, detections):
        matches = []
        for det in detections:
            matches.append({
                'detection': det,
                'source': None,
                'angular_error': None,
                'confidence': 0.0,
                'spatial_confidence': 0.0,
                'temporal_confidence': 0.0,
                'temporal_overlap_percent': 0.0,
                'detection_interval': 'N/A',
                'label': 'unknown',
                'match_type': 'unmatched'
            })
        return matches

    def _analyze_mic_array_session(self, mic_array_context, angle_threshold, time_pre=None, time_post=None):
        """Analyze a selected Mic Array session using uploaded GT when available."""
        tracks_path = mic_array_context.get('tracks_path')
        if not tracks_path or not Path(tracks_path).exists():
            st.error('Mic Array tracks JSON not found for the selected session.')
            return None

        detections = self._parse_mic_array_tracks(tracks_path)
        st.info(f"Parsed {len(detections)} detections from Mic Array tracks JSON")
        if not detections:
            st.warning('No detections found in the selected Mic Array session.')
            return None

        scene = mic_array_context.get('ground_truth_scene') or {
            'scene_name': mic_array_context.get('session_name', 'mic_array_session'),
            'directional_sources': [],
        }
        if scene.get('directional_sources'):
            matches, unmatched = self._match_detections_to_sources(
                detections, scene, angle_threshold, time_pre=time_pre, time_post=time_post
            )
        else:
            unmatched = list(detections)
            matches = self._build_unmatched_records(detections)

        stats = self._calculate_statistics(matches, unmatched, scene)
        return {
            'run_id': mic_array_context.get('analysis_id'),
            'render_id': mic_array_context.get('session_name', 'mic_array_session'),
            'scene_name': scene.get('scene_name', mic_array_context.get('session_name', 'mic_array_session')),
            'timestamp': datetime.now().isoformat(),
            'config': {
                'angular_threshold': angle_threshold,
                'save_unmatched': True,
                'source_type': mic_array_context.get('session_type'),
                'source_path': str(mic_array_context.get('active_session')),
                'tracks_path': str(tracks_path),
                'config_model_path': mic_array_context.get('config_model_path', ''),
                'selected_model_dir': mic_array_context.get('selected_model_dir', ''),
                'ground_truth_name': mic_array_context.get('ground_truth_name', ''),
            },
            'summary': stats['summary'],
            'by_source': stats['by_source'],
            'matches': matches,
            'unmatched': unmatched,
            'scene': scene,
            'run_metadata': {
                'mic_array': True,
                'latlong_path': str(mic_array_context.get('latlong_path', '')),
                'cfg_path': str(mic_array_context.get('cfg_path', '')),
                'notes_path': str(mic_array_context.get('notes_path', '')),
            }
        }

    def _render_mic_array_inputs(self):
        """Render Live/Passive Mic Array source selectors below the run picker."""
        st.markdown("**Mic Array Imports**")

        live_sources = self._list_mic_array_sources(self.live_audio_dir)
        passive_sources = self._list_mic_array_sources(self.passive_audio_dir)

        col1, col2 = st.columns(2)

        with col1:
            live_selection = st.selectbox(
                "Live Session",
                options=live_sources,
                format_func=lambda path: path.name,
                index=None,
                placeholder="Select a live session zip/folder",
                key="mic_array_live_session",
            )
            if live_selection is not None:
                st.caption(f"Selected: {live_selection}")
            elif not live_sources:
                st.caption(f"No live sessions found in {self.live_audio_dir}")

        with col2:
            passive_selection = st.selectbox(
                "Passive Session",
                options=passive_sources,
                format_func=lambda path: path.name,
                index=None,
                placeholder="Select a passive session zip/folder",
                key="mic_array_passive_session",
            )
            if passive_selection is not None:
                st.caption(f"Selected: {passive_selection}")
            elif not passive_sources:
                st.caption(f"No passive sessions found in {self.passive_audio_dir}")

        if live_selection and passive_selection:
            st.error('Select either a Live Session or a Passive Session, not both.')
            return {'active_session': None}

        active_session = live_selection or passive_selection
        if active_session is None:
            return {'active_session': None}

        session_type = 'live_session' if live_selection else 'passive_session'
        discovered = self._extract_mic_array_source(active_session)
        cfg_text = ''
        if discovered.get('cfg_path') and Path(discovered['cfg_path']).exists():
            cfg_text = Path(discovered['cfg_path']).read_text(encoding='utf-8', errors='replace')
        config_model_path = self._extract_model_path_from_cfg(cfg_text)
        config_model_name = Path(config_model_path).name if config_model_path else ''

        st.caption(f"Tracks JSON: {discovered.get('tracks_path') or 'Not found'}")
        st.caption(f"Config file: {discovered.get('cfg_path') or 'Not found'}")

        gt_upload = st.file_uploader(
            'Ground truth JSON',
            type=['json'],
            key=f'mic_array_gt_{session_type}',
            help='Optional. Upload GT JSON to enable source matching and richer reports.'
        )
        ground_truth_scene = None
        ground_truth_name = ''
        if gt_upload is not None:
            try:
                ground_truth_name = gt_upload.name
                ground_truth_scene = self._coerce_ground_truth_scene(
                    json.loads(gt_upload.getvalue().decode('utf-8')),
                    scene_name=Path(gt_upload.name).stem,
                )
                st.caption(f"Ground truth loaded: {ground_truth_name}")
            except Exception as exc:
                st.warning(f"Could not parse ground truth JSON: {exc}")

        local_model_dirs = self._list_local_model_dirs()
        model_mode = st.radio(
            'Model source',
            ['Use model from session config', 'Choose local model folder'],
            horizontal=True,
            key=f'mic_array_model_mode_{session_type}',
        )
        selected_model_dir = None
        selected_model_name = config_model_name or 'default'
        if model_mode == 'Use model from session config':
            if config_model_name:
                selected_model_dir = next((p for p in local_model_dirs if p.name == config_model_name), None)
                st.caption(f"Config model_path: {config_model_path}")
            else:
                st.caption('No model_path found in session config; using current reporter model path.')
        else:
            if local_model_dirs:
                selected_model_dir = st.selectbox(
                    'Local model folder',
                    options=local_model_dirs,
                    format_func=lambda p: p.name,
                    key=f'mic_array_local_model_{session_type}',
                )
                selected_model_name = selected_model_dir.name
            else:
                st.warning(f'No local model folders found under {self.odas_models_dir.parent}')

        if selected_model_dir is None:
            selected_model_dir = self.odas_models_dir
            selected_model_name = selected_model_name or selected_model_dir.name

        st.session_state['odas_model_override_dir'] = str(selected_model_dir)
        st.caption(f"Reporter TFLite path: {selected_model_dir / 'yamnet_core.tflite'}")

        session_name = active_session.stem if active_session.suffix.lower() == '.zip' else active_session.name
        analysis_id = f"mic_{session_type}_{session_name}"
        return {
            'active_session': active_session,
            'session_type': session_type,
            'session_name': session_name,
            'analysis_id': analysis_id,
            'tracks_path': discovered.get('tracks_path'),
            'cfg_path': discovered.get('cfg_path'),
            'latlong_path': discovered.get('latlong_path'),
            'notes_path': discovered.get('notes_path'),
            'ground_truth_scene': ground_truth_scene,
            'ground_truth_name': ground_truth_name,
            'config_model_path': config_model_path,
            'selected_model_dir': str(selected_model_dir),
            'selected_model_name': selected_model_name,
        }
    
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
            session_live_file = self._resolve_run_path(
                run_data.get('session_live_file'),
                search_dirs=[
                    self.odas_logs_dir,
                    self.project_root / 'ClassifierLogs',
                    Path.home() / 'chatak-odas' / 'build' / 'ClassifierLogs',
                    self.project_root.parent / 'chatak-odas' / 'build' / 'ClassifierLogs',
                    Path.home() / 'simulator' / 'ClassifierLogs',
                    Path.home() / 'Git_Dev' / 'simulator' / 'ClassifierLogs',
                ]
            )
            if not session_live_file or not os.path.exists(session_live_file):
                session_live_file = self._discover_session_live_file(run_data)

            if not session_live_file or not os.path.exists(session_live_file):
                st.error(f"Session live file not found: {session_live_file}")
                return None

            st.caption(f"Using session_live file: {session_live_file}")
            
            # Get scene file
            scene_file = self._resolve_run_path(
                run_data.get('scene_file'),
                search_dirs=[
                    self.project_root / 'config' / 'scenes',
                    self.project_root / 'config',
                ]
            )
            if not scene_file or not os.path.exists(scene_file):
                st.error(f"Scene file not found: {scene_file}")
                return None
            
            # Load scene
            with open(scene_file, 'r') as f:
                scene_data = json.load(f)
            
            # Parse ODAS output
            # warmup_seconds: silence prepended to render so ODAS can initialise
            # its spatial filters before the first source starts.  The renderer
            # stores this in the run metadata; default 0 for older renders.
            # NOTE: warmup_seconds is stored inside scene_metadata (because run
            # JSON copies the render metadata there), so we check both locations.
            warmup_seconds = float(
                run_data.get('warmup_seconds',
                    run_data.get('scene_metadata', {}).get('warmup_seconds', 0.0))
            )
            if warmup_seconds > 0:
                st.info(f"⏱ Warmup offset: subtracting {warmup_seconds:.0f}s from event timestamps")
            detections = self._parse_odas_output(session_live_file, warmup_seconds=warmup_seconds)
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
    
    def _parse_odas_output(self, session_live_file, warmup_seconds=0.0):
        """Parse ODAS session_live JSON file.

        Args:
            session_live_file: Path to the sst_session_live JSON.
            warmup_seconds: Silence prepended to the render before sending to
                ODAS (written by renderer into render metadata as
                'warmup_seconds').  Subtracted from every event timestamp so
                times align with the original scene GT windows.
        """
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
                    # hop_count × 8ms/hop, minus the silence prepended before
                    # the scene audio so that times align with GT windows.
                    line_timestamp = time_stamp * 0.008 - warmup_seconds
                    
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
                            # Number of *real* spectral frames written into the 96-slot buffer
                            # via spec_at_peak (SSL peak events).  Values << 96 mean the
                            # buffer is mostly zeros and the reconstructed audio will be
                            # near-silent even though the track direction is correct.
                            'spectral_count': src.get('spectral_count', 0),
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

        # ── Deduplication ──────────────────────────────────────────────────────
        # The SST JSON is emitted every ROLLING_HOPS frames (~48 ms) and every
        # src entry carries the *most-recent* YAMNet hop result (event_votes,
        # spectra_file, topk_history) persistently until the next hop fires.
        # Without deduplication every SST line produces a separate "detection"
        # for the same hop event, inflating a 70-second file to 15 000+
        # detections and poisoning GT matching.
        #
        # Key:   (track_id, hop_id)
        # hop_id = spectra_file path when non-empty (sim_mode=1, uniquely
        #          identifies one 96-frame patch per track), otherwise the
        #          frame-number timestamp of the first topk_history entry
        #          (field 'timestamp' is the ODAS hop frame counter which only
        #          changes when a new hop fires).
        # We keep the LAST occurrence of each hop so we get the most-accumulated
        # spectral_count / frame_count for that hop window.
        hop_index: dict = {}
        for d in detections:
            sf = d.get('spectra_file', '')
            if sf:
                hop_id = sf
            else:
                th = d.get('topk_history', [])
                hop_id = str(th[0]['timestamp']) if th else str(d.get('odas_timestamp', 0))
            key = (d['track_id'], hop_id)
            hop_index[key] = d          # overwrite → keep last (highest frame_count)

        return list(hop_index.values())
    
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
        """Match detected peaks to ground truth sources using interval overlap and event validity.

        Improvements over previous version:
          1. Event validity filtering: Only process detections with valid classifications
             (event_class_id != -1, event_votes >= 4, activity > 0.5, confidence > 0.6)
          2. Interval-based matching: Account for YAMNet's 960ms accumulation window
             using timing_compensator to calculate actual sound capture intervals
          3. Asymmetric windows: pre=10s (YAMNet latency), post=5s (Kalman persistence)

        Two-pass approach:
        1. For each source window [start - pre, end + post], match direction‑matched
           detections to that source (first match wins; prevents double-assigning).
        2. Label remaining unmatched detections as 'unknown'.
        """
        sources = scene.get('directional_sources', [])
        pre  = time_pre  if time_pre  is not None else CONFIG['time_window_pre_s']
        post = time_post if time_post is not None else CONFIG['time_window_post_s']
        
        # Check if detections have new event fields (firmware version check)
        has_event_fields = False
        if len(detections) > 0:
            sample_det = detections[0]
            has_event_fields = 'event_class_id' in sample_det and 'event_votes' in sample_det
        
        # Filter detections for valid events (only if new firmware with event fields)
        valid_detections = []
        filtered_stats = {'total': len(detections), 'no_classification': 0, 'low_votes': 0, 
                         'low_confidence': 0, 'valid': 0, 'legacy_mode': not has_event_fields}
        if has_event_fields:
            # New firmware: Apply event validity filtering
            for det in detections:
                # Check 1: Has valid classification (not -1)
                if det.get('event_class_id', -1) == -1:
                    filtered_stats['no_classification'] += 1
                    continue
                
                # Check 2: Has at least 1 hop vote (ODAS already gates at min_event_votes)
                # We accept votes >= 1 here; the curator/labeler decides quality.
                if det.get('event_votes', 0) < 1:
                    filtered_stats['low_votes'] += 1
                    continue
                
                # Check 3: Acoustic confidence gate (replaces Kalman activity threshold).
                # Kalman activity reflects track persistence, not acoustic strength —
                # it stays near zero during the Kalman warm-up phase (~150 hops).
                # event_avg_confidence is the mean YAMNet confidence across rolling
                # hops, which IS noise-robust: random noise rarely scores >= 0.1
                # consistently on any YAMNet class.
                if det.get('event_avg_confidence', 0.0) < 0.1:
                    filtered_stats['low_confidence'] += 1
                    continue
                
                valid_detections.append(det)
                filtered_stats['valid'] += 1
        else:
            # Legacy firmware: Use all detections (backward compatibility)
            valid_detections = list(detections)
            filtered_stats['valid'] = len(detections)
        
        # Log filtering stats
        if len(detections) > 0:
            if has_event_fields:
                validity_rate = (filtered_stats['valid'] / filtered_stats['total']) * 100
                print(f"Detection filtering: {filtered_stats['valid']}/{filtered_stats['total']} valid ({validity_rate:.1f}%)")
                print(f"  Filtered: no_class={filtered_stats['no_classification']}, "
                      f"low_votes={filtered_stats['low_votes']}, "
                      f"low_confidence={filtered_stats['low_confidence']}")
            else:
                print(f"Legacy mode: Using all {len(detections)} detections (no event fields in session data)")
        
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
            for idx, det in enumerate(valid_detections):
                det_time = det['timestamp']
                frame_count = det.get('frame_count', 1)
                
                # Calculate detection interval using timing compensator
                # (accounts for YAMNet's 960ms accumulation window)
                has_overlap, overlap_conf, overlap_info = self.timing_compensator.check_temporal_overlap(
                    gt_start=window_start,
                    gt_end=window_end,
                    detection=det
                )
                
                # Build overlap dict for downstream use
                overlap = {'has_overlap': has_overlap, 'confidence': overlap_conf}

                # Skip if no interval overlap with extended GT window
                if not has_overlap:
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
                    spatial_confidence = self._calculate_confidence(angle_diff)
                    
                    # Combine spatial and temporal confidence
                    # spatial_confidence: based on angular error (0-1)
                    # overlap_conf: based on temporal overlap percentage (0-1)
                    combined_confidence = (spatial_confidence + overlap_conf) / 2.0
                    
                    matches.append({
                        'detection': det,
                        'source': src,
                        'angular_error': angle_diff,
                        'confidence': combined_confidence,
                        'spatial_confidence': spatial_confidence,
                        'temporal_confidence': overlap_conf,
                        'temporal_overlap_percent': overlap_info.get('overlap_percent', 0.0),
                        'detection_interval': f"[{overlap_info.get('det_start', det_time):.2f}s - {overlap_info.get('det_end', det_time):.2f}s]",
                        'label': src_label,
                        'match_type': 'ground_truth'
                    })
                    
                    matched_detection_indices.add(idx)
        
        # PASS 2: Label unmatched valid detections as 'unknown'
        # Note: Only valid_detections are considered, so invalid events are implicitly filtered out
        unmatched = []
        for idx, det in enumerate(valid_detections):
            if idx not in matched_detection_indices:
                matches.append({
                    'detection': det,
                    'source': None,
                    'angular_error': None,
                    'confidence': 0.0,
                    'spatial_confidence': 0.0,
                    'temporal_confidence': 0.0,
                    'temporal_overlap_percent': 0.0,
                    'detection_interval': 'N/A',
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
                    model_dir = Path(st.session_state.get('odas_model_override_dir', str(self.odas_models_dir)))
                    if (not hasattr(self, '_py_yamnet') or
                            st.session_state.get('_py_yamnet_model_dir') != str(model_dir)):
                        model = model_dir / 'yamnet_core.tflite'
                        class_map = model_dir / 'yamnet_class_map.csv'
                        if not model.exists() or not class_map.exists():
                            return dict(class_id=-1, class_name='missing_model_files', confidence=0.0,
                                        votes=0, strategy_used='python_yamnet_missing_model')
                        self._py_yamnet = YAMNetSpectrumClassifier(str(model), str(class_map))
                        st.session_state['_py_yamnet_model_dir'] = str(model_dir)
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

        # ── Strategy 4: fine-tuned YAMNet TFLite ─────────────────────────────
        if strategy == 'Fine-tuned model':
            spectra_file = det.get('spectra_file', '')
            if spectra_file and os.path.exists(spectra_file):
                try:
                    import numpy as np
                    from yamnet_helper.yamnet_spectrum_classifier import YAMNetSpectrumClassifier
                    from yamnet_finetuner import YAMNetFinetuner

                    # Resolve active model paths from registry
                    finetuner = YAMNetFinetuner(str(self.base_output_dir))
                    tflite_path, class_map_path = finetuner.get_active_model_paths()

                    if tflite_path is None:
                        return dict(class_id=-1, class_name='no_active_model',
                                    confidence=0.0, votes=0,
                                    strategy_used='finetuned_no_model')

                    # Cache classifier in session_state; invalidate when model path changes
                    cached_path = st.session_state.get('_ft_yamnet_path')
                    if cached_path != tflite_path or '_ft_yamnet_obj' not in st.session_state:
                        st.session_state['_ft_yamnet_obj']  = YAMNetSpectrumClassifier(
                            tflite_path, class_map_path
                        )
                        st.session_state['_ft_yamnet_path'] = tflite_path

                    classifier = st.session_state['_ft_yamnet_obj']
                    patch = np.fromfile(spectra_file, dtype=np.float32).reshape(96, 257)
                    cid, cname, conf = classifier.classify_patch(patch)
                    return dict(class_id=cid, class_name=cname, confidence=float(conf),
                                votes=1, strategy_used='finetuned_yamnet_tflite')
                except Exception as exc:
                    return dict(class_id=-1, class_name=f'error:{exc}',
                                confidence=0.0, votes=0,
                                strategy_used='finetuned_error')
            return dict(class_id=-1, class_name='unclassified', confidence=0.0,
                        votes=0, strategy_used='finetuned_missing_bin')

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

            # ── Enrich in-memory match with patch_quality + spectral_count ──
            # These fields are computed by _build_match_record for JSON saving
            # but the curator runs on the LIVE match dicts before JSON is written,
            # so we must compute and set them here too.
            if 'patch_quality' not in match:
                src = match.get('source')
                ts  = det.get('timestamp', 0.0)
                fc  = det.get('frame_count', 0)
                track_start = det.get('track_start', ts - fc * 0.008)
                if src:
                    gt_s = float(src.get('start_time', ts))
                    gt_e = float(src.get('end_time',   ts))
                    if   track_start < gt_s:  match['patch_quality'] = 'pre_gt'
                    elif track_start <= gt_e: match['patch_quality'] = 'during_gt'
                    else:                     match['patch_quality'] = 'post_gt'
                else:
                    match['patch_quality'] = 'unknown'
            if 'spectral_count' not in match:
                match['spectral_count'] = int(det.get('spectral_count', 0))

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

    def _build_match_record(self, m):
        """Build the flat dict saved per matched detection.

        Adds patch-quality fields that flag whether the YAMNet spectra patch
        was captured during the ground-truth window or in the post-GT reverb tail.

        Fields added:
          track_start        – when ODAS first started accumulating spectra for
                               this track (timestamp − frame_count × 8 ms).
          detection_latency  – seconds between GT end and track_start.
                               0 means the track started before or during GT.
          patch_gt_overlap   – fraction (0–1) of the GT window duration that is
                               covered by the spectra patch interval.
          patch_quality      – "during_gt"   patch fully or partially overlaps GT
                               "post_gt"     track born after GT ended
                               "pre_gt"      track born before GT started (unusual)
        """
        det = m['detection']
        ts = float(det['timestamp'])
        fc = int(det.get('frame_count', 0))
        track_start = ts - fc * 0.008          # seconds: when track was born

        src = m.get('source') or {}
        gt_start = float(src.get('start_time', ts))
        gt_end   = float(src.get('end_time',   ts))
        gt_dur   = max(gt_end - gt_start, 1e-6)

        # How late (in seconds) the track started relative to GT end
        detection_latency = max(0.0, track_start - gt_end)

        # Overlap between [track_start, ts] and [gt_start, gt_end]
        patch_overlap_s = max(0.0, min(ts, gt_end) - max(track_start, gt_start))
        patch_gt_overlap = patch_overlap_s / gt_dur

        if track_start < gt_start:
            patch_quality = 'pre_gt'
        elif track_start <= gt_end:
            patch_quality = 'during_gt'
        else:
            patch_quality = 'post_gt'

        src_pos = None
        if src:
            sp = src.get('position', [src.get('x', 0), src.get('y', 0), src.get('z', 0)])
            src_pos = [float(x) for x in sp]

        return {
            'timestamp':          ts,
            'frame_count':        fc,
            'track_start':        round(track_start, 4),
            'position':           [float(det['x']), float(det['y']), float(det['z'])],
            'activity':           float(det['activity']),
            'source_label':       str(m['label']),
            'source_position':    src_pos,
            'gt_start':           gt_start,
            'gt_end':             gt_end,
            # ── Timing quality ──
            'detection_latency':  round(detection_latency, 4),
            'patch_gt_overlap':   round(patch_gt_overlap, 4),
            'patch_quality':      patch_quality,
            # ── Spatial ──
            'angular_error':      float(m['angular_error']) if m['angular_error'] is not None else None,
            'confidence':         float(m['confidence']),
            # spectral_count: number of real SSL-peak frames written into the 96-slot
            # YAMNet buffer.  0/1 means the buffer is nearly empty → audio ≈ silence.
            # bins_count kept for backward compat (legacy = 1 if spectra_file exists).
            'spectral_count':     int(det.get('spectral_count', 0)),
            'bins_count':         int(det.get('spectral_count', 0)) if det.get('spectral_count', 0) > 0
                                  else (1 if det.get('spectra_file') and os.path.exists(det.get('spectra_file', ''))
                                        else len(det.get('bins', []))),
            'spectra_file':       det.get('spectra_file', ''),
            # ── YAMNet top-voted class (for quick access) ──
            'model_prediction':   str(m['model_prediction']) if 'model_prediction' in m else None,
            'model_confidence':   float(m['model_confidence']) if 'model_confidence' in m else None,
            # ── Per-hop YAMNet voting summary ──
            'event_votes':        int(det.get('event_votes', 0)),
            'event_avg_confidence': float(det.get('event_avg_confidence', 0.0)),
            'event_max_confidence': float(det.get('event_max_confidence', 0.0)),
            # ── Full ranked candidate list: [{class_id, class_name, hop_votes, avg_confidence}] ──
            'event_candidates':   self._convert_to_native(det.get('event_candidates', [])),
            # ── Per-hop Top-K history: [{timestamp, class_ids[5], class_names[5], confidences[5]}] ──
            'topk_history':       self._convert_to_native(det.get('topk_history', [])),
            'match_type':         str(m.get('match_type', 'ground_truth')),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Concurrent-source analysis helpers
    # ──────────────────────────────────────────────────────────────────────

    def _compute_concurrency_buckets(self, src_windows, matches):
        """
        For every detection frame, count how many GT source windows were
        simultaneously active and group into buckets 0 / 1 / 2 / 3 / 4+.

        Parameters
        ----------
        src_windows : list of (start_time, end_time) float tuples
            One entry per unique GT source activation interval.
        matches : list of match dicts (saved JSON format)
        """
        from collections import Counter

        bucket_keys = ['0', '1', '2', '3', '4+']
        buckets = {k: {'total': 0, 'matched': 0,
                       'model_correct': 0, 'model_classified': 0,
                       'gt_labels': Counter()}
                   for k in bucket_keys}

        for m in matches:
            t = float(m.get('timestamp', 0.0))

            # Count how many source windows contain t
            active = sum(1 for (s, e) in src_windows if s <= t <= e)
            key = '0' if active == 0 else ('4+' if active >= 4 else str(active))

            b = buckets[key]
            b['total'] += 1

            if m.get('match_type') == 'ground_truth':
                b['matched'] += 1
                b['gt_labels'][m.get('source_label', 'unknown')] += 1

            mp = m.get('model_prediction', '')
            if mp and mp not in ('unclassified', ''):
                b['model_classified'] += 1
                if mp == m.get('source_label', ''):
                    b['model_correct'] += 1

        return buckets

    def _add_concurrent_source_section(self, html_parts, results):
        """
        Append an HTML section to html_parts showing detection performance
        broken down by how many GT sources were simultaneously active at each
        detection timestamp (single / double / triple / 4+).
        """
        matches = results['matches']

        # Build source windows from GT match records (gt_start / gt_end fields)
        # Each unique (source_label, gt_start, gt_end) triple is one activation.
        seen = set()
        src_windows = []
        for m in matches:
            if m.get('match_type') == 'ground_truth' and m.get('gt_start') is not None:
                key = (m.get('source_label'), round(m.get('gt_start', 0), 2),
                       round(m.get('gt_end', 0), 2))
                if key not in seen:
                    seen.add(key)
                    src_windows.append((float(m['gt_start']), float(m['gt_end'])))

        # Also try render sidecar for windows not covered by matches
        try:
            render_path = (self.base_output_dir / 'renders'
                           / f"{results.get('render_id', '')}.json")
            import json as _j
            for s in _j.loads(render_path.read_text()).get('source_sidecars', []):
                w = (float(s.get('start_time', 0)), float(s.get('end_time', 0)))
                src_windows.append(w)
        except Exception:
            pass

        # Deduplicate and remove degenerate windows
        src_windows = list(set((round(s, 2), round(e, 2))
                               for s, e in src_windows if e > s))

        if not src_windows:
            return

        buckets = self._compute_concurrency_buckets(src_windows, matches)

        # Bucket metadata for display
        bucket_meta = {
            '0':  ('🔇 None (ambient)',     '#95a5a6'),
            '1':  ('🔵 Single',              '#3498db'),
            '2':  ('🟢 Double',              '#2ecc71'),
            '3':  ('🟡 Triple',              '#f39c12'),
            '4+': ('🔴 Quad+',               '#e74c3c'),
        }

        # Build table rows
        rows_html = ''
        bar_labels, bar_total, bar_matched, bar_unmatched = [], [], [], []
        for key in ['0', '1', '2', '3', '4+']:
            b    = buckets[key]
            label, color = bucket_meta[key]
            total      = b['total']
            matched    = b['matched']
            unmatched  = total - matched
            match_rate = (matched / total * 100) if total else 0.0
            yam_cls    = b['model_classified']
            yam_corr   = b['model_correct']
            yam_acc    = (yam_corr / yam_cls * 100) if yam_cls else 0.0

            if total == 0:
                continue   # skip empty buckets in table

            top_gt = ', '.join(
                f'{lbl}×{cnt}' for lbl, cnt in b['gt_labels'].most_common(3)
            ) or '—'

            rows_html += f"""
            <tr>
                <td><span style="color:{color};font-weight:bold;">{label}</span></td>
                <td style="text-align:right;">{total}</td>
                <td style="text-align:right;">{matched}</td>
                <td style="text-align:right;">{unmatched}</td>
                <td style="text-align:right;">{match_rate:.1f}%</td>
                <td style="text-align:right;">{yam_cls}</td>
                <td style="text-align:right;">{yam_acc:.1f}%</td>
                <td style="font-size:12px;">{top_gt}</td>
            </tr>"""

            bar_labels.append(label)
            bar_total.append(total)
            bar_matched.append(matched)
            bar_unmatched.append(unmatched)

        import json as _json
        chart_data = _json.dumps({
            'labels':    bar_labels,
            'matched':   bar_matched,
            'unmatched': bar_unmatched,
        })

        html_parts.append(f"""
    <div class="section">
        <h2>🔢 Concurrent Source Analysis</h2>
        <p style="color:#555;">For each detection event, shows how many GT sources
        were <em>simultaneously active</em> at that moment.  A high false-positive
        rate in busy scenes (Double / Triple) usually means the classifier confuses
        overlapping sources rather than missing isolated ones.</p>

        <table>
            <tr>
                <th>Concurrency</th>
                <th style="text-align:right;">Total Events</th>
                <th style="text-align:right;">Matched</th>
                <th style="text-align:right;">Unmatched</th>
                <th style="text-align:right;">Match Rate</th>
                <th style="text-align:right;">Model Classified</th>
                <th style="text-align:right;">Model Accuracy</th>
                <th>Top GT Labels (matched)</th>
            </tr>
            {rows_html}
        </table>

        <div id="concurrency_chart" style="margin-top:20px;"></div>
    </div>
""")

        # Plotly stacked bar chart
        html_parts.append(f"""
    <script>
    (function() {{
        var d = {chart_data};
        Plotly.newPlot('concurrency_chart', [
            {{
                x: d.labels,
                y: d.matched,
                name: 'Matched (GT)',
                type: 'bar',
                marker: {{color: '#2ecc71'}}
            }},
            {{
                x: d.labels,
                y: d.unmatched,
                name: 'Unmatched',
                type: 'bar',
                marker: {{color: '#e74c3c'}}
            }}
        ], {{
            barmode: 'stack',
            title: 'Detection Counts by Source Concurrency',
            xaxis: {{title: 'Simultaneous Active GT Sources'}},
            yaxis: {{title: 'Detection Events'}},
            legend: {{orientation: 'h', y: -0.2}},
            margin: {{t: 50, b: 100}}
        }});
    }})();
    </script>
""")

    # ──────────────────────────────────────────────────────────────────────
    # Distance analysis helper
    # ──────────────────────────────────────────────────────────────────────

    _DIST_BUCKETS = [
        (0,   10,  '< 10 m'),
        (10,  25,  '10 – 25 m'),
        (25,  50,  '25 – 50 m'),
        (50,  100, '50 – 100 m'),
        (100, 200, '100 – 200 m'),
        (200, 9999,'> 200 m'),
    ]

    def _add_distance_analysis_section(self, html_parts, results):
        """
        Append an HTML section showing detection + accuracy stats broken down
        by the physical distance of the GT source from the microphone array.
        Only GT-matched detections have a known source position; unmatched
        detections are shown as a separate total row.
        """
        import math
        from collections import Counter

        matches   = results['matches']
        gt_matches = [m for m in matches if m.get('match_type') == 'ground_truth'
                      and m.get('source_position') and len(m['source_position']) == 3]
        if not gt_matches:
            return

        # Build per-bucket stats
        bucket_stats = []
        for lo, hi, label in self._DIST_BUCKETS:
            bm = [m for m in gt_matches
                  if lo <= math.sqrt(sum(x**2 for x in m['source_position'])) < hi]
            if not bm:
                continue
            dists      = [math.sqrt(sum(x**2 for x in m['source_position'])) for m in bm]
            errs       = [m.get('angular_error', 0) for m in bm]
            # "classified" = has a non-empty model_prediction
            classified = [m for m in bm if m.get('model_prediction') not in (None, 'unclassified', '')]
            correct    = sum(1 for m in classified
                             if m.get('model_prediction') == m.get('source_label'))
            top_labels = Counter(m.get('source_label', '?') for m in bm).most_common(3)
            bucket_stats.append({
                'label'      : label,
                'lo': lo, 'hi': hi,
                'n'          : len(bm),
                'avg_dist'   : sum(dists) / len(dists),
                'avg_err'    : sum(errs) / len(errs),
                'max_err'    : max(errs),
                'classified' : len(classified),
                'correct'    : correct,
                'acc'        : correct / len(classified) * 100 if classified else 0.0,
                'top_labels' : ', '.join(f'{l}×{c}' for l, c in top_labels),
            })

        if not bucket_stats:
            return

        # ── Table ─────────────────────────────────────────────────────────
        rows_html = ''
        # colour gradient: near=green, far=red
        colours = ['#27ae60','#2ecc71','#f39c12','#e67e22','#e74c3c','#c0392b']
        for i, b in enumerate(bucket_stats):
            col = colours[min(i, len(colours)-1)]
            rows_html += f"""
            <tr>
                <td><span style="color:{col};font-weight:bold;">{b['label']}</span></td>
                <td style="text-align:right;">{b['n']}</td>
                <td style="text-align:right;">{b['avg_dist']:.0f} m</td>
                <td style="text-align:right;">{b['avg_err']:.1f}°</td>
                <td style="text-align:right;">{b['max_err']:.1f}°</td>
                <td style="text-align:right;">{b['classified']}</td>
                <td style="text-align:right;">{b['acc']:.1f}%</td>
                <td style="font-size:12px;">{b['top_labels']}</td>
            </tr>"""

        # ── Plotly data ────────────────────────────────────────────────────
        import json as _json
        labels    = [b['label']   for b in bucket_stats]
        n_vals    = [b['n']       for b in bucket_stats]
        err_vals  = [b['avg_err'] for b in bucket_stats]
        acc_vals  = [b['acc']     for b in bucket_stats]
        chart_data = _json.dumps({'labels': labels, 'n': n_vals,
                                  'err': err_vals, 'acc': acc_vals})

        html_parts.append(f"""
    <div class="section">
        <h2>📏 Distance Analysis</h2>
        <p style="color:#555;">Detection and classification performance broken
        down by the physical distance from the microphone array to the GT source.
        Only spatially-matched (GT) events have a known distance.</p>

        <table>
            <tr>
                <th>Distance Band</th>
                <th style="text-align:right;">GT Events</th>
                <th style="text-align:right;">Avg Dist</th>
                <th style="text-align:right;">Avg Ang Err</th>
                <th style="text-align:right;">Max Ang Err</th>
                <th style="text-align:right;">Classified</th>
                <th style="text-align:right;">Model Accuracy</th>
                <th>Top GT Classes</th>
            </tr>
            {rows_html}
        </table>

        <div id="distance_chart" style="margin-top:20px;"></div>
    </div>
""")

        html_parts.append(f"""
    <script>
    (function() {{
        var d = {chart_data};
        var t1 = {{
            x: d.labels, y: d.n,
            name: 'GT matched events',
            type: 'bar',
            marker: {{color: '#3498db'}},
            yaxis: 'y'
        }};
        var t2 = {{
            x: d.labels, y: d.err,
            name: 'Avg angular error (°)',
            type: 'scatter', mode: 'lines+markers',
            marker: {{color: '#e74c3c', size: 8}},
            line: {{width: 2}},
            yaxis: 'y2'
        }};
        var t3 = {{
            x: d.labels, y: d.acc,
            name: 'Model accuracy (%)',
            type: 'scatter', mode: 'lines+markers',
            marker: {{color: '#2ecc71', size: 8}},
            line: {{width: 2, dash: 'dash'}},
            yaxis: 'y2'
        }};
        Plotly.newPlot('distance_chart', [t1, t2, t3], {{
            title: 'Detection Count · Angular Error · YAMNet Accuracy vs Distance',
            xaxis: {{title: 'Distance from microphone array'}},
            yaxis: {{title: 'GT Matched Events', side: 'left'}},
            yaxis2: {{title: 'Degrees / Accuracy %', overlaying: 'y', side: 'right',
                      range: [0, 100]}},
            legend: {{orientation: 'h', y: -0.25}},
            margin: {{t: 50, b: 120}}
        }});
    }})();
    </script>
""")

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
            'run_metadata': self._convert_to_native(results.get('run_metadata', {})),
            'summary': self._convert_to_native(results['summary']),
            'by_source': self._convert_to_native(results['by_source']),
            'model_stats': self._convert_to_native(results.get('model_stats', {})),
            'matches': [
                self._build_match_record(m)
                for m in results['matches']
            ],
            'unmatched': [
                {
                    'timestamp': float(u['timestamp']),
                    'frame_count': int(u['frame_count']),
                    'position': [float(u['x']), float(u['y']), float(u['z'])],
                    'activity': float(u['activity']),
                    'spectral_count': int(u.get('spectral_count', 0)),
                    'bins_count': int(u.get('spectral_count', 0)) if u.get('spectral_count', 0) > 0
                                  else (1 if u.get('spectra_file') and os.path.exists(u.get('spectra_file','')) else len(u.get('bins', []))),
                    'spectra_file': u.get('spectra_file', ''),
                    'model_prediction': str(u.get('event_class_name', 'unclassified')),
                    'model_confidence': float(u.get('event_max_confidence', 0.0)),
                    'event_votes': int(u.get('event_votes', 0)),
                    'event_avg_confidence': float(u.get('event_avg_confidence', 0.0)),
                    'event_max_confidence': float(u.get('event_max_confidence', 0.0)),
                    'event_candidates': self._convert_to_native(u.get('event_candidates', [])),
                    'topk_history': self._convert_to_native(u.get('topk_history', []))
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
            import numpy as _np

            # Read raw audio file (S16_LE format - signed 16-bit little-endian)
            # Use numpy memmap so we never load the full file into RAM.
            audio_mm = _np.memmap(raw_audio_file, dtype='<i2', mode='r')

            # Extract channel 3 (index 2) - can be made configurable
            channel_to_plot = 2  # Channel 3 (0-indexed)
            channel_samples_np = audio_mm[channel_to_plot::n_channels].astype(_np.float32)

            # Normalize to -1 to 1 range
            channel_samples_np /= 32768.0

            # Downsample for plotting — keep max 2000 points for small report size
            max_plot_points = 2000
            n_ch = len(channel_samples_np)
            step = max(1, n_ch // max_plot_points)
            channel_samples_np = channel_samples_np[::step]

            normalized_samples = channel_samples_np.tolist()
            n_pts = len(normalized_samples)
            time_axis = [i * step / sample_rate for i in range(n_pts)]
            
            # Add HTML section (no embedded audio — base64 audio adds ~2MB to report)
            html_parts.append(f"""
    <div class="section">
        <h2>🎵 Audio Waveform (Channel {channel_to_plot + 1})</h2>
        <p style="color: #666; margin-bottom: 15px;">
            Waveform of the raw capture aligned with the timeline above.
            Raw audio file: <code>{raw_audio_file}</code>
        </p>
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
    
    def _create_plotly_report(self, results):  # noqa: C901
        """
        Build a clean, comprehensive HTML analysis dashboard.

        Sections
        --------
        0  Header  +  KPI cards
        1  ODAS Detection Quality   – event-level detection rate, angular error
        2  Distance Analysis        – detection / accuracy vs source distance
        3  Simultaneous Sources     – existing concurrent-source section
        4  Timing                   – latency + patch-quality breakdown
        5  Model Classification     – confusion matrix, per-source accuracy
        6  Detection Timeline       – scrollable frame-by-frame slider
        """
        import math as _math
        import html as _html
        from collections import Counter, defaultdict

        # ── Normalise match records ───────────────────────────────────────────
        # In-memory results (from _analyze_run) store the GT label as m['label'].
        # Saved-JSON results (loaded from analysis_*.json) store it as
        # m['source_label'] (written by _build_match_record).
        # Similarly, the in-memory version nests the detection dict as
        # m['detection'], while the saved version flattens it into top-level
        # keys (timestamp, position, confidence, …).
        # Normalise to the saved-JSON convention so the rest of the report only
        # needs to look at one set of key names.
        def _norm(m):
            # ── source_label ─────────────────────────────────────────────────
            # In-memory uses 'label'; saved JSON uses 'source_label'.
            if 'source_label' not in m:
                m['source_label'] = m.get('label', '')

            # ── Flatten m['detection'] and m['source'] sub-dicts ─────────────
            # In-memory records nest raw ODAS fields under m['detection'] and
            # GT metadata under m['source'].  _build_match_record flattens all
            # of these to top-level keys when saving to JSON.  We replicate
            # that flattening here so the rest of the report only needs one
            # set of key names.
            if 'detection' in m and 'timestamp' not in m:
                det = m['detection']
                src = m.get('source') or {}

                ts = float(det.get('timestamp', 0))
                fc = int(det.get('frame_count', 0))

                # ODAS detection fields
                m['timestamp'] = ts
                m['position']  = [det.get('x', 0), det.get('y', 0), det.get('z', 0)]
                m['activity']  = det.get('activity', 0)
                # 'confidence' is the spatial+temporal combined score already
                # at top-level for in-memory records; don't overwrite it.
                m.setdefault('confidence', 0.0)

                # GT window bounds (critical for event-level detection rate)
                gt_start = float(src.get('start_time', ts))
                gt_end   = float(src.get('end_time',   ts))
                m.setdefault('gt_start', gt_start)
                m.setdefault('gt_end',   gt_end)

                # Source 3-D position (needed for distance analysis)
                if 'source_position' not in m and src:
                    sp = src.get('position')
                    if sp is None:
                        sp = [src.get('x', 0), src.get('y', 0), src.get('z', 0)]
                    m['source_position'] = [float(x) for x in sp]

                # Detection latency & patch quality
                if 'detection_latency' not in m:
                    track_start = ts - fc * 0.008
                    gt_dur = max(gt_end - gt_start, 1e-6)
                    patch_overlap_s = max(0.0, min(ts, gt_end) - max(track_start, gt_start))
                    m['track_start']       = round(track_start, 4)
                    m['detection_latency'] = round(max(0.0, track_start - gt_end), 4)
                    m['patch_gt_overlap']  = round(patch_overlap_s / gt_dur, 4)
                    if track_start < gt_start:
                        m['patch_quality'] = 'pre_gt'
                    elif track_start <= gt_end:
                        m['patch_quality'] = 'during_gt'
                    else:
                        m['patch_quality'] = 'post_gt'

            # model_prediction — same key in both in-memory and saved JSON ✓
            return m

        for _m in results.get('matches', []):
            _norm(_m)
        for _m in results.get('unmatched', []):
            _norm(_m)

        # ── Unpack ───────────────────────────────────────────────────────────
        run_id      = results.get('run_id', '')
        scene_name  = results.get('scene_name', '')
        render_id   = results.get('render_id', '')
        timestamp   = results.get('created_at', results.get('timestamp', ''))
        cfg         = results.get('config', {})
        summary     = results.get('summary', {})
        by_source   = results.get('by_source', {})
        matches     = results['matches']

        gt_matches  = [m for m in matches if m.get('match_type') == 'ground_truth']
        fp_matches  = [m for m in matches if m.get('match_type') != 'ground_truth']

        # ── Load render sidecar for GT event counts ───────────────────────────
        render_path = self.base_output_dir / 'renders' / f'{render_id}.json'
        total_gt_by_label = {}
        total_gt = 0
        scene_duration = summary.get('time_span_seconds', 60.0)
        try:
            rdata = json.loads(render_path.read_text())
            scene_duration = rdata.get('duration', scene_duration)
            for s in rdata.get('source_sidecars', []):
                lbl = s.get('label', '?')
                total_gt_by_label[lbl] = total_gt_by_label.get(lbl, 0) + 1
            total_gt = sum(total_gt_by_label.values())
        except Exception:
            pass

        # ── GT event-level detection rate ─────────────────────────────────────
        det_windows = defaultdict(set)
        for m in gt_matches:
            det_windows[m['source_label']].add(
                (m['source_label'], round(m.get('gt_start', 0), 1))
            )
        detected_event_count = sum(len(v) for v in det_windows.values())
        missed_event_count   = max(total_gt - detected_event_count, 0)
        odas_det_rate        = (detected_event_count / total_gt * 100
                                if total_gt else 0.0)
        fp_rate              = (len(fp_matches) / len(matches) * 100
                                if matches else 0.0)

        # ── Angular error stats ───────────────────────────────────────────────
        errs      = [m['angular_error'] for m in gt_matches
                     if m.get('angular_error') is not None]
        avg_err   = sum(errs) / len(errs) if errs else 0.0
        within_5  = (sum(1 for e in errs if e < 5)  / len(errs) * 100) if errs else 0
        within_10 = (sum(1 for e in errs if e < 10) / len(errs) * 100) if errs else 0

        # ── Model accuracy ────────────────────────────────────────────────────
        pred_m     = [m for m in gt_matches
                      if m.get('model_prediction') not in (None, 'unclassified', '')]
        correct_m  = [m for m in pred_m
                      if m['model_prediction'] == m['source_label']]
        model_acc  = len(correct_m) / len(pred_m) * 100 if pred_m else 0.0

        # ── Latency ───────────────────────────────────────────────────────────
        lats    = [m['detection_latency'] for m in gt_matches
                   if m.get('detection_latency') is not None]
        avg_lat = sum(lats) / len(lats) if lats else 0.0

        # ── No-GT fallback metrics ────────────────────────────────────────────
        classified_labels = [
            m.get('model_prediction') for m in matches
            if m.get('model_prediction') not in (None, '', 'unclassified')
        ]
        class_counter = Counter(classified_labels)
        classified_count = len(classified_labels)
        unclassified_count = max(len(matches) - classified_count, 0)
        ts_vals = [float(m.get('timestamp', 0.0)) for m in matches if m.get('timestamp') is not None]
        ts_min = min(ts_vals) if ts_vals else 0.0
        ts_max = max(ts_vals) if ts_vals else 0.0
        per_sec_counts = Counter(int(t) for t in ts_vals)
        no_gt_det_x = sorted(per_sec_counts.keys())
        no_gt_det_y = [per_sec_counts[k] for k in no_gt_det_x]
        class_rows_html = ''
        total_cls = max(classified_count, 1)
        for lbl, cnt in class_counter.most_common(12):
            pct = cnt / total_cls * 100
            class_rows_html += (
                f'<tr><td><b>{lbl}</b></td>'
                f'<td style="text-align:right">{cnt}</td>'
                f'<td style="text-align:right">{pct:.1f}%</td></tr>'
            )

        # ── Mic Array wall-clock timestamps (optional) ──────────────────────
        run_meta = results.get('run_metadata', {})
        selected_cfg_name, _, selected_model_name, _ = self._extract_runtime_selection(run_meta)
        selected_cfg_name = _html.escape(selected_cfg_name)
        selected_model_name = _html.escape(selected_model_name)
        notes_path = run_meta.get('notes_path', '')
        wallclock_start_iso = ''
        wallclock_points_iso = []
        wallclock_axis_for_classification = False
        if notes_path and os.path.exists(notes_path):
            try:
                with open(notes_path, 'r', encoding='utf-8', errors='replace') as nf:
                    note_lines = [ln.strip() for ln in nf.readlines() if ln.strip()]
                for ln in note_lines:
                    if ln.startswith('Recording started at:'):
                        wallclock_start_iso = ln.split('Recording started at:', 1)[1].strip()
                        break
                for ln in note_lines:
                    if 'T' in ln and len(ln) >= 19 and ln[:4].isdigit():
                        wallclock_points_iso.append(ln)
                wallclock_axis_for_classification = len(wallclock_points_iso) > 0
            except Exception:
                wallclock_start_iso = ''
                wallclock_points_iso = []
                wallclock_axis_for_classification = False

        has_ground_truth = len(gt_matches) > 0 and total_gt > 0

        # ── Helpers ───────────────────────────────────────────────────────────
        def pill(val, good, mid, fmt='%d%%'):
            cls = 'green' if val >= good else 'amber' if val >= mid else 'red'
            return f'<span class="pill {cls}">{fmt % val}</span>'

        def kpi_cls(val, good, mid):
            return 'green' if val >= good else 'amber' if val >= mid else 'red'

        # ═══════════════════════════════════════════════════════════════════════
        html_parts = []

        # ── Stylesheet ────────────────────────────────────────────────────────
        html_parts.append(f"""<!DOCTYPE html>
<html><head>
  <meta charset="utf-8">
  <title>ODAS Analysis — {run_id}</title>
  <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{font-family:'Segoe UI',sans-serif;background:#f0f2f5;color:#2d3436;}}
    .page{{max-width:1440px;margin:0 auto;padding:20px 24px;}}

    /* Banner */
    .banner{{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
             color:#fff;padding:26px 34px;border-radius:12px;margin-bottom:20px;}}
    .banner h1{{font-size:21px;margin-bottom:8px;letter-spacing:.3px;}}
    .banner .meta{{font-size:13px;opacity:.75;line-height:1.9;}}
    .banner .meta b{{color:#f9ca24;}}

    /* KPI row */
    .kpi-row{{display:grid;grid-template-columns:repeat(5,1fr);gap:14px;margin-bottom:20px;}}
    .kpi{{background:white;border-radius:10px;padding:15px 12px;text-align:center;
          box-shadow:0 2px 8px rgba(0,0,0,.06);border-top:4px solid #636e72;}}
    .kpi.green{{border-top-color:#00b894;}}
    .kpi.red{{border-top-color:#d63031;}}
    .kpi.amber{{border-top-color:#e17055;}}
    .kpi.blue{{border-top-color:#0984e3;}}
    .kpi.purple{{border-top-color:#6c5ce7;}}
    .kpi-val{{font-size:28px;font-weight:700;margin:5px 0 3px;}}
    .kpi-lbl{{font-size:11px;color:#636e72;text-transform:uppercase;letter-spacing:.6px;}}
    .kpi-sub{{font-size:11px;color:#b2bec3;margin-top:3px;}}

    /* Cards */
    .card{{background:white;border-radius:10px;padding:22px 26px;
           margin-bottom:18px;box-shadow:0 2px 8px rgba(0,0,0,.06);}}
    .card h2{{font-size:17px;margin-bottom:4px;}}
    .card .sub{{font-size:13px;color:#636e72;margin-bottom:14px;}}
    .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:18px;}}
    .three-col{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:18px;}}

    /* Tables */
    table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px;}}
    th{{background:#f8f9fa;color:#636e72;font-weight:600;text-transform:uppercase;
        font-size:11px;letter-spacing:.5px;padding:9px 12px;
        border-bottom:2px solid #dee2e6;text-align:left;}}
    td{{padding:8px 12px;border-bottom:1px solid #f1f3f5;vertical-align:middle;}}
    tr:last-child td{{border-bottom:none;}}
    tr:hover td{{background:#f8f9fa;}}

    /* Pills */
    .pill{{display:inline-block;padding:2px 9px;border-radius:20px;
           font-size:11px;font-weight:600;}}
    .pill.green{{background:#d4edda;color:#155724;}}
    .pill.red{{background:#f8d7da;color:#721c24;}}
    .pill.amber{{background:#fff3cd;color:#856404;}}
    .pill.blue{{background:#cce5ff;color:#004085;}}
    .pill.grey{{background:#e9ecef;color:#495057;}}

    /* Insight */
    .insight{{background:#f0f0fe;border-left:4px solid #6c5ce7;
              padding:10px 16px;border-radius:0 6px 6px 0;
              font-size:13px;line-height:1.6;margin:12px 0 4px;}}
    .insight b{{color:#6c5ce7;}}

    /* Section label */
    .sec-lbl{{font-size:11px;color:#b2bec3;text-transform:uppercase;
              letter-spacing:1px;margin:26px 0 8px;padding-left:2px;}}

    @media(max-width:900px){{
      .kpi-row{{grid-template-columns:repeat(2,1fr);}}
      .two-col,.three-col{{grid-template-columns:1fr;}}
    }}
  </style>
</head><body><div class="page">

  <!-- Banner -->
  <div class="banner">
    <h1>🎯 ODAS Pipeline Analysis Report</h1>
    <div class="meta">
      <b>Run:</b> {run_id} &nbsp;·&nbsp;
      <b>Scene:</b> {scene_name} &nbsp;·&nbsp;
      <b>Render:</b> {render_id}<br>
      <b>Analysed:</b> {str(timestamp)[:19].replace('T',' ')} &nbsp;·&nbsp;
      <b>Duration:</b> {scene_duration:.0f} s &nbsp;·&nbsp;
            <b>Threshold:</b> {cfg.get('angular_threshold','?')}°<br>
            <b>Config:</b> {selected_cfg_name} &nbsp;·&nbsp;
            <b>Model:</b> {selected_model_name}
    </div>
  </div>
""")

        # ── KPI Cards ─────────────────────────────────────────────────────────
        gt_str    = str(total_gt) if total_gt else 'N/A'
        det_str   = f'{odas_det_rate:.0f}%' if total_gt else f"{summary.get('match_rate',0)*100:.0f}%"
        if has_ground_truth:
            html_parts.append(f"""
  <!-- KPI Cards -->
  <div class="kpi-row">
    <div class="kpi purple">
      <div class="kpi-lbl">GT Sound Events</div>
      <div class="kpi-val">{gt_str}</div>
      <div class="kpi-sub">{summary.get('unique_sources',0)} sources · {scene_duration:.0f}s</div>
    </div>
    <div class="kpi {kpi_cls(odas_det_rate,75,50)}">
      <div class="kpi-lbl">ODAS Detection Rate</div>
      <div class="kpi-val">{det_str}</div>
      <div class="kpi-sub">{detected_event_count} detected · {missed_event_count} missed</div>
    </div>
    <div class="kpi {kpi_cls(100-fp_rate,60,40)}">
      <div class="kpi-lbl">False Positive Rate</div>
      <div class="kpi-val">{fp_rate:.0f}%</div>
      <div class="kpi-sub">{len(fp_matches)} FP · {len(gt_matches)} GT frames</div>
    </div>
    <div class="kpi blue">
      <div class="kpi-lbl">Avg Angular Error</div>
      <div class="kpi-val">{avg_err:.1f}°</div>
      <div class="kpi-sub">{within_5:.0f}% &lt;5° · {within_10:.0f}% &lt;10°</div>
    </div>
    <div class="kpi {kpi_cls(model_acc,60,30)}">
      <div class="kpi-lbl">Model Accuracy</div>
      <div class="kpi-val">{model_acc:.1f}%</div>
      <div class="kpi-sub">{len(correct_m)} correct · {len(pred_m)} classified</div>
    </div>
  </div>
""")
        else:
            html_parts.append(f"""
    <!-- KPI Cards (No Ground Truth) -->
    <div class="kpi-row">
        <div class="kpi blue">
            <div class="kpi-lbl">ODAS Frames</div>
            <div class="kpi-val">{len(matches)}</div>
            <div class="kpi-sub">Total detection frames in session</div>
        </div>
        <div class="kpi green">
            <div class="kpi-lbl">Classified Frames</div>
            <div class="kpi-val">{classified_count}</div>
            <div class="kpi-sub">Frames with non-empty class label</div>
        </div>
        <div class="kpi amber">
            <div class="kpi-lbl">Unclassified Frames</div>
            <div class="kpi-val">{unclassified_count}</div>
            <div class="kpi-sub">Frames without confident class output</div>
        </div>
        <div class="kpi purple">
            <div class="kpi-lbl">Time Span</div>
            <div class="kpi-val">{summary.get('time_span_seconds', 0):.1f}s</div>
            <div class="kpi-sub">Relative detection time range</div>
        </div>
        <div class="kpi blue">
            <div class="kpi-lbl">Detection Range</div>
            <div class="kpi-val">{ts_min:.1f}–{ts_max:.1f}s</div>
            <div class="kpi-sub">Min/max detection timestamps</div>
        </div>
    </div>
""")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 1 – ODAS Detection Quality
        # ═══════════════════════════════════════════════════════════════════════
        all_src = sorted(set(list(total_gt_by_label.keys()) + list(by_source.keys())))

        # Build per-source stats for table + chart
        ch_labels, ch_det, ch_miss, ch_err, ch_acc = [], [], [], [], []
        src_rows_html = ''
        for lbl in sorted(all_src,
                          key=lambda l: -(len(det_windows.get(l, set())) /
                                          max(total_gt_by_label.get(l, 1), 1))):
            gt_n  = total_gt_by_label.get(lbl, 0)
            det_n = len(det_windows.get(lbl, set()))
            mis_n = gt_n - det_n
            rate  = det_n / gt_n * 100 if gt_n else 0.0
            ae    = by_source.get(lbl, {}).get('avg_error', 0.0)

            src_gt   = [m for m in gt_matches if m.get('source_label') == lbl]
            src_pred = [m for m in src_gt
                        if m.get('model_prediction') not in (None, 'unclassified', '')]
            src_corr = sum(1 for m in src_pred
                           if m['model_prediction'] == m['source_label'])
            src_acc  = src_corr / len(src_pred) * 100 if src_pred else 0.0

            p_rate = 'green' if rate >= 80 else 'amber' if rate >= 60 else 'red'
            p_acc  = 'green' if src_acc >= 60 else 'amber' if src_acc >= 30 else 'red'

            src_rows_html += f"""
        <tr>
          <td><b>{lbl}</b></td>
          <td style="text-align:right">{gt_n}</td>
          <td style="text-align:right">{det_n}</td>
          <td style="text-align:right">{mis_n}</td>
          <td style="text-align:right"><span class="pill {p_rate}">{rate:.0f}%</span></td>
          <td style="text-align:right">{ae:.1f}°</td>
          <td style="text-align:right">{len(src_gt)}</td>
          <td style="text-align:right"><span class="pill {p_acc}">{src_acc:.1f}%</span></td>
        </tr>"""

            ch_labels.append(lbl)
            ch_det.append(det_n)
            ch_miss.append(mis_n)
            ch_err.append(round(ae, 2))
            ch_acc.append(round(src_acc, 1))

        if has_ground_truth:
            html_parts.append(f"""
  <div class="sec-lbl">01 — ODAS Spatial Detection</div>
  <div class="card">
    <h2>📡 Detection Quality by Source</h2>
    <p class="sub">How many GT sound events did ODAS spatially locate? Angular accuracy and model accuracy per source.</p>
    <div class="two-col">
      <div id="ch_det_rate" style="height:310px"></div>
      <div id="ch_ang_cdf"  style="height:310px"></div>
    </div>
    <table>
      <tr>
        <th>Source</th>
        <th style="text-align:right">GT Events</th>
        <th style="text-align:right">Detected</th>
        <th style="text-align:right">Missed</th>
        <th style="text-align:right">Det. Rate</th>
        <th style="text-align:right">Avg Ang. Err</th>
        <th style="text-align:right">ODAS Frames</th>
        <th style="text-align:right">Model Acc.</th>
      </tr>
      {src_rows_html}
    </table>
    </div>
""")
        else:
                        html_parts.append(f"""
    <div class="sec-lbl">01 — ODAS Spatial Detection</div>
    <div class="card">
                <h2>📡 Detection Overview (No Ground Truth)</h2>
                <p class="sub">Ground truth not provided. Showing detection volume and class distribution from the session JSON.</p>
                <div class="two-col">
                    <div id="ch_no_gt_det" style="height:310px"></div>
                    <div id="ch_no_gt_cls" style="height:310px"></div>
                </div>
                <table>
                    <tr>
                        <th>Class</th>
                        <th style="text-align:right">Frames</th>
                        <th style="text-align:right">Share</th>
                    </tr>
                    {class_rows_html}
                </table>
    </div>
""")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 2 – Distance Analysis
        # ═══════════════════════════════════════════════════════════════════════
        DIST_BANDS = [
            (0,   10,  '< 10 m'),
            (10,  25,  '10–25 m'),
            (25,  50,  '25–50 m'),
            (50,  100, '50–100 m'),
            (100, 200, '100–200 m'),
            (200, 9999,'> 200 m'),
        ]
        def _d3(sp):
            return _math.sqrt(sum(x**2 for x in sp)) if sp and len(sp) == 3 else None

        gt_pos = [m for m in gt_matches
                  if m.get('source_position') and len(m['source_position']) == 3]
        dist_stats = []
        if has_ground_truth:
            for lo, hi, band in DIST_BANDS:
                bm = [m for m in gt_pos if lo <= _d3(m['source_position']) < hi]
                if not bm:
                    continue
                dists  = [_d3(m['source_position']) for m in bm]
                b_errs = [m.get('angular_error', 0) for m in bm]
                b_pred = [m for m in bm
                          if m.get('model_prediction') not in (None, 'unclassified', '')]
                b_corr = sum(1 for m in b_pred
                             if m['model_prediction'] == m['source_label'])
                b_lats = [m['detection_latency'] for m in bm
                          if m.get('detection_latency') is not None]
                top    = Counter(m.get('source_label', '?') for m in bm).most_common(3)
                dist_stats.append({
                    'label':    band,
                    'n':        len(bm),
                    'avg_dist': sum(dists) / len(dists),
                    'avg_err':  sum(b_errs) / len(b_errs),
                    'max_err':  max(b_errs),
                    'acc':      b_corr / len(b_pred) * 100 if b_pred else 0.0,
                    'avg_lat':  sum(b_lats) / len(b_lats) if b_lats else 0.0,
                    'top':      ', '.join(f'{l}×{c}' for l, c in top),
                })
        else:
            det_pos = [m for m in matches if m.get('position') and len(m.get('position', [])) == 3]
            for lo, hi, band in DIST_BANDS:
                bm = [m for m in det_pos if lo <= _d3(m['position']) < hi]
                dists = [_d3(m['position']) for m in bm] if bm else []
                top = Counter(
                    m.get('model_prediction')
                    for m in bm
                    if m.get('model_prediction') not in (None, '', 'unclassified')
                ).most_common(3)
                dist_stats.append({
                    'label':    band,
                    'n':        len(bm),
                    'avg_dist': (sum(dists) / len(dists)) if dists else None,
                    'avg_err':  None,
                    'max_err':  None,
                    'acc':      None,
                    'avg_lat':  None,
                    'top':      ', '.join(f'{l}×{c}' for l, c in top),
                })

        if dist_stats:
            # Insight: is angular error flat or increasing with distance?
            if has_ground_truth and dist_stats[0]['avg_err'] is not None and dist_stats[-1]['avg_err'] is not None:
                first_err = dist_stats[0]['avg_err']
                last_err  = dist_stats[-1]['avg_err']
                if last_err > first_err * 1.25:
                    insight_d = (f"⚠️ <b>Angular error grows with distance</b> — "
                                 f"{dist_stats[0]['label']} avg {first_err:.1f}° vs "
                                 f"{dist_stats[-1]['label']} avg {last_err:.1f}°. "
                                 f"Consider tightening thresholds for near sources.")
                else:
                    insight_d = (f"✅ <b>Angular accuracy is distance-agnostic</b> — "
                                 f"error only {first_err:.1f}°–{last_err:.1f}° across all bands. "
                                 f"ODAS spatial localisation is robust to range.")
            elif has_ground_truth:
                insight_d = "ℹ️ <b>Ground truth available</b> — insufficient angular samples in one or more distance buckets."
            else:
                insight_d = "ℹ️ <b>No ground truth uploaded</b> — showing ODAS frame counts per distance bucket and top predicted classes."

            dist_rows = ''
            for b in dist_stats:
                p = 'green' if (b['acc'] is not None and b['acc'] >= 60) else 'amber' if (b['acc'] is not None and b['acc'] >= 30) else 'red'
                avg_dist_txt = f"{b['avg_dist']:.0f} m" if b['avg_dist'] is not None else "NA"
                avg_err_txt = f"{b['avg_err']:.1f}°" if b['avg_err'] is not None else "NA"
                max_err_txt = f"{b['max_err']:.1f}°" if b['max_err'] is not None else "NA"
                avg_lat_txt = f"{b['avg_lat']:.2f} s" if b['avg_lat'] is not None else "NA"
                acc_txt = f"<span class=\"pill {p}\">{b['acc']:.1f}%</span>" if b['acc'] is not None else "NA"
                top_txt = b['top'] if b['top'] else "NA"
                dist_rows += f"""
        <tr>
          <td><b>{b['label']}</b></td>
          <td style="text-align:right">{b['n']}</td>
          <td style="text-align:right">{avg_dist_txt}</td>
          <td style="text-align:right">{avg_err_txt}</td>
          <td style="text-align:right">{max_err_txt}</td>
          <td style="text-align:right">{avg_lat_txt}</td>
          <td style="text-align:right">{acc_txt}</td>
          <td style="font-size:12px">{top_txt}</td>
        </tr>"""
        else:
            insight_d = "ℹ️ <b>No detections available</b> — distance analysis has no rows for this run."
            dist_rows = '<tr><td colspan="8" style="text-align:center;color:#636e72">No distance data available.</td></tr>'

        frames_col_name = "GT Frames" if has_ground_truth else "ODAS Frames"
        dist_sub = "Does ODAS detect nearer events more reliably? Is classification accuracy distance-dependent?" if has_ground_truth else "No GT mode: only ODAS frame counts (blue bars) and top predicted classes by distance bucket."

        html_parts.append(f"""
  <div class="sec-lbl">02 — Distance Analysis</div>
  <div class="card">
    <h2>📏 Detection &amp; Accuracy vs Source Distance</h2>
    <p class="sub">{dist_sub}</p>
    <div id="ch_dist" style="height:330px"></div>
    <div class="insight">{insight_d}</div>
    <table>
      <tr>
        <th>Distance Band</th>
        <th style="text-align:right">{frames_col_name}</th>
        <th style="text-align:right">Avg Distance</th>
        <th style="text-align:right">Avg Ang. Err</th>
        <th style="text-align:right">Max Ang. Err</th>
        <th style="text-align:right">Avg Latency</th>
        <th style="text-align:right">Model Acc.</th>
        <th>Top Sources</th>
      </tr>
      {dist_rows}
    </table>
  </div>
""")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 3 – Simultaneous Sources (existing helper)
        # ═══════════════════════════════════════════════════════════════════════
        if has_ground_truth:
            html_parts.append('\n  <div class="sec-lbl">03 — Simultaneous Sources</div>')
            self._add_concurrent_source_section(html_parts, results)

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 4 – Timing
        # ═══════════════════════════════════════════════════════════════════════
        pq_counts  = Counter(m.get('patch_quality') for m in gt_matches)
        pq_labels  = [k for k, _ in pq_counts.most_common()]
        pq_vals    = [v for _, v in pq_counts.most_common()]
        po_vals    = [m['patch_gt_overlap'] for m in gt_matches
                      if m.get('patch_gt_overlap') is not None]
        avg_po     = sum(po_vals) / len(po_vals) if po_vals else 0.0

        pq_desc = {
            'during_gt': 'Detection fired while source was active',
            'pre_gt':    'Detection arrived before source started (early ODAS)',
            'post_gt':   'Detection arrived after source ended (Kalman tail)',
        }
        pq_rows = ''
        for k, v in pq_counts.most_common():
            p = v / len(gt_matches) * 100 if gt_matches else 0
            pq_rows += (f'<tr><td><b>{k}</b></td>'
                        f'<td style="text-align:right">{v}</td>'
                        f'<td style="text-align:right">{p:.0f}%</td>'
                        f'<td style="color:#636e72">{pq_desc.get(k,"—")}</td></tr>')

        lt100 = sum(1 for l in lats if l < 0.1)
        if has_ground_truth:
            timing_sub = "How quickly does ODAS detect events? Are detections temporally aligned with the GT windows?"
            timing_insight = f"""
    <div class="insight" style="margin-top:14px">
      <b>Avg latency:</b> {avg_lat:.2f} s &nbsp;·&nbsp;
      <b>Within 100 ms:</b> {lt100}/{len(lats)} ({lt100/max(len(lats),1)*100:.0f}% of GT frames) &nbsp;·&nbsp;
      <b>Avg GT overlap:</b> {avg_po:.0%}
    </div>
    <details style="margin-top:10px">
      <summary style="cursor:pointer;font-size:12px;color:#6c5ce7;font-weight:600">❓ Why can ODAS detect events <em>before</em> GT start? (pre_gt frames)</summary>
      <div class="insight" style="margin-top:8px;border-left-color:#0984e3">
        <b>Four reasons ODAS fires before the ground-truth window opens:</b><br>
        1. <b>Kalman noise tracking</b> — the filter continuously tracks ambient noise in all directions.
           When a GT source activates, an existing noise track that happens to point the right way is
           promoted rather than a new track started, so the "first seen" timestamp is earlier than GT start.<br>
        2. <b>YAMNet 960 ms buffer</b> — ODAS waits until its rolling audio buffer is full before
           emitting a classification. That buffer spans up to 960 ms, so a detection reported at
           time T contains audio from as far back as T&minus;960 ms — bridging back before GT start.<br>
        3. <b>Room acoustics / reverb</b> — early reflections from walls can reach the mic array
           slightly before the direct wavefront, letting ODAS lock onto the direction a fraction
           of a second before the sound source officially begins.<br>
        4. <b>Pre-window tolerance</b> — the matcher deliberately accepts detections
           up to <code>time_pre</code> seconds before GT start (default 10 s) to catch exactly
           these early starts. Only a tiny fraction (&lt;5 %) of frames fall in this bucket.
      </div>
    </details>
"""
        else:
            timing_sub = "Track activity over relative session time. Wall-clock timestamps are shown when available from session text logs."
            wc_note = f"Wall-clock start: {wallclock_start_iso}" if wallclock_start_iso else "Wall-clock start unavailable"
            timing_insight = f"""
    <div class="insight" style="margin-top:14px">
      <b>Detection frames:</b> {len(matches)} &nbsp;·&nbsp;
      <b>Time span:</b> {summary.get('time_span_seconds', 0):.1f}s &nbsp;·&nbsp;
      <b>{wc_note}</b>
    </div>
"""

        html_parts.append(f"""
  <div class="sec-lbl">04 — Timing</div>
  <div class="card">
    <h2>⏱️ Detection Timing</h2>
    <p class="sub">{timing_sub}</p>
    <div class="two-col">
      <div id="ch_latency"  style="height:290px"></div>
      <div id="ch_patch_q"  style="height:290px"></div>
    </div>
    <table>
      <tr><th>Patch Quality</th><th style="text-align:right">Frames</th>
          <th style="text-align:right">%</th><th>Meaning</th></tr>
      {pq_rows}
    </table>
        {timing_insight}
  </div>
""")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 5 – Model Classification
        # ═══════════════════════════════════════════════════════════════════════
        conf_labels = sorted(
            set(m['source_label'] for m in pred_m) |
            set(m['model_prediction'] for m in pred_m)
        )
        cl_idx = {l: i for i, l in enumerate(conf_labels)}
        conf_mat = [[0] * len(conf_labels) for _ in conf_labels]
        for m in pred_m:
            ti = cl_idx.get(m['source_label'], -1)
            pi = cl_idx.get(m['model_prediction'], -1)
            if ti >= 0 and pi >= 0:
                conf_mat[ti][pi] += 1

        # Per-source accuracy (bar)
        src_acc_lbl, src_acc_pct, src_corr_n, src_total_n = [], [], [], []
        for lbl in sorted(all_src):
            sp = [m for m in gt_matches if m.get('source_label') == lbl
                  and m.get('model_prediction') not in (None, 'unclassified', '')]
            sc = sum(1 for m in sp if m['model_prediction'] == m['source_label'])
            src_acc_lbl.append(lbl)
            src_acc_pct.append(round(sc / len(sp) * 100 if sp else 0, 1))
            src_corr_n.append(sc)
            src_total_n.append(len(sp))

        # Top-10 wrong pairs
        wrong_pairs = Counter(
            (m['source_label'], m['model_prediction'])
            for m in pred_m if m['model_prediction'] != m['source_label']
        )
        wrong_rows = ''
        for (tl, pl), cnt in wrong_pairs.most_common(10):
            pct = cnt / sum(1 for m in pred_m if m['source_label'] == tl) * 100
            wrong_rows += (f'<tr><td>{tl}</td><td style="color:#636e72">→</td>'
                           f'<td><b>{pl}</b></td>'
                           f'<td style="text-align:right">{cnt}</td>'
                           f'<td style="text-align:right">{pct:.0f}%</td></tr>')

        corr_conf  = [round(m.get('model_confidence', 0) or 0, 3) for m in correct_m]
        wrong_conf = [round(m.get('model_confidence', 0) or 0, 3)
                      for m in pred_m if m['model_prediction'] != m['source_label']]

        if has_ground_truth:
            html_parts.append(f"""
  <div class="sec-lbl">05 — Model Classification</div>
  <div class="card">
    <h2>🤖 Classification Performance</h2>
    <p class="sub">Of the ODAS-detected events, how accurately does the deployed model identify the sound class?</p>
    <div class="two-col">
      <div id="ch_confusion" style="height:390px"></div>
      <div id="ch_src_acc"   style="height:390px"></div>
    </div>
    <div class="two-col" style="margin-top:18px">
      <div>
        <h3 style="font-size:14px;margin-bottom:8px;color:#636e72">Top Misclassifications</h3>
        <table>
          <tr><th>True Class</th><th></th><th>Predicted As</th>
              <th style="text-align:right">Count</th>
              <th style="text-align:right">% of True</th></tr>
          {wrong_rows}
        </table>
      </div>
      <div id="ch_conf_dist" style="height:310px"></div>
    </div>
  </div>
""")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 7 – Directional & Temporal Overview
        # ═══════════════════════════════════════════════════════════════════════
        def _sph_deg(vec):
            """Cartesian unit/position vector → (azimuth_deg, elevation_deg)."""
            x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
            r = _math.sqrt(x**2 + y**2 + z**2)
            if r < 1e-9:
                return 0.0, 0.0
            el = round(_math.degrees(_math.asin(max(-1.0, min(1.0, z / r)))), 1)
            az = round(_math.degrees(_math.atan2(y, x)), 1)
            return az, el

        # Load all GT event positions from scene file ─────────────────────────
        gt_all_events = []   # {lbl, az, el, dist, start, end}
        try:
            _rdata = json.loads(render_path.read_text())
            _sf = _rdata.get('scene_file', '')
            if _sf and os.path.exists(_sf):
                _scene_d = json.loads(open(_sf).read())
                for _s in _scene_d.get('directional_sources', []):
                    _x, _y, _z = _s.get('x',0), _s.get('y',0), _s.get('z',0)
                    _d = _math.sqrt(_x**2 + _y**2 + _z**2)
                    _az, _el = _sph_deg([_x, _y, _z])
                    gt_all_events.append({'lbl': _s.get('label','?'),
                                          'az': _az, 'el': _el,
                                          'dist': round(_d, 1),
                                          'start': round(_s.get('start_time', 0), 2),
                                          'end':   round(_s.get('end_time',   0), 2)})
        except Exception:
            pass
        # Fallback: use GT-matched events only
        if not gt_all_events:
            _seen_k = set()
            for _m in gt_matches:
                _k = (_m.get('source_label',''), round(_m.get('gt_start',0), 1))
                if _k in _seen_k:
                    continue
                _seen_k.add(_k)
                _sp = _m.get('source_position') or [0, 0, 0]
                _d2 = _math.sqrt(sum(v**2 for v in _sp))
                _az2, _el2 = _sph_deg(_sp)
                gt_all_events.append({'lbl': _m.get('source_label',''),
                                      'az': _az2, 'el': _el2, 'dist': round(_d2, 1),
                                      'start': round(_m.get('gt_start',0), 2),
                                      'end':   round(_m.get('gt_end',  0), 2)})

        # ODAS direction vectors → az/el ───────────────────────────────────────
        odas_matched_dir = []
        for _m in gt_matches:
            _az, _el = _sph_deg(_m.get('position', [0, 0, 0]))
            odas_matched_dir.append({'az': _az, 'el': _el,
                                     'lbl': _m.get('source_label', '')})
        odas_fp_dir = []
        for _m in fp_matches:
            _az, _el = _sph_deg(_m.get('position', [0, 0, 0]))
            odas_fp_dir.append({'az': _az, 'el': _el})

        # Timeline: bands per source row ───────────────────────────────────────
        _tl_src_set = sorted(
            set(e['lbl'] for e in gt_all_events) |
            set(_m.get('source_label','') for _m in gt_matches)
        )
        _tl_src_idx = {lbl: i for i, lbl in enumerate(_tl_src_set)}
        tl_gt_bands = [{'lbl': e['lbl'], 'start': e['start'], 'end': e['end'],
                         'y': _tl_src_idx.get(e['lbl'], 0)}
                        for e in gt_all_events]
        tl_odas_gt  = [{'t': round(_m.get('timestamp',0), 2),
                         'y': _tl_src_idx.get(_m.get('source_label',''), -1),
                         'lbl': _m.get('source_label','')}
                        for _m in gt_matches]
        tl_odas_fp  = [{'t': round(_m.get('timestamp',0), 2)}
                        for _m in fp_matches]

        html_parts.append(f"""
  <div class="sec-lbl">07 \u2014 Spatial &amp; Temporal Overview</div>
  <div class="card">
    <h2>🗺️ Directional Distribution</h2>
    <p class="sub">
      GT sound events (♦ diamonds) positioned by azimuth &amp; elevation angle as seen from the mic
      array, colour-coded by source distance. Green circles = ODAS-matched detections;
      red circles = false-positive (unmatched) detections.
    </p>
    <div id="ch_dir" style="height:420px"></div>
  </div>
  <div class="card" style="margin-top:18px">
    <h2>⏰ Detection Timeline</h2>
    <p class="sub">
      Shaded bands = GT event windows per source label.
      Green dots = ODAS frames matched to GT. Red dots = false-positive ODAS frames
      (plotted below the source rows).
    </p>
    <div id="ch_timeline" style="height:{max(320, len(_tl_src_set)*55+100)}px"></div>
  </div>
""")

        # ═══════════════════════════════════════════════════════════════════════
        # SECTION 6 – Frame-by-Frame Timeline Slider
        # ═══════════════════════════════════════════════════════════════════════
        tl_data: dict = defaultdict(list)
        for m in matches:
            tk = f'{round(m.get("timestamp", 0), 1):.1f}'
            tl_data[tk].append({
                'mt':   m.get('match_type', 'unmatched'),
                'sl':   m.get('source_label', ''),
                'mp':   m.get('model_prediction', ''),
                'ae':   round(m.get('angular_error', 0) or 0, 2),
                'lat':  round(m.get('detection_latency', 0) or 0, 3),
                'conf': round(m.get('confidence', 0) or 0, 3),
            })
        tl_json = json.dumps(dict(tl_data))

        html_parts.append(f"""
  <div class="sec-lbl">06 — Detection Timeline</div>
  <div class="card">
    <h2>🔍 Frame-by-Frame Detection Timeline</h2>
    <p class="sub">Scrub through time to inspect individual ODAS detection frames vs ground truth.</p>
    <div style="margin:14px 0">
      <label style="font-weight:600;font-size:13px">
        Time: <span id="tl_val">0.0</span> s
      </label><br>
      <input type="range" id="tl_slider"
             min="0" max="{scene_duration:.0f}" step="0.1" value="0"
             style="width:100%;margin:8px 0;accent-color:#6c5ce7">
    </div>
    <div id="tl_table"></div>
  </div>
""")

        # ── Audio waveform (existing helper) ─────────────────────────────────
        self._add_audio_waveform_section(html_parts, results)

        # ═══════════════════════════════════════════════════════════════════════
        # JavaScript – all Plotly charts
        # ═══════════════════════════════════════════════════════════════════════
        # Pre-serialise Python data to JSON strings for embedding
        _ch_det_rate_data = json.dumps({
            'labels': ch_labels, 'det': ch_det, 'miss': ch_miss, 'err': ch_err,
        })
        _errs_json   = json.dumps(sorted(errs))
        _dist_json   = json.dumps([
            {'label': b['label'], 'n': b['n'], 'err': round(b['avg_err'], 1) if b['avg_err'] is not None else None,
               'acc': round(b['acc'], 1) if b['acc'] is not None else None,
               'has_gt': has_ground_truth} for b in dist_stats
        ] if dist_stats else [])
        _lat_json    = json.dumps(lats)
        _pq_json     = json.dumps({'labels': pq_labels, 'values': pq_vals})
        _conf_labels = json.dumps(conf_labels)
        _conf_mat    = json.dumps(conf_mat)
        _src_acc     = json.dumps({
            'labels': src_acc_lbl, 'pcts': src_acc_pct,
            'corr': src_corr_n, 'total': src_total_n,
        })
        _corr_conf   = json.dumps(corr_conf)
        _wrong_conf  = json.dumps(wrong_conf)
        _gt_dir_json       = json.dumps(gt_all_events)
        _odas_matched_dir_json = json.dumps(odas_matched_dir)
        _odas_fp_dir_json  = json.dumps(odas_fp_dir)
        _tl_bands_json     = json.dumps(tl_gt_bands)
        _tl_odas_gt_json   = json.dumps(tl_odas_gt)
        _tl_odas_fp_json   = json.dumps(tl_odas_fp)
        _tl_sources_json   = json.dumps(_tl_src_set)
        _wallclock_points_json = json.dumps(wallclock_points_iso)
        _no_gt_det_json = json.dumps({'x': no_gt_det_x, 'y': no_gt_det_y})
        _no_gt_cls_json = json.dumps({'labels': list(class_counter.keys()), 'values': list(class_counter.values())})

        html_parts.append(f"""
<script>
// ─────────────────────────────────────────────────────────────────────────────
// Ch1 · Detection rate per source (stacked bar + angular error overlay)
(function() {{
    if (!document.getElementById('ch_det_rate') || !document.getElementById('ch_ang_cdf')) return;
  var d = {_ch_det_rate_data};
  Plotly.newPlot('ch_det_rate', [
    {{ x:d.labels, y:d.det,  name:'Detected', type:'bar',
       marker:{{color:'#00b894'}},
       hovertemplate:'<b>%{{x}}</b><br>Detected: %{{y}}<extra></extra>' }},
    {{ x:d.labels, y:d.miss, name:'Missed', type:'bar',
       marker:{{color:'#d63031'}},
       hovertemplate:'<b>%{{x}}</b><br>Missed: %{{y}}<extra></extra>' }},
    {{ x:d.labels, y:d.err, name:'Avg Ang.Err (°)', type:'scatter',
       mode:'lines+markers', yaxis:'y2',
       marker:{{color:'#0984e3',size:8}}, line:{{dash:'dot',width:2}},
       hovertemplate:'<b>%{{x}}</b><br>Err: %{{y:.1f}}°<extra></extra>' }}
  ], {{
    barmode:'stack', height:310,
    title:{{text:'Event Detection Rate by Source',font:{{size:13}}}},
    xaxis:{{title:''}},
    yaxis:{{title:'GT Events', gridcolor:'#f1f3f5'}},
    yaxis2:{{title:'Angular Error (°)', overlaying:'y', side:'right', range:[0,20]}},
    legend:{{orientation:'h', y:-0.28}},
    plot_bgcolor:'#fafafa', paper_bgcolor:'white',
    margin:{{l:50, r:55, t:40, b:90}}
  }});
}})();

// Ch1b · No-GT detection overview
(function() {{
    if (!document.getElementById('ch_no_gt_det') || !document.getElementById('ch_no_gt_cls')) return;
    var d = {_no_gt_det_json};
    var c = {_no_gt_cls_json};
    Plotly.newPlot('ch_no_gt_det', [
        {{ x:d.x, y:d.y, type:'bar', marker:{{color:'#0984e3'}},
             hovertemplate:'t=%{{x}}s<br>Frames: %{{y}}<extra></extra>' }}
    ], {{
        height:310,
        title:{{text:'Detections per Second',font:{{size:13}}}},
        xaxis:{{title:'Relative Time (s)'}},
        yaxis:{{title:'Detection Frames', gridcolor:'#f1f3f5'}},
        plot_bgcolor:'#fafafa', paper_bgcolor:'white',
        margin:{{l:50, r:20, t:44, b:60}}
    }});

    Plotly.newPlot('ch_no_gt_cls', [
        {{ labels:c.labels, values:c.values, type:'pie', hole:0.45,
             marker:{{colors:['#00b894','#6c5ce7','#0984e3','#e17055','#d63031','#636e72']}},
             textinfo:'label+percent',
             hovertemplate:'<b>%{{label}}</b><br>%{{value}} frames (%{{percent}})<extra></extra>' }}
    ], {{
        height:310,
        title:{{text:'Class Distribution (classified frames)',font:{{size:13}}}},
        plot_bgcolor:'#fafafa', paper_bgcolor:'white',
        margin:{{l:20, r:20, t:44, b:20}}
    }});
}})();

// Ch2 · Angular error histogram + CDF
(function() {{
  var e = {_errs_json};
  var n = e.length;
  var cdf = e.map(function(_,i){{ return (i+1)/n*100; }});
  Plotly.newPlot('ch_ang_cdf', [
    {{ x:e, type:'histogram', name:'Frequency', nbinsx:30,
       marker:{{color:'#6c5ce7',opacity:0.7}},
       hovertemplate:'Error: %{{x:.1f}}°<br>Count: %{{y}}<extra></extra>' }},
    {{ x:e, y:cdf, type:'scatter', mode:'lines', name:'CDF (%)', yaxis:'y2',
       line:{{color:'#e17055',width:2}},
       hovertemplate:'Error: %{{x:.1f}}°<br>CDF: %{{y:.1f}}%<extra></extra>' }}
  ], {{
    height:310,
    title:{{text:'Angular Error Distribution & CDF',font:{{size:13}}}},
    xaxis:{{title:'Angular Error (°)'}},
    yaxis:{{title:'Detection Frames', gridcolor:'#f1f3f5'}},
    yaxis2:{{title:'Cumulative %', overlaying:'y', side:'right', range:[0,100]}},
    legend:{{orientation:'h', y:-0.28}},
    plot_bgcolor:'#fafafa', paper_bgcolor:'white',
    margin:{{l:50, r:55, t:40, b:90}}
  }});
}})();

// Ch3 · Distance: frames + angular error + model accuracy
(function() {{
    if (!document.getElementById('ch_dist')) return;
  var d = {_dist_json};
  if (!d.length) return;
  var lbl = d.map(function(x){{ return x.label; }});
    var hasGT = d.some(function(x){{ return !!x.has_gt; }});
    var traces = [
        {{ x:lbl, y:d.map(function(x){{return x.n;}}), name:'ODAS Frames', type:'bar',
             marker:{{color:'#0984e3',opacity:.8}},
             hovertemplate:'<b>%{{x}}</b><br>Frames: %{{y}}<extra></extra>' }}
    ];
    var layout = {{
        height:330,
        title:{{text: hasGT ? 'ODAS Frames · Angular Error · Model Accuracy vs Distance' : 'ODAS Frames by Distance Bucket', font:{{size:13}}}},
        xaxis:{{title:'Distance from mic array'}},
        yaxis:{{title:'ODAS Detection Frames', gridcolor:'#f1f3f5'}},
        legend:{{orientation:'h', y:-0.3}},
        plot_bgcolor:'#fafafa', paper_bgcolor:'white',
        margin:{{l:50, r:65, t:40, b:100}}
    }};

    if (hasGT) {{
        traces.push(
            {{ x:lbl, y:d.map(function(x){{return x.err;}}), name:'Avg Ang. Err (°)',
                 type:'scatter', mode:'lines+markers', yaxis:'y2',
                 marker:{{color:'#e17055',size:9}}, line:{{width:2}},
                 hovertemplate:'<b>%{{x}}</b><br>Err: %{{y:.1f}}°<extra></extra>' }},
            {{ x:lbl, y:d.map(function(x){{return x.acc;}}), name:'Model Acc (%)',
                 type:'scatter', mode:'lines+markers', yaxis:'y2',
                 marker:{{color:'#00b894',size:9,symbol:'diamond'}},
                 line:{{width:2,dash:'dot'}},
                 hovertemplate:'<b>%{{x}}</b><br>Acc: %{{y:.1f}}%<extra></extra>' }}
        );
        layout.yaxis2 = {{title:'Degrees / Accuracy %', overlaying:'y', side:'right', range:[0,100]}};
    }}

    Plotly.newPlot('ch_dist', traces, layout);
}})();

// Ch4 · Detection latency histogram
(function() {{
    if (!document.getElementById('ch_latency') || !document.getElementById('ch_patch_q')) return;
    var lat = {_lat_json};
    var wc = {_wallclock_points_json};
    var latSeries = lat;
    var xTitle = 'Latency (s)';
    if ((!lat || lat.length===0) && wc && wc.length>0) {{
        latSeries = wc;
        xTitle = 'Wall-clock timestamp';
    }}
  Plotly.newPlot('ch_latency', [
        {{ x:latSeries, type:'histogram', nbinsx:40, name:'Latency',
       marker:{{color:'#6c5ce7',opacity:0.75}},
       hovertemplate:'Latency: %{{x:.2f}}s<br>Count: %{{y}}<extra></extra>' }}
  ], {{
    height:290,
    title:{{text:'Detection Latency (time from GT start to first ODAS frame)',font:{{size:13}}}},
        xaxis:{{title:xTitle, type:(wc && wc.length>0 && (!lat || lat.length===0))?'date':'linear'}},
    yaxis:{{title:'Count', gridcolor:'#f1f3f5'}},
    plot_bgcolor:'#fafafa', paper_bgcolor:'white',
    margin:{{l:50, r:20, t:44, b:60}}
  }});
}})();

// Ch5 · Patch quality donut
(function() {{
  var pq = {_pq_json};
  Plotly.newPlot('ch_patch_q', [{{
    labels: pq.labels, values: pq.values, type:'pie', hole:0.45,
    marker:{{colors:['#00b894','#0984e3','#e17055','#636e72']}},
    textinfo:'label+percent',
    hovertemplate:'<b>%{{label}}</b><br>%{{value}} frames (%{{percent}})<extra></extra>'
  }}], {{
    height:290,
    title:{{text:'Patch Quality Distribution',font:{{size:13}}}},
    plot_bgcolor:'#fafafa', paper_bgcolor:'white',
    margin:{{l:20, r:20, t:44, b:20}}
  }});
}})();

// Ch6 · Confusion matrix heatmap (row-normalised %)
(function() {{
    if (!document.getElementById('ch_confusion') || !document.getElementById('ch_src_acc') || !document.getElementById('ch_conf_dist')) return;
  var labels = {_conf_labels};
  var z      = {_conf_mat};
  var zp = z.map(function(row) {{
    var s = row.reduce(function(a,b){{return a+b;}},0);
    return s > 0 ? row.map(function(v){{return Math.round(v/s*100);}})
                 : row.map(function(){{return 0;}});
  }});
  Plotly.newPlot('ch_confusion', [{{
    x:labels, y:labels, z:zp, type:'heatmap',
    colorscale:'Blues',
    text:zp.map(function(row){{return row.map(function(v){{return v+'%';}}); }}),
    texttemplate:'%{{text}}', textfont:{{size:11}},
    hovertemplate:'True: %{{y}}<br>Pred: %{{x}}<br>%{{z}}%<extra></extra>',
    colorbar:{{thickness:14, len:0.8}}
  }}], {{
    height:390,
    title:{{text:'Confusion Matrix — row-normalised (%)',font:{{size:13}}}},
    xaxis:{{title:'Predicted', tickangle:-35}},
    yaxis:{{title:'True', autorange:'reversed'}},
    plot_bgcolor:'#fafafa', paper_bgcolor:'white',
    margin:{{l:100, r:60, t:50, b:110}}
  }});
}})();

// Ch7 · Per-source model accuracy (horizontal bar)
(function() {{
  var d = {_src_acc};
  var colors = d.pcts.map(function(a){{
    return a>=60?'#00b894': a>=30?'#e17055':'#d63031';
  }});
  Plotly.newPlot('ch_src_acc', [{{
    x:d.pcts, y:d.labels, type:'bar', orientation:'h',
    marker:{{color:colors}},
    text:d.labels.map(function(_,i){{return d.corr[i]+'/'+d.total[i];}}),
    textposition:'outside',
    hovertemplate:'<b>%{{y}}</b><br>Acc: %{{x:.1f}}%<br>%{{text}} correct<extra></extra>'
  }}], {{
    height:390,
    title:{{text:'Model Accuracy by Source',font:{{size:13}}}},
    xaxis:{{title:'Accuracy (%)', range:[0,115]}},
    yaxis:{{title:''}},
    plot_bgcolor:'#fafafa', paper_bgcolor:'white',
    margin:{{l:100, r:60, t:50, b:60}}
  }});
}})();

// Ch8 · Model confidence: correct vs incorrect
(function() {{
  var cc = {_corr_conf};
  var wc = {_wrong_conf};
  Plotly.newPlot('ch_conf_dist', [
    {{ x:cc, type:'histogram', name:'Correct',   opacity:0.7, nbinsx:20,
       marker:{{color:'#00b894'}},
       hovertemplate:'Conf: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>' }},
    {{ x:wc, type:'histogram', name:'Incorrect', opacity:0.7, nbinsx:20,
       marker:{{color:'#d63031'}},
       hovertemplate:'Conf: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>' }}
  ], {{
    barmode:'overlay', height:310,
    title:{{text:'Model Confidence: Correct vs Incorrect',font:{{size:13}}}},
    xaxis:{{title:'Model Confidence'}},
    yaxis:{{title:'Count', gridcolor:'#f1f3f5'}},
    legend:{{x:0.65, y:0.9}},
    plot_bgcolor:'#fafafa', paper_bgcolor:'white',
    margin:{{l:50, r:20, t:44, b:60}}
  }});
}})();

// Ch9 · Directional distribution (az/el scatter)
(function() {{
  var gt = {_gt_dir_json};
  var om = {_odas_matched_dir_json};
  var fp = {_odas_fp_dir_json};

  // Jitter GT diamonds that land on the same rounded az+el cell so they don't
  // stack on top of each other. Spiral outward: odd indices shift right, even left.
  var gtAz = gt.map(function(d){{ return d.az; }});
  var gtEl = gt.map(function(d){{ return d.el; }});
  (function() {{
    var seen = {{}};
    for (var i = 0; i < gtAz.length; i++) {{
      var k = Math.round(gtAz[i]) + ',' + Math.round(gtEl[i]);
      var n = seen[k] || 0;
      if (n > 0) {{
        var step = Math.ceil(n / 2) * 2.5;
        gtAz[i] += (n % 2 === 0 ? 1 : -1) * step;
        gtEl[i] += (n % 4 < 2  ? 0.8 : -0.8) * Math.ceil(n / 2);
      }}
      seen[k] = n + 1;
    }}
  }})();

  Plotly.newPlot('ch_dir', [
    // FP — faint red dots; shows density/clusters of false positives
    {{ x:fp.map(function(d){{return d.az;}}),
       y:fp.map(function(d){{return d.el;}}),
       mode:'markers', type:'scatter', name:'False Positive',
       marker:{{symbol:'circle', size:4, color:'#d63031', opacity:0.22}},
       hovertemplate:'Az:%{{x:.1f}}°  El:%{{y:.1f}}°<extra>False Positive</extra>' }},
    // Matched ODAS directions — green circles
    {{ x:om.map(function(d){{return d.az;}}),
       y:om.map(function(d){{return d.el;}}),
       mode:'markers', type:'scatter', name:'ODAS Matched',
       text:om.map(function(d){{return d.lbl;}}),
       marker:{{symbol:'circle', size:6, color:'#00b894', opacity:0.45,
                line:{{color:'#00635a', width:0.7}}}},
       hovertemplate:'<b>%{{text}}</b><br>Az:%{{x:.1f}}°  El:%{{y:.1f}}°<extra>Matched</extra>' }},
    // GT events — diamonds, colour = source distance
    {{ x:gtAz, y:gtEl,
       mode:'markers', type:'scatter', name:'GT Event ♦',
       text:gt.map(function(d){{
         return d.lbl+' ('+d.dist+'m)  t='+d.start+'–'+d.end+'s';
       }}),
       marker:{{
         symbol:'diamond', size:13,
         color:gt.map(function(d){{return d.dist;}}),
         colorscale:'YlOrRd', showscale:true,
         colorbar:{{title:'Dist (m)', thickness:13, len:0.75, x:1.02}},
         line:{{color:'#2d3436', width:1.2}}
       }},
       hovertemplate:'<b>%{{text}}</b><br>Az:%{{x:.1f}}°  El:%{{y:.1f}}°<extra>GT Event</extra>' }}
  ], {{
    height:420,
    title:{{text:'Directional Distribution — GT Events ♦ (colour=distance) vs ODAS Detections',
            font:{{size:13}}}},
    xaxis:{{title:'Azimuth (°)', range:[-185,185], dtick:30,
            gridcolor:'#f1f3f5', zeroline:true,
            zerolinecolor:'#b2bec3', zerolinewidth:1.5}},
    yaxis:{{title:'Elevation (°)', range:[-55,75], dtick:15,
            gridcolor:'#f1f3f5', zeroline:true,
            zerolinecolor:'#b2bec3', zerolinewidth:1.5}},
    legend:{{orientation:'h', y:-0.22}},
    plot_bgcolor:'#fafafa', paper_bgcolor:'white',
    margin:{{l:60, r:85, t:50, b:80}}
  }});
}})();

// Ch10 · Detection timeline
(function() {{
  var bands  = {_tl_bands_json};
  var gtPts  = {_tl_odas_gt_json};
  var fpPts  = {_tl_odas_fp_json};
  var srcs   = {_tl_sources_json};
  var nSrc   = srcs.length;
  var shapes = bands.map(function(b) {{
    return {{type:'rect',
             x0:b.start, x1:b.end,
             y0:b.y-0.35, y1:b.y+0.35,
             fillcolor:'rgba(108,92,231,0.13)',
             line:{{color:'#6c5ce7',width:0.7}}}};
  }});
  var fpY = -0.7;
  Plotly.newPlot('ch_timeline', [
    {{ x:gtPts.map(function(d){{return d.t;}}),
       y:gtPts.map(function(d){{return d.y;}}),
       mode:'markers', type:'scatter', name:'ODAS Matched',
       text:gtPts.map(function(d){{return d.lbl;}}),
       marker:{{color:'#00b894',size:5,opacity:0.65,symbol:'circle'}},
       hovertemplate:'<b>%{{text}}</b><br>t=%{{x:.2f}}s<extra>Matched</extra>' }},
    {{ x:fpPts.map(function(d){{return d.t;}}),
       y:fpPts.map(function(){{return fpY;}}),
       mode:'markers', type:'scatter', name:'False Positive',
       marker:{{color:'#d63031',size:3,opacity:0.35,symbol:'circle'}},
       hovertemplate:'t=%{{x:.2f}}s<extra>False Positive</extra>' }}
  ], {{
    shapes:shapes,
    title:{{text:'Detection Timeline — GT windows (shaded) vs ODAS detections',font:{{size:13}}}},
    xaxis:{{title:'Time (s)',gridcolor:'#f1f3f5'}},
    yaxis:{{
      tickvals:srcs.map(function(_,i){{return i;}}).concat([fpY]),
      ticktext:srcs.concat(['FP / Unmatched']),
      range:[-1.2, nSrc],
      gridcolor:'#f1f3f5'
    }},
    legend:{{orientation:'h',y:-0.18}},
    plot_bgcolor:'#fafafa', paper_bgcolor:'white',
    margin:{{l:130,r:30,t:50,b:80}}
  }});
}})();

// Timeline slider
(function() {{
  var tl      = {tl_json};
  var slider  = document.getElementById('tl_slider');
  var valEl   = document.getElementById('tl_val');
  var tableEl = document.getElementById('tl_table');

  function render(t) {{
    var key = parseFloat(t).toFixed(1);
    valEl.textContent = key;
    var rows = tl[key] || [];
    if (!rows.length) {{
      tableEl.innerHTML = '<p style="color:#b2bec3;font-size:13px;padding:8px 0">No detections at this timestamp.</p>';
      return;
    }}
    var html = '<table><tr><th>Match Type</th><th>Source (GT)</th>'
             + '<th>Model Prediction</th>'
             + '<th style="text-align:right">Ang. Err</th>'
             + '<th style="text-align:right">Latency</th>'
             + '<th style="text-align:right">ODAS Conf</th></tr>';
    rows.forEach(function(r) {{
      var ok  = r.sl && r.mp && r.sl === r.mp;
      var bad = r.sl && r.mp && r.sl !== r.mp;
      var bg  = r.mt === 'ground_truth'
                  ? (ok ? '#f0fff4' : bad ? '#fff5f5' : '#f8f9fa')
                  : '#fff5f5';
      var mc  = ok ? '#00b894' : bad ? '#d63031' : '#636e72';
      html += '<tr style="background:' + bg + '">'
            + '<td><span class="pill '+(r.mt==='ground_truth'?'green':'red')+'">' + r.mt + '</span></td>'
            + '<td>' + (r.sl || '—') + '</td>'
            + '<td style="color:'+mc+';font-weight:600">' + (r.mp || '—') + '</td>'
            + '<td style="text-align:right">' + r.ae + '°</td>'
            + '<td style="text-align:right">' + r.lat + ' s</td>'
            + '<td style="text-align:right">' + r.conf + '</td>'
            + '</tr>';
    }});
    html += '</table>';
    tableEl.innerHTML = html;
  }}

  slider.addEventListener('input', function() {{ render(this.value); }});
  render(slider.value);
}})();
</script>
</div></body></html>""")

        return ''.join(html_parts)

    def _display_summary(self, analysis_data):
        """Display analysis summary in Streamlit"""
        summary = analysis_data['summary']
        run_meta = analysis_data.get('run_metadata', {})
        selected_cfg_name, selected_cfg_path, selected_model_name, _ = self._extract_runtime_selection(run_meta)
        
        st.subheader("📊 Analysis Summary")
        if selected_cfg_name != 'N/A' or selected_model_name != 'N/A':
            st.caption(f"Report provenance: config={selected_cfg_name} | model={selected_model_name}")
            if selected_cfg_path:
                st.caption(f"Config path: {selected_cfg_path}")
        
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
        
        # Compute event-level detection rate from matches + render sidecar
        _all_m   = analysis_data.get('matches', [])
        _gt_m    = [m for m in _all_m if m.get('match_type') == 'ground_truth']
        _fp_m    = [m for m in _all_m if m.get('match_type') != 'ground_truth']
        _det_evt = len(set(
            (m.get('source_label', m.get('label', '')), round(m.get('gt_start', 0), 1))
            for m in _gt_m
        ))
        _total_gt = 0
        try:
            import pathlib as _pl, json as _json
            _render_f = self.base_output_dir / 'renders' / f"{analysis_data.get('render_id','')}.json"
            _total_gt = len(_json.loads(_render_f.read_text()).get('source_sidecars', []))
        except Exception:
            pass
        _fp_rate = len(_fp_m) / max(len(_all_m), 1) * 100
        has_ground_truth = bool(analysis_data.get('config', {}).get('ground_truth_name') or _gt_m or _total_gt)

        # Check if YAMNet stats are available
        has_yamnet_stats = 'yamnet_stats' in analysis_data

        if has_yamnet_stats:
            # Show YAMNet stats prominently
            yamnet_stats = analysis_data['yamnet_stats']
            st.info(f"🎯 **Using YAMNet classifications**: {yamnet_stats['yamnet_classified']} classified detections")

            if has_ground_truth:
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
            else:
                st.metric("YAMNet Classified", yamnet_stats['yamnet_classified'])
                st.caption("Ground-truth-dependent YAMNet metrics are hidden until a GT JSON is uploaded.")

            # Show samples needing fine-tuning
            needs_training = yamnet_stats.get('needs_training', 0)
            if needs_training > 0:
                st.warning(f"⚠️ **{needs_training} samples marked for YAMNet fine-tuning dataset** (mismatches, low confidence, or unclassified)")

            st.markdown("---")

        if has_ground_truth:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("ODAS Frames", summary['total_detections'],
                          help="Total ODAS detection frames (GT-matched + false positives)")
            with col2:
                if _total_gt:
                    st.metric("Events Detected", f"{_det_evt}/{_total_gt}",
                              delta=f"{_det_evt/_total_gt*100:.0f}%",
                              help="GT sound events that ODAS picked up at least once")
                else:
                    st.metric("GT Frame Coverage", f"{summary['match_rate']*100:.1f}%",
                              help="Fraction of ODAS frames that matched a GT source")
            with col3:
                st.metric("False Positive Rate", f"{_fp_rate:.0f}%",
                          help="ODAS frames not matched to any GT source")
            with col4:
                st.metric("Avg Angular Error", f"{summary['avg_angular_error']:.2f}°")
            with col5:
                st.metric("Time Span", f"{summary['time_span_seconds']:.1f}s")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ODAS Frames", summary['total_detections'],
                          help="Total ODAS detection frames parsed from the session JSON")
            with col2:
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
    
    def _render_deployment_eval(self, analysis_data: dict, run_id: str):
        """Render the Deployment Evaluation tab using compute_deployment_metrics()."""
        st.markdown("### 🚀 Deployment Evaluation Metrics")
        st.caption(
            "These metrics measure how the *deployed* classifier would perform on this run: "
            "event-level precision/recall (direction-aware) and false-positive rate."
        )

        try:
            dep = self.yamnet_curator.compute_deployment_metrics(analysis_data)
        except Exception as exc:
            st.warning(f"Could not compute deployment metrics: {exc}")
            return

        if not dep:
            st.info("No deployment metrics available for this run.")
            return

        # ── Top-line metrics ────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Event Precision", f"{dep.get('event_precision', 0):.2%}")
        with col2:
            st.metric("Event Recall", f"{dep.get('event_recall', 0):.2%}")
        with col3:
            st.metric("Event F1", f"{dep.get('event_f1', 0):.2%}")
        with col4:
            fp_min = dep.get("fp_per_min", dep.get("fp_per_minute", 0))
            st.metric("FP / min", f"{fp_min:.2f}")

        direction_acc = dep.get("correct_class_and_direction_pct", dep.get("direction_accuracy"))
        if direction_acc is not None:
            st.metric("Correct class + direction %", f"{direction_acc:.1%}")

        # ── Confusion matrix ─────────────────────────────────────────────────
        cm = dep.get("confusion_matrix")
        if cm:
            st.markdown("#### Confusion Matrix")
            import pandas as pd
            st.dataframe(pd.DataFrame(cm), width="stretch")

        # ── Per-label breakdown ──────────────────────────────────────────────
        per_label = dep.get("per_label")
        if per_label:
            st.markdown("#### Per-Label Breakdown")
            import pandas as pd
            rows = []
            for label, stats in per_label.items():
                rows.append({
                    "Label":     label,
                    "TP":        stats.get("tp", 0),
                    "FP":        stats.get("fp", 0),
                    "FN":        stats.get("fn", 0),
                    "Precision": f"{stats.get('precision', 0):.2%}",
                    "Recall":    f"{stats.get('recall', 0):.2%}",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch")

        # ── Raw JSON dump for debugging ──────────────────────────────────────
        with st.expander("📄 Raw deployment metrics JSON"):
            st.json({k: v for k, v in dep.items() if k != "confusion_matrix"})

    def _load_run_metadata(self, run_id: str, analysis_data: dict):
        """Load run JSON metadata for this analysis."""
        run_meta = analysis_data.get('run_metadata')
        if isinstance(run_meta, dict) and run_meta:
            return run_meta

        run_path = self.runs_dir / f"{run_id}.json"
        if not run_path.exists():
            return {}
        try:
            return json.loads(run_path.read_text())
        except Exception:
            return {}

    def _load_render_metadata(self, render_id: str):
        """Load render sidecar JSON from outputs/renders/{render_id}.json."""
        if not render_id:
            return {}
        render_path = self.base_output_dir / 'renders' / f"{render_id}.json"
        if not render_path.exists():
            return {}
        try:
            return json.loads(render_path.read_text())
        except Exception:
            return {}

    def _extract_runtime_selection(self, run_meta: dict):
        """Return selected ODAS config/model provenance from run metadata."""
        run_meta = run_meta or {}
        scene_meta = run_meta.get('scene_metadata', {}) or {}

        selected_cfg_path = (
            run_meta.get('selected_odas_config')
            or scene_meta.get('selected_odas_config')
            or run_meta.get('odas_config')
            or ''
        )
        if Path(str(selected_cfg_path)).name.startswith('runtime_cfg_'):
            selected_cfg_path = (
                scene_meta.get('selected_odas_config')
                or run_meta.get('selected_odas_config')
                or ''
            )
        if not selected_cfg_path:
            selected_cfg_path = run_meta.get('odas_runtime_config') or ''
        selected_cfg_name = Path(selected_cfg_path).name if selected_cfg_path else 'N/A'

        selected_model_dir = (
            run_meta.get('selected_model_dir')
            or scene_meta.get('selected_model_dir')
            or ''
        )
        selected_model_name = (
            run_meta.get('selected_model_name')
            or (Path(selected_model_dir).name if selected_model_dir else '')
            or scene_meta.get('selected_model_name')
            or 'N/A'
        )

        return selected_cfg_name, selected_cfg_path, selected_model_name, selected_model_dir

    def _extract_mono_from_raw_window(self, raw_audio_file, start_time, end_time,
                                      warmup_seconds=0.0, sr=16000, n_channels=6):
        """Extract a mono (avg of 4 mic channels) clip from interleaved raw PCM."""
        if not raw_audio_file or not os.path.exists(raw_audio_file):
            return None

        try:
            render_start = float(start_time) + float(warmup_seconds)
            render_end = float(end_time) + float(warmup_seconds)
            if render_end <= render_start:
                return None

            start_frame = int(render_start * sr)
            end_frame = int(render_end * sr)
            n_frames = max(0, end_frame - start_frame)
            if n_frames <= 0:
                return None

            bytes_per_sample = 2  # S16_LE
            start_byte = start_frame * n_channels * bytes_per_sample
            n_bytes = n_frames * n_channels * bytes_per_sample

            with open(raw_audio_file, 'rb') as f:
                f.seek(start_byte)
                raw_data = f.read(n_bytes)

            if len(raw_data) < n_channels * bytes_per_sample:
                return None

            n_frames_actual = len(raw_data) // (n_channels * bytes_per_sample)
            data = np.frombuffer(raw_data, dtype='<i2').reshape(n_frames_actual, n_channels)

            # Channels 1:5 are the 4 microphone channels in renderer.py.
            if data.shape[1] >= 5:
                mono = data[:, 1:5].mean(axis=1).astype(np.float32) / 32768.0
            else:
                mono = data.mean(axis=1).astype(np.float32) / 32768.0
            return mono
        except Exception:
            return None

    def _waveform_to_spectrogram(self, waveform, sr=16000, n_fft=512, hop=128):
        """Compute magnitude spectrogram from waveform."""
        if waveform is None or len(waveform) == 0:
            return None
        try:
            import librosa
            spec = np.abs(librosa.stft(waveform.astype(np.float32), n_fft=n_fft, hop_length=hop))
            return spec
        except Exception:
            return None

    def _plot_spectrogram(self, spec, title, sr=16000, hop=128, key=None):
        """Render a spectrogram (linear magnitude) as Plotly heatmap."""
        if spec is None or spec.size == 0:
            st.caption("No spectrogram available")
            return

        # Expect shape (freq_bins, time_frames)
        db = 20.0 * np.log10(np.maximum(spec, 1e-8))
        times = np.arange(db.shape[1]) * (hop / float(sr))
        freqs = np.linspace(0, sr / 2.0, db.shape[0])

        fig = go.Figure(
            data=go.Heatmap(
                z=db,
                x=times,
                y=freqs,
                colorscale='Viridis',
                colorbar=dict(title='dB')
            )
        )
        fig.update_layout(
            title=title,
            xaxis_title='Time (s)',
            yaxis_title='Frequency (Hz)',
            height=320,
            margin=dict(l=10, r=10, t=45, b=10)
        )
        st.plotly_chart(fig, width="stretch", key=key)

    def _plot_slice_gt_direction_radar(self, scene_file, slice_start, slice_end, key=None):
        """Plot a simple polar (radar-style) chart of GT source directions in a slice."""
        if not scene_file or not os.path.exists(scene_file):
            st.caption("GT direction radar unavailable (scene file missing).")
            return

        try:
            scene = json.loads(Path(scene_file).read_text())
        except Exception:
            st.caption("GT direction radar unavailable (could not read scene file).")
            return

        active = []
        for src in scene.get('directional_sources', []):
            try:
                s = float(src.get('start_time', 0.0))
                e = float(src.get('end_time', 0.0))
            except Exception:
                continue
            if e <= slice_start or s >= slice_end:
                continue

            x = float(src.get('x', 0.0))
            y = float(src.get('y', 0.0))
            z = float(src.get('z', 0.0))
            az, _el = self._cartesian_to_spherical(x, y, z)
            az_deg = (np.degrees(az) + 360.0) % 360.0
            dist_xy = float(np.sqrt(x * x + y * y))
            active.append({
                'label': src.get('label', 'unknown'),
                'az_deg': az_deg,
                'dist_xy': max(dist_xy, 1e-6),
            })

        if not active:
            st.caption("No active GT sources in this slice.")
            return

        max_r = max(a['dist_xy'] for a in active)
        thetas = [a['az_deg'] for a in active]
        rs = [a['dist_xy'] / max_r for a in active]
        texts = [f"{a['label']} ({a['az_deg']:.0f}°)" for a in active]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=rs,
            theta=thetas,
            mode='markers+text',
            text=texts,
            textposition='top center',
            marker=dict(size=10),
            name='GT sources'
        ))
        fig.update_layout(
            title='GT source directions (slice)',
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1.05], title='Relative distance'),
                angularaxis=dict(direction='counterclockwise', rotation=90)
            ),
            height=360,
            margin=dict(l=10, r=10, t=45, b=10),
            showlegend=False
        )
        st.plotly_chart(fig, width="stretch", key=key)

    def _extract_gt_source_clips(self, scene_file, win_start, win_end, target_sr=16000):
        """Extract source-local GT clips that overlap a selected window."""
        if not scene_file or not os.path.exists(scene_file):
            return []

        try:
            scene = json.loads(Path(scene_file).read_text())
        except Exception:
            return []

        clips = []
        for src in scene.get('directional_sources', []):
            s = float(src.get('start_time', 0.0))
            e = float(src.get('end_time', 0.0))
            if e <= win_start or s >= win_end:
                continue

            overlap_start = max(win_start, s)
            overlap_end = min(win_end, e)
            if overlap_end <= overlap_start:
                continue

            wav_path = src.get('wav_path')
            if not wav_path or not os.path.exists(wav_path):
                continue

            try:
                audio, sr = sf.read(wav_path, always_2d=False)
                if audio is None or len(audio) == 0:
                    continue
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                audio = audio.astype(np.float32)

                src_offset = max(0.0, overlap_start - s)
                src_dur = max(0.0, overlap_end - overlap_start)
                i0 = int(src_offset * sr)
                i1 = int((src_offset + src_dur) * sr)
                clip = audio[i0:i1]
                if len(clip) == 0:
                    continue

                # Optional resample to target_sr for consistent spectrogram axes.
                if int(sr) != int(target_sr):
                    import librosa
                    clip = librosa.resample(clip, orig_sr=int(sr), target_sr=int(target_sr))
                    sr = target_sr

                clips.append({
                    'label': src.get('label', 'unknown'),
                    'wav_path': wav_path,
                    'start': overlap_start,
                    'end': overlap_end,
                    'audio': clip,
                    'sample_rate': int(sr),
                })
            except Exception:
                continue

        return clips

    def _save_window_bundle(self, run_id, window_id, payload):
        """Persist per-window wav outputs and lookup JSON for offline listening."""
        out_root = self.base_output_dir / 'analysis' / 'window_audio' / run_id / window_id
        out_root.mkdir(parents=True, exist_ok=True)

        lookup = {
            'run_id': run_id,
            'window_id': window_id,
            'window': payload.get('window', {}),
            'files': []
        }

        # Save peak clips
        for i, p in enumerate(payload.get('peaks', [])):
            audio = p.get('audio')
            sr = int(p.get('sample_rate', 16000))
            if audio is None or len(audio) == 0:
                continue
            fn = out_root / f"peak_{i:02d}_t{p.get('timestamp', 0.0):.3f}.wav"
            sf.write(str(fn), np.asarray(audio, dtype=np.float32), sr)
            lookup['files'].append({'type': 'peak', 'timestamp': p.get('timestamp', 0.0), 'path': str(fn)})

        # Save rendered GT mix clip for this window
        rendered_audio = payload.get('rendered_gt_audio')
        if rendered_audio is not None and len(rendered_audio) > 0:
            fn = out_root / 'window_rendered_gt_mix.wav'
            sf.write(str(fn), np.asarray(rendered_audio, dtype=np.float32), int(payload.get('sample_rate', 16000)))
            lookup['files'].append({'type': 'rendered_gt_mix', 'path': str(fn)})

        # Save source-local GT clips used to render this window
        for i, c in enumerate(payload.get('source_clips', [])):
            audio = c.get('audio')
            if audio is None or len(audio) == 0:
                continue
            safe_lbl = str(c.get('label', 'source')).replace(' ', '_')
            fn = out_root / f"source_{i:02d}_{safe_lbl}.wav"
            sf.write(str(fn), np.asarray(audio, dtype=np.float32), int(c.get('sample_rate', 16000)))
            lookup['files'].append({
                'type': 'source_gt',
                'label': c.get('label', 'unknown'),
                'start': c.get('start'),
                'end': c.get('end'),
                'path': str(fn)
            })

        (out_root / 'lookup.json').write_text(json.dumps(lookup, indent=2))

        # Maintain run-level index for easy lookup by window.
        run_index_path = self.base_output_dir / 'analysis' / 'window_audio' / run_id / 'lookup_index.json'
        run_index = {}
        if run_index_path.exists():
            try:
                run_index = json.loads(run_index_path.read_text())
            except Exception:
                run_index = {}
        run_index[window_id] = {'lookup': str(out_root / 'lookup.json')}
        run_index_path.write_text(json.dumps(run_index, indent=2))

        return str(out_root), str(out_root / 'lookup.json')

    def _build_slice_timeline_figure(self, windows, peaks, slice_start=None, slice_end=None):
        """Build a compact timeline showing GT windows and peak timestamps."""
        fig = go.Figure()

        labels = sorted({str(w.get('label', 'unknown')) for w in windows})
        y_map = {label: idx + 1 for idx, label in enumerate(labels)}
        unmatched_y = 0

        for w in windows:
            label = str(w.get('label', 'unknown'))
            y = y_map.get(label, len(y_map) + 1)
            fig.add_trace(go.Scatter(
                x=[w['start'], w['end']],
                y=[y, y],
                mode='lines',
                line=dict(width=12),
                name=f"GT: {label}",
                legendgroup=f"gt_{label}",
                showlegend=False,
                hovertemplate=(
                    f"GT {label}<br>start={w['start']:.3f}s<br>end={w['end']:.3f}s<extra></extra>"
                )
            ))

        matched = [p for p in peaks if p.get('match_type') == 'ground_truth']
        unmatched = [p for p in peaks if p.get('match_type') != 'ground_truth']

        if matched:
            fig.add_trace(go.Scatter(
                x=[p['timestamp'] for p in matched],
                y=[y_map.get(str(p.get('source_label', 'unknown')), unmatched_y) for p in matched],
                mode='markers',
                marker=dict(size=8, color='#1f77b4', symbol='circle'),
                name='Matched peaks',
                hovertemplate='Matched peak<br>t=%{x:.3f}s<extra></extra>'
            ))

        if unmatched:
            fig.add_trace(go.Scatter(
                x=[p['timestamp'] for p in unmatched],
                y=[unmatched_y for _ in unmatched],
                mode='markers',
                marker=dict(size=8, color='#d62728', symbol='x'),
                name='Other peaks',
                hovertemplate='Other peak<br>t=%{x:.3f}s<extra></extra>'
            ))

        if slice_start is not None and slice_end is not None:
            fig.add_vrect(
                x0=slice_start,
                x1=slice_end,
                fillcolor='rgba(46, 204, 113, 0.18)',
                line_width=1,
                line_color='rgba(39, 174, 96, 0.7)'
            )

        tickvals = [unmatched_y] + [y_map[label] for label in labels]
        ticktext = ['other/unmatched'] + labels
        fig.update_layout(
            title='GT windows and detected peaks over time',
            xaxis_title='Time (s)',
            yaxis=dict(title='GT label', tickmode='array', tickvals=tickvals, ticktext=ticktext),
            height=360,
            margin=dict(l=10, r=10, t=45, b=10),
            legend=dict(orientation='h')
        )
        return fig

    def _render_window_explorer(self, analysis_data: dict, run_id: str):
        """Inspect per-window peak vs GT spectrograms and export window WAV bundles."""
        st.markdown("### 🪟 Peak/GT Window Explorer")
        st.caption(
            "Compare ODAS peak spectrograms against GT rendered mix and source-local GT clips "
            "for each window that contains detections."
        )

        matches = analysis_data.get('matches', []) or []
        if not matches:
            st.info("No detections found in analysis data.")
            return

        run_meta = self._load_run_metadata(run_id, analysis_data)
        render_meta = self._load_render_metadata(analysis_data.get('render_id', ''))

        source_sidecars = render_meta.get('source_sidecars', []) if isinstance(render_meta, dict) else []
        windows = []
        for i, s in enumerate(source_sidecars):
            try:
                ws = float(s.get('start_time', 0.0))
                we = float(s.get('end_time', 0.0))
                if we > ws:
                    windows.append({
                        'id': f"w{i:03d}",
                        'label': s.get('label', 'unknown'),
                        'start': ws,
                        'end': we,
                        'source_idx': s.get('source_idx', i),
                    })
            except Exception:
                continue

        # Fallback window list from GT bounds stored in matches.
        if not windows:
            seen = set()
            for m in matches:
                gs = m.get('gt_start')
                ge = m.get('gt_end')
                if gs is None or ge is None:
                    continue
                key = (m.get('source_label', 'unknown'), round(float(gs), 3), round(float(ge), 3))
                if key in seen:
                    continue
                seen.add(key)
                windows.append({
                    'id': f"w{len(windows):03d}",
                    'label': key[0],
                    'start': float(gs),
                    'end': float(ge),
                    'source_idx': None,
                })

        if not windows:
            st.warning("No GT windows available (render sidecar missing and no GT bounds in matches).")
            return

        # Detections/peaks across the run. Include all peaks; some may not have
        # spectra sidecars, in which case they still appear in tables/timeline
        # but won't have spectrogram/audio reconstruction available.
        peaks = []
        for m in matches:
            ts = m.get('timestamp', m.get('detection', {}).get('timestamp'))
            if ts is None:
                continue
            sf_path = m.get('spectra_file', m.get('detection', {}).get('spectra_file', ''))
            frame_count = int(m.get('frame_count', m.get('detection', {}).get('frame_count', 0) or 0))
            track_start = m.get('track_start', None)
            if track_start is None:
                track_start = float(ts) - frame_count * 0.008
            peaks.append({
                'timestamp': float(ts),
                'spectra_file': sf_path,
                'has_spectra': bool(sf_path and os.path.exists(sf_path)),
                'source_label': m.get('source_label', m.get('label', 'unknown')),
                'match_type': m.get('match_type', 'unknown'),
                'model_prediction': m.get('model_prediction', ''),
                'event_votes': int(m.get('event_votes', 0)),
                'event_max_confidence': float(m.get('event_max_confidence', 0.0)),
                'frame_count': frame_count,
                'track_start': float(track_start),
                'gt_start': m.get('gt_start'),
                'gt_end': m.get('gt_end'),
            })

        if not peaks:
            st.info("No peaks/detections were found in this analysis.")
            return

        def _same_gt_window(peak, window, tol=0.05):
            if peak.get('match_type') != 'ground_truth':
                return False
            if str(peak.get('source_label', '')) != str(window.get('label', '')):
                return False
            gs = peak.get('gt_start')
            ge = peak.get('gt_end')
            if gs is None or ge is None:
                return False
            return abs(float(gs) - float(window['start'])) <= tol and abs(float(ge) - float(window['end'])) <= tol

        # Peak count per window.
        for w in windows:
            w['peak_count'] = sum(1 for p in peaks if _same_gt_window(p, w))
            w['time_window_peak_count'] = sum(1 for p in peaks if w['start'] <= p['timestamp'] <= w['end'])

        st.markdown("#### ⏱️ Time-slice explorer")
        overall_start = min([w['start'] for w in windows] + [p['timestamp'] for p in peaks])
        overall_end = max([w['end'] for w in windows] + [p['timestamp'] for p in peaks])
        default_duration = min(2.0, max(0.5, overall_end - overall_start))

        slice_duration = st.slider(
            "Slice duration (s)",
            min_value=0.25,
            max_value=max(0.25, float(max(0.25, overall_end - overall_start))),
            value=float(default_duration),
            step=0.05,
            key="slice_duration_slider"
        )
        slice_start_min = float(overall_start)
        slice_start_max = float(max(overall_start, overall_end - slice_duration))
        if slice_start_max <= slice_start_min:
            # Zero-duration sessions/ranges can make slider bounds equal.
            slice_start = slice_start_min
            st.caption(f"Slice start fixed at {slice_start:.3f}s (no selectable range).")
        else:
            slice_start = st.slider(
                "Slice start (s)",
                min_value=slice_start_min,
                max_value=slice_start_max,
                value=slice_start_min,
                step=0.05,
                key="slice_start_slider"
            )
        slice_end = min(float(overall_end), float(slice_start + slice_duration))
        st.caption(f"Selected slice: {slice_start:.3f}s → {slice_end:.3f}s")

        slice_fig = self._build_slice_timeline_figure(windows, peaks, slice_start=slice_start, slice_end=slice_end)
        st.plotly_chart(slice_fig, width="stretch")

        slice_peaks = [p for p in peaks if slice_start <= p['timestamp'] <= slice_end]
        slice_windows = [w for w in windows if not (w['end'] < slice_start or w['start'] > slice_end)]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Peaks in slice", len(slice_peaks))
        with col2:
            st.metric("GT windows in slice", len(slice_windows))
        with col3:
            st.metric("Peaks with spectra", sum(1 for p in slice_peaks if p.get('has_spectra')))

        if slice_peaks:
            peak_rows = [{
                't (s)': round(p['timestamp'], 3),
                'GT': p.get('source_label', 'unknown'),
                'match_type': p.get('match_type', 'unknown'),
                'pred': p.get('model_prediction', ''),
                'votes': p.get('event_votes', 0),
                'has_spectra': p.get('has_spectra', False),
            } for p in slice_peaks]
            st.dataframe(pd.DataFrame(peak_rows), width='stretch')
        else:
            st.info("No peaks fall inside the selected slice.")

        st.markdown("#### 📦 Slice records")

        scene_meta = run_meta.get('scene_metadata', {}) if isinstance(run_meta, dict) else {}
        raw_audio_file = run_meta.get('raw_audio_file', '') if isinstance(run_meta, dict) else ''
        warmup_s = float(run_meta.get('warmup_seconds', scene_meta.get('warmup_seconds', 0.0))) if isinstance(run_meta, dict) else 0.0
        sr = int(scene_meta.get('sample_rate', 16000)) if isinstance(scene_meta, dict) else 16000
        n_channels = int(scene_meta.get('n_channels', 6)) if isinstance(scene_meta, dict) else 6
        scene_file = run_meta.get('scene_file') if isinstance(run_meta, dict) else None

        if not hasattr(self, '_window_reconstructor'):
            self._window_reconstructor = AudioReconstructor(sample_rate=16000, n_fft=512, hop_length=128)
        recon = self._window_reconstructor

        slice_render_audio = self._extract_mono_from_raw_window(
            raw_audio_file, slice_start, slice_end,
            warmup_seconds=warmup_s, sr=sr, n_channels=n_channels
        )
        slice_render_spec = self._waveform_to_spectrogram(slice_render_audio, sr=sr)
        slice_source_clips = self._extract_gt_source_clips(scene_file, slice_start, slice_end, target_sr=sr)

        st.markdown("#### ✅ Clear slice overview")
        st.caption(
            "Read top-to-bottom: peak list for this slice, one GT rendered mix for the whole slice, "
            "then all peak cards, then all GT source cards."
        )

        sum_col1, sum_col2 = st.columns([1.4, 1])
        with sum_col1:
            if slice_peaks:
                overview_rows = [{
                    'Peak t (s)': round(p['timestamp'], 3),
                    'Associated GT': p.get('source_label', 'unknown'),
                    'Match type': p.get('match_type', 'unknown'),
                    'Predicted class': p.get('model_prediction') or 'n/a',
                    'Votes': p.get('event_votes', 0),
                    'Has spectra': 'yes' if p.get('has_spectra') else 'no',
                } for p in slice_peaks]
                st.dataframe(pd.DataFrame(overview_rows), width='stretch', hide_index=True)
            else:
                st.info("No peaks in the selected slice.")
        with sum_col2:
            gt_rows = [{
                'GT label': w['label'],
                'Start (s)': round(max(slice_start, w['start']), 3),
                'End (s)': round(min(slice_end, w['end']), 3),
                'Matched peaks': sum(1 for p in slice_peaks if _same_gt_window(p, w)),
            } for w in slice_windows]
            if gt_rows:
                st.dataframe(pd.DataFrame(gt_rows), width='stretch', hide_index=True)
            else:
                st.info("No GT windows overlap this slice.")

        st.markdown("**GT rendered mix for the entire selected slice**")
        self._plot_spectrogram(slice_render_spec, "GT rendered mix for selected slice", sr=sr, key=f"slice_render_{slice_start:.3f}_{slice_end:.3f}")
        if slice_render_audio is not None and len(slice_render_audio) > 0:
            st.audio(slice_render_audio, sample_rate=sr)

        st.markdown("**GT source direction radar for this slice**")
        self._plot_slice_gt_direction_radar(
            scene_file=scene_file,
            slice_start=slice_start,
            slice_end=slice_end,
            key=f"slice_gt_radar_{slice_start:.3f}_{slice_end:.3f}"
        )

        def _render_peak_card(container, peak, card_key):
            with container:
                st.markdown(
                    f"**Peak @ {peak['timestamp']:.3f}s**  \\n"
                    f"GT: {peak.get('source_label', 'unknown')} · Pred: {peak.get('model_prediction') or 'n/a'} · Votes: {peak.get('event_votes', 0)}"
                )
                if peak.get('has_spectra'):
                    try:
                        raw = np.fromfile(peak['spectra_file'], dtype=np.float32)
                        n_frames = raw.size // 257
                        if n_frames > 0:
                            spec = raw[:n_frames * 257].reshape(n_frames, 257).T
                            self._plot_spectrogram(spec, f"Peak {peak['timestamp']:.3f}s", key=f"{card_key}_spec")
                        rr = recon.reconstruct_from_spectra_file(peak['spectra_file'])
                        if rr is not None and rr.get('audio') is not None and len(rr.get('audio')) > 0:
                            st.audio(rr['audio'], sample_rate=16000)
                    except Exception:
                        st.caption("Peak spectrogram/audio unavailable.")
                else:
                    st.caption("No ODAS spectra sidecar for this peak.")

        def _render_source_card(container, clip, card_key):
            with container:
                st.markdown(
                    f"**{clip['label']}**  \\n"
                    f"{clip['start']:.3f}s → {clip['end']:.3f}s"
                )
                clip_spec = self._waveform_to_spectrogram(clip['audio'], sr=clip['sample_rate'])
                self._plot_spectrogram(clip_spec, f"GT source: {clip['label']}", sr=clip['sample_rate'], key=f"{card_key}_spec")
                st.audio(clip['audio'], sample_rate=clip['sample_rate'])

        st.markdown("**All peaks in this slice**")
        if slice_peaks:
            peak_cols_per_row = 2
            for row_start in range(0, len(slice_peaks), peak_cols_per_row):
                cols = st.columns(peak_cols_per_row)
                for offset, peak in enumerate(slice_peaks[row_start:row_start + peak_cols_per_row]):
                    _render_peak_card(cols[offset], peak, f"slice_peak_{row_start+offset}_{slice_start:.3f}_{slice_end:.3f}")
        else:
            st.caption("No peaks to display for this slice.")

        st.markdown("**All GT source clips in this slice**")
        if slice_source_clips:
            src_cols_per_row = 2
            for row_start in range(0, len(slice_source_clips), src_cols_per_row):
                cols = st.columns(src_cols_per_row)
                for offset, clip in enumerate(slice_source_clips[row_start:row_start + src_cols_per_row]):
                    _render_source_card(cols[offset], clip, f"slice_src_{row_start+offset}_{slice_start:.3f}_{slice_end:.3f}")
        else:
            st.caption("No GT source clips overlap this slice.")

        with st.expander("Advanced diagnostic views", expanded=False):

            def _render_peaks_for_record(record_peaks, key_prefix):
                if not record_peaks:
                    st.caption("No peaks in this record.")
                    return
                for i, p in enumerate(record_peaks):
                    st.markdown(
                        f"Peak {i+1}: t={p['timestamp']:.3f}s · GT={p.get('source_label', 'unknown')} · "
                        f"pred={p.get('model_prediction') or 'n/a'} · votes={p.get('event_votes', 0)}"
                    )
                    if p.get('has_spectra'):
                        try:
                            raw = np.fromfile(p['spectra_file'], dtype=np.float32)
                            n_frames = raw.size // 257
                            if n_frames > 0:
                                spec = raw[:n_frames * 257].reshape(n_frames, 257).T
                                self._plot_spectrogram(spec, f"{key_prefix} peak {i+1}", key=f"{key_prefix}_peak_{i}_spec")
                                rr = recon.reconstruct_from_spectra_file(p['spectra_file'])
                                if rr is not None and rr.get('audio') is not None and len(rr.get('audio')) > 0:
                                    st.audio(rr['audio'], sample_rate=16000)
                        except Exception:
                            st.caption("Peak spectrogram reconstruction failed.")
                    else:
                        st.caption("No ODAS spectra sidecar available for this peak.")

            if not slice_windows and slice_peaks:
                st.info("This slice has peaks but no overlapping GT window; treat these as ambient/unmatched detections.")

            for idx, w in enumerate(slice_windows):
                rec_start = max(slice_start, w['start'])
                rec_end = min(slice_end, w['end'])
                record_peaks = [
                    p for p in slice_peaks
                    if _same_gt_window(p, w) or (w['start'] <= p['timestamp'] <= w['end'] and p.get('source_label') == w.get('label'))
                ]
                with st.expander(
                    f"Record {idx+1}: {w['label']} · {rec_start:.3f}s → {rec_end:.3f}s · peaks={len(record_peaks)}",
                    expanded=(idx == 0)
                ):
                    _render_peaks_for_record(record_peaks, f"record_{idx}")

                    st.markdown("**GT rendered mix for this record**")
                    rec_render_audio = self._extract_mono_from_raw_window(
                        raw_audio_file, rec_start, rec_end,
                        warmup_seconds=warmup_s, sr=sr, n_channels=n_channels
                    )
                    rec_render_spec = self._waveform_to_spectrogram(rec_render_audio, sr=sr)
                    self._plot_spectrogram(rec_render_spec, f"Rendered GT mix: {w['label']}", sr=sr, key=f"record_{idx}_render_mix")
                    if rec_render_audio is not None and len(rec_render_audio) > 0:
                        st.audio(rec_render_audio, sample_rate=sr)

                    st.markdown("**Overlapping source clips for this record**")
                    record_source_clips = self._extract_gt_source_clips(scene_file, rec_start, rec_end, target_sr=sr)
                    if not record_source_clips:
                        st.caption("No GT source clips overlap this record.")
                    else:
                        for j, clip in enumerate(record_source_clips):
                            st.markdown(
                                f"Source {j+1}: {clip['label']} · {clip['start']:.3f}s → {clip['end']:.3f}s"
                            )
                            clip_spec = self._waveform_to_spectrogram(clip['audio'], sr=clip['sample_rate'])
                            self._plot_spectrogram(clip_spec, f"Source clip: {clip['label']}", sr=clip['sample_rate'], key=f"record_{idx}_source_{j}_{clip['label']}")
                            st.audio(clip['audio'], sample_rate=clip['sample_rate'])

            unmatched_slice_peaks = [
                p for p in slice_peaks
                if not any(_same_gt_window(p, w) for w in slice_windows)
            ]
            if unmatched_slice_peaks:
                with st.expander(
                    f"Record: ambient/unmatched · peaks={len(unmatched_slice_peaks)}",
                    expanded=False
                ):
                    _render_peaks_for_record(unmatched_slice_peaks, 'ambient')

        show_empty = st.checkbox("Show windows without peaks", value=False)
        filtered = windows if show_empty else [w for w in windows if w['peak_count'] > 0]
        if not filtered:
            st.info("No windows with peaks under current filter.")
            return

        selected = st.selectbox(
            "Window",
            filtered,
            format_func=lambda w: (
                f"{w['id']} | {w['label']} | {w['start']:.2f}s–{w['end']:.2f}s "
                f"| matched_peaks={w['peak_count']}"
            )
        )

        win_start, win_end = selected['start'], selected['end']
        include_other_time_peaks = st.checkbox(
            "Include unrelated peaks that happen in the same time window",
            value=False,
            help="Off by default: only peaks matched to this exact GT event are shown. Turn on to also inspect other peaks that occur during the same time span."
        )

        matched_peaks_for_window = [p for p in peaks if _same_gt_window(p, selected)]
        time_window_peaks = [p for p in peaks if win_start <= p['timestamp'] <= win_end]
        peaks_in_window = time_window_peaks if include_other_time_peaks else matched_peaks_for_window
        peaks_in_window.sort(key=lambda p: p['timestamp'])

        st.markdown(
            f"**Window:** {selected['id']} · **Label:** {selected['label']} · "
            f"**Range:** {win_start:.2f}s → {win_end:.2f}s · "
            f"**Matched peaks:** {len(matched_peaks_for_window)} · "
            f"**All peaks in time span:** {len(time_window_peaks)}"
        )

        if not include_other_time_peaks and not peaks_in_window:
            st.info("No peaks were spatially/temporally matched to this exact GT event. Turn on the checkbox above to inspect all peaks that occurred during the same time range.")
            return

        # Peak-centric view: choose one peak and show only assets associated
        # with that peak interval.
        st.markdown("#### 🎯 Selected peak details")
        sel_peak = st.selectbox(
            "Peak",
            peaks_in_window,
            format_func=lambda p: (
                f"t={p['timestamp']:.3f}s | GT={p['source_label']} | "
                f"pred={p['model_prediction'] or 'n/a'} | votes={p['event_votes']}"
            ),
            key=f"peak_sel_{selected['id']}"
        )

        scene_meta = run_meta.get('scene_metadata', {}) if isinstance(run_meta, dict) else {}
        raw_audio_file = run_meta.get('raw_audio_file', '') if isinstance(run_meta, dict) else ''
        warmup_s = float(run_meta.get('warmup_seconds', scene_meta.get('warmup_seconds', 0.0))) if isinstance(run_meta, dict) else 0.0
        sr = int(scene_meta.get('sample_rate', 16000)) if isinstance(scene_meta, dict) else 16000
        n_channels = int(scene_meta.get('n_channels', 6)) if isinstance(scene_meta, dict) else 6
        scene_file = run_meta.get('scene_file') if isinstance(run_meta, dict) else None

        peak_ts = float(sel_peak['timestamp'])
        peak_track_start = float(sel_peak.get('track_start', peak_ts))
        assoc_start = max(win_start, peak_track_start)
        assoc_end = min(win_end, peak_ts)
        if assoc_end <= assoc_start:
            # Fallback to a small fixed context if interval is degenerate.
            assoc_start = max(win_start, peak_ts - 0.96)
            assoc_end = min(win_end, peak_ts)

        st.caption(
            f"Associated interval for selected peak: {assoc_start:.3f}s → {assoc_end:.3f}s "
            f"(track_start={peak_track_start:.3f}s, peak={peak_ts:.3f}s)"
        )

        # Selected peak spectrogram/audio
        if not hasattr(self, '_window_reconstructor'):
            self._window_reconstructor = AudioReconstructor(sample_rate=16000, n_fft=512, hop_length=128)
        recon = self._window_reconstructor

        if sel_peak.get('has_spectra'):
            raw_sel = np.fromfile(sel_peak['spectra_file'], dtype=np.float32)
            n_sel = raw_sel.size // 257
            if n_sel > 0:
                spec_sel = raw_sel[:n_sel * 257].reshape(n_sel, 257).T
                self._plot_spectrogram(spec_sel, f"Selected peak spectrogram ({n_sel} frames)", key=f"selected_peak_{selected['id']}_{peak_ts:.3f}")
                try:
                    rr_sel = recon.reconstruct_from_spectra_file(sel_peak['spectra_file'])
                    if rr_sel is not None and rr_sel.get('audio') is not None and len(rr_sel.get('audio')) > 0:
                        st.audio(rr_sel['audio'], sample_rate=16000)
                except Exception:
                    pass
        else:
            st.caption("Selected peak has no ODAS spectra sidecar, so only metadata is available.")

        # GT rendered mix associated with selected peak interval
        st.markdown("**Associated GT rendered mix (selected peak interval)**")
        assoc_render_audio = self._extract_mono_from_raw_window(
            raw_audio_file, assoc_start, assoc_end,
            warmup_seconds=warmup_s, sr=sr, n_channels=n_channels
        )
        assoc_render_spec = self._waveform_to_spectrogram(assoc_render_audio, sr=sr)
        self._plot_spectrogram(assoc_render_spec, "Associated GT rendered mix", sr=sr, key=f"assoc_render_{selected['id']}_{peak_ts:.3f}")
        if assoc_render_audio is not None and len(assoc_render_audio) > 0:
            st.audio(assoc_render_audio, sample_rate=sr)

        # Individual GT sounds associated with selected peak interval
        st.markdown("**Individual GT source clips associated with selected peak**")
        associated_source_clips = self._extract_gt_source_clips(
            scene_file, assoc_start, assoc_end, target_sr=sr
        )
        if not associated_source_clips:
            st.caption("No GT source overlaps this selected peak interval.")
        else:
            for i, clip in enumerate(associated_source_clips):
                st.markdown(
                    f"GT source {i+1}: {clip['label']} ({clip['start']:.3f}s→{clip['end']:.3f}s)"
                )
                clip_spec = self._waveform_to_spectrogram(clip['audio'], sr=clip['sample_rate'])
                self._plot_spectrogram(clip_spec, f"Associated GT source: {clip['label']}", sr=clip['sample_rate'], key=f"assoc_src_{selected['id']}_{i}_{clip['label']}")
                st.audio(clip['audio'], sample_rate=clip['sample_rate'])

        max_peaks = st.slider("Max peaks to display", min_value=1, max_value=20, value=min(6, max(1, len(peaks_in_window))))
        show_peaks = peaks_in_window[:max_peaks]

        # 1) Spectrogram for each peak
        st.markdown("#### 1) ODAS peak spectrogram(s)")
        if not hasattr(self, '_window_reconstructor'):
            self._window_reconstructor = AudioReconstructor(sample_rate=16000, n_fft=512, hop_length=128)
        recon = self._window_reconstructor

        payload_peaks = []
        for i, p in enumerate(show_peaks):
            st.markdown(
                f"**Peak {i+1}** · t={p['timestamp']:.3f}s · GT={p['source_label']} · "
                f"Pred={p['model_prediction'] or 'n/a'} · votes={p['event_votes']}"
            )

            peak_audio = None
            if p.get('has_spectra'):
                raw = np.fromfile(p['spectra_file'], dtype=np.float32)
                n_frames = raw.size // 257
                if n_frames > 0:
                    spec_frames = raw[:n_frames * 257].reshape(n_frames, 257)
                    spec = spec_frames.T  # (257, T)
                    self._plot_spectrogram(spec, f"Peak {i+1} spectrogram ({n_frames} frames)", key=f"window_{selected['id']}_peak_{i}_spec")
                try:
                    rr = recon.reconstruct_from_spectra_file(p['spectra_file'])
                    if rr is not None:
                        peak_audio = rr.get('audio')
                except Exception:
                    peak_audio = None
            else:
                st.caption("No ODAS spectra sidecar available for this peak.")

            if peak_audio is not None and len(peak_audio) > 0:
                st.audio(peak_audio, sample_rate=16000)

            payload_peaks.append({
                'timestamp': p['timestamp'],
                'audio': peak_audio,
                'sample_rate': 16000
            })

        # 2) Spectrogram of GT rendered audio in that window
        st.markdown("#### 2) GT rendered mix spectrogram (window) ")
        scene_meta = run_meta.get('scene_metadata', {}) if isinstance(run_meta, dict) else {}
        raw_audio_file = run_meta.get('raw_audio_file', '') if isinstance(run_meta, dict) else ''
        warmup_s = float(run_meta.get('warmup_seconds', scene_meta.get('warmup_seconds', 0.0))) if isinstance(run_meta, dict) else 0.0
        sr = int(scene_meta.get('sample_rate', 16000)) if isinstance(scene_meta, dict) else 16000
        n_channels = int(scene_meta.get('n_channels', 6)) if isinstance(scene_meta, dict) else 6

        rendered_gt_audio = self._extract_mono_from_raw_window(
            raw_audio_file, win_start, win_end, warmup_seconds=warmup_s, sr=sr, n_channels=n_channels
        )
        rendered_spec = self._waveform_to_spectrogram(rendered_gt_audio, sr=sr)
        self._plot_spectrogram(rendered_spec, "GT rendered mix (raw render window)", sr=sr, key=f"window_{selected['id']}_render_mix")
        if rendered_gt_audio is not None and len(rendered_gt_audio) > 0:
            st.audio(rendered_gt_audio, sample_rate=sr)
        else:
            st.caption("Rendered GT window audio is unavailable (run metadata/raw file missing).")

        # 3) Spectrogram of source-local GT audio used to render this window
        st.markdown("#### 3) Source-local GT audio used in this window")
        scene_file = run_meta.get('scene_file') if isinstance(run_meta, dict) else None
        gt_source_clips = self._extract_gt_source_clips(scene_file, win_start, win_end, target_sr=sr)
        if not gt_source_clips:
            st.caption("No overlapping source-local GT clips found for this window.")
        else:
            for i, clip in enumerate(gt_source_clips):
                st.markdown(
                    f"**GT clip {i+1}** · {clip['label']} · {clip['start']:.2f}s→{clip['end']:.2f}s"
                )
                clip_spec = self._waveform_to_spectrogram(clip['audio'], sr=clip['sample_rate'])
                self._plot_spectrogram(clip_spec, f"GT source clip: {clip['label']}", sr=clip['sample_rate'], key=f"window_{selected['id']}_gtclip_{i}_{clip['label']}")
                st.audio(clip['audio'], sample_rate=clip['sample_rate'])

        # Export WAV bundle + lookup JSON for this window.
        if st.button("💾 Export WAV bundle for this window"):
            out_dir, lookup_path = self._save_window_bundle(
                run_id=run_id,
                window_id=selected['id'],
                payload={
                    'window': selected,
                    'peaks': payload_peaks,
                    'rendered_gt_audio': rendered_gt_audio,
                    'source_clips': gt_source_clips,
                    'sample_rate': sr,
                }
            )
            st.success(f"Saved window bundle to: {out_dir}")
            st.caption(f"Lookup JSON: {lookup_path}")

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
