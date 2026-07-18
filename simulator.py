"""
Simulator module to run ODAS on rendered audio.

This orchestrates:
1. Starting the socket server (vm_socket_emit.py) to stream the raw audio
2. Starting ODAS (odaslive) to process the audio stream
3. Monitoring the logs and output
4. Creating a run file with metadata linking scene config to ODAS output

Output files are saved in ~/sodas/ClassifierLogs/:
- sst_classify_events_<timestamp>.json
- sst_session_live.json_<timestamp>.json

IMPORTANT: Simulation processes run in a background thread to survive
Streamlit script re-runs (caused by fastReruns=true).  Process handles
are stored in st.session_state['sim_state'] so they persist across reruns.
"""

import streamlit as st
import subprocess
import os
import time
import json
import shutil
import re
import zipfile
from pathlib import Path
from datetime import datetime
import threading
import signal

# ── session-state key used by the background thread ───────────────────────────
_SIM_STATE_KEY = "sim_state"

# ── Named ODAS SST presets ────────────────────────────────────────────────────
# These patch the key parameters into the config file before each run.
# Param descriptions: see docs/odas_sst_tuning.md
SST_PRESETS = {
    "Balanced (default)": {
        "description": "Best balance for dataset collection. N_prob=6, ~0.63 FP/s.",
        "Pnew": 0.06,
        "theta_new": 0.80,
        "N_prob": 6,
        "theta_prob": 0.65,
        "Pfalse": 0.1,
        "gainMin": 0.40,
        "theta_inactive": 0.80,
    },
    "High-Recall (dataset collection)": {
        "description": "Maximises event detection rate. More FPs (~1.3/s). Use ONLY for building training datasets.",
        "Pnew": 0.15,
        "theta_new": 0.60,
        "N_prob": 3,
        "theta_prob": 0.60,
        "Pfalse": 0.2,
        "gainMin": 0.30,
        "theta_inactive": 0.70,
    },
    "Low-FP (deployment test)": {
        "description": "Suppresses spurious tracks. Higher miss rate for short/quiet events. Use for deployment simulation.",
        "Pnew": 0.03,
        "theta_new": 0.85,
        "N_prob": 8,
        "theta_prob": 0.75,
        "Pfalse": 0.05,
        "gainMin": 0.45,
        "theta_inactive": 0.85,
    },
}

def _get_sim_state():
    """Return the sim_state dict from session_state (creates it if absent)."""
    if _SIM_STATE_KEY not in st.session_state:
        st.session_state[_SIM_STATE_KEY] = {
            "running": False,
            "status": "idle",
            "log_lines": [],
            "socket_process": None,
            "tracked_sink_process": None,
            "odas_process": None,
            "run_name": None,
            "log_file": None,
            "start_time": None,
            "duration": None,
            "elapsed": 0.0,
        }
    return st.session_state[_SIM_STATE_KEY]

class SimulationRunner:
    def __init__(self, output_dir, odas_logs_dir):
        self.base_output_dir = Path(output_dir)
        self.renders_dir = self.base_output_dir / 'renders'
        self.runs_dir = self.base_output_dir / 'runs'
        self.imported_gt_dir = self.base_output_dir / 'imported_ground_truth'
        self.mic_array_cache_dir = self.base_output_dir / 'mic_array_imports'
        self.project_root = Path(__file__).resolve().parent
        self.mic_array_root = self.project_root / 'Mic_Array'
        self.live_audio_dir = self.mic_array_root / 'Live_Audio'
        self.passive_audio_dir = self.mic_array_root / 'Passive_Audio'
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.imported_gt_dir.mkdir(parents=True, exist_ok=True)
        self.mic_array_cache_dir.mkdir(parents=True, exist_ok=True)
        self.odas_logs_dir = Path(odas_logs_dir)
        self.odas_logs_dir.mkdir(parents=True, exist_ok=True)
        self.odas_config_dir = self.project_root / 'odas_config'
        self.models_dir = self.project_root / 'models'

        odas_root_env = os.getenv('ODAS_ROOT')
        odas_root_candidates = [
            Path(odas_root_env) if odas_root_env else None,
            Path.home() / 'chatak-odas',
            self.project_root.parent / 'chatak-odas',
            self.project_root.parent.parent / 'chatak-odas',
        ]
        odas_root_candidates = [p for p in odas_root_candidates if p is not None]
        self.odas_root = next((p for p in odas_root_candidates if p.exists()), odas_root_candidates[0])

        odas_build_env = os.getenv('ODAS_BUILD_DIR')
        odas_build_candidates = [
            Path(odas_build_env) if odas_build_env else None,
            self.odas_root / 'build',
            self.odas_root / 'build-release',
            self.odas_root / 'build-debug',
        ]
        odas_build_candidates = [p for p in odas_build_candidates if p is not None]
        self.odas_build_dir = next((p for p in odas_build_candidates if p.exists()), odas_build_candidates[0])

        # Paths
        self.socket_emit_script = os.getenv('ODAS_SOCKET_EMIT_SCRIPT', str(self.odas_root / 'vm_socket_emit.py'))

        odas_cfg_env = os.getenv('ODAS_CONFIG_PATH')
        cfg_candidates = [Path(odas_cfg_env)] if odas_cfg_env else []
        cfg_candidates.extend([
            self.odas_config_dir / 'yammnetsocket.cfg',
            self.odas_config_dir / 'yammnetterminal.cfg',
            self.odas_root / 'config' / 'runtime' / 'local_socket.cfg',
        ])
        first_project_cfg = next((p for p in self._list_odas_configs() if p.exists()), None)
        if first_project_cfg is not None:
            cfg_candidates.append(first_project_cfg)
        self.odas_config = str(next((p for p in cfg_candidates if p.exists()), cfg_candidates[0]))

        odaslive_env = os.getenv('ODASLIVE_BIN')
        odaslive_candidates = [Path(odaslive_env)] if odaslive_env else []
        odaslive_candidates.extend([
            self.odas_build_dir / 'bin' / 'odaslive',
            self.odas_root / 'build' / 'bin' / 'odaslive',
            self.odas_root / 'build-release' / 'bin' / 'odaslive',
            self.odas_root / 'build-debug' / 'bin' / 'odaslive',
        ])
        odaslive_from_path = shutil.which('odaslive')
        if odaslive_from_path:
            odaslive_candidates.append(Path(odaslive_from_path))
        self.odaslive_bin = str(next((p for p in odaslive_candidates if p.exists()), odaslive_candidates[0]))

    def _list_odas_configs(self):
        """Return sorted runnable ODAS .cfg files from project odas_config directory."""
        if not self.odas_config_dir.exists():
            return []
        cfgs = [p for p in self.odas_config_dir.rglob('*') if p.is_file() and p.suffix.lower() == '.cfg']

        def _is_runnable_cfg(path: Path) -> bool:
            """Keep only top-level ODAS runtime configs (exclude helper includes)."""
            name = path.name.lower()
            # Common include/helper cfgs that are not valid odaslive entrypoints.
            if any(tok in name for tok in ('bandpass', 'postfilter', 'filter', 'diag')):
                return False
            try:
                text = path.read_text(encoding='utf-8', errors='replace')
            except Exception:
                return False
            has_raw = re.search(r'\braw\s*:\s*\{', text) is not None
            has_nbits = re.search(r'\bnBits\s*=', text) is not None
            has_iface = re.search(r'\binterface\s*:\s*\{', text) is not None
            return has_raw and has_nbits and has_iface

        cfgs = [p for p in cfgs if _is_runnable_cfg(p)]
        return sorted(cfgs, key=lambda p: p.name.lower())

    def _list_models(self):
        """Return sorted model directories from project models directory."""
        if not self.models_dir.exists():
            return []
        models = [p for p in self.models_dir.iterdir() if p.is_dir()]
        return sorted(models, key=lambda p: p.name.lower())

    # ── convenience: pull live process handles from session state ────────────
    @property
    def socket_process(self):
        return _get_sim_state().get("socket_process")

    @property
    def odas_process(self):
        return _get_sim_state().get("odas_process")
        
    def render(self):
        """Render the simulation runner interface"""
        st.subheader("ODAS Simulation")
        st.markdown("Run ODAS on rendered audio to generate peak detection data")

        sim = _get_sim_state()

        # ── If a simulation is currently running, show status and return ────
        if sim["running"]:
            self._render_running_status(sim)
            return

        # ── If the last run just finished, show a summary banner ────────────
        if sim["status"] == "done":
            st.success(f"✅ Simulation finished: **{sim['run_name']}**")
            if sim.get("log_file"):
                st.info(f"📝 Log: {sim['log_file']}")
            if st.button("Clear status"):
                sim.update({"status": "idle", "run_name": None, "log_lines": [],
                            "log_file": None, "start_time": None,
                            "duration": None, "elapsed": 0.0})
                st.rerun()

        # ── Normal launch UI ─────────────────────────────────────────────────
        use_mic_array_imports = st.toggle(
            "Use Live/Passive Mic Array ZIP (optional)",
            value=False,
            help=(
                "Enable when running ODAS on external live/passive captures. "
                "For synthetic rendered runs, keep this off."
            ),
            key="sim_use_mic_array_imports",
        )

        selected_raw_file = None
        metadata = {}
        selected_cfg_from_zip = None

        if use_mic_array_imports:
            mic_ctx = self._render_mic_array_inputs()
            if mic_ctx.get('active_session') is None:
                return
            if not mic_ctx.get('raw_path'):
                st.error("Selected session does not contain a .raw file.")
                return

            selected_raw_file = Path(mic_ctx['raw_path'])
            selected_cfg_from_zip = mic_ctx.get('cfg_path')
            metadata = {
                'render_id': mic_ctx.get('session_name', selected_raw_file.stem),
                'scene_name': mic_ctx.get('session_name', selected_raw_file.stem),
                'duration': mic_ctx.get('duration_seconds', 0.0),
                'sample_rate': 16000,
                'scene_file': mic_ctx.get('ground_truth_scene_file', ''),
                'source_type': mic_ctx.get('session_type', ''),
                'active_session': str(mic_ctx.get('active_session', '')),
                'ground_truth_source': str(mic_ctx.get('tracks_path', '')),
            }

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Session", metadata.get('scene_name', 'Unknown'))
            with col2:
                st.metric("Duration", f"{metadata.get('duration', 0):.1f}s")
            with col3:
                gt_state = "Ready" if metadata.get('scene_file') else "Missing"
                st.metric("Ground Truth", gt_state)

            if metadata.get('scene_file'):
                st.info(
                    "Using tracks JSON from the selected ZIP/folder as default ground truth. "
                    "It is parsed with the same concatenated-JSON parser logic as Results Analyzer."
                )
        else:
            # Select rendered audio
            raw_files = list(self.renders_dir.glob("*.raw"))

            if not raw_files:
                st.warning("No rendered audio found. Please render audio first.")
                return

            # Sort by modification time, newest first
            raw_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

            selected_raw_file = st.selectbox(
                "Select Rendered Audio",
                raw_files,
                format_func=lambda x: x.stem  # Show filename without extension
            )

            # Load metadata
            metadata_path = str(selected_raw_file).replace('.raw', '.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Scene", metadata.get('scene_name', 'Unknown'))
                with col2:
                    st.metric("Duration", f"{metadata.get('duration', 0)}s")
                with col3:
                    st.metric("Sample Rate", f"{metadata.get('sample_rate', 16000)} Hz")

        st.markdown("#### Runtime Model & Config")
        cfg_options = self._list_odas_configs()
        model_options = self._list_models()

        selected_cfg_from_render = metadata.get('selected_odas_config', '')
        default_cfg = Path(selected_cfg_from_render) if selected_cfg_from_render else None
        cfg_default_idx = 0
        if cfg_options and default_cfg is not None:
            for i, cfg in enumerate(cfg_options):
                try:
                    if cfg.resolve() == default_cfg.resolve():
                        cfg_default_idx = i
                        break
                except Exception:
                    continue

        selected_model_name_from_render = metadata.get('selected_model_name', '')
        model_default_idx = 0
        if model_options and selected_model_name_from_render:
            for i, model_dir in enumerate(model_options):
                if model_dir.name == selected_model_name_from_render:
                    model_default_idx = i
                    break

        runtime_col1, runtime_col2 = st.columns(2)
        with runtime_col1:
            if use_mic_array_imports and selected_cfg_from_zip:
                cfg_union = []
                seen = set()
                for candidate in [Path(selected_cfg_from_zip), *cfg_options]:
                    resolved = str(candidate.resolve()) if candidate.exists() else str(candidate)
                    if resolved in seen:
                        continue
                    cfg_union.append(candidate)
                    seen.add(resolved)

                selected_odas_cfg = st.selectbox(
                    "ODAS Config (.cfg)",
                    cfg_union,
                    index=0,
                    format_func=lambda p: p.name,
                    help="Default comes from selected ZIP/folder. Override if needed."
                )
                st.info(
                    f"Default config loaded from session ZIP/folder: {Path(selected_cfg_from_zip).name}. "
                    "You can override it here."
                )
            elif cfg_options:
                selected_odas_cfg = st.selectbox(
                    "ODAS Config (.cfg)",
                    cfg_options,
                    index=cfg_default_idx,
                    format_func=lambda p: p.name,
                    help=f"Configs from {self.odas_config_dir}"
                )
            else:
                selected_odas_cfg = Path(self.odas_config)
                st.warning(f"No .cfg files found in {self.odas_config_dir}. Using default: {self.odas_config}")

        with runtime_col2:
            if model_options:
                selected_model_dir = st.selectbox(
                    "Model Directory",
                    model_options,
                    index=model_default_idx,
                    format_func=lambda p: p.name,
                    help=f"Models from {self.models_dir}"
                )
            else:
                selected_model_dir = None
                st.warning(f"No model directories found in {self.models_dir}")
        
        # ── Experiment settings (stub) ───────────────────────────────────────
        user_selected_cfg = bool(cfg_options)
        preset_name = "Balanced (default)"
        experiment_tag = ""
        sst_overrides = None
        apply_sst_preset = not user_selected_cfg

        with st.expander("🧪 Experiment Settings", expanded=False):
            st.caption("Stub: this section is kept for future experiment workflows.")
            if user_selected_cfg:
                st.info("Using SST parameters directly from the selected ODAS config file. Preset patching is bypassed.")
            else:
                st.info("No project config file selected; preset-based SST patching remains enabled.")
                preset_name = st.selectbox(
                    "ODAS SST Preset",
                    list(SST_PRESETS.keys()),
                    index=0,
                )
                preset = SST_PRESETS[preset_name]
                st.caption(f"ℹ️ {preset['description']}")
                experiment_tag = st.text_input(
                    "Experiment tag (optional)",
                    placeholder="e.g. exp_b4_hard_negatives",
                )

        # Port configuration
        port = st.number_input("Socket Port", 10000, 20000, 10000, 1)
        
        # Run simulation
        col1, col2 = st.columns(2)
        with col1:
            run_button = st.button("▶️ Run Simulation", type="primary")
        with col2:
            stop_button = st.button("⏹️ Stop Simulation", type="secondary")
        
        if run_button:
            started = self._run_simulation(str(selected_raw_file), port, metadata,
                                           preset_name=preset_name,
                                           sst_overrides=sst_overrides,
                                           apply_sst_preset=apply_sst_preset,
                                           experiment_tag=experiment_tag.strip() or None,
                                           odas_config_path=str(selected_odas_cfg),
                                           selected_model_dir=str(selected_model_dir) if selected_model_dir else '',
                                           selected_model_name=selected_model_dir.name if selected_model_dir else '')
            if started:
                st.rerun()  # switch to the status view immediately

        if stop_button:
            self._stop_simulation()
        
        # Show previous runs
        st.subheader("Previous Runs")
        self._show_previous_runs()

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
        raw_files = sorted(session_root.rglob('*.raw'))
        txt_files = sorted(session_root.rglob('*.txt'))

        return {
            'session_root': session_root,
            'tracks_path': tracks_files[0] if tracks_files else None,
            'cfg_path': cfg_files[0] if cfg_files else None,
            'cfg_candidates': cfg_files,
            'latlong_path': latlong_files[0] if latlong_files else None,
            'raw_path': raw_files[0] if raw_files else None,
            'notes_path': next((p for p in txt_files if p not in latlong_files), None),
        }

    def _parse_concatenated_json_objects(self, text):
        """Parse JSON streams where objects are concatenated without commas/newlines."""
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
        """Parse Mic Array *_tracks.json using the same logic as Results Analyzer."""
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
                detections.append({
                    'timestamp': float(rel_ts),
                    'frame_index': frame_index,
                    'track_id': int(src.get('id', 0)),
                    'class_name': src.get('class') or 'unclassified',
                    'x': float(src.get('x', 0.0)),
                    'y': float(src.get('y', 0.0)),
                    'z': float(src.get('z', 0.0)),
                    'raw_ts': raw_ts,
                })
        return detections

    def _build_ground_truth_scene_from_tracks(self, tracks_path, session_name):
        """Convert tracks JSON into a synthetic-style scene JSON for analyzer matching."""
        detections = self._parse_mic_array_tracks(tracks_path)
        if not detections:
            return None, 0.0

        detections.sort(key=lambda d: d['timestamp'])
        grouped = {}
        for det in detections:
            key = (det['class_name'], det['track_id'])
            grouped.setdefault(key, []).append(det)

        directional_sources = []
        merge_gap_s = 0.7
        min_event_s = 0.08

        for (class_name, track_id), rows in grouped.items():
            rows.sort(key=lambda d: d['timestamp'])
            cur = None

            for row in rows:
                if cur is None:
                    cur = {
                        'start': row['timestamp'],
                        'end': row['timestamp'],
                        'sum_x': row['x'],
                        'sum_y': row['y'],
                        'sum_z': row['z'],
                        'count': 1,
                    }
                    continue

                if row['timestamp'] - cur['end'] <= merge_gap_s:
                    cur['end'] = row['timestamp']
                    cur['sum_x'] += row['x']
                    cur['sum_y'] += row['y']
                    cur['sum_z'] += row['z']
                    cur['count'] += 1
                else:
                    start = float(cur['start'])
                    end = float(max(cur['end'], start + min_event_s))
                    directional_sources.append({
                        'label': class_name if class_name else 'unknown',
                        'start_time': start,
                        'end_time': end,
                        'position': [
                            cur['sum_x'] / max(1, cur['count']),
                            cur['sum_y'] / max(1, cur['count']),
                            cur['sum_z'] / max(1, cur['count']),
                        ],
                        'track_id': track_id,
                    })
                    cur = {
                        'start': row['timestamp'],
                        'end': row['timestamp'],
                        'sum_x': row['x'],
                        'sum_y': row['y'],
                        'sum_z': row['z'],
                        'count': 1,
                    }

            if cur is not None:
                start = float(cur['start'])
                end = float(max(cur['end'], start + min_event_s))
                directional_sources.append({
                    'label': class_name if class_name else 'unknown',
                    'start_time': start,
                    'end_time': end,
                    'position': [
                        cur['sum_x'] / max(1, cur['count']),
                        cur['sum_y'] / max(1, cur['count']),
                        cur['sum_z'] / max(1, cur['count']),
                    ],
                    'track_id': track_id,
                })

        scene_payload = {
            'scene_name': session_name,
            'directional_sources': directional_sources,
            'source': 'mic_array_tracks_default_ground_truth',
        }
        approx_duration = float(detections[-1]['timestamp']) if detections else 0.0
        return scene_payload, approx_duration

    def _estimate_raw_duration_seconds(self, raw_path, n_channels: int = 4):
        """Estimate session duration from raw size using the provided channel count."""
        try:
            byte_count = Path(raw_path).stat().st_size
            sample_count = byte_count / 2.0  # int16
            frames = sample_count / max(int(n_channels or 4), 1)
            return max(0.0, frames / 16000.0)
        except Exception:
            return 0.0

    def _extract_n_channels_from_cfg(self, cfg_path: str):
        """Extract raw nChannels from an ODAS config file when available."""
        try:
            text = Path(cfg_path).read_text(encoding='utf-8', errors='replace')
        except Exception:
            return None

        match = re.search(r'(?m)^\s*nChannels\s*=\s*(\d+)\s*;', text)
        if not match:
            return None

        try:
            return int(match.group(1))
        except Exception:
            return None

    def _render_mic_array_inputs(self):
        """Render Live/Passive Mic Array selectors for simulator input mode."""
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
                key="sim_mic_array_live_session",
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
                key="sim_mic_array_passive_session",
            )
            if passive_selection is not None:
                st.caption(f"Selected: {passive_selection}")
            elif not passive_sources:
                st.caption(f"No passive sessions found in {self.passive_audio_dir}")

        if live_selection and passive_selection:
            st.error("Select either a Live Session or a Passive Session, not both.")
            return {'active_session': None}

        active_session = live_selection or passive_selection
        if active_session is None:
            return {'active_session': None}

        session_type = 'live_session' if live_selection else 'passive_session'
        discovered = self._extract_mic_array_source(active_session)
        session_name = active_session.stem if active_session.suffix.lower() == '.zip' else active_session.name

        cfg_n_channels = None
        if discovered.get('cfg_path'):
            cfg_n_channels = self._extract_n_channels_from_cfg(discovered['cfg_path'])

        tracks_path = discovered.get('tracks_path')
        gt_scene_file = ''
        duration_from_tracks = 0.0
        if tracks_path and Path(tracks_path).exists():
            try:
                scene_payload, duration_from_tracks = self._build_ground_truth_scene_from_tracks(tracks_path, session_name)
                if scene_payload:
                    gt_scene_file = str(self.imported_gt_dir / f"{session_name}_ground_truth_scene.json")
                    with open(gt_scene_file, 'w', encoding='utf-8') as f:
                        json.dump(scene_payload, f, indent=2)
            except Exception as exc:
                st.warning(f"Could not parse tracks JSON for default ground truth: {exc}")

        st.caption(f"Raw audio: {discovered.get('raw_path') or 'Not found'}")
        st.caption(f"Tracks JSON: {discovered.get('tracks_path') or 'Not found'}")
        st.caption(f"Config file: {discovered.get('cfg_path') or 'Not found'}")

        duration_seconds = duration_from_tracks
        if not duration_seconds and discovered.get('raw_path'):
            duration_seconds = self._estimate_raw_duration_seconds(
                discovered['raw_path'],
                cfg_n_channels or 4,
            )

        return {
            'active_session': active_session,
            'session_type': session_type,
            'session_name': session_name,
            'tracks_path': discovered.get('tracks_path'),
            'cfg_path': discovered.get('cfg_path'),
            'cfg_candidates': discovered.get('cfg_candidates', []),
            'n_channels': cfg_n_channels,
            'latlong_path': discovered.get('latlong_path'),
            'notes_path': discovered.get('notes_path'),
            'raw_path': str(discovered.get('raw_path')) if discovered.get('raw_path') else '',
            'ground_truth_scene_file': gt_scene_file,
            'duration_seconds': duration_seconds,
        }

    def _render_running_status(self, sim):
        """Display live status while a simulation is running in the background."""
        st.info(f"🔄 Simulation running: **{sim.get('run_name', '...')}**")

        elapsed = sim.get("elapsed", 0.0)
        audio_duration = sim.get("duration") or 1.0
        # Wall-clock time is longer than audio duration:
        #   socket streams at ~1.25× real-time + up to 90 s ODAS drain
        expected_wallclock = audio_duration * 1.25 + 90
        progress = min(elapsed / expected_wallclock, 1.0)

        st.progress(progress)
        st.caption(f"Elapsed: {elapsed:.0f}s / ~{expected_wallclock:.0f}s est.  "
                   f"({progress * 100:.1f}%)  — audio: {audio_duration:.0f}s")

        log_lines = sim.get("log_lines", [])
        if log_lines:
            st.text_area("Recent log", "\n".join(log_lines[-20:]), height=200)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("⏹️ Stop Simulation", type="secondary"):
                self._stop_simulation()
                st.rerun()
        with col2:
            if st.button("🔄 Refresh Status"):
                st.rerun()

        # Auto-refresh every 5 s while running
        time.sleep(5)
        st.rerun()
    
    def _apply_sst_preset(self, cfg_path: str, preset: dict):
        """Patch the SST parameters in the ODAS config file to match the selected preset."""
        import re as _re
        try:
            text = Path(cfg_path).read_text()
            param_map = {
                'Pnew':           str(preset['Pnew']),
                'theta_new':      str(preset['theta_new']),
                'N_prob':         str(preset['N_prob']),
                'theta_prob':     str(preset['theta_prob']),
                'Pfalse':         str(preset['Pfalse']),
                'gainMin':        str(preset['gainMin']),
                'theta_inactive': str(preset['theta_inactive']),
            }
            for param, value in param_map.items():
                # Matches: Pnew = 0.06;  (with optional spaces)
                text = _re.sub(
                    rf'({_re.escape(param)}\s*=\s*)[^;]+;',
                    rf'\g<1>{value};',
                    text
                )
            Path(cfg_path).write_text(text)
        except Exception as exc:
            st.warning(f"⚠️ Could not patch ODAS config with preset: {exc}")

    def _prepare_runtime_cfg_for_simulation(
        self,
        cfg_path: str,
        port: int,
        run_timestamp: str,
        raw_file_path: str,
        raw_n_channels: int | None = None,
        selected_model_dir: str = '',
    ) -> str:
        """Create a run-local cfg that enforces raw.interface file replay.

        The user-selected config remains the source of truth for SST and all other
        ODAS sections; only the RAW transport is normalized for simulator runs.
        """
        src_path = Path(cfg_path)
        runtime_cfg = self.runs_dir / f"runtime_cfg_{run_timestamp}.cfg"

        try:
            text = src_path.read_text(encoding='utf-8', errors='replace')

            # Guardrail: reject helper configs (e.g. bandpass.cfg) as odaslive entry cfg.
            if (
                re.search(r'\braw\s*:\s*\{', text) is None
                or re.search(r'\bnBits\s*=', text) is None
                or re.search(r'\binterface\s*:\s*\{', text) is None
            ):
                fallback = Path(self.odas_config)
                st.warning(
                    f"Selected config is not a runnable ODAS entry config: {src_path.name}. "
                    f"Falling back to {fallback.name}."
                )
                src_path = fallback
                text = src_path.read_text(encoding='utf-8', errors='replace')

            pattern = r'(raw\s*:\s*\{.*?interface\s*:\s*\{)(.*?)(\}\s*)'
            match = re.search(pattern, text, flags=re.S)

            if match:
                file_block = (
                    "\n"
                    "        type = \"file\";\n"
                    f"        path = \"{raw_file_path}\";\n"
                    "\n"
                )
                text = text[:match.start(2)] + file_block + text[match.end(2):]

            # Remap known file-system paths from device configs to local VM paths.
            local_model_dir = selected_model_dir.strip() if selected_model_dir else str(self.models_dir)
            local_bandpass = self.odas_config_dir / 'bandpass.cfg'
            local_bandpass_path = str(local_bandpass) if local_bandpass.exists() else ''

            path_overrides = {
                'model_path': local_model_dir,
                'liveRecordPath': str(self.live_audio_dir),
                'passiveRecordPath': str(self.passive_audio_dir),
                'audioRecordPath': str(self.mic_array_root),
            }
            if local_bandpass_path:
                path_overrides['bandpass'] = local_bandpass_path

            for key, value in path_overrides.items():
                text = re.sub(
                    rf'(?m)^(\s*{re.escape(key)}\s*=\s*)"[^"]*"\s*;',
                    rf'\1"{value}";',
                    text,
                    count=1,
                )

            if raw_n_channels in (4, 6):
                text = re.sub(
                    r'(?m)^(\s*nChannels\s*=\s*)\d+(\s*;)',
                    rf'\g<1>{int(raw_n_channels)}\2',
                    text,
                    count=1,
                )

                # Live/Passive imported .raw files are expected as 4-channel streams.
                # In file mode, ODAS should read channels directly as (0,1,2,3).
                if int(raw_n_channels) == 4:
                    text = re.sub(
                        r'(?m)^(\s*map\s*:\s*\()\s*[^\)]*(\)\s*;)',
                        r'\g<1>0, 1, 2, 3\2',
                        text,
                        count=1,
                    )

            def _set_section_format(cfg_text: str, section_name: str, fmt_value: str) -> str:
                section_pattern = rf'({section_name}\s*:\s*\{{)(.*?)(\n\s*\}}\s*;?)'
                m = re.search(section_pattern, cfg_text, flags=re.S)
                if not m:
                    return cfg_text
                body = m.group(2)
                body = re.sub(
                    r'(?m)^(\s*format\s*=\s*)"[^"]*"\s*;',
                    rf'\1"{fmt_value}";',
                    body,
                    count=1,
                )
                return cfg_text[:m.start(2)] + body + cfg_text[m.end(2):]

            # Keep the SST JSON stream available for post-run comparison while
            # suppressing the SSL potential stream during simulator file replay.
            text = _set_section_format(text, 'potential', 'undefined')
            text = re.sub(
                r'(potential\s*:\s*\{.*?interface\s*:\s*\{)(.*?)(\}\s*;)',
                '\\1\n            type = "blackhole";\n        \\3',
                text,
                count=1,
                flags=re.S,
            )

            # SST output stays as JSON so ODAS can emit the live session stream.
            # Terminal output is captured in the per-run log file and the
            # classifier log directory is redirected to the simulator workspace.
            text = _set_section_format(text, 'tracked', 'json')
            text = re.sub(
                r'(tracked\s*:\s*\{.*?interface\s*:\s*\{)(.*?)(\}\s*;)',
                '\\1\n             type = "terminal";\n        \\3',
                text,
                count=1,
                flags=re.S,
            )

            text = re.sub(
                r'(separated\s*:\s*\{.*?interface\s*:\s*\{)(.*?)(\}\s*;)',
                '\\1\n            type = "blackhole";\n        \\3',
                text,
                count=1,
                flags=re.S,
            )
            text = re.sub(
                r'(postfiltered\s*:\s*\{.*?interface\s*:\s*\{)(.*?)(\}\s*;)',
                '\\1\n            type = "blackhole";\n        \\3',
                text,
                count=1,
                flags=re.S,
            )

            # Write classifier artifacts into simulator logs directory so run metadata
            # can resolve files deterministically.
            text = re.sub(
                r'(?m)^(\s*classifier_log_dir\s*=\s*)"[^"]*"\s*;',
                rf'\1"{str(self.odas_logs_dir)}";',
                text,
                count=1,
            )

            runtime_cfg.write_text(text)
            return str(runtime_cfg)
        except Exception as exc:
            st.warning(f"⚠️ Could not prepare runtime socket config, using original file: {exc}")
            return str(src_path)

    def _start_socket_drain_server(self, port: int):
        """Start a tiny TCP sink that accepts and discards tracked socket output."""
        sink_code = (
            "import socket,sys\n"
            "port=int(sys.argv[1])\n"
            "s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)\n"
            "s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n"
            "s.bind(('127.0.0.1',port))\n"
            "s.listen(1)\n"
            "while True:\n"
            "    conn,_=s.accept()\n"
            "    with conn:\n"
            "        while conn.recv(65536):\n"
            "            pass\n"
        )
        return subprocess.Popen(
            ["python3", "-u", "-c", sink_code, str(int(port))],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=str(self.project_root),
        )

    def _run_simulation(self, raw_file_path, port, metadata,
                        preset_name: str = "Balanced (default)",
                        sst_overrides: dict | None = None,
                        apply_sst_preset: bool = True,
                        experiment_tag: str | None = None,
                        odas_config_path: str | None = None,
                        selected_model_dir: str = '',
                        selected_model_name: str = ''):
        """Start the ODAS simulation — processes run in a background thread so
        Streamlit re-renders cannot kill them prematurely."""

        sim = _get_sim_state()
        if sim["running"]:
            st.warning("A simulation is already running.")
            return False

        odas_cfg_source = odas_config_path or self.odas_config

        # ── derive names / durations ────────────────────────────────────────
        render_id = metadata.get('render_id', Path(raw_file_path).stem)
        scene_name = metadata.get('scene_name', 'unknown')
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{render_id}_run_{run_timestamp}"
        warmup_seconds = metadata.get('warmup_seconds', 0)
        tail_seconds   = metadata.get('tail_silence_seconds', 0)
        duration = metadata.get('duration', 10) + warmup_seconds + tail_seconds
        log_file_path = str(self.runs_dir / f"odas_log_{run_timestamp}.txt")

        # Force simulator transport to file replay using the selected raw audio.
        raw_n_channels = None
        source_type = str(metadata.get('source_type', '')).strip().lower()
        try:
            raw_n_channels = int(metadata.get('n_channels', 0)) or None
        except Exception:
            raw_n_channels = None

        # Imported Mic Array sessions are frequently 6-channel raw captures.
        # Preserve the actual channel count from the session config instead of
        # forcing 4 channels, which can desync ODAS file replay and suppress
        # classifier output artifacts.
        if raw_n_channels is None and source_type in ('live_session', 'passive_session'):
            cfg_path = metadata.get('cfg_path', '')
            if cfg_path:
                raw_n_channels = self._extract_n_channels_from_cfg(cfg_path)

        odas_cfg = self._prepare_runtime_cfg_for_simulation(
            odas_cfg_source,
            port,
            run_timestamp,
            raw_file_path,
            raw_n_channels=raw_n_channels,
            selected_model_dir=selected_model_dir,
        )

        if apply_sst_preset:
            # Overrides take precedence over preset when patching is enabled.
            base_preset = SST_PRESETS.get(preset_name, SST_PRESETS["Balanced (default)"])
            effective = {**base_preset, **(sst_overrides or {})}
            self._apply_sst_preset(odas_cfg, effective)
            st.info(f"⚙️ ODAS SST preset applied: **{preset_name}**")
        else:
            st.info(f"⚙️ Using SST settings from selected config: **{Path(odas_cfg_source).name}**")

        required_paths = [
            odas_cfg,
            self.odaslive_bin,
        ]
        missing = [p for p in required_paths if not Path(p).exists()]
        if missing:
            st.error("Missing ODAS runtime files. Set ODAS_ROOT/ODAS_BUILD_DIR/ODAS_CONFIG_PATH to valid paths.")
            st.caption(
                f"Resolved paths -> ODAS_ROOT: {self.odas_root}, "
                f"ODAS_BUILD_DIR: {self.odas_build_dir}, "
                f"ODAS_CONFIG_PATH: {odas_cfg}, "
                f"ODASLIVE_BIN: {self.odaslive_bin}"
            )
            st.code("\n".join(missing))
            return False

        # ── start ODAS ───────────────────────────────────────────────────────
        run_start_time = time.time()
        odas_cmd = [
            self.odaslive_bin,
            "-v",
            "-c", odas_cfg
        ]
        log_fh = open(log_file_path, 'w')
        odas_process = subprocess.Popen(
            odas_cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(self.odas_build_dir),
        )

        # early crash check
        time.sleep(3)
        if odas_process.poll() is not None:
            log_fh.close()
            with open(log_file_path) as _lf:
                tail = _lf.read()[-800:]
            st.error("❌ ODAS crashed during initialisation.")
            st.code(tail)
            return False

        # ── persist handles in session state ─────────────────────────────────
        sim.update({
            "running": True,
            "status": "running",
            "log_lines": [
                "✅ Raw file input mode enabled (ODAS reads file directly)",
                "✅ ODAS started"
            ],
            "socket_process": None,
            "tracked_sink_process": None,
            "odas_process": odas_process,
            "run_name": run_name,
            "log_file": log_file_path,
            "start_time": run_start_time,
            "duration": duration,
            "elapsed": 0.0,
        })

        # ── launch background monitor thread ─────────────────────────────────
        monitor_thread = threading.Thread(
            target=self._monitor_background,
            args=(sim, log_fh, run_name, render_id, scene_name,
                  metadata, raw_file_path, log_file_path,
                  run_start_time, run_timestamp, duration,
                preset_name, experiment_tag, odas_cfg_source, odas_cfg,
                selected_model_dir, selected_model_name),
            daemon=True,
        )
        monitor_thread.start()
        return True

    # ── background monitor ────────────────────────────────────────────────────
    def _monitor_background(self, sim, log_fh, run_name, render_id, scene_name,
                             metadata, raw_file_path, log_file_path,
                             run_start_time, run_timestamp, duration,
                             preset_name: str = "Balanced (default)",
                             experiment_tag: str | None = None,
                             selected_odas_cfg: str = '',
                             odas_cfg: str | None = None,
                             selected_model_dir: str = '',
                             selected_model_name: str = ''):
        """Runs in a daemon thread and cleans up when ODAS processing completes.
        Updates sim dict in-place so the Streamlit polling loop can display
        progress without touching the processes.
        """
        socket_process = sim.get("socket_process")
        tracked_sink_process = sim.get("tracked_sink_process")
        odas_process   = sim["odas_process"]
        start_time     = sim["start_time"]

        try:
            while True:
                elapsed = time.time() - start_time
                sim["elapsed"] = elapsed

                # Append a log heartbeat every ~10 s
                if int(elapsed) % 10 == 0:
                    expected = duration * 1.25 + 90
                    sim["log_lines"].append(f"⏳ {elapsed:.0f}s elapsed / ~{expected:.0f}s est. (audio: {duration:.0f}s)")

                if socket_process is None:
                    # File mode: ODAS owns end-of-file and exits by itself.
                    if odas_process.poll() is not None:
                        sim["log_lines"].append("✅ ODAS completed file processing.")
                        break
                else:
                    # Socket mode: socket completion indicates all frames sent.
                    if socket_process.poll() is not None:
                        socket_rc = socket_process.returncode
                        if socket_rc != 0:
                            out, err = socket_process.communicate()
                            sim["log_lines"].append(f"❌ Socket streamer failed (exit {socket_rc}).")
                            if err:
                                sim["log_lines"].append(err.decode(errors='replace')[-240:])
                            elif out:
                                sim["log_lines"].append(out.decode(errors='replace')[-240:])
                            break

                        # ODAS processes at ~0.79× real-time, so when the socket
                        # finishes there may still be tens of seconds of audio left
                        # in ODAS's processing queue.  Give it up to 90 s to drain.
                        sim["log_lines"].append("✅ Socket server completed — waiting up to 90 s for ODAS to drain...")
                        for _ in range(90):
                            if odas_process.poll() is not None:
                                break
                            time.sleep(1)
                        break

                # Hard timeout: the socket streams at 10 ms/frame so it takes
                # ~1.25× the audio duration.  Add a generous margin on top.
                if elapsed > duration * 1.35 + 120:
                    sim["log_lines"].append("⏱️ Hard timeout reached")
                    break

                time.sleep(1)

            # ── graceful ODAS shutdown ────────────────────────────────────
            sim["log_lines"].append("⏹️ Stopping ODAS...")
            time.sleep(2)
            try:
                odas_process.terminate()
                odas_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                odas_process.kill()
                odas_process.wait()

        finally:
            log_fh.close()
            if tracked_sink_process:
                try:
                    tracked_sink_process.terminate()
                    tracked_sink_process.wait(timeout=3)
                except Exception:
                    try:
                        tracked_sink_process.kill()
                    except Exception:
                        pass

        # ── collect output files ──────────────────────────────────────────
        time.sleep(2)
        log_dirs = [
            Path(self.odas_logs_dir),
            self.odas_build_dir / 'ClassifierLogs',
            self.odas_root / 'build' / 'ClassifierLogs',
        ]

        classify_events_files = []
        session_live_files = []
        for d in log_dirs:
            if d.exists():
                classify_events_files.extend(d.glob("sst_classify_events_*.json"))
                session_live_files.extend(d.glob("sst_session_live.json_*.json"))

        classify_events_files = sorted(classify_events_files, key=os.path.getmtime, reverse=True)
        session_live_files = sorted(session_live_files, key=os.path.getmtime, reverse=True)

        # Prefer files produced during/after this run.
        classify_events_files = [p for p in classify_events_files if os.path.getmtime(p) >= run_start_time] or classify_events_files
        session_live_files = [p for p in session_live_files if os.path.getmtime(p) >= run_start_time] or session_live_files

        classify_events_file = str(classify_events_files[0]) if classify_events_files else None
        session_live_file    = str(session_live_files[0])    if session_live_files    else None

        if session_live_file and os.path.getmtime(session_live_file) < run_start_time:
            sim["log_lines"].append(
                f"⚠️ Session file is stale (mtime predates run start). "
                f"ODAS may have crashed."
            )
            session_live_file = None

        # ── save run JSON ─────────────────────────────────────────────────
        run_data = {
            'run_id': run_name,
            'render_id': render_id,
            'scene_name': scene_name,
            'timestamp': run_timestamp,
            'raw_audio_file': raw_file_path,
            'scene_metadata': metadata,
            'scene_file': metadata.get('scene_file', None),
            'odas_log_file': log_file_path,
            'classify_events_file': classify_events_file,
            'session_live_file': session_live_file,
            'port': 10000,
            # Keep both paths: the config the user selected and the run-local
            # runtime copy patched for direct file replay.
            'odas_config': selected_odas_cfg,
            'selected_odas_config': selected_odas_cfg,
            'selected_odas_config_name': Path(selected_odas_cfg).name if selected_odas_cfg else '',
            'odas_runtime_config': odas_cfg or self.odas_config,
            'warmup_seconds': metadata.get('warmup_seconds', 0),
            'odas_preset': preset_name,
            'experiment_tag': experiment_tag or '',
            'selected_model_dir': selected_model_dir,
            'selected_model_name': selected_model_name,
        }
        run_file_path = str(self.runs_dir / f"{run_name}.json")
        with open(run_file_path, 'w') as f:
            json.dump(run_data, f, indent=2)

        sim["log_lines"].append(f"✅ Run file saved: {run_file_path}")
        sim["log_lines"].append("✅ Simulation complete!")

        # Mark as done so the UI shows the summary banner
        sim.update({
            "running": False,
            "status": "done",
            "socket_process": None,
            "tracked_sink_process": None,
            "odas_process": None,
        })
    
    def _stop_simulation(self):
        """Stop running processes and clear session state."""
        sim = _get_sim_state()
        for key in ("socket_process", "tracked_sink_process", "odas_process"):
            proc = sim.get(key)
            if proc:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            sim[key] = None

        sim.update({"running": False, "status": "idle"})
        st.info("Processes stopped")

    def _free_port(self, port):
        """Kill any process holding the given TCP port."""
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"],
                           capture_output=True, timeout=5)
            time.sleep(0.5)
        except Exception:
            pass
    
    def _show_previous_runs(self):
        """Display previous simulation runs"""
        run_files = sorted(
            Path(self.runs_dir).glob("*.json"),
            key=os.path.getmtime,
            reverse=True
        )
        
        if not run_files:
            st.info("No previous runs found")
            return
        
        # Show recent runs in table
        run_data_list = []
        for run_file in run_files[:10]:  # Show last 10
            try:
                with open(run_file, 'r') as f:
                    run_data = json.load(f)

                odas_cfg_path = (
                    run_data.get('selected_odas_config')
                    or run_data.get('odas_config', '')
                )
                if Path(str(odas_cfg_path)).name.startswith('runtime_cfg_'):
                    odas_cfg_path = run_data.get('scene_metadata', {}).get('selected_odas_config', '')
                cfg_name = Path(odas_cfg_path).name if odas_cfg_path else ''

                model_name = (
                    run_data.get('selected_model_name')
                    or run_data.get('scene_metadata', {}).get('selected_model_name')
                    or (Path(run_data.get('selected_model_dir', '')).name if run_data.get('selected_model_dir') else '')
                    or 'N/A'
                )

                run_data_list.append({
                    'Run ID': run_data.get('run_id', run_data.get('run_name', '')),
                    'Scene': run_data.get('scene_name', run_data.get('scene_metadata', {}).get('scene_name', 'Unknown')),
                    'Render ID': run_data.get('render_id', 'N/A'),
                    'Config': cfg_name or 'N/A',
                    'Model': model_name,
                    'Duration': f"{run_data.get('scene_metadata', {}).get('duration', 0)}s",
                    'Timestamp': run_data.get('timestamp', '')
                })
            except:
                continue
        
        if run_data_list:
            import pandas as pd
            st.dataframe(pd.DataFrame(run_data_list), width='stretch')
