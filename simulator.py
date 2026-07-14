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
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.odas_logs_dir = Path(odas_logs_dir)
        self.odas_logs_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(__file__).resolve().parent
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
        metadata = {}
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
            if cfg_options:
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

    def _prepare_runtime_cfg_for_simulation(self, cfg_path: str, port: int, run_timestamp: str) -> str:
        """Create a run-local cfg that enforces raw.interface socket replay.

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
                socket_block = (
                    "\n"
                    "        type = \"socket\";\n"
                    "        ip = \"127.0.0.1\";\n"
                    f"        port = {int(port)};\n"
                    "\n"
                )
                text = text[:match.start(2)] + socket_block + text[match.end(2):]

            # Write classifier artifacts into simulator logs directory so run metadata
            # can resolve files deterministically.
            text = re.sub(
                r'(?m)^(\s*classifier_log_dir\s*=\s*)"[^"]*"\s*;',
                rf'\1"{str(self.odas_logs_dir)}";',
                text,
                count=1,
            )

            # Force tracked sink to a dedicated local socket for simulator runs.
            tracked_if_pattern = r'(tracked\s*:\s*\{.*?interface\s*:\s*\{)(.*?)(\}\s*;)'
            tracked_if_match = re.search(tracked_if_pattern, text, flags=re.S)
            if tracked_if_match:
                tracked_block = (
                    "\n"
                    "            type = \"socket\";\n"
                    "            ip = \"127.0.0.1\";\n"
                    f"            port = {int(port) + 2};\n"
                    "\n"
                )
                text = text[:tracked_if_match.start(2)] + tracked_block + text[tracked_if_match.end(2):]

            def _remap_section_socket_port(cfg_text: str, section_name: str, desired_port: int) -> tuple[str, bool]:
                section_pattern = (
                    rf'({section_name}\s*:\s*\{{.*?interface\s*:\s*\{{.*?'
                    rf'type\s*=\s*"socket"\s*;.*?port\s*=\s*)(\d+)(\s*;)'
                )

                changed = False

                def _repl(m):
                    nonlocal changed
                    old_port = int(m.group(2))
                    # Only remap when this section collides with RAW input socket.
                    if old_port == int(port):
                        changed = True
                        return f"{m.group(1)}{desired_port}{m.group(3)}"
                    return m.group(0)

                updated = re.sub(section_pattern, _repl, cfg_text, count=1, flags=re.S)
                return updated, changed

            text, tracked_changed = _remap_section_socket_port(text, 'tracked', int(port) + 2)
            text, potential_changed = _remap_section_socket_port(text, 'potential', int(port) + 3)

            if tracked_changed or potential_changed:
                st.caption(
                    f"Adjusted ODAS socket sinks to avoid RAW port collision (RAW={int(port)}, "
                    f"tracked={int(port)+2}, potential={int(port)+3})."
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

        # Force simulator transport to socket replay, regardless of cfg capture mode.
        odas_cfg = self._prepare_runtime_cfg_for_simulation(odas_cfg_source, port, run_timestamp)

        if apply_sst_preset:
            # Overrides take precedence over preset when patching is enabled.
            base_preset = SST_PRESETS.get(preset_name, SST_PRESETS["Balanced (default)"])
            effective = {**base_preset, **(sst_overrides or {})}
            self._apply_sst_preset(odas_cfg, effective)
            st.info(f"⚙️ ODAS SST preset applied: **{preset_name}**")
        else:
            st.info(f"⚙️ Using SST settings from selected config: **{Path(odas_cfg_source).name}**")

        required_paths = [
            self.socket_emit_script,
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

        # ── release stale port ───────────────────────────────────────────────
        tracked_sink_port = int(port) + 2
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"],
                           capture_output=True, timeout=5)
            subprocess.run(["fuser", "-k", f"{tracked_sink_port}/tcp"],
                           capture_output=True, timeout=5)
            time.sleep(1)
        except Exception:
            pass

        # ── start tracked sink receiver (ODAS tracked socket consumer) ───────
        tracked_sink_process = self._start_socket_drain_server(tracked_sink_port)
        time.sleep(0.5)
        if tracked_sink_process.poll() is not None:
            st.error(f"Tracked sink receiver failed on port {tracked_sink_port}.")
            return False

        # ── start socket server ──────────────────────────────────────────────
        socket_cmd = [
            "python3",
            self.socket_emit_script,
            "--audio", raw_file_path,
            "--port", str(port)
        ]
        socket_process = subprocess.Popen(
            socket_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.odas_root)
        )
        time.sleep(2)

        if socket_process.poll() is not None:
            stdout, stderr = socket_process.communicate()
            st.error("Socket server failed to start!")
            st.code(stderr.decode())
            tracked_sink_process.terminate()
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
            socket_process.terminate()
            tracked_sink_process.terminate()
            return False

        # ── persist handles in session state ─────────────────────────────────
        sim.update({
            "running": True,
            "status": "running",
            "log_lines": [
                f"✅ Tracked sink receiver started on {tracked_sink_port}",
                "✅ Socket server started",
                "✅ ODAS started"
            ],
            "socket_process": socket_process,
            "tracked_sink_process": tracked_sink_process,
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
        """Runs in a daemon thread — waits for socket to finish, then cleans up.
        Updates sim dict in-place so the Streamlit polling loop can display
        progress without touching the processes.
        """
        socket_process = sim["socket_process"]
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

                # Check if socket finished (all audio sent)
                if socket_process.poll() is not None:
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
                if elapsed > duration * 2 + 60:
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
            # runtime copy patched for socket replay.
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
