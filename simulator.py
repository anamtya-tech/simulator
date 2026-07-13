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

        odas_root_candidates = [
            Path(os.getenv('ODAS_ROOT', '')) if os.getenv('ODAS_ROOT') else None,
            Path.home() / 'chatak-odas',
            self.base_output_dir.parent.parent / 'chatak-odas',
        ]
        odas_root_candidates = [p for p in odas_root_candidates if p is not None]
        self.odas_root = next((p for p in odas_root_candidates if p.exists()), odas_root_candidates[0])
        self.odas_build_dir = Path(os.getenv('ODAS_BUILD_DIR', str(self.odas_root / 'build')))

        # Paths
        self.socket_emit_script = os.getenv('ODAS_SOCKET_EMIT_SCRIPT', str(self.odas_root / 'vm_socket_emit.py'))
        self.odas_config = os.getenv('ODAS_CONFIG_PATH', str(self.odas_root / 'config' / 'runtime' / 'local_socket.cfg'))
        self.odaslive_bin = os.getenv('ODASLIVE_BIN', str(self.odas_build_dir / 'bin' / 'odaslive'))

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
        
        # ── Experiment settings ───────────────────────────────────────────────
        with st.expander("🧪 Experiment Settings", expanded=True):
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                preset_name = st.selectbox(
                    "ODAS SST Preset",
                    list(SST_PRESETS.keys()),
                    index=0,
                    help=(
                        "Quick-start configuration for ODAS Sound Source Tracking (SST).  "
                        "Each preset is a named combination of the 7 parameters below — "
                        "expand **Advanced** to inspect or override individual values.\n\n"
                        "**Balanced (default)** — `N_prob=6 / theta_prob=0.65 / Pnew=0.06` — "
                        "best for general dataset collection; ~0.63 FP/s.\n\n"
                        "**High-Recall** — `N_prob=3 / theta_prob=0.60 / Pnew=0.15` — "
                        "catches short/quiet events at the cost of more FPs (~1.3/s); "
                        "use *only* for building training datasets (EXP-A/B).\n\n"
                        "**Low-FP (deployment test)** — `N_prob=8 / theta_prob=0.75 / Pnew=0.03` — "
                        "suppresses geometry-driven hotspot FPs; higher miss rate; "
                        "use to simulate live-device conditions (EXP-C/D)."
                    )
                )
                preset = SST_PRESETS[preset_name]
                st.caption(f"ℹ️ {preset['description']}")
            with exp_col2:
                experiment_tag = st.text_input(
                    "Experiment tag (optional)",
                    placeholder="e.g. exp_b4_hard_negatives",
                    help=(
                        "Free-text tag written into the run JSON and propagated to datasets/models. "
                        "Use the tags from docs/experiments.md so results are traceable."
                    )
                )

            # ── Advanced per-parameter overrides ─────────────────────────────
            with st.expander("⚙️ Advanced — override individual SST parameters"):
                st.caption(
                    "Values are pre-filled from the chosen preset. "
                    "Any change here overrides the preset for this run only — the preset name is "
                    "still saved in the run JSON for traceability."
                )
                adv_col1, adv_col2 = st.columns(2)
                with adv_col1:
                    ov_Pnew = st.slider(
                        "Pnew — track-birth probability",
                        0.01, 0.30, float(preset["Pnew"]), 0.01,
                        help=(
                            "Probability threshold for spawning a *new* tracked source. "
                            "Lower → more new tracks born (higher recall, more FPs). "
                            "Higher → only strong candidates become tracks (fewer FPs, miss quiet sources)."
                        )
                    )
                    ov_theta_new = st.slider(
                        "theta_new — spatial gate for new tracks",
                        0.40, 0.99, float(preset["theta_new"]), 0.01,
                        help=(
                            "Angular coherence required before a candidate SSL peak is promoted "
                            "to a new SST track. Higher → stricter spatial gate."
                        )
                    )
                    ov_Pfalse = st.slider(
                        "Pfalse — false-source probability",
                        0.01, 0.30, float(preset["Pfalse"]), 0.01,
                        help=(
                            "Prior probability that any given observation is a false alarm. "
                            "Higher → tracker is more sceptical, prunes more aggressively."
                        )
                    )
                    ov_gainMin = st.slider(
                        "gainMin — minimum beamformer gain",
                        0.10, 0.90, float(preset["gainMin"]), 0.05,
                        help=(
                            "Minimum gain applied to the beamformed output for each tracked source. "
                            "Lower → more aggressive noise suppression; may clip quiet signals. "
                            "Higher → more signal retained but also more residual noise in the clip."
                        )
                    )
                with adv_col2:
                    ov_N_prob = st.slider(
                        "N_prob — confirmation frames",
                        1, 15, int(preset["N_prob"]), 1,
                        help=(
                            "Number of consecutive ODAS frames a candidate must remain active "
                            "before being confirmed as a real source. "
                            "Higher → fewer FPs but misses short events. "
                            "Lower → catches brief sounds but increases FP rate."
                        )
                    )
                    ov_theta_prob = st.slider(
                        "theta_prob — confirmation coherence",
                        0.40, 0.99, float(preset["theta_prob"]), 0.01,
                        help=(
                            "Minimum spatial coherence required during the N_prob confirmation "
                            "window. Higher → stricter confirmation gate."
                        )
                    )
                    ov_theta_inactive = st.slider(
                        "theta_inactive — track-death threshold",
                        0.40, 0.99, float(preset["theta_inactive"]), 0.01,
                        help=(
                            "A confirmed track is killed when its coherence drops below this "
                            "value for several frames. Higher → tracks die sooner after the "
                            "source stops (less trailing FP activity)."
                        )
                    )

                # Build override dict — passed to _run_simulation
                sst_overrides = {
                    "Pnew":           ov_Pnew,
                    "theta_new":      ov_theta_new,
                    "N_prob":         ov_N_prob,
                    "theta_prob":     ov_theta_prob,
                    "Pfalse":         ov_Pfalse,
                    "gainMin":        ov_gainMin,
                    "theta_inactive": ov_theta_inactive,
                }

        # Port configuration
        port = st.number_input("Socket Port", 10000, 20000, 10000, 1)
        
        # Run simulation
        col1, col2 = st.columns(2)
        with col1:
            run_button = st.button("▶️ Run Simulation", type="primary")
        with col2:
            stop_button = st.button("⏹️ Stop Simulation", type="secondary")
        
        if run_button:
            self._run_simulation(str(selected_raw_file), port, metadata,
                                 preset_name=preset_name,
                                 sst_overrides=sst_overrides,
                                 experiment_tag=experiment_tag.strip() or None)
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
    
    def _apply_sst_preset(self, preset: dict):
        """Patch the SST parameters in the ODAS config file to match the selected preset."""
        import re as _re
        cfg_path = self.odas_config
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

    def _run_simulation(self, raw_file_path, port, metadata,
                        preset_name: str = "Balanced (default)",
                        sst_overrides: dict | None = None,
                        experiment_tag: str | None = None):
        """Start the ODAS simulation — processes run in a background thread so
        Streamlit re-renders cannot kill them prematurely."""

        sim = _get_sim_state()
        if sim["running"]:
            st.warning("A simulation is already running.")
            return

        # ── apply SST config — overrides take precedence over preset ─────────
        base_preset = SST_PRESETS.get(preset_name, SST_PRESETS["Balanced (default)"])
        effective = {**base_preset, **(sst_overrides or {})}
        self._apply_sst_preset(effective)
        st.info(f"⚙️ ODAS SST preset applied: **{preset_name}**")

        # ── derive names / durations ────────────────────────────────────────
        render_id = metadata.get('render_id', Path(raw_file_path).stem)
        scene_name = metadata.get('scene_name', 'unknown')
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{render_id}_run_{run_timestamp}"
        warmup_seconds = metadata.get('warmup_seconds', 0)
        tail_seconds   = metadata.get('tail_silence_seconds', 0)
        duration = metadata.get('duration', 10) + warmup_seconds + tail_seconds
        log_file_path = str(self.runs_dir / f"odas_log_{run_timestamp}.txt")

        required_paths = [
            self.socket_emit_script,
            self.odas_config,
            self.odaslive_bin,
        ]
        missing = [p for p in required_paths if not Path(p).exists()]
        if missing:
            st.error("Missing ODAS runtime files. Set ODAS_ROOT/ODAS_BUILD_DIR/ODAS_CONFIG_PATH to valid paths.")
            st.code("\n".join(missing))
            return

        # ── release stale port ───────────────────────────────────────────────
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"],
                           capture_output=True, timeout=5)
            time.sleep(1)
        except Exception:
            pass

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
            return

        # ── start ODAS ───────────────────────────────────────────────────────
        run_start_time = time.time()
        odas_cmd = [
            self.odaslive_bin,
            "-v",
            "-c", self.odas_config
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
            return

        # ── persist handles in session state ─────────────────────────────────
        sim.update({
            "running": True,
            "status": "running",
            "log_lines": ["✅ Socket server started", "✅ ODAS started"],
            "socket_process": socket_process,
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
                  preset_name, experiment_tag),
            daemon=True,
        )
        monitor_thread.start()

    # ── background monitor ────────────────────────────────────────────────────
    def _monitor_background(self, sim, log_fh, run_name, render_id, scene_name,
                             metadata, raw_file_path, log_file_path,
                             run_start_time, run_timestamp, duration,
                             preset_name: str = "Balanced (default)",
                             experiment_tag: str | None = None):
        """Runs in a daemon thread — waits for socket to finish, then cleans up.
        Updates sim dict in-place so the Streamlit polling loop can display
        progress without touching the processes.
        """
        socket_process = sim["socket_process"]
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

        # ── collect output files ──────────────────────────────────────────
        time.sleep(2)
        classify_events_files = sorted(
            Path(self.odas_logs_dir).glob("sst_classify_events_*.json"),
            key=os.path.getmtime, reverse=True
        )
        session_live_files = sorted(
            Path(self.odas_logs_dir).glob("sst_session_live.json_*.json"),
            key=os.path.getmtime, reverse=True
        )
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
            'odas_config': self.odas_config,
            'warmup_seconds': metadata.get('warmup_seconds', 0),
            'odas_preset': preset_name,
            'experiment_tag': experiment_tag or '',
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
            "odas_process": None,
        })
    
    def _stop_simulation(self):
        """Stop running processes and clear session state."""
        sim = _get_sim_state()
        for key in ("socket_process", "odas_process"):
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
                run_data_list.append({
                    'Run ID': run_data.get('run_id', run_data.get('run_name', '')),
                    'Scene': run_data.get('scene_name', run_data.get('scene_metadata', {}).get('scene_name', 'Unknown')),
                    'Render ID': run_data.get('render_id', 'N/A'),
                    'Duration': f"{run_data.get('scene_metadata', {}).get('duration', 0)}s",
                    'Timestamp': run_data.get('timestamp', '')
                })
            except:
                continue
        
        if run_data_list:
            import pandas as pd
            st.dataframe(pd.DataFrame(run_data_list), width='stretch')
