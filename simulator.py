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
        self.odas_logs_dir = odas_logs_dir
        
        # Paths
        self.socket_emit_script = "/home/azureuser/z_odas_newbeamform/vm_socket_emit.py"
        self.odas_config = "/home/azureuser/z_odas_newbeamform/config/runtime/local_socket.cfg"
        self.odaslive_bin = "/home/azureuser/z_odas_newbeamform/build/bin/odaslive"

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
        
        # Port configuration
        port = st.number_input("Socket Port", 10000, 20000, 10000, 1)
        
        # Run simulation
        col1, col2 = st.columns(2)
        with col1:
            run_button = st.button("▶️ Run Simulation", type="primary")
        with col2:
            stop_button = st.button("⏹️ Stop Simulation", type="secondary")
        
        if run_button:
            self._run_simulation(str(selected_raw_file), port, metadata)
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
    
    def _run_simulation(self, raw_file_path, port, metadata):
        """Start the ODAS simulation — processes run in a background thread so
        Streamlit re-runs cannot kill them prematurely."""

        sim = _get_sim_state()
        if sim["running"]:
            st.warning("A simulation is already running.")
            return

        # ── derive names / durations ────────────────────────────────────────
        render_id = metadata.get('render_id', Path(raw_file_path).stem)
        scene_name = metadata.get('scene_name', 'unknown')
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{render_id}_run_{run_timestamp}"
        warmup_seconds = metadata.get('warmup_seconds', 0)
        tail_seconds   = metadata.get('tail_silence_seconds', 0)
        duration = metadata.get('duration', 10) + warmup_seconds + tail_seconds
        log_file_path = str(self.runs_dir / f"odas_log_{run_timestamp}.txt")

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
            cwd="/home/azureuser/sodas"
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
            cwd="/home/azureuser/z_odas_newbeamform/build",
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
                  run_start_time, run_timestamp, duration),
            daemon=True,
        )
        monitor_thread.start()

    # ── background monitor ────────────────────────────────────────────────────
    def _monitor_background(self, sim, log_fh, run_name, render_id, scene_name,
                             metadata, raw_file_path, log_file_path,
                             run_start_time, run_timestamp, duration):
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
