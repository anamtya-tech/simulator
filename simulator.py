"""
Simulator module to run ODAS on rendered audio.

This orchestrates:
1. Starting the socket server (vm_socket_emit.py) to stream the raw audio
2. Starting ODAS (odaslive) to process the audio stream
3. Monitoring the logs and output
4. Creating a run file with metadata linking scene config to ODAS output

Output files are saved in z_odas/ClassifierLogs/:
- sst_classify_events_<timestamp>.json
- sst_session_live.json_<timestamp>.json
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

class SimulationRunner:
    def __init__(self, output_dir, odas_logs_dir):
        self.base_output_dir = Path(output_dir)
        self.renders_dir = self.base_output_dir / 'renders'
        self.runs_dir = self.base_output_dir / 'runs'
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.odas_logs_dir = odas_logs_dir
        
        # Paths
        self.socket_emit_script = "/home/azureuser/sodas/vm_socket_emit.py"
        self.odas_config = "/home/azureuser/sodas/local_socket.cfg"
        self.odaslive_bin = "/home/azureuser/z_odas_newbeamform/build/bin/odaslive"
        #self.odaslive_bin = "/home/azureuser/z_odas/build/bin/odaslive"
        
        # Process handles
        self.socket_process = None
        self.odas_process = None
        
    def render(self):
        """Render the simulation runner interface"""
        st.subheader("ODAS Simulation")
        st.markdown("Run ODAS on rendered audio to generate peak detection data")
        
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
        
        # Port configuration
        port = st.number_input("Socket Port", 10000, 20000, 10000, 1)
        
        # Run simulation
        col1, col2 = st.columns(2)
        with col1:
            run_button = st.button("▶️ Run Simulation", type="primary")
        with col2:
            stop_button = st.button("⏹️ Stop Simulation", type="secondary")
        
        if run_button:
            self._run_simulation(str(selected_raw_file), port, metadata if os.path.exists(metadata_path) else {})
        
        if stop_button:
            self._stop_simulation()
        
        # Show previous runs
        st.subheader("Previous Runs")
        self._show_previous_runs()
    
    def _run_simulation(self, raw_file_path, port, metadata):
        """Run the ODAS simulation"""
        st.info("Starting simulation...")
        
        # Get render_id from metadata, or create from filename
        render_id = metadata.get('render_id', Path(raw_file_path).stem)
        scene_name = metadata.get('scene_name', 'unknown')
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{render_id}_run_{run_timestamp}"
        
        try:
            # Start socket server in background
            st.write("🔌 Starting socket server...")
            socket_cmd = [
                "python3",
                self.socket_emit_script,
                "--audio", raw_file_path,
                "--port", str(port)
            ]
            
            # Start socket server
            self.socket_process = subprocess.Popen(
                socket_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="/home/azureuser/sodas"
            )
            
            # Wait a bit for socket to start
            time.sleep(2)
            
            # Check if socket process is still running
            if self.socket_process.poll() is not None:
                stdout, stderr = self.socket_process.communicate()
                st.error("Socket server failed to start!")
                st.code(stderr.decode())
                return
            
            st.write("✅ Socket server started")
            
            # Start ODAS
            st.write("🎵 Starting ODAS...")
            log_file_path = str(self.runs_dir / f"odas_log_{run_timestamp}.txt")
            
            odas_cmd = [
                self.odaslive_bin,
                "-v",  # Verbose mode
                "-c", self.odas_config
            ]
            
            with open(log_file_path, 'w') as log_file:
                self.odas_process = subprocess.Popen(
                    odas_cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd="/home/azureuser/z_odas_newbeamform/build",
                    # Commented out: allows core dumps for debugging
                    # preexec_fn=lambda: __import__('resource').setrlimit(
                    #     __import__('resource').RLIMIT_CORE, (0, 0)
                    # )  # Disable core dumps for this process
                )
            
            st.write("✅ ODAS started")
            st.write(f"📝 Logs: {log_file_path}")
            
            # Monitor processes
            st.write("⏳ Waiting for simulation to complete...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            duration = metadata.get('duration', 10)
            start_time = time.time()
            
            while True:
                elapsed = time.time() - start_time
                progress = min(elapsed / duration, 1.0)
                progress_bar.progress(progress)
                status_text.text(f"Elapsed: {elapsed:.1f}s / {duration}s")
                
                # Check if socket process finished
                if self.socket_process and self.socket_process.poll() is not None:
                    st.write("✅ Socket server completed")
                    # Give ODAS extra time to process buffered audio and write outputs
                    st.write("⏳ Waiting for ODAS to process remaining audio (5 seconds)...")
                    time.sleep(5)
                    break
                
                if elapsed > duration + 5:  # Give 5 extra seconds
                    st.write("⏱️ Timeout reached")
                    break
                
                time.sleep(0.5)
            
            # Stop ODAS gracefully
            st.write("⏹️ Stopping ODAS...")
            if self.odas_process:
                # Give ODAS extra time to flush final JSON outputs
                st.write("💾 Flushing ODAS outputs (2 seconds)...")
                time.sleep(2)
                # Use terminate (SIGTERM) for graceful shutdown
                self.odas_process.terminate()
                try:
                    self.odas_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # If it doesn't stop gracefully, force kill
                    st.write("⚠️ Force stopping ODAS...")
                    self.odas_process.kill()
                    self.odas_process.wait()
            
            # Find output files
            st.write("🔍 Looking for output files...")
            time.sleep(2)  # Wait for files to be written
            
            # Find the most recent classifier logs
            classify_events_files = sorted(
                Path(self.odas_logs_dir).glob("sst_classify_events_*.json"),
                key=os.path.getmtime,
                reverse=True
            )
            session_live_files = sorted(
                Path(self.odas_logs_dir).glob("sst_session_live.json_*.json"),
                key=os.path.getmtime,
                reverse=True
            )
            
            classify_events_file = str(classify_events_files[0]) if classify_events_files else None
            session_live_file = str(session_live_files[0]) if session_live_files else None
            
            if not session_live_file:
                st.warning("⚠️ Could not find session output file")
            else:
                st.success("✅ Found output files!")
                if classify_events_file:
                    st.write(f"- Classify events: {Path(classify_events_file).name}")
                st.write(f"- Session live: {Path(session_live_file).name}")
            
            # Create run file
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
                'port': port,
                'odas_config': self.odas_config
            }
            
            run_file_path = str(self.runs_dir / f"{run_name}.json")
            with open(run_file_path, 'w') as f:
                json.dump(run_data, f, indent=2)
            
            st.success(f"✅ Simulation complete!")
            st.info(f"Run ID: {run_name}")
            st.info(f"Run file: {run_file_path}")
            
            # Store in session state
            st.session_state.last_run_file = run_file_path
            
        except Exception as e:
            st.error(f"Error running simulation: {e}")
            import traceback
            st.code(traceback.format_exc())
        finally:
            # Cleanup
            self._stop_simulation()
    
    def _stop_simulation(self):
        """Stop running processes"""
        if self.socket_process:
            try:
                self.socket_process.terminate()
                self.socket_process.wait(timeout=2)
            except:
                try:
                    self.socket_process.kill()
                except:
                    pass
            self.socket_process = None
        
        if self.odas_process:
            try:
                self.odas_process.terminate()
                self.odas_process.wait(timeout=2)
            except:
                try:
                    self.odas_process.kill()
                except:
                    pass
            self.odas_process = None
        
        st.info("Processes stopped")
    
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
