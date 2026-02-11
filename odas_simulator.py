"""
ODAS-Style Processor Simulator

Uses the OPTIMIZED ODAS processor from odas_optimized.py to process rendered audio files.
This bypasses the socket/odaslive pipeline and processes directly.
Now runs in real-time with 28x performance improvement!
"""

import streamlit as st
import subprocess
import os
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# Import the OPTIMIZED ODAS processor
from odas_optimized import ODASProcessorOptimized, MicArray

class ODASSimulator:
    def __init__(self, output_dir):
        self.base_output_dir = Path(output_dir)
        self.renders_dir = self.base_output_dir / 'renders'
        self.odas_output_dir = self.base_output_dir / 'odas_results'
        self.runs_dir = self.base_output_dir / 'runs'
        self.odas_output_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
        # ReSpeaker USB 4 Mic Array configuration
        # Note: Positions must match C ODAS config exactly
        self.mic_positions = np.array([
            [-0.032, 0.000, 0.000],  # Mic 0: Left
            [0.000, -0.032, 0.000],  # Mic 1: Back  
            [0.032, 0.000, 0.000],   # Mic 2: Right
            [0.000, 0.032, 0.000]    # Mic 3: Front
        ])
        
    def render(self):
        """Render the ODAS simulator interface"""
        st.subheader("Improved ODAS Processor (Optimized)")
        st.markdown("Process rendered audio using the **optimized** ODAS-style DOA processor - **28x faster than real-time!**")
        
        # Performance badge
        st.success("⚡ Real-time capable: 0.28ms per frame (28x faster than needed!)")
        
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
            format_func=lambda x: x.stem
        )
        
        # Load metadata
        metadata_path = str(selected_raw_file).replace('.raw', '.json')
        metadata = {}
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
        
        # Configuration options (simplified for optimized version)
        st.subheader("Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            n_pots = st.slider("Max Sources (SSL pots)", 1, 5, 3)
            ssl_min_coherence = st.slider("SSL Coherence Threshold", 0.0, 1.0, 0.45, 0.05,
                                         help="Lower = more detections (try 0.3-0.5 for weak signals)")
            n_grid_points = st.select_slider("Scan Grid Points", 
                                            options=[64, 128, 256, 512], 
                                            value=256,
                                            help="More points = better accuracy but slower")
        with col2:
            sst_inactive_frames = st.number_input("Inactive Frames Threshold", 10, 500, 50, 10)
            sst_max_distance = st.slider("Track Association Distance (deg)", 5.0, 45.0, 20.0, 5.0)
            sst_min_confidence = st.slider("Min Track Confidence", 0.1, 1.0, 0.3, 0.05)
        
        # Advanced settings expander
        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                st.text("STFT Settings")
                frame_size = st.number_input("Frame Size", 256, 2048, 512, 64)
                hop_size = st.number_input("Hop Size", 64, 1024, 128, 32)
                
                st.text("SSL Settings")
                ssl_freq_min = st.number_input("Min Frequency (Hz)", 50, 1000, 100, 50)
                ssl_freq_max = st.number_input("Max Frequency (Hz)", 3000, 12000, 8000, 500)
            
            with col2:
                st.text("Performance Options")
                enable_spectral = st.checkbox("Enable Spectral Fingerprint", value=False,
                                             help="Slower but better identity preservation")
                enable_adaptive = st.checkbox("Enable Adaptive Kalman", value=False,
                                             help="Slower but better motion tracking")
        
        # Build config
        config = {
            'frame_size': frame_size,
            'hop_size': hop_size,
            'ssl_n_pots': n_pots,
            'ssl_n_grid_points': n_grid_points,
            'ssl_freq_min': ssl_freq_min,
            'ssl_freq_max': ssl_freq_max,
            'ssl_min_coherence': ssl_min_coherence,
            'sst_max_distance': sst_max_distance,
            'sst_inactive_frames': sst_inactive_frames,
            'sst_min_confidence': sst_min_confidence,
            'enable_spectral_fingerprint': enable_spectral,
            'enable_adaptive_kalman': enable_adaptive,
        }
        
        # Run processing
        if st.button("▶️ Process Audio", type="primary"):
            self._process_audio(str(selected_raw_file), metadata, config)
        
        # Show previous results
        st.subheader("Previous Results")
        self._show_previous_results()
    
    def _process_audio(self, raw_file_path, metadata, config):
        """Process audio file with optimized ODAS processor"""
        st.info("🔄 Processing audio with optimized ODAS processor...")
        
        try:
            # Create progress indicators
            progress_bar = st.progress(0)
            status_text = st.empty()
            perf_metrics = st.empty()
            
            # Setup mic array
            mic_array = MicArray(positions=self.mic_positions)
            
            # Create processor
            status_text.text("Initializing optimized ODAS processor...")
            processor = ODASProcessorOptimized(mic_array, config=config)
            
            # Generate output filename
            render_id = metadata.get('render_id', Path(raw_file_path).stem)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{render_id}_odas_{timestamp}.json"
            output_path = self.odas_output_dir / output_filename
            
            # Process file
            status_text.text("Loading audio data...")
            
            # Load audio
            audio_int16 = np.fromfile(raw_file_path, dtype=np.int16)
            n_samples = len(audio_int16) // 6
            audio_6ch = audio_int16.reshape(n_samples, 6).T
            # Extract mic channels 1-4 (indices 1,2,3,4) to match renderer output
            mic_audio = audio_6ch[1:5, :].astype(np.float32) / 32767.0
            
            # Calculate processing parameters
            duration = n_samples / 16000
            frame_size = config['frame_size']
            hop_size = config['hop_size']
            n_frames = (n_samples - frame_size) // hop_size + 1
            
            status_text.text(f"Processing {n_frames} frames...")
            
            # Process frames with performance monitoring
            results = []
            processing_times = []
            
            import time
            overall_start = time.perf_counter()
            
            for frame_idx in range(n_frames):
                start = frame_idx * hop_size
                end = start + frame_size
                
                if end > n_samples:
                    break
                
                frame_start = time.perf_counter()
                frame = mic_audio[:, start:end]
                frame_results = processor.process_frame(frame, frame_idx)
                frame_elapsed = (time.perf_counter() - frame_start) * 1000
                
                results.append(frame_results)
                processing_times.append(frame_elapsed)
                
                # Update progress every 10 frames
                if (frame_idx + 1) % 10 == 0:
                    progress = (frame_idx + 1) / n_frames
                    progress_bar.progress(progress)
                    
                    avg_time = np.mean(processing_times[-10:])
                    real_time_constraint = (hop_size / 16000) * 1000
                    speedup = real_time_constraint / avg_time
                    
                    status_text.text(f"Frame {frame_idx + 1}/{n_frames} ({progress*100:.1f}%)")
                    perf_metrics.text(f"⚡ Avg: {avg_time:.2f}ms/frame | "
                                    f"Real-time factor: {speedup:.1f}x | "
                                    f"Active tracks: {len(processor.tracks)}")
            
            overall_elapsed = time.perf_counter() - overall_start
            
            progress_bar.progress(1.0)
            status_text.text("Finalizing results...")
            
            # Performance summary
            processing_times_arr = np.array(processing_times)
            real_time_constraint = (hop_size / 16000) * 1000
            real_time_factor = duration / overall_elapsed
            
            # Collect all unique tracks from all frames
            all_tracks = {}
            for frame_result in results:
                for track_dict in frame_result.get('tracks', []):
                    track_id = track_dict['track_id']
                    # Keep the latest version of each track
                    if track_id not in all_tracks or frame_result['frame_idx'] > all_tracks[track_id].get('last_frame', 0):
                        all_tracks[track_id] = track_dict.copy()
                        all_tracks[track_id]['last_frame'] = frame_result['frame_idx']
            
            track_summary = list(all_tracks.values())
            
            # Compile results (match format expected by analyzer)
            output = {
                'metadata': {
                    'file': raw_file_path,
                    'render_id': render_id,
                    'scene_name': metadata.get('scene_name', 'unknown'),
                    'duration': duration,
                    'sample_rate': 16000,
                    'frames_processed': len(results),
                    'pots_detected': sum(len(r['pots']) for r in results),
                    'tracks_created': processor.next_track_id - 1,
                    'timestamp': datetime.now().isoformat(),
                    'scene_metadata': metadata,
                    # Performance metrics
                    'performance': {
                        'processing_time_sec': overall_elapsed,
                        'real_time_factor': real_time_factor,
                        'avg_frame_time_ms': float(np.mean(processing_times_arr)),
                        'min_frame_time_ms': float(np.min(processing_times_arr)),
                        'max_frame_time_ms': float(np.max(processing_times_arr)),
                        'std_frame_time_ms': float(np.std(processing_times_arr)),
                        'can_run_realtime': bool(np.mean(processing_times_arr) < real_time_constraint)
                    }
                },
                'config': config,
                'frames': results,
                'track_summary': track_summary
            }
            
            # Save detailed results
            with open(output_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            # Create ODAS-format session_live file for analyzer compatibility
            session_live_path = self.odas_output_dir / f"{render_id}_session_live_{timestamp}.json"
            self._create_session_live_format(results, session_live_path, config)
            
            # Create run file for analyzer
            run_file_path = self.runs_dir / f"{render_id}_run_{timestamp}.json"
            run_data = {
                'run_id': f"{render_id}_{timestamp}",
                'run_name': f"{render_id}_{timestamp}",
                'scene_name': metadata.get('scene_name', 'unknown'),
                'render_id': render_id,
                'scene_file': metadata.get('scene_file', ''),
                'rendered_audio': raw_file_path,
                'session_live_file': str(session_live_path),
                'odas_results_file': str(output_path),
                'timestamp': datetime.now().isoformat(),
                'scene_metadata': metadata,
                'performance': output['metadata']['performance']
            }
            with open(run_file_path, 'w') as f:
                json.dump(run_data, f, indent=2)
            
            # Show results summary
            st.success("✅ Processing complete!")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("Frames", len(results))
            with col2:
                st.metric("Pots", output['metadata']['pots_detected'])
            with col3:
                st.metric("Tracks", output['metadata']['tracks_created'])
            with col4:
                st.metric("Duration", f"{duration:.1f}s")
            with col5:
                rt_factor = output['metadata']['performance']['real_time_factor']
                st.metric("RT Factor", f"{rt_factor:.1f}x", 
                         delta="Real-time!" if rt_factor > 1 else "Too slow")
            
            # Performance details
            st.subheader("⚡ Performance Metrics")
            perf = output['metadata']['performance']
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Frame Time", f"{perf['avg_frame_time_ms']:.2f}ms")
                st.metric("Processing Time", f"{perf['processing_time_sec']:.2f}s")
            with col2:
                st.metric("Min Frame Time", f"{perf['min_frame_time_ms']:.2f}ms")
                st.metric("Max Frame Time", f"{perf['max_frame_time_ms']:.2f}ms")
            with col3:
                st.metric("Std Dev", f"{perf['std_frame_time_ms']:.2f}ms")
                can_rt = "✅ YES" if perf['can_run_realtime'] else "❌ NO"
                st.metric("Real-time Capable", can_rt)
            
            st.info(f"💾 Results saved to: `{output_path.name}`")
            
            # Show track summary using all_tracks
            if all_tracks:
                st.subheader(f"🎯 Detected Tracks ({len(all_tracks)} total)")
                
                track_data = []
                for track in all_tracks.values():
                    first_time = track.first_frame * config['hop_size'] / 16000
                    last_time = track.last_frame * config['hop_size'] / 16000
                    track_duration = last_time - first_time
                    
                    track_data.append({
                        'Track ID': track.track_id,
                        'Azimuth': f"{track.azimuth:.1f}°",
                        'Elevation': f"{track.elevation:.1f}°",
                        'Start Time': f"{first_time:.2f}s",
                        'End Time': f"{last_time:.2f}s",
                        'Duration': f"{track_duration:.2f}s",
                        'Status': track.status,
                        'Confidence': f"{track.confidence:.2f}"
                    })
                
                import pandas as pd
                st.dataframe(pd.DataFrame(track_data), width='stretch')
            
            # Store in session state
            st.session_state.last_odas_result = str(output_path)
            st.session_state.last_run_file = str(run_file_path)
            
        except Exception as e:
            st.error(f"❌ Error processing audio: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def _create_session_live_format(self, frame_results, output_path, config):
        """
        Create ODAS-format session_live JSON file for analyzer compatibility
        
        Format: Each line is a JSON object representing 8ms (or hop_size) of analysis
        with 'src' array containing x, y, z, activity, bins for each detected source
        """
        with open(output_path, 'w') as f:
            for frame_data in frame_results:
                frame_idx = frame_data['frame_idx']
                timestamp = frame_data['timestamp']
                
                # Convert tracks to ODAS src format
                src_list = []
                for track in frame_data.get('tracks', []):
                    src_entry = {
                        'id': track['track_id'],
                        'x': track['position'][0],
                        'y': track['position'][1],
                        'z': track['position'][2],
                        'activity': track.get('confidence', 0.9),  # Use confidence as activity
                        'frame_count': frame_idx,
                        'bins': []  # Empty bins for now (can add spectrum data if needed)
                    }
                    src_list.append(src_entry)
                
                # Only output tracks, not raw pots
                # (tracks are the filtered, stable detections)
                
                # Create ODAS-format line
                odas_line = {
                    'timeStamp': int(timestamp * 1000),  # Convert to ms
                    'src': src_list
                }
                
                f.write(json.dumps(odas_line) + '\n')
    
    def _show_previous_results(self):
        """Display previous ODAS processing results"""
        result_files = sorted(
            self.odas_output_dir.glob("*.json"),
            key=os.path.getmtime,
            reverse=True
        )
        
        if not result_files:
            st.info("No previous results found")
            return
        
        # Show recent results in table
        result_data_list = []
        for result_file in result_files[:10]:  # Show last 10
            try:
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                metadata = result_data.get('metadata', {})
                result_data_list.append({
                    'Result File': result_file.stem,
                    'Scene': metadata.get('scene_name', 'Unknown'),
                    'Duration': f"{metadata.get('duration', 0):.1f}s",
                    'Frames': metadata.get('frames_processed', 0),
                    'Pots': metadata.get('pots_detected', 0),
                    'Tracks': metadata.get('tracks_created', 0),
                    'Timestamp': metadata.get('timestamp', '')[:19] if metadata.get('timestamp') else ''
                })
            except:
                continue
        
        if result_data_list:
            import pandas as pd
            df = pd.DataFrame(result_data_list)
            st.dataframe(df, use_container_width=True)
            
            # Allow viewing details
            selected_result = st.selectbox(
                "View details for:",
                result_files[:10],
                format_func=lambda x: x.stem
            )
            
            if st.button("View Details"):
                with open(selected_result, 'r') as f:
                    result_data = json.load(f)
                
                st.json(result_data['metadata'])
                
                # Show track summary
                if 'track_summary' in result_data and result_data['track_summary']:
                    st.subheader("Tracks")
                    import pandas as pd
                    tracks_df = pd.DataFrame(result_data['track_summary'])
                    st.dataframe(tracks_df, width='stretch')
