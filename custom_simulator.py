"""
Custom DOA Simulator Module

Provides UI for running custom DOA processing and comparing with SODAS output.
"""

import streamlit as st
import json
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from custom_doa_processor import CustomDOAProcessor, ProcessingConfig, process_audio_file


class CustomSimulator:
    def __init__(self, output_dir, renders_dir):
        self.base_output_dir = Path(output_dir)
        self.renders_dir = Path(renders_dir)
        self.custom_output_dir = self.base_output_dir / 'custom_doa'
        self.custom_output_dir.mkdir(parents=True, exist_ok=True)
        
    def render(self):
        """Render the custom simulator interface"""
        st.subheader("🔬 Custom DOA Processor")
        st.markdown("""
        Process rendered audio with custom phase-based direction-of-arrival estimation.
        Compare results with SODAS/ODAS output.
        """)
        
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
        
        # Processing configuration
        st.subheader("⚙️ Processing Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            frame_size = st.selectbox("Frame Size (samples)", 
                                     [64, 128, 256, 512], 
                                     index=1,
                                     help="8ms = 128 samples @ 16kHz")
            
            confidence_window = st.slider("Confidence Window (frames)", 
                                         3, 20, 8,
                                         help="Number of frames for temporal validation")
            
            confidence_threshold = st.slider("Confidence Threshold", 
                                           0.3, 1.0, 0.625, 0.05,
                                           help="Minimum ratio of frames to validate detection")
        
        with col2:
            min_snr = st.slider("Min Peak SNR (dB)", 
                               5.0, 30.0, 12.0, 1.0,
                               help="Minimum signal-to-noise ratio for peak detection")
            
            freq_range = st.slider("Frequency Range (Hz)", 
                                  100, 8000, (200, 6000), 100,
                                  help="Frequency range to analyze")
            
            direction_tolerance = st.slider("Direction Tolerance (degrees)", 
                                          5.0, 45.0, 20.0, 5.0,
                                          help="Angular tolerance for matching detections")
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            with col1:
                fft_size = st.selectbox("FFT Size", [128, 256, 512, 1024], index=1)
                hop_size = st.number_input("Hop Size (samples)", 
                                          16, 256, frame_size // 2, 16)
            with col2:
                peak_prominence = st.slider("Peak Prominence (dB)", 
                                           1.0, 10.0, 3.0, 0.5)
                frequency_tolerance = st.slider("Frequency Tolerance (Hz)", 
                                              50, 500, 100, 25)
        
        # Create config
        config = ProcessingConfig(
            sample_rate=metadata.get('sample_rate', 16000),
            frame_size=frame_size,
            hop_size=hop_size,
            fft_size=fft_size,
            min_peak_snr=min_snr,
            min_frequency=freq_range[0],
            max_frequency=freq_range[1],
            peak_prominence=peak_prominence,
            confidence_window_size=confidence_window,
            confidence_threshold=confidence_threshold,
            direction_tolerance=direction_tolerance,
            frequency_tolerance=frequency_tolerance
        )
        
        # Run processing
        col1, col2 = st.columns([2, 1])
        with col1:
            if st.button("▶️ Run Custom DOA Processing", type="primary"):
                self._run_processing(str(selected_raw_file), config, metadata)
        
        # Show previous results
        st.subheader("📊 Previous Results")
        self._show_previous_results()
        
        # Comparison with SODAS
        st.subheader("🔄 Compare with SODAS")
        self._show_comparison_interface()
    
    def _run_processing(self, raw_file_path: str, config: ProcessingConfig, metadata: dict):
        """Run custom DOA processing"""
        st.info("🔄 Processing audio...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Generate output filename
            render_id = metadata.get('render_id', Path(raw_file_path).stem)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{render_id}_custom_{timestamp}.json"
            output_path = self.custom_output_dir / output_filename
            
            status_text.text("Processing frames...")
            progress_bar.progress(0.3)
            
            # Process file
            results = process_audio_file(
                raw_file_path,
                str(output_path),
                config
            )
            
            progress_bar.progress(1.0)
            status_text.text("✅ Processing complete!")
            
            st.success(f"✅ Processing complete!")
            st.info(f"Results saved to: {output_path.name}")
            
            # Display results
            self._display_results(results, output_path)
            
        except Exception as e:
            st.error(f"Error during processing: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def _display_results(self, results: dict, output_path: Path):
        """Display processing results"""
        st.subheader("📈 Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Frames Processed", results['metadata']['frames_processed'])
        with col2:
            st.metric("Peaks Detected", results['metadata']['total_peaks_detected'])
        with col3:
            st.metric("Peaks Validated", results['metadata']['total_peaks_validated'])
        with col4:
            validation_rate = results['metadata']['validation_rate'] * 100
            st.metric("Validation Rate", f"{validation_rate:.1f}%")
        
        # Unique sources
        st.markdown("### 🎯 Detected Sources")
        if results['summary']['unique_sources'] > 0:
            sources_df = pd.DataFrame(results['summary']['sources'])
            st.dataframe(sources_df, width='stretch')
            
            # Visualize sources
            self._plot_sources(results['summary']['sources'])
        else:
            st.warning("No sources detected with current parameters. Try adjusting thresholds.")
        
        # Timeline visualization
        st.markdown("### ⏱️ Detection Timeline")
        self._plot_timeline(results['frames'])
    
    def _plot_sources(self, sources: list):
        """Plot detected sources on polar plot"""
        if not sources:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Polar plot of directions
        ax_polar = plt.subplot(121, projection='polar')
        for source in sources[:10]:  # Show top 10
            azimuth_rad = np.radians(source['azimuth'])
            # Use occurrence count as radius
            radius = np.log10(source['occurrence_count'] + 1)
            ax_polar.scatter(azimuth_rad, radius, 
                           s=100 * source['avg_confidence'],
                           alpha=0.7,
                           label=f"{source['frequency']:.0f} Hz")
        
        ax_polar.set_theta_zero_location('N')
        ax_polar.set_theta_direction(-1)
        ax_polar.set_title('Source Directions\n(Front=0°, Right=90°)')
        ax_polar.legend(bbox_to_anchor=(1.1, 1.05), fontsize=8)
        
        # Frequency distribution
        frequencies = [s['frequency'] for s in sources]
        confidences = [s['avg_confidence'] for s in sources]
        counts = [s['occurrence_count'] for s in sources]
        
        ax2.scatter(frequencies, counts, s=np.array(confidences)*200, alpha=0.6)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Occurrence Count')
        ax2.set_title('Source Frequency Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def _plot_timeline(self, frames: list):
        """Plot detection timeline"""
        if not frames:
            return
        
        # Extract timeline data
        times = []
        detection_counts = []
        avg_confidences = []
        
        for frame in frames:
            if frame['detections']:
                times.append(frame['time'])
                detection_counts.append(len(frame['detections']))
                avg_conf = np.mean([d['confidence'] for d in frame['detections']])
                avg_confidences.append(avg_conf)
        
        if not times:
            st.info("No detections to display in timeline")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        
        # Detection counts
        ax1.plot(times, detection_counts, 'b-', alpha=0.7, linewidth=2)
        ax1.fill_between(times, detection_counts, alpha=0.3)
        ax1.set_ylabel('Detections per Frame')
        ax1.set_title('Detection Timeline')
        ax1.grid(True, alpha=0.3)
        
        # Average confidence
        ax2.plot(times, avg_confidences, 'g-', alpha=0.7, linewidth=2)
        ax2.fill_between(times, avg_confidences, alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Avg Confidence')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
    
    def _show_previous_results(self):
        """Display previous processing results"""
        result_files = sorted(
            self.custom_output_dir.glob("*.json"),
            key=os.path.getmtime,
            reverse=True
        )
        
        if not result_files:
            st.info("No previous results found")
            return
        
        # Show recent results in table
        results_data = []
        for result_file in result_files[:10]:  # Show last 10
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                results_data.append({
                    'Filename': result_file.name,
                    'Duration': f"{result['metadata']['duration']:.1f}s",
                    'Sources': result['summary']['unique_sources'],
                    'Detections': result['metadata']['total_peaks_validated'],
                    'Validation Rate': f"{result['metadata']['validation_rate']*100:.1f}%",
                    'Timestamp': result['metadata']['timestamp'][:19]
                })
            except:
                continue
        
        if results_data:
            df = pd.DataFrame(results_data)
            
            # Add selection for detailed view
            selected_result = st.selectbox(
                "View Details",
                result_files[:10],
                format_func=lambda x: x.stem
            )
            
            if st.button("📊 Show Details"):
                with open(selected_result, 'r') as f:
                    result = json.load(f)
                self._display_results(result, selected_result)
            
            st.dataframe(df, width='stretch')
    
    def _show_comparison_interface(self):
        """Interface for comparing custom results with SODAS"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Custom DOA Results**")
            custom_files = list(self.custom_output_dir.glob("*.json"))
            if custom_files:
                selected_custom = st.selectbox(
                    "Select Custom Result",
                    custom_files,
                    format_func=lambda x: x.stem
                )
            else:
                st.warning("No custom results available")
                return
        
        with col2:
            st.markdown("**SODAS Results**")
            sodas_dir = Path("/home/azureuser/z_odas/ClassifierLogs")
            sodas_files = list(sodas_dir.glob("sst_classify_events_*.json"))
            if sodas_files:
                selected_sodas = st.selectbox(
                    "Select SODAS Result",
                    sodas_files,
                    format_func=lambda x: x.stem
                )
            else:
                st.warning("No SODAS results available")
                return
        
        if st.button("🔄 Compare Results"):
            self._compare_results(selected_custom, selected_sodas)
    
    def _compare_results(self, custom_file: Path, sodas_file: Path):
        """Compare custom and SODAS results"""
        st.subheader("📊 Comparison Results")
        
        try:
            # Load custom results
            with open(custom_file, 'r') as f:
                custom_data = json.load(f)
            
            # Load SODAS results
            with open(sodas_file, 'r') as f:
                sodas_data = json.load(f)
            
            # Comparison metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Custom DOA**")
                st.metric("Sources Detected", custom_data['summary']['unique_sources'])
                st.metric("Total Detections", custom_data['metadata']['total_peaks_validated'])
                st.metric("Validation Rate", 
                         f"{custom_data['metadata']['validation_rate']*100:.1f}%")
            
            with col2:
                st.markdown("**SODAS**")
                # Parse SODAS data structure
                sodas_events = sodas_data if isinstance(sodas_data, list) else []
                st.metric("Events", len(sodas_events))
                st.info("SODAS event count")
            
            with col3:
                st.markdown("**Difference**")
                diff = custom_data['summary']['unique_sources'] - len(sodas_events)
                st.metric("Source Count Δ", diff, delta=diff)
            
            # Detailed comparison
            st.markdown("### Detailed Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Custom Sources**")
                if custom_data['summary']['sources']:
                    custom_df = pd.DataFrame(custom_data['summary']['sources'])
                    st.dataframe(custom_df, width='stretch')
            
            with col2:
                st.markdown("**SODAS Events**")
                if sodas_events:
                    st.json(sodas_events[:5])  # Show first 5 events
                else:
                    st.warning("No SODAS events detected")
            
            # Visualization comparison
            st.markdown("### Visual Comparison")
            self._plot_comparison(custom_data, sodas_events)
            
        except Exception as e:
            st.error(f"Error comparing results: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    def _plot_comparison(self, custom_data: dict, sodas_events: list):
        """Plot comparison visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Custom detections over time
        ax1 = axes[0]
        times = [f['time'] for f in custom_data['frames'] if f['detections']]
        counts = [len(f['detections']) for f in custom_data['frames'] if f['detections']]
        
        if times:
            ax1.plot(times, counts, 'b-', label='Custom DOA', linewidth=2)
            ax1.fill_between(times, counts, alpha=0.3)
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Detections per Frame')
        ax1.set_title('Custom DOA Timeline')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Frequency comparison
        ax2 = axes[1]
        if custom_data['summary']['sources']:
            custom_freqs = [s['frequency'] for s in custom_data['summary']['sources']]
            custom_counts = [s['occurrence_count'] for s in custom_data['summary']['sources']]
            ax2.bar(range(len(custom_freqs)), custom_counts, alpha=0.7, label='Custom DOA')
            ax2.set_xticks(range(len(custom_freqs)))
            ax2.set_xticklabels([f"{f:.0f}" for f in custom_freqs], rotation=45)
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Occurrence Count')
            ax2.set_title('Source Frequency Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
