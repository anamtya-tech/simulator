"""
Renderer module to generate multi-channel audio using pyroomacoustics.

Given a scene configuration JSON from the configurator module:
- Loads directional and ambient audio sources
- Simulates room acoustics (open forest environment)
- Generates 6-channel raw PCM audio (16kHz, S16_LE)
  - Channels 2,3,4,5 contain the 4 mic signals
  - Channels 1 and 6 are zeros (no mic there)

ReSpeaker USB 4 Mic Array geometry (in meters):
- Mic 1 (Ch 2): [-0.032, 0.000, 0.000]  # Left
- Mic 2 (Ch 3): [0.000, -0.032, 0.000]  # Back
- Mic 3 (Ch 4): [0.032, 0.000, 0.000]   # Right
- Mic 4 (Ch 5): [0.000, 0.032, 0.000]   # Front

Output format: ${date}_{scene_name}_ChatakX_sim.raw
"""

import streamlit as st
import numpy as np
import json
import os
from pathlib import Path
import librosa
import soundfile as sf
from datetime import datetime
import pyroomacoustics as pra

class AudioRenderer:
    def __init__(self, scenes_dir, output_dir):
        self.scenes_dir = scenes_dir
        self.output_dir = Path(output_dir) / 'renders'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mic array geometry (ReSpeaker USB 4 Mic Array)
        self.mic_positions = np.array([
            [-0.032, 0.000, 0.000],  # Mic 1: Left
            [0.000, -0.032, 0.000],  # Mic 2: Back
            [0.032, 0.000, 0.000],   # Mic 3: Right
            [0.000, 0.032, 0.000]    # Mic 4: Front
        ]).T  # Shape: (3, 4)
        
        # Audio parameters
        self.sample_rate = 16000
        self.n_channels_output = 6  # 6 channels total (1, 2-5 (mics), 6)
    
    def _find_existing_renders(self, scene_name):
        """Find all existing renders for a scene"""
        scene_name_clean = scene_name.replace(' ', '_')
        pattern = f"{scene_name_clean}_*.raw"
        raw_files = list(Path(self.output_dir).glob(pattern))
        
        renders = []
        for raw_file in raw_files:
            metadata_file = raw_file.with_suffix('.json')
            
            render_info = {
                'path': str(raw_file),
                'filename': raw_file.name,
                'mtime': os.path.getmtime(raw_file),
                'size_mb': os.path.getsize(raw_file) / (1024 * 1024)
            }
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    render_info['timestamp'] = metadata.get('timestamp', 'Unknown')
                    render_info['render_id'] = metadata.get('render_id', raw_file.stem)
            else:
                render_info['timestamp'] = datetime.fromtimestamp(render_info['mtime']).strftime("%Y%m%d_%H%M%S")
                render_info['render_id'] = raw_file.stem
            
            renders.append(render_info)
        
        # Sort by modification time, newest first
        renders.sort(key=lambda x: x['mtime'], reverse=True)
        return renders
    
    def _delete_render(self, raw_path):
        """Delete a render and its metadata"""
        try:
            # Delete raw file
            if os.path.exists(raw_path):
                os.remove(raw_path)
            
            # Delete metadata file
            metadata_path = raw_path.replace('.raw', '.json')
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
            
            # Clear preview if this was being previewed
            if st.session_state.get('preview_path') == raw_path:
                st.session_state.preview_path = None
            
            st.success(f"✅ Deleted render: {os.path.basename(raw_path)}")
        except Exception as e:
            st.error(f"Error deleting render: {e}")
        
    def render(self):
        """Render the audio renderer interface"""
        st.subheader("Audio Rendering")
        st.markdown("Generate multi-channel audio from scene configuration using pyroomacoustics")
        
        # Load scene selection
        scene_files = list(Path(self.scenes_dir).glob("*.json"))
        
        if not scene_files:
            st.warning("No scenes found. Please create a scene first in the Scene Configurator.")
            return
        
        selected_scene_file = st.selectbox(
            "Select Scene",
            scene_files,
            format_func=lambda x: x.stem
        )
        
        # Load scene
        with open(selected_scene_file, 'r') as f:
            scene = json.load(f)
        
        scene_mtime = os.path.getmtime(selected_scene_file)
        
        # Check if already rendered
        existing_renders = self._find_existing_renders(scene['name'])
        
        # Display scene info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Duration", f"{scene['duration']}s")
        with col2:
            st.metric("Directional Sources", len(scene['directional_sources']))
        with col3:
            st.metric("Ambient Sources", len(scene['ambient_sources']))
        with col4:
            st.metric("Max Radius", f"{scene['max_radius']}m")
        
        # Show existing renders if any
        if existing_renders:
            st.subheader("📁 Existing Renders")
            
            for render_info in existing_renders:
                render_time = render_info['timestamp']
                render_path = render_info['path']
                is_outdated = render_info['mtime'] < scene_mtime
                
                with st.expander(
                    f"{'⚠️ ' if is_outdated else '✅ '}{render_info['filename']} - {render_time}",
                    expanded=not is_outdated
                ):
                    if is_outdated:
                        st.warning("⚠️ Scene has been modified since this render. Consider re-rendering.")
                    else:
                        st.success("✅ Up to date with current scene configuration")
                    
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.caption(f"Rendered: {render_time}")
                        st.caption(f"Size: {render_info['size_mb']:.2f} MB")
                    with col2:
                        if st.button("🎧 Preview", key=f"preview_{render_info['filename']}"):
                            st.session_state.preview_path = render_path
                    with col3:
                        if st.button("🗑️ Delete", key=f"delete_{render_info['filename']}"):
                            self._delete_render(render_path)
                            st.rerun()
                    
                    if st.session_state.get('preview_path') == render_path:
                        self._show_preview(render_path)
        
        # Rendering parameters
        st.subheader("Rendering Parameters")
        col1, col2 = st.columns(2)
        with col1:
            room_dim_x = st.number_input("Room X dimension (m)", 10.0, 1000.0, scene['max_radius'] * 2.5, 10.0)
            room_dim_y = st.number_input("Room Y dimension (m)", 10.0, 1000.0, scene['max_radius'] * 2.5, 10.0)
            room_dim_z = st.number_input("Room Z dimension (m)", 3.0, 200.0, max(20.0, scene['max_height'] * 1.5), 5.0)
        with col2:
            absorption = st.slider("Wall Absorption (0=reflective, 1=absorptive)", 0.0, 1.0, 0.7, 0.05,
                                   help="Forest is open, so high absorption")
            max_order = st.slider("Max reflection order", 0, 10, 3, 1,
                                  help="Number of wall reflections to simulate")
            add_noise = st.checkbox("Add background noise", value=False)
            if add_noise:
                noise_level = st.slider("Noise level (dB)", -60, -20, -40, 5)
        
        # Render button
        if st.button("🎨 Render Audio", type="primary"):
            with st.spinner("Rendering audio..."):
                try:
                    output_path = self._render_scene(
                        scene, 
                        room_dim_x, 
                        room_dim_y, 
                        room_dim_z, 
                        absorption, 
                        max_order,
                        add_noise if add_noise else False,
                        noise_level if add_noise else -40
                    )
                    st.success(f"✅ Audio rendered successfully!")
                    st.info(f"Output: {output_path}")
                    
                    # Store in session state
                    st.session_state.rendered_audio_path = output_path
                    
                    # Offer preview (convert to mono for web playback)
                    self._show_preview(output_path)
                    
                except Exception as e:
                    st.error(f"Error rendering audio: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    def _render_scene(self, scene, room_x, room_y, room_z, absorption, max_order, add_noise, noise_level):
        """Render the scene using pyroomacoustics"""
        duration = scene['duration']
        n_samples = int(duration * self.sample_rate)
        
        # Create room
        room = pra.ShoeBox(
            [room_x, room_y, room_z],
            fs=self.sample_rate,
            materials=pra.Material(absorption),
            max_order=max_order
        )
        
        # Add microphone array at origin (shifted to room center)
        mic_center = np.array([room_x / 2, room_y / 2, 1.5])  # 1.5m height
        mic_array_pos = self.mic_positions + mic_center[:, np.newaxis]
        room.add_microphone_array(mic_array_pos)
        
        # Process directional sources
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, source_config in enumerate(scene['directional_sources']):
            status_text.text(f"Processing directional source {idx + 1}/{len(scene['directional_sources'])}...")
            
            # Load audio
            audio_path = source_config['wav_path']
            if not os.path.exists(audio_path):
                st.warning(f"Audio file not found: {audio_path}")
                continue
            
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Handle timing
            start_sample = int(source_config['start_time'] * self.sample_rate)
            end_sample = int(source_config['end_time'] * self.sample_rate)
            duration_samples = end_sample - start_sample
            
            # Repeat or trim audio to fit time window
            if source_config.get('repeat', False) and len(audio) < duration_samples:
                n_repeats = int(np.ceil(duration_samples / len(audio)))
                audio = np.tile(audio, n_repeats)[:duration_samples]
            elif len(audio) > duration_samples:
                audio = audio[:duration_samples]
            else:
                # Pad if needed
                audio = np.pad(audio, (0, max(0, duration_samples - len(audio))))
            
            # Ensure audio is exactly the right length
            audio = audio[:duration_samples]
            
            # Create full-length signal with silence
            full_signal = np.zeros(n_samples)
            actual_end = min(start_sample + len(audio), n_samples)
            full_signal[start_sample:actual_end] = audio[:actual_end - start_sample]
            
            # Add source to room
            source_pos = np.array([
                source_config['x'] + room_x / 2,
                source_config['y'] + room_y / 2,
                source_config['z'] + room_z / 2
            ])
            room.add_source(source_pos, signal=full_signal)
            
            progress_bar.progress((idx + 1) / (len(scene['directional_sources']) + 1))
        
        # Simulate
        status_text.text("Running room simulation...")
        room.simulate()
        
        # Get mic signals (shape: [n_mics, n_samples])
        mic_signals = room.mic_array.signals
        
        # Ensure mic_signals matches expected length (pyroomacoustics may produce slightly different length)
        actual_samples = mic_signals.shape[1]
        if actual_samples != n_samples:
            if actual_samples > n_samples:
                mic_signals = mic_signals[:, :n_samples]
            else:
                # Pad with zeros if shorter
                padding = np.zeros((mic_signals.shape[0], n_samples - actual_samples))
                mic_signals = np.concatenate([mic_signals, padding], axis=1)
            actual_samples = n_samples
        
        # Process ambient sources (add equally to all mics)
        if scene['ambient_sources']:
            status_text.text("Adding ambient sources...")
            ambient_mix = np.zeros(actual_samples)
            
            for amb_source in scene['ambient_sources']:
                audio_path = amb_source['wav_path']
                if not os.path.exists(audio_path):
                    st.warning(f"Ambient audio not found: {audio_path}")
                    continue
                
                audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                
                # Loop ambient to fill duration
                if len(audio) < actual_samples:
                    n_repeats = int(np.ceil(actual_samples / len(audio)))
                    audio = np.tile(audio, n_repeats)
                
                # Ensure exact length
                audio = audio[:actual_samples]
                
                # Apply volume
                audio *= amb_source.get('volume', 0.5)
                ambient_mix += audio
            
            # Add ambient to all mics
            for i in range(mic_signals.shape[0]):
                mic_signals[i, :] += ambient_mix
        
        # Add noise if requested
        if add_noise:
            noise_amplitude = 10 ** (noise_level / 20)
            noise = np.random.randn(*mic_signals.shape) * noise_amplitude
            mic_signals += noise
        
        # Normalize to prevent clipping
        max_val = np.abs(mic_signals).max()
        if max_val > 0:
            mic_signals = mic_signals / max_val * 0.95
        
        # Create 6-channel output (channel 1 and 6 are zeros)
        six_channel = np.zeros((self.n_channels_output, actual_samples))
        six_channel[1:5, :] = mic_signals  # Channels 2-5 are the 4 mics
        
        # Convert to int16
        audio_int16 = (six_channel * 32767).astype(np.int16)
        
        # Interleave channels: [ch1_s1, ch2_s1, ..., ch6_s1, ch1_s2, ch2_s2, ...]
        interleaved = audio_int16.T.flatten()
        
        # Save as raw PCM
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_name_clean = scene['name'].replace(' ', '_')
        output_filename = f"{scene_name_clean}_{timestamp}.raw"
        output_path = self.output_dir / output_filename
        
        interleaved.tofile(output_path)
        
        # Also save metadata
        metadata = {
            'scene_name': scene['name'],
            'timestamp': timestamp,
            'render_id': f"{scene_name_clean}_{timestamp}",
            'duration': duration,
            'sample_rate': self.sample_rate,
            'n_channels': self.n_channels_output,
            'format': 'S16_LE',
            'room_dimensions': [room_x, room_y, room_z],
            'absorption': absorption,
            'max_order': max_order,
            'scene_file': str(Path(self.scenes_dir) / f"{scene['name']}.json"),
            'output_file': str(output_path)
        }
        
        metadata_path = str(output_path).replace('.raw', '.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Rendering complete!")
        
        return output_path
    
    def _show_preview(self, raw_path):
        """Show audio preview"""
        st.subheader("Preview")
        
        try:
            # Read raw file
            audio_int16 = np.fromfile(raw_path, dtype=np.int16)
            
            # Reshape to 6 channels
            n_samples = len(audio_int16) // self.n_channels_output
            audio_6ch = audio_int16.reshape(n_samples, self.n_channels_output).T
            
            # Convert to float
            audio_float = audio_6ch.astype(np.float32) / 32767
            
            # Show waveforms
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(4, 1, figsize=(12, 8))
            time_axis = np.arange(n_samples) / self.sample_rate
            
            mic_labels = ['Mic 1 (Left, Ch2)', 'Mic 2 (Back, Ch3)', 
                          'Mic 3 (Right, Ch4)', 'Mic 4 (Front, Ch5)']
            
            for i in range(4):
                axes[i].plot(time_axis, audio_float[i + 1])  # Channels 2-5
                axes[i].set_ylabel(mic_labels[i])
                axes[i].grid(True, alpha=0.3)
                if i == 3:
                    axes[i].set_xlabel('Time (s)')
            
            plt.suptitle('Multi-channel Waveforms')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Audio playback for all 4 mics
            st.markdown("**🎧 Listen to Individual Microphones**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Mic 1 (Left, Ch2)**")
                st.audio(audio_float[1], sample_rate=self.sample_rate)
                
                st.markdown("**Mic 2 (Back, Ch3)**")
                st.audio(audio_float[2], sample_rate=self.sample_rate)
            
            with col2:
                st.markdown("**Mic 3 (Right, Ch4)**")
                st.audio(audio_float[3], sample_rate=self.sample_rate)
                
                st.markdown("**Mic 4 (Front, Ch5)**")
                st.audio(audio_float[4], sample_rate=self.sample_rate)
            
            # Mixed mono for comparison
            st.markdown("**Mixed (All Mics Average)**")
            mixed_audio = np.mean(audio_float[1:5], axis=0)
            st.audio(mixed_audio, sample_rate=self.sample_rate)
            
        except Exception as e:
            st.error(f"Error previewing audio: {e}")
