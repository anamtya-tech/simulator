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

import re
import streamlit as st
import numpy as np
import json
import os
import tempfile
from pathlib import Path
import librosa
import soundfile as sf
from datetime import datetime
import pyroomacoustics as pra

class AudioRenderer:
    def __init__(self, scenes_dir, output_dir, odas_config_dir=None, models_dir=None):
        self.scenes_dir = scenes_dir
        self.output_dir = Path(output_dir) / 'renders'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        project_root = Path(__file__).resolve().parent
        self.odas_config_dir = Path(odas_config_dir) if odas_config_dir else (project_root / 'odas_config')
        self.models_dir = Path(models_dir) if models_dir else (project_root / 'models')
        
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

    def _list_odas_configs(self):
        """Return sorted .cfg files under odas_config directory (recursive)."""
        if not self.odas_config_dir.exists():
            return []
        cfg_files = [p for p in self.odas_config_dir.rglob('*') if p.is_file() and p.suffix.lower() == '.cfg']
        return sorted(cfg_files, key=lambda p: p.name.lower())

    def _list_models(self):
        """Return sorted model directories under models directory."""
        if not self.models_dir.exists():
            return []
        model_dirs = [p for p in self.models_dir.iterdir() if p.is_dir()]
        return sorted(model_dirs, key=lambda p: p.name.lower())

    def _resolve_source_level(self, source_cfg):
        """Return (mode, value) where mode is 'spl' or 'dbfs', else (None, None).

        Preferred field is `spl_db_1m` (physical dB SPL @ 1 m).
        Backward compatibility:
        - old scenes may use `ref_dbfs` (digital RMS target in dBFS)
        - if `ref_dbfs` looks like SPL (>20 dB), treat it as SPL
        """
        spl_db_1m = source_cfg.get('spl_db_1m', None)
        if spl_db_1m is not None:
            return 'spl', float(spl_db_1m)

        legacy_ref = source_cfg.get('ref_dbfs', None)
        if legacy_ref is None:
            return None, None

        legacy_ref = float(legacy_ref)
        if legacy_ref > 20.0:
            return 'spl', legacy_ref
        return 'dbfs', legacy_ref

    def _apply_level_normalization(self, audio, source_cfg, spl_baseline_db, base_target_dbfs=-24.0):
        """Normalise clip RMS using SPL@1m metadata when available.

        Uncalibrated user clips can have arbitrary loudness. We first measure each
        clip RMS, then scale it to a target RMS derived from class SPL metadata.
        This enforces consistent relative loudness between labels while keeping
        overall digital levels bounded.
        """
        mode, level_value = self._resolve_source_level(source_cfg)
        if mode is None:
            return audio

        current_rms = float(np.sqrt(np.mean(audio ** 2)))
        if current_rms <= 1e-10:
            return audio

        if mode == 'spl':
            target_dbfs = base_target_dbfs + (level_value - spl_baseline_db)
        else:
            target_dbfs = level_value

        target_rms = 10.0 ** (target_dbfs / 20.0)
        audio *= target_rms / current_rms
        return audio
    
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

        st.markdown("#### ODAS Runtime Selection")
        runtime_col1, runtime_col2 = st.columns(2)
        with runtime_col1:
            odas_cfg_options = self._list_odas_configs()
            if odas_cfg_options:
                selected_odas_cfg = st.selectbox(
                    "ODAS Config (.cfg)",
                    odas_cfg_options,
                    format_func=lambda p: p.name,
                    help=f"Configs from {self.odas_config_dir}"
                )
            else:
                selected_odas_cfg = None
                st.warning(f"No .cfg files found in {self.odas_config_dir}")

        with runtime_col2:
            model_options = self._list_models()
            if model_options:
                selected_model_dir = st.selectbox(
                    "Model Directory",
                    model_options,
                    format_func=lambda p: p.name,
                    help=f"Models from {self.models_dir}"
                )
            else:
                selected_model_dir = None
                st.warning(f"No model directories found in {self.models_dir}")
        
        # ── Pre-flight warnings (outside button — checkboxes must live at top level) ──
        # Real peak RAM with the streaming/memmap renderer:
        #   2 mmaps (mic + ambient, paged) + one source window + FFT headroom
        max_src_dur   = max((s['end_time'] - s['start_time'])
                            for s in scene['directional_sources']) if scene['directional_sources'] else 0
        mmap_gb       = 2 * (scene['duration'] * self.sample_rate * 4 * 4) / (1024**3)
        src_window_gb = (max_src_dur * self.sample_rate * 4) / (1024**3)
        estimated_gb  = mmap_gb + src_window_gb * 4  # ×4 FFT headroom

        confirm_ram      = True
        confirm_duration = True
        confirm_sources  = True

        if estimated_gb > 8:
            st.warning(
                f"⚠️ Estimated peak RAM: {estimated_gb:.1f} GB (two mmap accumulators + FFT headroom). "
                f"The renderer streams one source at a time and never holds all sources in RAM at once."
            )
            confirm_ram = st.checkbox("I understand the RAM warning and want to continue",
                                      key="confirm_ram")

        if scene['duration'] > 3600:
            st.warning(f"⚠️ Scene duration is {scene['duration']} s "
                       f"({scene['duration']/3600:.1f} h). Rendering may take a very long time.")
            confirm_duration = st.checkbox("I understand this will take a long time",
                                           key="confirm_duration")

        if len(scene['directional_sources']) > 100:
            st.warning(
                f"⚠️ {len(scene['directional_sources'])} directional sources. "
                f"This is fine — sources are rendered one at a time with no memory explosion. "
                f"It will just take longer."
            )
            confirm_sources = st.checkbox("I understand, render anyway",
                                          key="confirm_sources")

        all_confirmed = confirm_ram and confirm_duration and confirm_sources

        # Render button
        if st.button("🎨 Render Audio", type="primary", disabled=not all_confirmed):
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
                        noise_level if add_noise else -40,
                        selected_odas_cfg=str(selected_odas_cfg) if selected_odas_cfg else '',
                        selected_model_dir=str(selected_model_dir) if selected_model_dir else '',
                        selected_model_name=selected_model_dir.name if selected_model_dir else ''
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
    
    def _render_scene(self, scene, room_x, room_y, room_z, absorption, max_order, add_noise, noise_level,
                      selected_odas_cfg='', selected_model_dir='', selected_model_name=''):
        """Render the scene using pyroomacoustics"""
        duration = scene['duration']
        n_samples = int(duration * self.sample_rate)

        # Scene-level SPL anchor (median of known class SPLs @ 1 m).
        # This lets us preserve relative loudness between classes without
        # requiring absolute calibration of uploaded clips.
        spl_values = []
        for src in scene.get('directional_sources', []):
            mode, value = self._resolve_source_level(src)
            if mode == 'spl':
                spl_values.append(value)
        for src in scene.get('ambient_sources', []):
            mode, value = self._resolve_source_level(src)
            if mode == 'spl':
                spl_values.append(value)
        spl_baseline_db = float(np.median(spl_values)) if spl_values else 70.0
        
        # Memory safety check — streaming renderer never holds all sources at once.
        # Peak RAM = 2 mmaps (mic + ambient, both paged) + one source window + FFT temps.
        max_src_dur        = max((s['end_time'] - s['start_time'])
                                 for s in scene['directional_sources']) if scene['directional_sources'] else 0
        mmap_gb            = 2 * (n_samples * 4 * 4) / (1024**3)   # mic + ambient mmaps
        src_window_gb      = (max_src_dur * self.sample_rate * 4) / (1024**3)
        estimated_memory_gb = mmap_gb + src_window_gb * 4           # ×4 FFT headroom

        if estimated_memory_gb > 16:  # true RAM limit — mmaps keep most data on disk
            raise MemoryError(f"Estimated peak RAM ({estimated_memory_gb:.1f} GB) exceeds safe limit. "
                            f"Try reducing scene duration.")
        
        actual_samples = n_samples

        # Mic array position (shifted to room center)
        mic_center = np.array([room_x / 2, room_y / 2, 1.5])  # 1.5m height
        mic_array_pos = self.mic_positions + mic_center[:, np.newaxis]

        # Use float32 throughout to halve memory vs float64.
        # Each source is simulated in its own Room that is freed immediately
        # after accumulation so we never hold all N source signals at once.
        # With 25 sources × 600 s × 16 kHz this would otherwise allocate
        # ~1.9 GB of source signals before simulation even begins.
        #
        # The accumulator is backed by a temp file via np.memmap so the OS
        # can page it out between source iterations.  Only the pages being
        # written/read at any moment need to reside in RAM.
        _mic_tmp_fd, _mic_tmp_path = tempfile.mkstemp(suffix='_mic_acc.f32')
        os.close(_mic_tmp_fd)
        mic_signals = np.memmap(_mic_tmp_path, dtype=np.float32, mode='w+', shape=(4, n_samples))

        # ── Output path (computed now so sidecar filenames can reference it) ─
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        scene_name_clean = scene['name'].replace(' ', '_')
        output_filename  = f"{scene_name_clean}_{timestamp}.raw"
        output_path      = self.output_dir / output_filename

        # ── GT sidecar: separate ambient accumulator ──────────────────────────
        # ambient_signals holds ONLY the background (no directional partials,
        # no noise).  GT clip = partial_i[mic_ch] + ambient[mic_ch, t_start:t_end]
        amb_sidecar_path = str(output_path).replace('.raw', '.ambient.f32')
        ambient_signals  = np.memmap(amb_sidecar_path, dtype=np.float32,
                                     mode='w+', shape=(4, n_samples))
        source_sidecars  = []  # populated during the directional source loop

        # Process directional sources
        progress_bar = st.progress(0)
        status_text = st.empty()

        n_directional = len(scene['directional_sources'])

        for idx, source_config in enumerate(scene['directional_sources']):
            status_text.text(f"Processing directional source {idx + 1}/{n_directional}...")

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

            # Capture the true content length BEFORE final trim/conversion
            # (used by GT builder to avoid chunking silent padding)
            audio_active_samples = int(np.count_nonzero(
                np.abs(audio) > 1e-6  # anything above noise floor
            ) > 0 and min(len(audio), duration_samples) or min(len(audio), duration_samples))
            # Simpler: non-zero content is however many samples loaded from wav
            audio_active_samples = int(min(len(audio), duration_samples))

            # Ensure audio is exactly the right length
            audio = audio[:duration_samples]

            # Normalise by per-class SPL metadata (label.txt line 3) so clips
            # recorded at arbitrary loudness become comparable before rendering.
            audio = self._apply_level_normalization(audio, source_config, spl_baseline_db)

            # Apply per-source volume gain (default 1.0) — acts as a fine-tune
            # trim on top of the reference level above (or as an absolute gain
            # when no SPL/reference metadata is set).
            audio *= source_config.get('volume', 1.0)

            # Windowed signal: only allocate an array for the active window
            # (duration_samples), not the full n_samples.  We pass this shorter
            # signal to pyroomacoustics, then write the convolution result back
            # into the right slice of the mmap accumulator.  For a 30-second
            # bird call in a 2-hour scene this saves ~450 MB vs the old approach.
            window_signal = audio.astype(np.float32)
            del audio  # free trimmed clip before FFT temporaries are allocated

            # Simulate this one source in its own room, then discard the room
            # immediately.  Peak RAM = 1 source window + FFT temps + OS pages of
            # the mmap accumulator, not the full-duration accumulator in RAM.
            src_room = pra.ShoeBox(
                [room_x, room_y, room_z],
                fs=self.sample_rate,
                materials=pra.Material(absorption),
                max_order=max_order
            )
            src_room.add_microphone_array(mic_array_pos)

            source_pos = np.array([
                source_config['x'] + room_x / 2,
                source_config['y'] + room_y / 2,
                source_config['z'] + room_z / 2
            ])
            # Ensure source is inside the room
            source_pos = np.clip(source_pos, [0, 0, 0], [room_x, room_y, room_z])
            src_room.add_source(source_pos, signal=window_signal)
            src_room.simulate()

            # partial is (4, window_samples + rir_tail); write it at start_sample.
            partial = src_room.mic_array.signals.astype(np.float32)  # (4, sim_samples)
            p_len = partial.shape[1]
            acc_start = start_sample
            acc_end   = min(acc_start + p_len, n_samples)
            sidecar_frames = acc_end - acc_start
            mic_signals[:, acc_start:acc_end] += partial[:, :sidecar_frames]

            # ── GT sidecar: isolated room-processed mic signals for this source
            _lbl_clean    = re.sub(r'[^\w\-]', '_', source_config.get('label', f'source_{idx}'))
            _sidecar_path = self.output_dir / f"{output_path.stem}_src{idx:02d}_{_lbl_clean}.f32"
            partial[:, :sidecar_frames].tofile(str(_sidecar_path))
            source_sidecars.append({
                'source_idx':          idx,
                'label':               source_config.get('label', f'source_{idx}'),
                'start_time':          source_config['start_time'],
                'end_time':            source_config['end_time'],
                'start_sample':        int(acc_start),
                'end_sample':          int(acc_end),
                'sidecar_path':        str(_sidecar_path),
                'n_frames':            int(sidecar_frames),
                'audio_active_samples': int(audio_active_samples),
            })

            del src_room, window_signal, partial  # releases RIRs, FFT buffers, everything
            mic_signals.flush()  # ensure written pages are pushed to disk

            progress_bar.progress((idx + 1) / max(n_directional, 1))
        
        # ── Ambient background ───────────────────────────────────────────────
        ambient_mode = scene.get('ambient_mode', 'synthetic')

        if ambient_mode == 'capture':
            # ── Real Capture mode ─────────────────────────────────────────────
            # Mix real 6-channel .raw background directly onto mic_signals.
            # Channels are 0-indexed; mics are on ch1-ch4 (same layout as
            # the 6-channel output and ODAS config map).
            cap_cfg = scene.get('ambient_capture', {})
            cap_path = cap_cfg.get('path', '')
            if cap_path and os.path.exists(cap_path):
                status_text.text("Mixing real capture background...")
                cap_volume = float(cap_cfg.get('volume', 1.0))
                start_off  = int(float(cap_cfg.get('start_offset', 0.0)) * self.sample_rate)

                # Load entire raw file: S16_LE interleaved, 6 channels.
                # Supports both bare PCM and WAV-wrapped (RIFF header) files.
                # Only read the frames we actually need (start_offset + scene
                # duration) so a multi-hour capture file doesn't get loaded
                # entirely into RAM, which could trigger an OOM kill.
                _frames_needed = start_off + actual_samples
                _bytes_per_frame = 6 * 2  # 6 channels × int16

                with open(cap_path, 'rb') as _fh:
                    _magic = _fh.read(4)
                if _magic == b'RIFF':
                    import wave as _wave
                    with _wave.open(cap_path, 'rb') as _w:
                        _raw_bytes = _w.readframes(min(_frames_needed, _w.getnframes()))
                else:
                    with open(cap_path, 'rb') as _fh:
                        _raw_bytes = _fh.read(_frames_needed * _bytes_per_frame)
                raw_int16 = np.frombuffer(_raw_bytes, dtype=np.int16
                                          ).astype(np.float32) / 32768.0  # normalise to [-1, +1]
                del _raw_bytes  # astype() made an independent copy; free the raw bytes now
                total_cap_frames = raw_int16.size // 6
                raw_6ch = raw_int16[:total_cap_frames * 6].reshape(total_cap_frames, 6)

                # Slice from start_offset, then loop/trim to actual_samples
                raw_6ch = raw_6ch[start_off:]
                if len(raw_6ch) < actual_samples:
                    n_repeats = int(np.ceil(actual_samples / max(len(raw_6ch), 1)))
                    raw_6ch = np.tile(raw_6ch, (n_repeats, 1))
                raw_6ch = raw_6ch[:actual_samples]        # (actual_samples, 6)

                # Channels 1-4 (0-indexed) are the 4 mics.
                cap_mics = raw_6ch[:, 1:5].T             # (4, actual_samples)

                # ── Step 1: de-spike ─────────────────────────────────────────
                # Field recordings often contain brief impulse artifacts
                # (mic handling, static discharge, clipping events) that are
                # only 1-2 samples wide but orders of magnitude above the
                # ambient floor.  If left in, these spikes dominate the global
                # normaliser and crush the directional animal signals.
                # We hard-clip each channel at ±6σ (covers >99.9999% of a
                # Gaussian; any sample beyond that is an artifact, not signal).
                for _ch in range(cap_mics.shape[0]):
                    _sigma = cap_mics[_ch].std()
                    if _sigma > 0:
                        _limit = 6.0 * _sigma
                        cap_mics[_ch] = np.clip(cap_mics[_ch], -_limit, _limit)

                # ── Step 2: per-channel RMS-normalise ────────────────────────
                # Scale each mic channel to the same mean RMS so that
                # physical mic-to-mic sensitivity differences in the capture
                # don't cause one mic to dominate the ambient background.
                ch_rms = np.sqrt(np.mean(cap_mics ** 2, axis=1, keepdims=True))  # (4, 1)
                mean_rms = ch_rms.mean()
                if mean_rms > 0:
                    ch_rms = np.where(ch_rms > 0, ch_rms, mean_rms)  # avoid /0
                    cap_mics = cap_mics / ch_rms * mean_rms

                mic_signals     += cap_mics * cap_volume
                ambient_signals += cap_mics * cap_volume
            else:
                st.warning('Real Capture mode selected but no valid capture file found. Continuing without ambient background.')

        else:
            # ── Synthetic ambient mode ────────────────────────────────────────
            if scene.get('ambient_sources'):
                status_text.text("Adding ambient sources...")
                ambient_mix = np.zeros(actual_samples, dtype=np.float32)

                for amb_source in scene['ambient_sources']:
                    audio_path = amb_source['wav_path']
                    if not os.path.exists(audio_path):
                        st.warning(f"Ambient audio not found: {audio_path}")
                        continue

                    audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

                    # Loop to fill duration
                    if len(audio) < actual_samples:
                        n_repeats = int(np.ceil(actual_samples / len(audio)))
                        audio = np.tile(audio, n_repeats)
                    audio = audio[:actual_samples]

                    # Same SPL-aware normalisation for ambient clips.
                    audio = self._apply_level_normalization(audio, amb_source, spl_baseline_db)

                    audio *= amb_source.get('volume', 0.5)
                    ambient_mix += audio

                for i in range(mic_signals.shape[0]):
                    mic_signals[i, :]     += ambient_mix
                    ambient_signals[i, :] += ambient_mix

        # Flush ambient sidecar before noise (noise is NOT part of GT ambient)
        ambient_signals.flush()
        del ambient_signals

        # Add noise if requested
        if add_noise:
            noise_amplitude = 10 ** (noise_level / 20)
            noise = np.random.randn(*mic_signals.shape).astype(np.float32) * noise_amplitude
            mic_signals += noise
            del noise
        
        # Normalize to prevent clipping
        max_val = np.abs(mic_signals).max()
        if max_val > 0:
            mic_signals /= max_val / 0.95  # in-place to avoid a copy
        mic_signals.flush()

        # ── ODAS warm-up / tail constants ────────────────────────────────────
        WARMUP_SECONDS = 10
        TAIL_SECONDS   = 10
        warmup_samples = int(WARMUP_SECONDS * self.sample_rate)
        tail_samples   = int(TAIL_SECONDS   * self.sample_rate)

        # ── Chunked streaming file writer ─────────────────────────────────────
        # Instead of:
        #   1. Building a 6-ch float64 six_channel array (2-3× mic_signals size)
        #   2. np.concatenate for warmup + tail (creates another full copy)
        #   3. Converting to int16 array (another full copy)
        #   4. .flatten() — yet another full copy
        # We open the output file once and stream through it in CHUNK_FRAMES-frame
        # blocks.  Peak working RAM = one chunk × 6 ch × 2 bytes ~ a few MB.
        CHUNK_FRAMES = 16000 * 30  # 30-second chunks
        N_CH = self.n_channels_output

        status_text.text("Writing output file...")

        with open(output_path, 'wb') as out_fh:
            # 1. Warmup silence
            silence_chunk = np.zeros((CHUNK_FRAMES, N_CH), dtype=np.int16)
            remaining = warmup_samples
            while remaining > 0:
                n = min(remaining, CHUNK_FRAMES)
                out_fh.write(silence_chunk[:n].tobytes())
                remaining -= n

            # 2. Mic content — read mic_signals in row-chunks, build 6-ch frame
            #    Channels 1-4 (0-indexed) get the mic data; 0 and 5 stay zero.
            remaining = actual_samples
            offset = 0
            while remaining > 0:
                n = min(remaining, CHUNK_FRAMES)
                # mic_signals is (4, actual_samples) mmap; slice columns
                chunk_mics = mic_signals[:, offset:offset + n]  # (4, n) float32
                chunk_6ch  = np.zeros((n, N_CH), dtype=np.int16)
                # Clip to [-1, 1] before int16 conversion
                chunk_int16 = np.clip(chunk_mics, -1.0, 1.0)
                chunk_int16 = (chunk_int16 * 32767).astype(np.int16)  # (4, n)
                chunk_6ch[:, 1:5] = chunk_int16.T  # (n, 4) into cols 1-4
                out_fh.write(chunk_6ch.tobytes())
                offset    += n
                remaining -= n

            # 3. Tail silence
            remaining = tail_samples
            while remaining > 0:
                n = min(remaining, CHUNK_FRAMES)
                out_fh.write(silence_chunk[:n].tobytes())
                remaining -= n

        # Release the memmap and delete its temp file
        del mic_signals
        try:
            os.unlink(_mic_tmp_path)
        except OSError:
            pass
        
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
            'output_file': str(output_path),
            'warmup_seconds': WARMUP_SECONDS,
            'tail_silence_seconds': TAIL_SECONDS,
            'source_sidecars': source_sidecars,
            'ambient_sidecar_path': amb_sidecar_path,
            'selected_odas_config': selected_odas_cfg,
            'selected_model_dir': selected_model_dir,
            'selected_model_name': selected_model_name,
        }
        
        metadata_path = str(output_path).replace('.raw', '.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        progress_bar.progress(1.0)
        status_text.text("✅ Rendering complete!")
        
        return output_path
    
    def _save_as_wav(self, raw_path):
        """Convert the full-length raw PCM file to a 4-channel WAV.

        Streams through the file in 30-second chunks so RAM stays bounded
        regardless of the render duration.  Channels saved are the 4 mic
        channels (columns 1-4 of the 6-channel interleaved layout).

        Returns the path of the saved WAV file.
        """
        wav_path = str(raw_path).rsplit('.raw', 1)[0] + '.wav'
        file_size = os.path.getsize(raw_path)
        n_frames  = file_size // (self.n_channels_output * 2)  # int16 = 2 bytes

        CHUNK_FRAMES = self.sample_rate * 30  # 30-second chunks

        audio_mmap = np.memmap(raw_path, dtype=np.int16, mode='r',
                               shape=(n_frames, self.n_channels_output))
        try:
            with sf.SoundFile(wav_path, mode='w', samplerate=self.sample_rate,
                              channels=4, subtype='PCM_16') as wav_out:
                offset = 0
                while offset < n_frames:
                    end   = min(offset + CHUNK_FRAMES, n_frames)
                    chunk = np.array(audio_mmap[offset:end, 1:5])  # mic channels
                    wav_out.write(chunk)
                    offset = end
        finally:
            del audio_mmap

        return wav_path

    def _show_preview(self, raw_path):
        """Scrubbing preview — fixed RAM cost (~8 MB) regardless of file size.

        Architecture:
        - Full-duration waveform plot via 16× downsampled mmap read  (tiny RAM)
        - A scrub slider lets the user seek to any position in the file
        - Only a WINDOW_S-second slice around the scrub point is decoded and
          sent to the browser, so RAM stays constant for files of any length.
        (Streamlit's st.audio() always sends bytes to the browser; true HTTP
        range streaming isn't possible in Streamlit's model, so a fixed window
        with a seek control is the correct low-RAM equivalent.)
        """
        st.subheader("Preview")

        WINDOW_S    = 30   # seconds decoded per audio player render
        PLOT_STRIDE = 16   # downsample factor for the waveform overview plot

        try:
            import matplotlib.pyplot as plt

            file_size      = os.path.getsize(raw_path)
            n_frames       = file_size // (self.n_channels_output * 2)  # int16 = 2 B
            total_duration = n_frames / self.sample_rate

            # ── Open mmap once; used for both plot and audio slice ────────────
            audio_mmap = np.memmap(raw_path, dtype=np.int16, mode='r',
                                   shape=(n_frames, self.n_channels_output))

            # ── Full-duration waveform plot (downsampled, tiny RAM) ───────────
            plot_data = audio_mmap[::PLOT_STRIDE, 1:5].astype(np.float32) / 32767
            time_axis = np.arange(len(plot_data)) * PLOT_STRIDE / self.sample_rate

            mic_labels = ['Mic 1 (Left, Ch2)', 'Mic 2 (Back, Ch3)',
                          'Mic 3 (Right, Ch4)', 'Mic 4 (Front, Ch5)']
            fig, axes = plt.subplots(4, 1, figsize=(12, 8))
            for i in range(4):
                axes[i].plot(time_axis, plot_data[:, i])
                axes[i].set_ylabel(mic_labels[i])
                axes[i].grid(True, alpha=0.3)
                if i == 3:
                    axes[i].set_xlabel('Time (s)')
            plt.suptitle(
                f'Full waveform — {total_duration:.1f} s  '
                f'(plot {PLOT_STRIDE}× downsampled, {len(plot_data):,} pts)'
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            del plot_data

            # ── Scrub slider ──────────────────────────────────────────────────
            # The slider key is tied to the file path so each render gets its
            # own independent scrub position in Streamlit session state.
            slider_key  = f'scrub_{abs(hash(raw_path))}'
            max_start_s = max(0.0, total_duration - WINDOW_S)

            if max_start_s > 0:
                scrub_s = st.slider(
                    f'🎚️ Scrub position  (loads {WINDOW_S} s window)',
                    min_value=0.0,
                    max_value=float(max_start_s),
                    value=float(st.session_state.get(slider_key, 0.0)),
                    step=1.0,
                    format='%.0f s',
                    key=slider_key,
                )
            else:
                scrub_s = 0.0
                st.caption(
                    f'ℹ️ File is {total_duration:.1f} s — shorter than the '
                    f'{WINDOW_S} s window, playing in full.'
                )

            # ── Decode only the window slice ──────────────────────────────────
            start_frame  = int(scrub_s * self.sample_rate)
            end_frame    = min(int((scrub_s + WINDOW_S) * self.sample_rate), n_frames)
            window_frames = end_frame - start_frame
            ram_mb        = window_frames * 4 * 4 / 1e6   # 4 mics × float32

            st.caption(
                f'📍 {scrub_s:.0f} s → {scrub_s + window_frames / self.sample_rate:.0f} s  '
                f'({window_frames:,} frames decoded, ~{ram_mb:.1f} MB in RAM)'
            )

            # Read the window (mmap pages, not whole file) then release mmap
            play_slice = audio_mmap[start_frame:end_frame, 1:5].astype(np.float32) / 32767
            del audio_mmap   # mmap released; play_slice is an independent copy

            # ── Audio players ─────────────────────────────────────────────────
            st.markdown("**🎧 Listen to Individual Microphones**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Mic 1 (Left, Ch2)**")
                st.audio(play_slice[:, 0], sample_rate=self.sample_rate)
                st.markdown("**Mic 2 (Back, Ch3)**")
                st.audio(play_slice[:, 1], sample_rate=self.sample_rate)
            with col2:
                st.markdown("**Mic 3 (Right, Ch4)**")
                st.audio(play_slice[:, 2], sample_rate=self.sample_rate)
                st.markdown("**Mic 4 (Front, Ch5)**")
                st.audio(play_slice[:, 3], sample_rate=self.sample_rate)

            st.markdown("**Mixed (All Mics Average)**")
            st.audio(np.mean(play_slice, axis=1), sample_rate=self.sample_rate)
            del play_slice

            # ── Save full audio to WAV ─────────────────────────────────────────
            st.markdown("---")
            st.markdown("**💾 Save Full Audio as WAV**")
            wav_path = str(raw_path).rsplit('.raw', 1)[0] + '.wav'
            _wav_key_save    = f'savewav_{abs(hash(str(raw_path)))}'
            _wav_key_resave  = f'resavewav_{abs(hash(str(raw_path)))}'
            if os.path.exists(wav_path):
                wav_size_mb = os.path.getsize(wav_path) / (1024 * 1024)
                st.info(
                    f"WAV already saved: `{os.path.basename(wav_path)}` "
                    f"({wav_size_mb:.1f} MB)  →  `{wav_path}`"
                )
                if st.button("🔄 Re-save WAV", key=_wav_key_resave):
                    with st.spinner(f"Saving {total_duration:.0f} s → WAV …"):
                        saved = self._save_as_wav(raw_path)
                    st.success(f"✅ WAV saved: {saved}")
            else:
                st.caption(
                    f"Export the full {total_duration:.0f} s render as a "
                    f"4-channel WAV (Mic 1-4) for playback in any audio player."
                )
                if st.button("💾 Save to WAV", key=_wav_key_save):
                    with st.spinner(f"Saving {total_duration:.0f} s → WAV …"):
                        saved = self._save_as_wav(raw_path)
                    st.success(f"✅ WAV saved: {saved}")

        except Exception as e:
            st.error(f"Error previewing audio: {e}")
