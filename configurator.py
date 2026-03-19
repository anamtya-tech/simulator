"""
Streamlit interface to configure scenes for audio simulation.

A scene contains:
1. Duration (in seconds)
2. Directional sources with position (azimuth, distance, height) and timing
3. Ambient sources (omnidirectional background)
4. Max bounds (radius and height range)

Sources are picked from sources.csv with columns: wav_path, source_type, label

TODO: moving directional sources. Here we should be able to configure a vector of movement 
for each directional source (dx,dy,dz) and the speed (m/s). The source should move along 
that vector at that speed during the duration of the scene.
"""

import streamlit as st
import pandas as pd
import json
import os
import shutil
import numpy as np
import wave
from datetime import datetime
from pathlib import Path
import random

DEFAULT_SOUNDS_DIR = Path('/home/azureuser/sounds')
CAPTURES_DIR       = Path('/home/azureuser/audio_cache/ambient_captures')

class SceneConfigurator:
    def __init__(self, scenes_dir, sounds_dir=None):
        self.scenes_dir = scenes_dir
        # Use provided path or default; persist across Streamlit reruns via session state
        initial_path = str(sounds_dir or DEFAULT_SOUNDS_DIR)
        if 'library_path' not in st.session_state:
            st.session_state.library_path = initial_path
        self._load_library(st.session_state.library_path)

        # Initialize session state for scene configuration
        if 'scene_config' not in st.session_state:
            st.session_state.scene_config = self._create_default_scene()
    
    def _create_default_scene(self):
        """Create a default scene configuration"""
        return {
            "name": "untitled_scene",
            "duration": 10.0,  # seconds
            "max_radius": 50.0,  # meters
            "max_height": 10.0,  # meters
            "min_height": -2.0,  # meters
            "directional_sources": [],
            "ambient_sources": []
        }
    
    def _azimuth_elevation_to_cartesian(self, azimuth_deg, distance, height):
        """Convert azimuth (degrees), distance, and height to cartesian coordinates"""
        azimuth_rad = np.deg2rad(azimuth_deg)
        x = distance * np.cos(azimuth_rad)
        y = distance * np.sin(azimuth_rad)
        z = height
        return x, y, z
    
    def _cartesian_to_azimuth_elevation(self, x, y, z):
        """Convert cartesian to azimuth (degrees), distance, and height"""
        distance = np.sqrt(x**2 + y**2)
        azimuth_deg = np.rad2deg(np.arctan2(y, x))
        return azimuth_deg, distance, z
    
    def _get_available_files_for_label(self, label, source_type=None):
        """Return all audio files for a label from the scanned library."""
        entry = self.library.get(label)
        if entry is None:
            return []
        return list(entry['files'])

    def _scan_library(self, root: Path) -> tuple:
        """Walk a directory tree and build a label→{source_type, files} lookup.

        Rules:
          - For each .wav, walk up toward root to find nearest ancestor label.txt
          - label.txt line 1 = label name; line 2 (optional) = directional|ambient
          - A deeper label.txt always overrides an ancestor's for its sub-tree
          - Files with no ancestor label.txt are silently skipped
          - Same label name in multiple branches → files merged under one key
        """
        label_cache: dict = {}  # folder Path → (label, source_type) | None

        def find_label(folder: Path):
            current = folder
            while True:
                if current in label_cache:
                    return label_cache[current]
                txt = current / 'label.txt'
                if txt.exists():
                    lines = txt.read_text().strip().splitlines()
                    lbl   = lines[0].strip() if lines else ''
                    stype = lines[1].strip() if len(lines) > 1 else 'directional'
                    if stype not in ('directional', 'ambient'):
                        stype = 'directional'
                    result = (lbl, stype) if lbl else None
                    label_cache[current] = result
                    return result
                if current == root or current.parent == current:
                    label_cache[current] = None
                    return None
                current = current.parent

        library: dict = {}
        skipped: list = []
        for wav in sorted(root.rglob('*.wav')):
            entry = find_label(wav.parent)
            if entry is None:
                skipped.append(wav.name)
                continue
            lbl, stype = entry
            if lbl not in library:
                library[lbl] = {'source_type': stype, 'files': []}
            library[lbl]['files'].append(str(wav))
        return library, skipped

    def _load_library(self, path: str) -> str | None:
        """Scan the given sounds directory and store results in self.library.
        Returns an error string on failure, or None on success.
        """
        p = Path(path).expanduser().resolve()
        self.library_path = str(p)
        if not p.exists():
            self.library = {}
            return f'Path not found: {p}'
        self.library, skipped = self._scan_library(p)
        return None

    def _list_captures(self):
        """Return list of .raw files in CAPTURES_DIR with basic metadata.
        Handles both bare S16_LE PCM and WAV-wrapped (RIFF header) files.
        """
        import wave as _wave
        CAPTURES_DIR.mkdir(parents=True, exist_ok=True)
        captures = []
        for f in sorted(CAPTURES_DIR.glob('*.raw')):
            size_mb = f.stat().st_size / 1e6
            # Detect WAV-wrapped vs bare S16_LE 6ch 16kHz
            with open(f, 'rb') as fh:
                magic = fh.read(4)
            if magic == b'RIFF':
                try:
                    with _wave.open(str(f), 'rb') as w:
                        n_samples = w.getnframes()
                        sr = w.getframerate()
                except Exception:
                    n_samples = f.stat().st_size // (2 * 6)
                    sr = 16000
            else:
                # Bare PCM: S16_LE 6ch 16kHz
                n_samples = f.stat().st_size // (2 * 6)
                sr = 16000
            dur_s = n_samples / sr
            captures.append({'path': str(f), 'name': f.name, 'size_mb': size_mb, 'dur_s': dur_s})
        return captures

    def render(self):
        """Render the scene configurator interface"""
        scene = st.session_state.scene_config
        
        # Top controls
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        with col1:
            scene['name'] = st.text_input("Scene Name", scene['name'])
        with col2:
            if st.button("💾 Save Scene"):
                self._save_scene(scene)
        with col3:
            if st.button("📂 Load Scene"):
                st.session_state.show_load_dialog = True
        with col4:
            if st.button("🔄 New Scene"):
                st.session_state.scene_config = self._create_default_scene()
                st.rerun()
        
        # Load scene dialog in expander
        if st.session_state.get('show_load_dialog', False):
            with st.expander("📂 Load Scene", expanded=True):
                self._show_load_scene_dialog()

        # ── Sound Library ────────────────────────────────────────────────────
        with st.expander('🔊 Sound Library', expanded=False):
            col_path, col_btn = st.columns([4, 1])
            with col_path:
                st.text_input(
                    'Library path',
                    value=st.session_state.library_path,
                    key='library_path_input',
                    help='Point to any folder organised with label.txt sub-folders'
                )
            with col_btn:
                st.write('')  # vertical align
                reload_clicked = st.button('🔄 Reload', key='reload_library_btn')
            if reload_clicked:
                err = self._load_library(st.session_state.library_path_input)
                st.session_state.library_path = self.library_path
                if err:
                    st.error(err)
                else:
                    st.rerun()
            dir_count = sum(1 for e in self.library.values() if e['source_type'] == 'directional')
            amb_count = sum(1 for e in self.library.values() if e['source_type'] == 'ambient')
            st.caption(
                f'📂 `{self.library_path}` — **{len(self.library)}** labels '
                f'({dir_count} directional, {amb_count} ambient)'
            )

        # Scene parameters
        st.subheader("Scene Parameters")
        col1, col2 = st.columns(2)
        with col1:
            scene['duration'] = st.number_input(
                "Duration (seconds)", 
                min_value=1.0, 
                max_value=600.0, 
                value=scene['duration'],
                step=1.0
            )
            scene['max_radius'] = st.number_input(
                "Max Radius (meters)", 
                min_value=1.0, 
                max_value=500.0, 
                value=scene['max_radius'],
                step=1.0
            )
        with col2:
            scene['max_height'] = st.number_input(
                "Max Height (meters)", 
                min_value=0.0, 
                max_value=100.0, 
                value=scene['max_height'],
                step=1.0
            )
            scene['min_height'] = st.number_input(
                "Min Height (meters)", 
                min_value=-10.0, 
                max_value=0.0, 
                value=scene['min_height'],
                step=0.5
            )
        
        # Tabs for directional and ambient sources
        tab1, tab2 = st.tabs(["🎯 Directional Sources", "🌊 Ambient Sources"])
        
        with tab1:
            self._render_directional_sources(scene)
        
        with tab2:
            self._render_ambient_sources(scene)
        
        # Visualization
        st.subheader("Scene Visualization")
        self._visualize_scene(scene)
    
    def _render_directional_sources(self, scene):
        """Render directional sources configuration"""
        st.markdown("### Directional Sources")

        directional_labels = sorted(
            lbl for lbl, e in self.library.items() if e['source_type'] == 'directional'
        )

        col_add, col_clear = st.columns([1, 1])
        with col_add:
            if st.button('➕ Add Source', key='add_dir_btn'):
                self._add_directional_source(scene)
                st.rerun()
        with col_clear:
            if scene['directional_sources'] and st.button('🗑️ Clear All', key='clear_dir_btn'):
                scene['directional_sources'].clear()
                st.rerun()

        # ── Random generation panel ────────────────────────────────────────
        with st.expander('🎲 Generate Random Sources', expanded=False):
            st.markdown('Select which labels to include and set spatial / timing constraints.')

            # Label picker
            chosen_labels = st.multiselect(
                'Labels to include',
                options=directional_labels,
                default=directional_labels[:min(3, len(directional_labels))],
                key='rand_labels'
            )

            col1, col2 = st.columns(2)
            with col1:
                rand_n = st.number_input(
                    'Number of sources to generate', 1, 50, 5, key='rand_n'
                )
                rand_min_dist = st.number_input(
                    'Min distance (m)', 0.5, float(scene['max_radius']),
                    1.0, step=0.5, key='rand_min_dist'
                )
                rand_max_dist = st.number_input(
                    'Max distance (m)', 0.5, float(scene['max_radius']),
                    float(scene['max_radius']), step=0.5, key='rand_max_dist'
                )
            with col2:
                rand_min_height = st.number_input(
                    'Min height (m)', float(scene['min_height']), float(scene['max_height']),
                    float(scene['min_height']), step=0.5, key='rand_min_height'
                )
                rand_max_height = st.number_input(
                    'Max height (m)', float(scene['min_height']), float(scene['max_height']),
                    float(scene['max_height']), step=0.5, key='rand_max_height'
                )

            # ── Timing mode (mutually exclusive) ─────────────────────────
            timing_mode = st.radio(
                'Timing mode',
                options=['🔁 Repeat to fill window', '✂️ Limit window to file duration'],
                index=0,
                key='rand_timing_mode',
                horizontal=True,
                help='Repeat: clip loops for the whole random time window. '
                     'Limit: window is sized to match the actual WAV file length.'
            )
            rand_repeat        = (timing_mode == '🔁 Repeat to fill window')
            rand_limit_to_file = not rand_repeat

            # Duration sliders only matter when not limiting to file
            if not rand_limit_to_file:
                dcol1, dcol2 = st.columns(2)
                with dcol1:
                    rand_min_dur = st.number_input(
                        'Min instance duration (s)', 0.5, float(scene['duration']),
                        2.0, step=0.5, key='rand_min_dur'
                    )
                with dcol2:
                    rand_max_dur = st.number_input(
                        'Max instance duration (s)', 0.5, float(scene['duration']),
                        min(10.0, float(scene['duration'])), step=0.5, key='rand_max_dur'
                    )
            else:
                rand_min_dur = rand_max_dur = None
                st.caption('⏱ Window duration will be read from each chosen WAV file.')

            if st.button('🎲 Generate', key='rand_gen_btn', type='primary'):
                if not chosen_labels:
                    st.warning('Pick at least one label.')
                else:
                    constraints = dict(
                        labels=chosen_labels,
                        min_dist=rand_min_dist, max_dist=rand_max_dist,
                        min_height=rand_min_height, max_height=rand_max_height,
                        min_dur=rand_min_dur, max_dur=rand_max_dur,
                        repeat=rand_repeat,
                        limit_to_file=rand_limit_to_file,
                    )
                    for _ in range(int(rand_n)):
                        self._add_directional_source(scene, randomize=True,
                                                     constraints=constraints)
                    st.rerun()

        # ── Existing sources list ──────────────────────────────────────────
        if not scene['directional_sources']:
            st.info("No directional sources added yet.")
            return

        for idx, source in enumerate(scene['directional_sources']):
            with st.expander(f"🎯 Source {idx + 1}: {source['label']}", expanded=False):
                self._render_directional_source_editor(scene, idx)

    def _render_directional_source_editor(self, scene, idx):
        """Render editor for a single directional source"""
        source = scene['directional_sources'][idx]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # Label selection
            directional_labels = sorted(
                lbl for lbl, e in self.library.items() if e['source_type'] == 'directional'
            )
            source['label'] = st.selectbox(
                "Label",
                directional_labels,
                index=directional_labels.index(source['label']) if source['label'] in directional_labels else 0,
                key=f"dir_label_{idx}"
            )
        with col2:
            if st.button("🗑️ Remove", key=f"remove_dir_{idx}"):
                scene['directional_sources'].pop(idx)
                st.rerun()
        
        # File selection
        available_files = self._get_available_files_for_label(source['label'], 'directional')
        if available_files:
            file_options = ["Random"] + available_files
            current_file = source.get('wav_path', 'Random')
            if current_file not in file_options:
                current_file = 'Random'
            
            source['wav_path'] = st.selectbox(
                "Audio File",
                file_options,
                index=file_options.index(current_file),
                key=f"dir_file_{idx}"
            )
        
        # Timing
        col1, col2 = st.columns(2)
        with col1:
            source['start_time'] = st.number_input(
                "Start Time (s)",
                min_value=0.0,
                max_value=float(scene['duration']),
                value=float(source['start_time']),
                step=0.1,
                key=f"dir_start_{idx}"
            )
        with col2:
            source['end_time'] = st.number_input(
                "End Time (s)",
                min_value=float(source['start_time']),
                max_value=float(scene['duration']),
                value=float(source['end_time']),
                step=0.1,
                key=f"dir_end_{idx}"
            )
        
        # Position
        st.markdown("**Position**")
        azimuth, distance, height = self._cartesian_to_azimuth_elevation(
            source['x'], source['y'], source['z']
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            new_azimuth = st.slider(
                "Azimuth (°)",
                -180.0, 180.0,
                float(azimuth),
                step=1.0,
                key=f"dir_azimuth_{idx}"
            )
        with col2:
            new_distance = st.slider(
                "Distance (m)",
                0.1, float(scene['max_radius']),
                float(min(distance, scene['max_radius'])),
                step=0.1,
                key=f"dir_distance_{idx}"
            )
        with col3:
            new_height = st.slider(
                "Height (m)",
                float(scene['min_height']), float(scene['max_height']),
                float(np.clip(height, scene['min_height'], scene['max_height'])),
                step=0.1,
                key=f"dir_height_{idx}"
            )
        
        # Update cartesian coordinates
        source['x'], source['y'], source['z'] = self._azimuth_elevation_to_cartesian(
            new_azimuth, new_distance, new_height
        )
        
        # Display cartesian coordinates
        st.caption(f"Cartesian: x={source['x']:.2f}m, y={source['y']:.2f}m, z={source['z']:.2f}m")
        
        # Repeat option for short sounds
        source['repeat'] = st.checkbox(
            "Repeat audio to fill time window",
            value=source.get('repeat', False),
            key=f"dir_repeat_{idx}"
        )

        # Per-source volume
        source['volume'] = st.slider(
            'Volume (gain)', 0.1, 10.0,
            float(source.get('volume', 1.0)),
            step=0.1, key=f'dir_volume_{idx}',
            help='Multiply signal amplitude. Use >1 to compensate range attenuation.'
        )
    
    def _visualize_capture(self, path, start_offset_s=0.0, preview_s=10.0):
        """Show per-channel waveform + RMS bar chart for a .raw capture file.
        Handles both WAV-wrapped (RIFF) and bare S16_LE 6-ch 16kHz files.
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        SR = 16000
        N_CH = 6

        # ── Load samples ────────────────────────────────────────────────────
        try:
            with open(path, 'rb') as fh:
                magic = fh.read(4)
            if magic == b'RIFF':
                with wave.open(path, 'rb') as w:
                    SR = w.getframerate()
                    N_CH = w.getnchannels()
                    start_frame = int(start_offset_s * SR)
                    w.setpos(min(start_frame, w.getnframes()))
                    n_preview = int(preview_s * SR)
                    raw_bytes = w.readframes(n_preview)
            else:
                bytes_per_frame = 2 * N_CH
                start_byte = int(start_offset_s * SR) * bytes_per_frame
                n_bytes = int(preview_s * SR) * bytes_per_frame
                with open(path, 'rb') as fh:
                    fh.seek(start_byte)
                    raw_bytes = fh.read(n_bytes)

            data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            n_frames = len(data) // N_CH
            data = data[:n_frames * N_CH].reshape(n_frames, N_CH)  # (frames, 6)
        except Exception as e:
            st.error(f'Could not read capture file: {e}')
            return

        t = np.linspace(start_offset_s, start_offset_s + n_frames / SR, n_frames)

        CH_LABELS = [
            'Ch 1 (DSP/processed)',
            'Ch 2 (mic 1)',
            'Ch 3 (mic 2)',
            'Ch 4 (mic 3)',
            'Ch 5 (mic 4)',
            'Ch 6 (unused/silent)',
        ]
        COLORS = ['#e74c3c', '#2ecc71', '#3498db', '#9b59b6', '#f39c12', '#95a5a6']
        # Channels currently used by the renderer (0-indexed 1–4)
        USED = {1, 2, 3, 4}

        rms_vals = np.sqrt(np.mean(data ** 2, axis=0))
        max_vals = np.abs(data).max(axis=0)

        # ── Waveform subplots ────────────────────────────────────────────────
        fig_wave = make_subplots(
            rows=N_CH, cols=1,
            shared_xaxes=True,
            subplot_titles=[f'{CH_LABELS[i]}  |  RMS={rms_vals[i]:.0f}  MAX={max_vals[i]:.0f}'
                            + ('  ✅ used' if i in USED else '  ⛔ skipped')
                            for i in range(N_CH)],
            vertical_spacing=0.04,
        )
        # Downsample for display (max ~4000 points per channel)
        step = max(1, n_frames // 4000)
        for i in range(N_CH):
            fig_wave.add_trace(
                go.Scatter(
                    x=t[::step], y=data[::step, i],
                    mode='lines', line=dict(color=COLORS[i], width=0.8),
                    name=CH_LABELS[i], showlegend=False,
                ),
                row=i + 1, col=1,
            )
        fig_wave.update_layout(
            height=120 * N_CH,
            margin=dict(l=10, r=10, t=30, b=10),
            title_text=f'Waveforms — {preview_s:.0f}s from offset {start_offset_s:.0f}s',
        )
        fig_wave.update_xaxes(title_text='Time (s)', row=N_CH, col=1)
        st.plotly_chart(fig_wave, use_container_width=True)

        # ── RMS bar chart ────────────────────────────────────────────────────
        fig_rms = go.Figure(go.Bar(
            x=[f'Ch {i+1}' for i in range(N_CH)],
            y=rms_vals,
            marker_color=[COLORS[i] if i in USED else '#bdc3c7' for i in range(N_CH)],
            text=[f'{v:.0f}' for v in rms_vals],
            textposition='outside',
        ))
        fig_rms.update_layout(
            title='RMS per channel  (green/blue = used as mic input, grey = skipped)',
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            yaxis_title='RMS (int16 units)',
        )
        st.plotly_chart(fig_rms, use_container_width=True)

        # ── Summary table ────────────────────────────────────────────────────
        rows = []
        for i in range(N_CH):
            rows.append({
                'Channel': f'Ch {i+1}',
                'Role': CH_LABELS[i],
                'RMS': f'{rms_vals[i]:.1f}',
                'Max': f'{max_vals[i]:.0f}',
                'Used by renderer': '✅' if i in USED else '⛔',
                'Has signal': '✅' if rms_vals[i] > 5 else '❌ silent',
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    def _render_ambient_sources(self, scene):
        """Render ambient sources configuration — synthetic or real capture."""
        st.markdown('### Ambient Background')

        # ── Mode toggle ────────────────────────────────────────────────────
        mode = st.radio(
            'Background mode',
            ['🔬 Synthetic (simulated)', '🎙️ Real Capture (.raw)'],
            index=0 if scene.get('ambient_mode', 'synthetic') == 'synthetic' else 1,
            horizontal=True, key='ambient_mode_radio'
        )
        scene['ambient_mode'] = 'synthetic' if mode.startswith('🔬') else 'capture'

        # ── Real Capture mode ──────────────────────────────────────────────
        if scene['ambient_mode'] == 'capture':
            st.info(
                '🎙️ **Real Capture mode** — the simulator will mix your '
                'directional sources on top of the real 6-channel background. '
                'Synthetic ambient sources below are ignored.'
            )
            captures = self._list_captures()

            # Upload a new capture
            with st.expander('⬆️ Upload Capture File', expanded=not captures):
                up_file = st.file_uploader(
                    '6-channel 16 kHz S16_LE .raw file',
                    type=['raw'], key='capture_upload'
                )
                up_tag  = st.text_input('Optional tag (e.g. midday-clear, dawn-windy)', key='capture_tag')
                if st.button('Save capture', key='save_capture_btn'):
                    if up_file is None:
                        st.error('Select a file first.')
                    else:
                        tag_part = f'_{up_tag.strip().replace(" ","-")}' if up_tag.strip() else ''
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        dest = CAPTURES_DIR / f'capture_{ts}{tag_part}.raw'
                        dest.write_bytes(up_file.read())
                        st.success(f'Saved as `{dest.name}`.')
                        st.rerun()

            # Capture drop folder guide
            with st.expander('📂 Drop files manually', expanded=False):
                st.markdown('Place `.raw` files here (6-ch 16 kHz S16_LE):')
                st.code(str(CAPTURES_DIR))
                st.markdown('Name them anything — shown by filename in the picker below.')

            if not captures:
                st.warning('No capture files found in `ambient_captures/`. Upload one above or drop files into the folder.')
                scene.pop('ambient_capture', None)
                return

            # Capture picker
            cap_names = [f"{c['name']}  ({c['dur_s']:.0f}s  {c['size_mb']:.1f} MB)" for c in captures]
            cap_idx = 0
            current_path = scene.get('ambient_capture', {}).get('path', '')
            for i, c in enumerate(captures):
                if c['path'] == current_path:
                    cap_idx = i
                    break

            chosen_idx = st.selectbox('Capture file', range(len(cap_names)),
                                      format_func=lambda i: cap_names[i],
                                      index=cap_idx, key='capture_picker')
            chosen = captures[chosen_idx]
            cap_dur = chosen['dur_s']

            # Trim controls
            col1, col2, col3 = st.columns(3)
            with col1:
                start_off = st.number_input(
                    'Start offset (s)', min_value=0.0,
                    max_value=max(0.0, cap_dur - 1.0),
                    value=float(scene.get('ambient_capture', {}).get('start_offset', 0.0)),
                    step=1.0, key='cap_start_off'
                )
            with col2:
                cap_volume = st.slider(
                    'Volume', 0.1, 3.0,
                    float(scene.get('ambient_capture', {}).get('volume', 1.0)),
                    step=0.05, key='cap_volume'
                )
            with col3:
                avail = cap_dur - start_off
                st.metric('Available from offset', f'{avail:.0f}s')
                if avail < scene['duration']:
                    st.caption('⚠️ Shorter than scene — will loop.')

            scene['ambient_capture'] = {
                'path': chosen['path'],
                'start_offset': start_off,
                'volume': cap_volume,
            }

            # ── Per-channel visualizer ─────────────────────────────────────
            with st.expander('📊 Inspect capture channels', expanded=False):
                prev_s = st.slider(
                    'Preview duration (s)', 2.0, 30.0, 10.0, step=1.0,
                    key='cap_preview_dur',
                    help='How many seconds to show in the waveform preview'
                )
                st.caption(
                    'Ch 2–5 (0-indexed 1–4) are used as the 4 mic inputs. '
                    'Ch 1 (DSP/processed) and Ch 6 (silent) are skipped.'
                )
                self._visualize_capture(chosen['path'], start_offset_s=start_off, preview_s=prev_s)

        # ── Synthetic mode ─────────────────────────────────────────────────
        st.markdown('---')
        synth_label = '### Synthetic Ambient Sources'
        if scene['ambient_mode'] == 'capture':
            synth_label += '  *(ignored in Real Capture mode)*'
        st.markdown(synth_label)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button('➕ Add Ambient'):
                self._add_ambient_source(scene)
        with col2:
            num_random = st.number_input('Number to generate', 1, 10, 1, key='num_random_amb')
        with col3:
            if st.button('🎲 Generate Random', key='gen_random_amb'):
                for _ in range(num_random):
                    self._add_ambient_source(scene, randomize=True)

        if not scene['ambient_sources']:
            st.info('No synthetic ambient sources added.')
        else:
            for idx, source in enumerate(scene['ambient_sources']):
                with st.expander(f"🌊 Ambient {idx + 1}: {source['label']}", expanded=False):
                    self._render_ambient_source_editor(scene, idx)
    
    def _render_ambient_source_editor(self, scene, idx):
        """Render editor for a single ambient source"""
        source = scene['ambient_sources'][idx]
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # Label selection
            ambient_labels = sorted(
                lbl for lbl, e in self.library.items() if e['source_type'] == 'ambient'
            )
            source['label'] = st.selectbox(
                "Label",
                ambient_labels,
                index=ambient_labels.index(source['label']) if source['label'] in ambient_labels else 0,
                key=f"amb_label_{idx}"
            )
        with col2:
            if st.button("🗑️ Remove", key=f"remove_amb_{idx}"):
                scene['ambient_sources'].pop(idx)
                st.rerun()
        
        # File selection
        available_files = self._get_available_files_for_label(source['label'], 'ambient')
        if available_files:
            file_options = ["Random"] + available_files
            current_file = source.get('wav_path', 'Random')
            if current_file not in file_options:
                current_file = 'Random'
            
            source['wav_path'] = st.selectbox(
                "Audio File",
                file_options,
                index=file_options.index(current_file),
                key=f"amb_file_{idx}"
            )
        
        # Volume control
        source['volume'] = st.slider(
            "Volume",
            0.0, 1.0,
            source.get('volume', 0.5),
            step=0.05,
            key=f"amb_volume_{idx}"
        )
    
    def _add_directional_source(self, scene, randomize=False, constraints=None):
        """Add a new directional source.

        When randomize=True, constraints dict can supply:
          labels          – list of labels to pick from
          min_dist / max_dist   – distance range in metres
          min_height / max_height
          min_dur / max_dur     – duration of the time window (ignored when limit_to_file=True)
          repeat          – bool, whether to loop the clip to fill the window
          limit_to_file   – bool, size the window to the actual WAV duration
        """
        directional_labels = [
            lbl for lbl, e in self.library.items() if e['source_type'] == 'directional'
        ]

        if randomize:
            c = constraints or {}
            pool            = c.get('labels', directional_labels) or directional_labels
            min_dist        = float(c.get('min_dist',   1.0))
            max_dist        = float(c.get('max_dist',   scene['max_radius']))
            min_h           = float(c.get('min_height', scene['min_height']))
            max_h           = float(c.get('max_height', scene['max_height']))
            do_repeat       = bool(c.get('repeat',        False))
            limit_to_file   = bool(c.get('limit_to_file', False))

            label    = random.choice(pool)
            azimuth  = random.uniform(-180, 180)
            distance = random.uniform(min_dist, max(min_dist, max_dist))
            height   = random.uniform(min_h,    max(min_h, max_h))

            # Pick the wav file now so we can read its duration if needed
            files    = self._get_available_files_for_label(label)
            wav_path = random.choice(files) if files else 'Random'

            if limit_to_file and wav_path != 'Random':
                # Read actual duration of the chosen file
                try:
                    with wave.open(wav_path, 'rb') as wf:
                        file_dur = wf.getnframes() / wf.getframerate()
                except Exception:
                    file_dur = float(c.get('min_dur') or 2.0)
                max_start  = max(0.0, scene['duration'] - file_dur)
                start_time = random.uniform(0, max_start) if max_start > 0 else 0.0
                end_time   = min(start_time + file_dur, scene['duration'])
            else:
                min_dur    = float(c.get('min_dur') or 1.0)
                max_dur    = float(c.get('max_dur') or scene['duration'])
                max_start  = max(0.0, scene['duration'] - min_dur)
                start_time = random.uniform(0, max_start)
                dur        = random.uniform(min_dur, min(max_dur, scene['duration'] - start_time))
                end_time   = min(start_time + dur, scene['duration'])
        else:
            label      = directional_labels[0] if directional_labels else 'Unknown'
            azimuth    = 0
            distance   = scene['max_radius'] / 2
            height     = 0
            start_time = 0
            end_time   = scene['duration']
            wav_path   = 'Random'
            do_repeat  = False

        x, y, z = self._azimuth_elevation_to_cartesian(azimuth, distance, height)

        source = {
            'label':      label,
            'wav_path':   wav_path,
            'x': x, 'y': y, 'z': z,
            'start_time': round(start_time, 2),
            'end_time':   round(end_time,   2),
            'repeat':     do_repeat,
        }
        
        scene['directional_sources'].append(source)
    
    def _add_ambient_source(self, scene, randomize=False):
        """Add a new ambient source"""
        ambient_labels = [
            lbl for lbl, e in self.library.items() if e['source_type'] == 'ambient'
        ]
        
        if randomize:
            label = random.choice(ambient_labels)
            volume = random.uniform(0.3, 0.7)
        else:
            label = ambient_labels[0]
            volume = 0.5
        
        source = {
            'label': label,
            'wav_path': 'Random',
            'volume': volume
        }
        
        scene['ambient_sources'].append(source)
    
    def _visualize_scene(self, scene):
        """Visualize the scene: top-down spatial map + timeline Gantt."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            import matplotlib.cm as cm

            sources   = scene['directional_sources']
            ambients  = scene['ambient_sources']
            duration  = scene['duration']
            radius    = scene['max_radius']

            # ── colour palette – one colour per unique label ──────────────
            all_labels = sorted({s['label'] for s in sources})
            cmap       = cm.get_cmap('tab20', max(len(all_labels), 1))
            label_colour = {lbl: cmap(i) for i, lbl in enumerate(all_labels)}

            # ── figure: spatial map (left) + timeline (right) ─────────────
            # Timeline height scales with number of sources (min 3 rows)
            n_rows    = max(3, len(sources) + len(ambients))
            fig_h     = max(5, 0.45 * n_rows + 2)
            fig, (ax_map, ax_time) = plt.subplots(
                1, 2,
                figsize=(14, fig_h),
                gridspec_kw={'width_ratios': [1, 1.6]}
            )

            # ── LEFT: top-down spatial map ────────────────────────────────
            ax_map.set_xlim(-radius, radius)
            ax_map.set_ylim(-radius, radius)
            ax_map.set_aspect('equal')
            ax_map.set_xlabel('X (m)')
            ax_map.set_ylabel('Y (m)')
            ax_map.set_title('Top View (XY)')
            ax_map.grid(True, alpha=0.25)
            ax_map.axhline(0, color='k', linewidth=0.4)
            ax_map.axvline(0, color='k', linewidth=0.4)

            # range rings
            for r in [radius * 0.25, radius * 0.5, radius * 0.75, radius]:
                circle = plt.Circle((0, 0), r, fill=False,
                                    color='grey', linewidth=0.4, linestyle='--')
                ax_map.add_patch(circle)
                ax_map.text(0, r, f'{r:.0f}m', fontsize=6,
                            ha='center', va='bottom', color='grey')

            ax_map.plot(0, 0, 'r*', markersize=14, zorder=5, label='Mic Array')

            for idx, src in enumerate(sources):
                c = label_colour[src['label']]
                ax_map.plot(src['x'], src['y'], 'o', color=c,
                            markersize=9, zorder=4)
                ax_map.annotate(
                    f"{idx+1}",
                    (src['x'], src['y']),
                    fontsize=7, fontweight='bold',
                    ha='center', va='center', color='white', zorder=5
                )
                ax_map.annotate(
                    f" {src['label']}",
                    (src['x'], src['y']),
                    xytext=(6, 6), textcoords='offset points',
                    fontsize=7, color=c
                )

            # legend for labels
            legend_patches = [
                mpatches.Patch(color=label_colour[lbl], label=lbl)
                for lbl in all_labels
            ]
            if legend_patches:
                ax_map.legend(handles=legend_patches, fontsize=7,
                              loc='lower right', framealpha=0.7)
            else:
                ax_map.legend(fontsize=7)

            # ── RIGHT: timeline (Gantt) ───────────────────────────────────
            ax_time.set_xlim(0, duration)
            ax_time.set_xlabel('Time (s)')
            ax_time.set_title('Timeline')
            ax_time.grid(axis='x', alpha=0.3)
            ax_time.axvline(0,        color='k', linewidth=0.5)
            ax_time.axvline(duration, color='k', linewidth=0.5)

            row_labels = []

            # directional sources
            for idx, src in enumerate(sources):
                y    = idx
                c    = label_colour[src['label']]
                t0   = src['start_time']
                t1   = src['end_time']
                ax_time.barh(y, t1 - t0, left=t0, height=0.6,
                             color=c, alpha=0.85, edgecolor='white', linewidth=0.5)
                ax_time.text(
                    t0 + (t1 - t0) / 2, y,
                    f"{src['label']}  {t0:.1f}–{t1:.1f}s",
                    ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white'
                )
                row_labels.append(f"#{idx+1}")

            # ambient sources (shown as full-width faded bars below)
            n_dir = len(sources)
            for idx, amb in enumerate(ambients):
                y  = n_dir + idx
                vol = amb.get('volume', 0.5)
                ax_time.barh(y, duration, left=0, height=0.6,
                             color='steelblue', alpha=max(0.15, vol * 0.6),
                             edgecolor='steelblue', linewidth=0.5,
                             linestyle='--')
                ax_time.text(
                    duration / 2, y,
                    f"[ambient] {amb['label']}  vol={vol:.2f}",
                    ha='center', va='center', fontsize=7, color='steelblue'
                )
                row_labels.append(f"~{amb['label']}")

            total_rows = n_dir + len(ambients)
            if total_rows == 0:
                ax_time.set_yticks([])
                ax_time.text(duration / 2, 0, 'No sources',
                             ha='center', va='center',
                             fontsize=10, color='grey')
            else:
                ax_time.set_ylim(-0.6, total_rows - 0.4)
                ax_time.set_yticks(range(total_rows))
                ax_time.set_yticklabels(row_labels, fontsize=8)

            # second x-axis tick every 5 s for longer scenes
            tick_step = 5 if duration > 30 else (2 if duration > 10 else 1)
            ax_time.set_xticks(
                list(range(0, int(duration) + 1, tick_step)) + [duration]
            )

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        except Exception as e:
            st.error(f"Error visualizing scene: {e}")
    
    def _save_scene(self, scene):
        """Save scene configuration to JSON file"""
        try:
            # Resolve random file selections
            saved_scene = scene.copy()
            
            for source in saved_scene['directional_sources']:
                if source['wav_path'] == 'Random':
                    available = self._get_available_files_for_label(source['label'], 'directional')
                    if available:
                        source['wav_path'] = random.choice(available)
            
            for source in saved_scene['ambient_sources']:
                if source['wav_path'] == 'Random':
                    available = self._get_available_files_for_label(source['label'], 'ambient')
                    if available:
                        source['wav_path'] = random.choice(available)
            
            # Add metadata
            saved_scene['created_at'] = datetime.now().isoformat()
            saved_scene['version'] = '1.0'
            
            # Save to file
            filename = f"{scene['name']}.json"
            filepath = os.path.join(self.scenes_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(saved_scene, f, indent=2)
            
            st.success(f"✅ Scene saved to: {filepath}")
            
        except Exception as e:
            st.error(f"Error saving scene: {e}")
    
    def _show_load_scene_dialog(self):
        """Show dialog to load existing scene"""
        scene_files = list(Path(self.scenes_dir).glob("*.json"))
        
        if not scene_files:
            st.warning("No saved scenes found.")
            return
        
        selected_file = st.selectbox(
            "Select scene to load",
            scene_files,
            format_func=lambda x: x.stem,
            key="scene_selector"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Load", type="primary", key="load_confirm"):
                try:
                    with open(selected_file, 'r') as f:
                        loaded_scene = json.load(f)
                    st.session_state.scene_config = loaded_scene
                    st.session_state.show_load_dialog = False
                    st.success(f"✅ Loaded scene: {selected_file.stem}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error loading scene: {e}")
        with col2:
            if st.button("❌ Cancel", key="load_cancel"):
                st.session_state.show_load_dialog = False
                st.rerun()


class DatasetConfigurator:
    """Configure YAMNet dataset curation settings"""
    
    def __init__(self, output_dir='outputs'):
        from yamnet_dataset_curator import YAMNetDatasetCurator
        from dataset_visualizer import DatasetVisualizer
        
        self.curator = YAMNetDatasetCurator(output_dir=f'{output_dir}/yamnet_datasets')
        self.visualizer = DatasetVisualizer(curator=self.curator)
    
    def render(self):
        """Render dataset configuration interface"""
        st.subheader("🎯 YAMNet Dataset Management")
        st.markdown("Manage datasets for fine-tuning YAMNet")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Datasets", "⚙️ Settings", "📈 Visualizer", "📖 Guide"])
        
        with tab1:
            self._render_dataset_list()
        
        with tab2:
            self._render_settings()
        
        with tab3:
            self.visualizer.render()
        
        with tab4:
            self._render_guide()
    
    def _render_dataset_list(self):
        """Render list of datasets with management options"""
        st.markdown("### Available Datasets")
        
        datasets = self.curator.list_datasets()
        
        if not datasets:
            st.info("No datasets created yet. Run analysis with YAMNet curation enabled to create datasets.")
            
            # Create new dataset
            st.markdown("---")
            st.markdown("#### Create New Dataset")
            new_name = st.text_input("Dataset name", "yamnet_train_001")
            if st.button("Create Dataset"):
                self.curator.set_active_dataset(new_name)
                st.success(f"Created dataset: {new_name}")
                st.rerun()
            return
        
        # Display datasets
        for dataset_name in datasets:
            stats = self.curator.get_dataset_stats(dataset_name)
            
            if stats is None:
                continue
            
            is_active = dataset_name == self.curator.get_active_dataset()
            
            with st.expander(
                f"{'🟢' if is_active else '⚪'} {dataset_name} ({stats['sample_count']} samples)",
                expanded=is_active
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples", stats['sample_count'])
                
                with col2:
                    st.metric("Audio Files", stats['actual_audio_files'])
                
                with col3:
                    st.metric("Unique Labels", len(stats['samples_by_label']))
                
                # Label distribution
                if stats['samples_by_label']:
                    st.markdown("**Label Distribution:**")
                    label_df = pd.DataFrame([
                        {'Label': label, 'Count': count}
                        for label, count in stats['samples_by_label'].items()
                    ])
                    st.dataframe(label_df, use_container_width=True)
                
                # Actions
                st.markdown("**Actions:**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("Set as Active", key=f"activate_{dataset_name}"):
                        self.curator.set_active_dataset(dataset_name)
                        st.success(f"Activated: {dataset_name}")
                        st.rerun()
                
                with col2:
                    if st.button("Prepare for TensorFlow", key=f"tf_{dataset_name}"):
                        with st.spinner("Preparing dataset..."):
                            result = self.curator.create_tensorflow_dataset(dataset_name)
                            st.success("✅ Dataset prepared!")
                            st.json(result)
                
                with col3:
                    st.text(f"Path: {stats['path']}")
        
        # Create new dataset
        st.markdown("---")
        st.markdown("### Create New Dataset")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            new_name = st.text_input("Dataset name", key="new_dataset_name")
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            if st.button("➕ Create"):
                if new_name:
                    self.curator.set_active_dataset(new_name)
                    st.success(f"Created and activated: {new_name}")
                    st.rerun()
        
        # Merge datasets
        st.markdown("---")
        st.markdown("### Merge Datasets")
        
        if len(datasets) >= 2:
            selected_datasets = st.multiselect(
                "Select datasets to merge",
                datasets
            )
            
            merged_name = st.text_input(
                "Name for merged dataset",
                f"merged_{datetime.now().strftime('%Y%m%d')}"
            )
            
            if st.button("🔀 Merge Selected"):
                if len(selected_datasets) >= 2 and merged_name:
                    with st.spinner("Merging datasets..."):
                        result = self.curator.merge_datasets(selected_datasets, merged_name)
                        if result:
                            st.success(f"✅ Merged {result['total_samples']} samples into {merged_name}")
                            st.rerun()
                else:
                    st.warning("Select at least 2 datasets to merge")
    
    def _render_settings(self):
        """Render curation settings"""
        st.markdown("### Curation Settings")
        st.markdown("Configure which samples to include in YAMNet datasets")
        
        config = self.curator.config
        criteria = config['curation_criteria']
        
        # Curation criteria
        st.markdown("#### Selection Criteria")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_mismatches = st.checkbox(
                "Include Mismatches",
                value=criteria['include_mismatches'],
                help="Save samples where YAMNet prediction doesn't match ground truth (if aligned)"
            )
            
            include_unclassified = st.checkbox(
                "Include Unclassified",
                value=criteria['include_unclassified'],
                help="Save samples YAMNet couldn't classify (if aligned)"
            )
            
            min_activity = st.slider(
                "Minimum Activity Level",
                0.0, 1.0, criteria['min_activity'],
                help="Skip samples with activity below this threshold"
            )
        
        with col2:
            include_low_confidence = st.checkbox(
                "Include Low Confidence",
                value=criteria['include_low_confidence'],
                help="Save samples with low YAMNet confidence (if aligned)"
            )
            
            save_unknown = st.checkbox(
                "Save Misaligned for Manual Review",
                value=criteria.get('save_unknown', True),
                help="Save samples outside thresholds for manual verification"
            )
        
        st.info("📌 **Note**: Direction and confidence thresholds are configured in Analysis Settings (used for both matching and curation)")
        
        # Audio reconstruction settings
        st.markdown("---")
        st.markdown("#### Audio Reconstruction")
        
        audio_params = config['audio_params']
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_duration = st.number_input(
                "Target Duration (seconds)",
                min_value=0.5,
                max_value=10.0,
                value=audio_params['target_duration'],
                step=0.5,
                help="Target length for reconstructed audio clips"
            )
        
        with col2:
            overlap_frames = st.number_input(
                "Overlap Frames",
                min_value=1,
                max_value=10,
                value=audio_params['overlap_frames'],
                help="Number of frames to overlap for smooth reconstruction"
            )
        
        # Save button
        if st.button("💾 Save Settings", use_container_width=True):
            config['curation_criteria'].update({
                'include_mismatches': include_mismatches,
                'include_unclassified': include_unclassified,
                'include_low_confidence': include_low_confidence,
                'min_activity': min_activity,
                'save_unknown': save_unknown
                # Note: confidence_threshold and direction_threshold_deg come from Analysis Settings
            })
            
            config['audio_params'].update({
                'target_duration': target_duration,
                'overlap_frames': overlap_frames
            })
            
            self.curator._save_config(config)
            st.success("✅ Settings saved!")
        
        # Active dataset info
        st.markdown("---")
        st.markdown("#### Active Dataset")
        
        active = self.curator.get_active_dataset()
        stats = self.curator.get_dataset_stats(active)
        
        if stats:
            st.info(f"**{active}** - {stats['sample_count']} samples")
    
    def _render_guide(self):
        """Render usage guide"""
        st.markdown("""
        ### 📖 YAMNet Dataset Curation Guide
        
        #### Overview
        This feature curates training datasets for fine-tuning YAMNet based on ODAS analysis results.
        
        #### How It Works
        
        1. **Analysis Phase**
           - ODAS processes audio and provides YAMNet classifications
           - Ground truth labels come from your scene configuration
           - System compares YAMNet predictions with ground truth
        
        2. **Curation Phase**
           - Samples are selected based on curation criteria:
             - **Mismatches**: YAMNet prediction ≠ ground truth
             - **Unclassified**: YAMNet didn't provide classification
             - **Low Confidence**: YAMNet confidence below threshold
           - Audio is reconstructed from frequency bins (1024 bins from ODAS)
           - Samples saved as WAV files with metadata
        
        3. **Dataset Organization**
           - Each dataset contains:
             - `audio/`: WAV files (16kHz mono)
             - `spectrograms/`: Visual representations
             - `metadata/`: CSV files with labels and metadata
             - `labels.csv`: Master label file for training
        
        #### Dataset Format
        
        The dataset follows TensorFlow Hub YAMNet format:
        - **Audio**: 16kHz mono WAV files
        - **Labels**: CSV with columns: filename, label, fold, yamnet_class, confidence, etc.
        - **Splits**: train/val/test folds for proper evaluation
        
        #### Fine-Tuning Workflow
        
        1. **Curate Data**: Run multiple simulations, system automatically curates samples
        2. **Review**: Use Dataset Visualizer to listen and verify samples
        3. **Prepare**: Click "Prepare for TensorFlow" to create train/val/test splits
        4. **Train**: Use TensorFlow/Keras to fine-tune YAMNet
        5. **Evaluate**: Test on validation set
        6. **Deploy**: Update ODAS with fine-tuned model
        
        #### Best Practices
        
        - **Diverse Data**: Include samples from various runs and conditions
        - **Balanced Labels**: Try to get similar counts for each class
        - **Quality Check**: Review samples in visualizer before training
        - **Iterative**: Fine-tune → test → curate more data → repeat
        
        #### Audio Reconstruction
        
        Since ODAS provides only magnitude spectra (1024 frequency bins), we use:
        - **Griffin-Lim Algorithm**: Iterative phase reconstruction
        - **Overlap-Add**: For temporal continuity across frames
        - Quality is sufficient for training, though not perfect for human listening
        
        #### TensorFlow Training Example
        
        ```python
        import tensorflow as tf
        import tensorflow_hub as hub
        import pandas as pd
        
        # Load YAMNet
        model = hub.load('https://tfhub.dev/google/yamnet/1')
        
        # Load your dataset
        df = pd.read_csv('outputs/yamnet_datasets/yamnet_train_001/labels.csv')
        train_df = df[df['fold'] == 'train']
        
        # Create dataset
        def load_audio(filename):
            audio, sr = tf.audio.decode_wav(tf.io.read_file(filename))
            return audio[:, 0]  # mono
        
        # Fine-tune transfer learning style
        # (Add classification head on top of YAMNet embeddings)
        ```
        
        #### Troubleshooting
        
        - **No samples curated**: Adjust thresholds, ensure YAMNet is classifying
        - **Audio quality poor**: Increase overlap_frames, check ODAS bin quality
        - **Imbalanced labels**: Collect more data for underrepresented classes
        
        #### Next Steps
        
        - See [TensorFlow YAMNet Tutorial](https://www.tensorflow.org/hub/tutorials/yamnet)
        - Check `outputs/yamnet_datasets/` for your curated data
        - Use Dataset Visualizer tab to explore samples
        """)