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
import numpy as np
from datetime import datetime
from pathlib import Path
import random

class SceneConfigurator:
    def __init__(self, sources_csv_path, scenes_dir):
        self.sources_csv_path = sources_csv_path
        self.scenes_dir = scenes_dir
        self.sources_df = pd.read_csv(sources_csv_path)
        
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
    
    def _get_available_files_for_label(self, label, source_type):
        """Get all available files for a given label"""
        filtered = self.sources_df[
            (self.sources_df['label'] == label) & 
            (self.sources_df['source_type'] == source_type)
        ]
        return filtered['wav_path'].tolist()
    
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
        
        # Add/Generate sources
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Add Source"):
                self._add_directional_source(scene)
        with col2:
            num_random = st.number_input("Number to generate", 1, 20, 1, key="num_random_dir")
        with col3:
            if st.button("🎲 Generate Random"):
                for _ in range(num_random):
                    self._add_directional_source(scene, randomize=True)
        
        # Display and edit existing sources
        if not scene['directional_sources']:
            st.info("No directional sources added. Click '➕ Add Source' to begin.")
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
                self.sources_df[self.sources_df['source_type'] == 'directional']['label'].unique()
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
    
    def _render_ambient_sources(self, scene):
        """Render ambient sources configuration"""
        st.markdown("### Ambient Sources")
        
        # Add/Generate sources
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ Add Ambient"):
                self._add_ambient_source(scene)
        with col2:
            num_random = st.number_input("Number to generate", 1, 10, 1, key="num_random_amb")
        with col3:
            if st.button("🎲 Generate Random", key="gen_random_amb"):
                for _ in range(num_random):
                    self._add_ambient_source(scene, randomize=True)
        
        # Display and edit existing sources
        if not scene['ambient_sources']:
            st.info("No ambient sources added.")
            return
        
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
                self.sources_df[self.sources_df['source_type'] == 'ambient']['label'].unique()
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
    
    def _add_directional_source(self, scene, randomize=False):
        """Add a new directional source"""
        directional_labels = self.sources_df[
            self.sources_df['source_type'] == 'directional'
        ]['label'].unique()
        
        if randomize:
            label = random.choice(directional_labels)
            azimuth = random.uniform(-180, 180)
            distance = random.uniform(1, scene['max_radius'])
            height = random.uniform(scene['min_height'], scene['max_height'])
            start_time = random.uniform(0, scene['duration'] * 0.7)
            duration = random.uniform(1, scene['duration'] - start_time)
            end_time = min(start_time + duration, scene['duration'])
        else:
            label = directional_labels[0]
            azimuth = 0
            distance = scene['max_radius'] / 2
            height = 0
            start_time = 0
            end_time = scene['duration']
        
        x, y, z = self._azimuth_elevation_to_cartesian(azimuth, distance, height)
        
        source = {
            'label': label,
            'wav_path': 'Random',
            'x': x,
            'y': y,
            'z': z,
            'start_time': start_time,
            'end_time': end_time,
            'repeat': False
        }
        
        scene['directional_sources'].append(source)
    
    def _add_ambient_source(self, scene, randomize=False):
        """Add a new ambient source"""
        ambient_labels = self.sources_df[
            self.sources_df['source_type'] == 'ambient'
        ]['label'].unique()
        
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
        """Visualize the scene configuration"""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(12, 5))
            
            # Top view (XY plane)
            ax1 = fig.add_subplot(121)
            ax1.set_xlim(-scene['max_radius'], scene['max_radius'])
            ax1.set_ylim(-scene['max_radius'], scene['max_radius'])
            ax1.set_xlabel('X (meters)')
            ax1.set_ylabel('Y (meters)')
            ax1.set_title('Top View (XY Plane)')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(0, color='k', linewidth=0.5)
            ax1.axvline(0, color='k', linewidth=0.5)
            
            # Draw mic array at origin
            ax1.plot(0, 0, 'r*', markersize=15, label='Mic Array')
            
            # Draw directional sources
            for idx, source in enumerate(scene['directional_sources']):
                ax1.plot(source['x'], source['y'], 'bo', markersize=8)
                ax1.annotate(
                    f"{idx+1}: {source['label']}\n({source['start_time']:.1f}s-{source['end_time']:.1f}s)",
                    (source['x'], source['y']),
                    fontsize=8,
                    ha='center'
                )
            
            # 3D view
            ax2 = fig.add_subplot(122, projection='3d')
            ax2.set_xlim(-scene['max_radius'], scene['max_radius'])
            ax2.set_ylim(-scene['max_radius'], scene['max_radius'])
            ax2.set_zlim(scene['min_height'], scene['max_height'])
            ax2.set_xlabel('X (meters)')
            ax2.set_ylabel('Y (meters)')
            ax2.set_zlabel('Z (meters)')
            ax2.set_title('3D View')
            
            # Draw mic array at origin
            ax2.plot([0], [0], [0], 'r*', markersize=15, label='Mic Array')
            
            # Draw directional sources
            for idx, source in enumerate(scene['directional_sources']):
                ax2.plot([source['x']], [source['y']], [source['z']], 'bo', markersize=8)
                ax2.text(source['x'], source['y'], source['z'], f"{idx+1}", fontsize=8)
            
            ax1.legend()
            ax2.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
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