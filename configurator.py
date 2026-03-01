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