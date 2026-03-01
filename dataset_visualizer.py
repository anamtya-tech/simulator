"""
Dataset Visualizer - Interactive visualization and audio playback for YAMNet datasets

Features:
1. Browse dataset samples with metadata
2. Play reconstructed audio directly in browser
3. View spectrograms and frequency bins
4. Filter by label, confidence, curation reason
5. Export subsets for fine-tuning
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import base64
from yamnet_dataset_curator import YAMNetDatasetCurator
from audio_reconstructor import AudioReconstructor


class DatasetVisualizer:
    """Interactive visualizer for YAMNet training datasets"""
    
    def __init__(self, curator=None):
        """
        Initialize visualizer.
        
        Args:
            curator: YAMNetDatasetCurator instance
        """
        if curator is None:
            curator = YAMNetDatasetCurator()
        
        self.curator = curator
        # n_fft=512 matches ODAS frameSize=512 (halfFrameSize=257 bins)
        self.reconstructor = AudioReconstructor(
            sample_rate=16000, n_fft=512, hop_length=128
        )
    
    def render(self):
        """Render the visualizer interface"""
        st.subheader("📊 YAMNet Dataset Visualizer")
        st.markdown("Browse, listen to, and analyze training datasets for YAMNet fine-tuning")
        
        # Dataset selection
        datasets = self.curator.list_datasets()
        
        if not datasets:
            st.warning("No datasets found. Create a dataset by running analysis with dataset curation enabled.")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_dataset = st.selectbox(
                "Select Dataset",
                datasets,
                index=datasets.index(self.curator.get_active_dataset()) 
                      if self.curator.get_active_dataset() in datasets else 0
            )
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        
        # Load dataset
        dataset_stats = self.curator.get_dataset_stats(selected_dataset)
        
        if dataset_stats is None:
            st.error("Failed to load dataset statistics")
            return
        
        # Display dataset overview
        self._display_dataset_overview(dataset_stats)
        
        # Load samples
        dataset_path = Path(dataset_stats['path'])
        labels_csv = dataset_path / 'labels.csv'
        
        if not labels_csv.exists():
            st.warning("No samples found in dataset")
            return
        
        df = pd.read_csv(labels_csv)
        
        if df.empty:
            st.warning("Dataset is empty")
            return
        
        # Filters
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Label filter
            all_labels = ['All'] + sorted(df['label'].unique().tolist())
            selected_label = st.selectbox("Label", all_labels)
        
        with col2:
            # Confidence filter
            min_conf = float(df['yamnet_confidence'].min())
            max_conf = float(df['yamnet_confidence'].max())
            
            # Only show slider if there's variation in confidence
            if max_conf > min_conf:
                conf_range = st.slider(
                    "YAMNet Confidence Range",
                    min_conf, max_conf, (min_conf, max_conf),
                    step=0.01
                )
            else:
                # All samples have same confidence
                st.info(f"All samples: {min_conf:.2f}")
                conf_range = (min_conf, max_conf)
        
        with col3:
            # Curation reason filter
            if 'curation_reason' in df.columns:
                all_reasons = ['All'] + sorted(df['curation_reason'].dropna().unique().tolist())
                selected_reason = st.selectbox("Curation Reason", all_reasons)
            else:
                selected_reason = 'All'
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_label != 'All':
            filtered_df = filtered_df[filtered_df['label'] == selected_label]
        
        filtered_df = filtered_df[
            (filtered_df['yamnet_confidence'] >= conf_range[0]) &
            (filtered_df['yamnet_confidence'] <= conf_range[1])
        ]
        
        if selected_reason != 'All' and 'curation_reason' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['curation_reason'].str.contains(selected_reason, na=False)]
        
        st.info(f"📊 Showing {len(filtered_df)} of {len(df)} samples")
        
        # Sample browser
        st.markdown("---")
        st.markdown("### 🎵 Sample Browser")
        
        if filtered_df.empty:
            st.warning("No samples match the filters")
            return
        
        # Pagination
        samples_per_page = 10
        total_pages = (len(filtered_df) + samples_per_page - 1) // samples_per_page
        
        col1, col2, col3 = st.columns([2, 1, 2])
        with col2:
            page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=total_pages,
                value=1,
                step=1
            )
        
        start_idx = (page - 1) * samples_per_page
        end_idx = min(start_idx + samples_per_page, len(filtered_df))
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        # Display samples
        for idx, row in page_df.iterrows():
            self._display_sample(row, dataset_path)
            st.markdown("---")
        
        # Visualizations
        st.markdown("### 📈 Dataset Analytics")
        self._display_analytics(df, filtered_df, dataset_stats)
        
        # Export options
        st.markdown("---")
        st.markdown("### 💾 Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export filtered samples
            if st.button("📥 Export Filtered Samples CSV", use_container_width=True):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    file_name=f"{selected_dataset}_filtered.csv",
                    mime="text/csv"
                )
        
        with col2:
            # Create TensorFlow dataset
            if st.button("🤖 Prepare TensorFlow Dataset", use_container_width=True):
                with st.spinner("Preparing dataset..."):
                    result = self.curator.create_tensorflow_dataset(selected_dataset)
                    st.success("✅ Dataset prepared for TensorFlow!")
                    st.json(result)
    
    def _display_dataset_overview(self, stats):
        """Display dataset overview statistics"""
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", stats['sample_count'])
        
        with col2:
            st.metric("Audio Files", stats['actual_audio_files'])
        
        with col3:
            st.metric("Unique Labels", len(stats['samples_by_label']))
        
        with col4:
            st.metric("Runs Processed", stats['runs_processed'])
        
        # Label distribution
        with st.expander("📊 Label Distribution", expanded=False):
            if stats['samples_by_label']:
                fig = px.bar(
                    x=list(stats['samples_by_label'].keys()),
                    y=list(stats['samples_by_label'].values()),
                    labels={'x': 'Label', 'y': 'Count'},
                    title='Samples per Label'
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Recent runs
        with st.expander("🏃 Recent Runs", expanded=False):
            if stats.get('recent_runs'):
                recent_df = pd.DataFrame(stats['recent_runs'])
                st.dataframe(recent_df, use_container_width=True)
    
    def _display_sample(self, row, dataset_path):
        """Display a single sample with audio playback"""
        # Create expandable section for each sample
        timestamp_str = f"{row.get('timestamp', 0):.3f}s"
        label = row['label']
        yamnet_class = row.get('yamnet_class', 'N/A')
        confidence = row.get('yamnet_confidence', 0)
        
        # Header
        header = f"🎵 {label} @ {timestamp_str} (YAMNet: {yamnet_class}, conf: {confidence:.2f})"
        
        with st.expander(header, expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Metadata
                st.markdown("**Metadata:**")
                meta_col1, meta_col2 = st.columns(2)
                
                with meta_col1:
                    st.text(f"Label: {label}")
                    st.text(f"YAMNet: {yamnet_class}")
                    st.text(f"Confidence: {confidence:.3f}")
                
                with meta_col2:
                    st.text(f"Run: {row.get('run_id', 'N/A')}")
                    st.text(f"Activity: {row.get('activity', 0):.3f}")
                    if 'curation_reason' in row:
                        st.text(f"Reason: {row.get('curation_reason', 'N/A')}")
                
                # Position
                if 'position' in row and isinstance(row['position'], str):
                    try:
                        pos = json.loads(row['position'].replace("'", '"'))
                        st.text(f"Position: ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})")
                    except:
                        pass
            
            with col2:
                # Comparison
                gt = row.get('ground_truth', 'unknown')
                if gt != 'unknown' and gt != yamnet_class:
                    st.markdown("**⚠️ Mismatch:**")
                    st.text(f"Ground Truth: {gt}")
                    st.text(f"YAMNet: {yamnet_class}")
                else:
                    st.markdown("**✅ Match**" if gt != 'unknown' else "**❓ Unknown GT**")
            
            # Audio playback
            audio_path = dataset_path / 'audio' / row['filename']
            
            if audio_path.exists():
                st.markdown("**🔊 Audio:**")
                
                # Load and display audio
                try:
                    with open(audio_path, 'rb') as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format='audio/wav')
                except Exception as e:
                    st.error(f"Failed to load audio: {e}")
            else:
                st.warning(f"Audio file not found: {row['filename']}")
            
            # Spectrogram visualization
            spec_path = dataset_path / 'spectrograms' / row['filename'].replace('.wav', '.png')
            
            if spec_path.exists():
                st.markdown("**📊 Spectrogram:**")
                st.image(str(spec_path), use_column_width=True)
    
    def _display_analytics(self, full_df, filtered_df, stats):
        """Display analytics and visualizations"""
        tab1, tab2, tab3 = st.tabs(["Label Distribution", "Confidence Analysis", "Temporal Distribution"])
        
        with tab1:
            # Label distribution comparison
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=("Full Dataset", "Filtered Dataset"),
                specs=[[{"type": "pie"}, {"type": "pie"}]]
            )
            
            # Full dataset
            full_label_counts = full_df['label'].value_counts()
            fig.add_trace(
                go.Pie(labels=full_label_counts.index, values=full_label_counts.values, name="Full"),
                row=1, col=1
            )
            
            # Filtered dataset
            if not filtered_df.empty:
                filtered_label_counts = filtered_df['label'].value_counts()
                fig.add_trace(
                    go.Pie(labels=filtered_label_counts.index, values=filtered_label_counts.values, name="Filtered"),
                    row=1, col=2
                )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Confidence distribution
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=full_df['yamnet_confidence'],
                name='Full Dataset',
                opacity=0.7,
                nbinsx=50
            ))
            
            if not filtered_df.empty:
                fig.add_trace(go.Histogram(
                    x=filtered_df['yamnet_confidence'],
                    name='Filtered',
                    opacity=0.7,
                    nbinsx=50
                ))
            
            fig.update_layout(
                title='YAMNet Confidence Distribution',
                xaxis_title='Confidence',
                yaxis_title='Count',
                barmode='overlay',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence by label
            if not filtered_df.empty:
                fig2 = px.box(
                    filtered_df,
                    x='label',
                    y='yamnet_confidence',
                    title='Confidence by Label',
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)
        
        with tab3:
            # Temporal distribution
            if 'timestamp' in filtered_df.columns and not filtered_df.empty:
                fig = px.scatter(
                    filtered_df,
                    x='timestamp',
                    y='yamnet_confidence',
                    color='label',
                    title='Samples Over Time',
                    height=400,
                    hover_data=['run_id', 'yamnet_class']
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No temporal data available")
            
            # Samples by run
            if 'run_id' in filtered_df.columns and not filtered_df.empty:
                run_counts = filtered_df['run_id'].value_counts()
                fig2 = px.bar(
                    x=run_counts.index,
                    y=run_counts.values,
                    title='Samples per Run',
                    labels={'x': 'Run ID', 'y': 'Count'},
                    height=400
                )
                st.plotly_chart(fig2, use_container_width=True)


def render_dataset_visualizer():
    """Standalone rendering function for Streamlit integration"""
    visualizer = DatasetVisualizer()
    visualizer.render()


if __name__ == '__main__':
    # For standalone testing
    st.set_page_config(
        page_title="YAMNet Dataset Visualizer",
        page_icon="🎵",
        layout="wide"
    )
    
    render_dataset_visualizer()
