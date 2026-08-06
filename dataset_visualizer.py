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

        df = self._prepare_sample_dataframe(df)
        
        # Filters
        st.markdown("---")
        st.markdown("### 🔍 Filters")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            # Dataset label filter
            all_labels = ['All'] + sorted(df['label'].astype(str).unique().tolist())
            selected_label = st.selectbox("Dataset Label", all_labels)
        
        with col2:
            # Class / category filter (includes YAMNet prediction and GT label)
            all_classes = ['All'] + self._get_class_options(df)
            selected_class = st.selectbox("Class / Category", all_classes)
        
        with col3:
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
        
        with col4:
            # Curation reason filter
            if 'curation_reason' in df.columns:
                all_reasons = ['All'] + sorted(df['curation_reason'].dropna().unique().tolist())
                selected_reason = st.selectbox("Curation Reason", all_reasons)
            else:
                selected_reason = 'All'

        with col5:
            # Minimum source distance filter
            selected_min_dist = st.slider(
                "Min Distance (m)",
                min_value=0.0,
                max_value=125.0,
                value=0.0,
                step=1.0,
                help="Ground-truth distance filter in meters (0-125m). Keeps clips with distance >= selected value."
            )

        group_by_class = st.checkbox(
            "Group samples by class/category",
            value=True,
            help="Show all samples for one class together so you can inspect the full set of spectrograms at once."
        )
        overlap_mode = st.selectbox(
            "Overlap mode",
            [
                "All clips",
                "Singletons only",
                "Overlap with 1",
                "Overlap with 2",
                "Overlap with 3+",
            ],
            index=0,
            help="Choose whether to browse isolated single clips or clips that overlap with exactly 1, 2, or 3+ other clips."
        )

        # Do not render heavy sample content until filters are explicitly applied.
        filter_token = (
            selected_label,
            selected_class,
            float(conf_range[0]),
            float(conf_range[1]),
            selected_reason,
            float(selected_min_dist),
            overlap_mode,
            bool(group_by_class),
        )
        apply_key = f"apply_filters_{selected_dataset}"
        token_key = f"applied_filter_token_{selected_dataset}"

        if st.button("Apply Filters / Load Clips", key=apply_key, type="primary"):
            st.session_state[token_key] = filter_token

        if st.session_state.get(token_key) != filter_token:
            st.info("Select filters, then click 'Apply Filters / Load Clips' to load sample cards.")
            return
        
        # Apply filters
        filtered_df = df.copy()
        
        if selected_label != 'All':
            filtered_df = filtered_df[filtered_df['label'].astype(str) == selected_label]

        if selected_class != 'All':
            filtered_df = filtered_df[
                filtered_df['class_group'].astype(str).eq(selected_class) |
                filtered_df['label'].astype(str).eq(selected_class) |
                filtered_df['yamnet_class'].astype(str).eq(selected_class) |
                filtered_df['ground_truth'].astype(str).eq(selected_class)
            ]
        
        filtered_df = filtered_df[
            (filtered_df['yamnet_confidence'] >= conf_range[0]) &
            (filtered_df['yamnet_confidence'] <= conf_range[1])
        ]
        
        if selected_reason != 'All' and 'curation_reason' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['curation_reason'].str.contains(selected_reason, na=False)]

        if 'source_distance_m' in filtered_df.columns:
            dist_numeric = pd.to_numeric(filtered_df['source_distance_m'], errors='coerce')
            filtered_df = filtered_df[dist_numeric.fillna(-np.inf) >= float(selected_min_dist)]

        overlap_lookup = self._build_overlap_lookup(df)
        if overlap_mode == 'Singletons only':
            filtered_df = filtered_df[
                filtered_df['filename'].astype(str).map(lambda name: len(overlap_lookup.get(str(name), [])) == 0)
            ]
        elif overlap_mode == 'Overlap with 1':
            filtered_df = filtered_df[
                filtered_df['filename'].astype(str).map(lambda name: len(overlap_lookup.get(str(name), [])) == 1)
            ]
        elif overlap_mode == 'Overlap with 2':
            filtered_df = filtered_df[
                filtered_df['filename'].astype(str).map(lambda name: len(overlap_lookup.get(str(name), [])) == 2)
            ]
        elif overlap_mode == 'Overlap with 3+':
            filtered_df = filtered_df[
                filtered_df['filename'].astype(str).map(lambda name: len(overlap_lookup.get(str(name), [])) >= 3)
            ]
        
        st.info(f"📊 Showing {len(filtered_df)} of {len(df)} samples")
        
        # Sample browser
        st.markdown("---")
        st.markdown("### 🎵 Sample Browser")
        
        if filtered_df.empty:
            st.warning("No samples match the filters")
            return

        overlap_lookup = self._build_overlap_lookup(df)
        
        if group_by_class:
            if selected_class == 'All':
                st.info("Choose a specific value in 'Class / Category' to enable clip selection mode.")
                return

            class_rows = filtered_df.sort_values('timestamp').reset_index(drop=True)
            st.markdown(f"#### {selected_class} ({len(class_rows)} samples)")

            if class_rows.empty:
                st.warning("No clips match current filters.")
                return

            selected_idx_key = f"selected_clip_idx_{selected_dataset}_{selected_class}_{overlap_mode}"
            if selected_idx_key not in st.session_state:
                st.session_state[selected_idx_key] = 0
            if int(st.session_state[selected_idx_key]) >= len(class_rows):
                st.session_state[selected_idx_key] = 0

            st.markdown("**Clip candidates:**")
            clip_options = list(range(len(class_rows)))

            def _clip_label(idx):
                clip_row = class_rows.iloc[idx]
                overlaps = overlap_lookup.get(str(clip_row.get('filename', '')), [])
                overlap_count = len(overlaps)
                start_s = float(clip_row.get('timestamp', 0.0) or 0.0)
                duration_s = float(clip_row.get('duration_s', 0.5) or 0.5)
                end_s = start_s + duration_s
                overlap_classes = '/'.join(sorted({str(ov.get('label', 'unknown')) for ov in overlaps[:4]})) or 'none'
                overlap_durations = '/'.join(f"{float(ov.get('duration_s', 0.0) or 0.0):.2f}s" for ov in overlaps[:3]) or 'none'
                return (
                    f"#{idx + 1} | {clip_row.get('label', 'unknown')} | "
                    f"YAMNet {clip_row.get('yamnet_class', 'unknown')} | "
                    f"conf {float(clip_row.get('yamnet_confidence', 0.0)):.2f} | ov {overlap_count}:{overlap_classes} | "
                    f"start {start_s:.3f}s end {end_s:.3f}s dur {duration_s:.2f}s | "
                    f"ovdur {overlap_durations} | dist {float(clip_row.get('source_distance_m', 0.0) or 0.0):.2f}m"
                )

            selected_idx = st.selectbox(
                "Clip selection",
                clip_options,
                index=int(st.session_state[selected_idx_key]),
                format_func=_clip_label,
                key=f"clip_select_{selected_dataset}_{selected_class}_{overlap_mode}"
            )
            st.session_state[selected_idx_key] = int(selected_idx)

            selected_row = class_rows.iloc[selected_idx]
            overlaps = overlap_lookup.get(str(selected_row.get('filename', '')), [])
            st.markdown(f"### Selected Clip Details (Clip #{selected_idx + 1})")
            self._render_selected_clip_detail(selected_row, dataset_path, overlaps)
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
        for _, row in page_df.iterrows():
            overlaps = overlap_lookup.get(str(row.get('filename', '')), [])
            self._render_summary_card(row, dataset_path, overlaps)
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
    
    def _prepare_sample_dataframe(self, df):
        """Normalize and enrich sample metadata for browsing."""
        enriched = df.copy()

        def _safe_str(value):
            if pd.isna(value):
                return ''
            if isinstance(value, str):
                return value.strip()
            return str(value)

        def _parse_position(value):
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    parsed = json.loads(value.replace("'", '"'))
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    return {}
            return {}

        def _distance_m(position_dict):
            try:
                x = float(position_dict.get('x', 0.0))
                y = float(position_dict.get('y', 0.0))
                z = float(position_dict.get('z', 0.0))
                return float(np.sqrt(x * x + y * y + z * z))
            except Exception:
                return None

        def _duration_seconds(row):
            if 'duration_s' in row.index and not pd.isna(row['duration_s']):
                try:
                    return float(row['duration_s'])
                except Exception:
                    pass
            if 'stitched_duration_s' in row.index and not pd.isna(row['stitched_duration_s']):
                try:
                    return float(row['stitched_duration_s'])
                except Exception:
                    pass
            if 'n_frames' in row.index and not pd.isna(row['n_frames']):
                try:
                    return float(int(row['n_frames']) * 128 / 16000)
                except Exception:
                    pass
            return 0.5

        enriched['label'] = enriched['label'].fillna('unknown').astype(str)
        enriched['yamnet_class'] = enriched['yamnet_class'].fillna('unknown').astype(str)
        enriched['ground_truth'] = enriched['ground_truth'].fillna('unknown').astype(str)
        enriched['source_label'] = enriched.apply(
            lambda r: _safe_str(r.get('source_label')) or _safe_str(r.get('ground_truth')) or _safe_str(r.get('label')) or _safe_str(r.get('yamnet_class')),
            axis=1,
        )
        enriched['class_group'] = enriched.apply(
            lambda r: _safe_str(r.get('label')) or _safe_str(r.get('yamnet_class')) or _safe_str(r.get('ground_truth')) or _safe_str(r.get('source_label')) or 'Unknown',
            axis=1,
        )
        enriched['position_dict'] = enriched['position'].apply(_parse_position)
        enriched['source_distance_m'] = enriched['position_dict'].apply(_distance_m)
        enriched['duration_s'] = enriched.apply(_duration_seconds, axis=1)
        enriched['time_end_s'] = enriched['timestamp'].fillna(0).astype(float) + enriched['duration_s']
        return enriched

    def _get_class_options(self, df):
        """Return sorted class options derived from labels and predictions."""
        values = []
        for col in ['label', 'yamnet_class', 'ground_truth', 'source_label', 'class_group']:
            if col in df.columns:
                values.extend([str(v) for v in df[col].dropna().tolist() if str(v).strip()])
        uniq = sorted({v for v in values if v not in {'', 'unknown', 'Unknown', 'unclassified', 'N/A'}})
        return uniq

    def _build_overlap_lookup(self, df):
        """Find samples that overlap in time, across any class."""
        lookup = {}
        rows = df.reset_index(drop=True)

        for idx, row in rows.iterrows():
            overlaps = []
            row_start = float(row.get('timestamp', 0.0) or 0.0)
            row_end = row_start + max(float(row.get('duration_s', 0.5) or 0.5), 0.1)

            for jdx, other in rows.iterrows():
                if idx == jdx:
                    continue

                other_start = float(other.get('timestamp', 0.0) or 0.0)
                other_end = other_start + max(float(other.get('duration_s', 0.5) or 0.5), 0.1)
                if row_start < other_end and other_start < row_end:
                    overlaps.append({
                        'filename': str(other.get('filename', '')),
                        'label': str(other.get('label', 'unknown')),
                        'yamnet_class': str(other.get('yamnet_class', 'unknown')),
                        'timestamp': float(other.get('timestamp', 0.0) or 0.0),
                        'duration_s': float(other.get('duration_s', 0.5) or 0.5),
                        'source_distance_m': other.get('source_distance_m'),
                        'spectra_file': other.get('spectra_file', ''),
                        'position_dict': other.get('position_dict', {}),
                    })

            lookup[str(row.get('filename', ''))] = overlaps[:4]

        return lookup

    def _display_dataset_overview(self, stats):
        """Display dataset overview statistics"""
        st.markdown("### 📋 Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Samples", stats['sample_count'])
        
        with col2:
            st.metric("Audio Files", stats['actual_audio_files'])
        
        with col3:
            st.metric("Unique Labels", len(stats['samples_by_label']))
        
        with col4:
            st.metric("Runs Processed", stats['runs_processed'])
        
        with col5:
            # Count how many samples have .bin spectral files
            bins_dir = Path(stats['path']) / 'bins'
            bin_count = len(list(bins_dir.glob('*.bin'))) if bins_dir.exists() else 0
            st.metric("Bin Files", bin_count)
        
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
    
    def _render_summary_card(self, row, dataset_path, overlaps=None):
        """Render a compact summary card with visible metadata, overlap info, and a spectrogram preview."""
        timestamp_str = f"{row.get('timestamp', 0):.3f}s"
        label = str(row.get('label', 'unknown'))
        yamnet_class = str(row.get('yamnet_class', 'unknown'))
        confidence = float(row.get('yamnet_confidence', 0.0))
        source_label = str(row.get('source_label', row.get('ground_truth', 'unknown')))
        source_distance = row.get('source_distance_m')
        overlap_count = len(overlaps or [])
        duration_s = float(row.get('duration_s', 0.5) or 0.5)
        main_filename = Path(str(row.get('filename', ''))).name
        spec_path = dataset_path / 'spectrograms' / Path(str(row.get('filename', ''))).with_suffix('.png').name
        audio_path = dataset_path / 'audio' / main_filename

        with st.container():
            st.markdown(
                f"**{label}** @ {timestamp_str} | YAMNet: {yamnet_class} | conf: {confidence:.2f} | "
                f"source: {source_label} | dist: {source_distance:.2f}m | duration: {duration_s:.2f}s | overlaps: {overlap_count}"
            )
            if overlaps:
                overlap_text = ' | '.join(
                    f"{ov['label']}@{ov['timestamp']:.2f}s({ov['yamnet_class']})" for ov in overlaps[:3]
                )
                st.caption(f"Overlap partners: {overlap_text}")
            else:
                st.caption("No nearby overlapping calls")

            if spec_path.exists():
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.caption("Spectrogram")
                    st.image(str(spec_path), use_container_width=True)
                with col_b:
                    st.caption("Quick view")
                    st.text(f"Run: {row.get('run_id', 'N/A')}")
                    st.text(f"Activity: {row.get('activity', 0):.3f}")
                    st.text(f"Reason: {row.get('curation_reason', 'N/A')}")
            else:
                st.caption(f"Run: {row.get('run_id', 'N/A')} | activity: {row.get('activity', 0):.3f} | reason: {row.get('curation_reason', 'N/A')}")

            if audio_path.exists():
                st.caption("Main clip audio")
                try:
                    with open(audio_path, 'rb') as f:
                        st.audio(f.read(), format='audio/wav')
                except Exception as e:
                    st.warning(f"Could not load main clip audio: {e}")
            else:
                st.caption(f"Main clip audio missing: {main_filename}")

            if overlaps:
                st.markdown("**Overlap comparisons (main vs overlap):**")
                for ov in overlaps:
                    ov_filename = Path(str(ov.get('filename', ''))).name
                    ov_spec_path = dataset_path / 'spectrograms' / Path(ov_filename).with_suffix('.png').name
                    ov_audio_path = dataset_path / 'audio' / ov_filename

                    st.caption(
                        f"Overlap: {ov.get('label', 'unknown')} @ {float(ov.get('timestamp', 0.0)):.3f}s | "
                        f"YAMNet: {ov.get('yamnet_class', 'unknown')}"
                    )

                    col_main, col_overlap = st.columns(2)
                    with col_main:
                        st.caption(f"Main: {Path(main_filename).stem} | {label} | {yamnet_class}")
                        if spec_path.exists():
                            st.image(str(spec_path), use_container_width=True)
                        else:
                            st.caption("Main spectrogram missing")

                        if audio_path.exists():
                            try:
                                with open(audio_path, 'rb') as f:
                                    st.audio(f.read(), format='audio/wav')
                            except Exception as e:
                                st.warning(f"Could not load main audio: {e}")

                    with col_overlap:
                        st.caption(f"Overlap: {Path(ov_filename).stem} | {ov.get('label', 'unknown')} | {ov.get('yamnet_class', 'unknown')}")
                        if ov_spec_path.exists():
                            st.image(str(ov_spec_path), use_container_width=True)
                        else:
                            st.caption("Overlap spectrogram missing")

                        if ov_audio_path.exists():
                            try:
                                with open(ov_audio_path, 'rb') as f:
                                    st.audio(f.read(), format='audio/wav')
                            except Exception as e:
                                st.warning(f"Could not load overlap audio: {e}")
                        else:
                            st.caption(f"Overlap audio missing: {ov_filename}")

    def _render_selected_clip_detail(self, row, dataset_path, overlaps=None):
        """Render the focused clip inspector for the selected class item."""
        overlaps = overlaps or []
        label = str(row.get('label', 'unknown'))
        yamnet_class = str(row.get('yamnet_class', 'unknown'))
        timestamp = float(row.get('timestamp', 0.0) or 0.0)
        duration_s = float(row.get('duration_s', 0.5) or 0.5)
        end_time = timestamp + duration_s

        st.markdown("**Selected clip metadata**")
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        with meta_col1:
            st.text(f"Label: {label}")
            st.text(f"YAMNet: {yamnet_class}")
            st.text(f"Confidence: {float(row.get('yamnet_confidence', 0.0)):.3f}")
            st.text(f"Run: {row.get('run_id', 'N/A')}")
            st.text(f"Start: {timestamp:.3f}s")
            st.text(f"End: {end_time:.3f}s")
            st.text(f"Duration: {duration_s:.2f}s")
        with meta_col2:
            st.text(f"Source label: {row.get('source_label', row.get('ground_truth', 'unknown'))}")
            st.text(f"Ground truth: {row.get('ground_truth', 'unknown')}")
            st.text(f"Activity: {float(row.get('activity', 0.0)):.3f}")
            st.text(f"Reason: {row.get('curation_reason', 'N/A')}")
            st.text(f"Source distance: {row.get('source_distance_m', 'N/A')}")
            if 'position_dict' in row and isinstance(row['position_dict'], dict):
                pos = row['position_dict']
                st.text(f"Main position: ({float(pos.get('x', 0.0)):.2f}, {float(pos.get('y', 0.0)):.2f}, {float(pos.get('z', 0.0)):.2f})")
        with meta_col3:
            st.text(f"Overlap count: {len(overlaps)}")
            if overlaps:
                st.text("Overlap sources:")
                for ov in overlaps[:4]:
                    st.text(f"- {ov.get('label', 'unknown')} @ {float(ov.get('timestamp', 0.0)):.3f}s")

        left_col, right_col = st.columns([1, 1])
        with left_col:
            main_spec_path = dataset_path / 'spectrograms' / Path(str(row.get('filename', ''))).with_suffix('.png').name
            main_audio_path = dataset_path / 'audio' / Path(str(row.get('filename', ''))).name
            st.markdown("**Main clip**")
            main_spec_col, main_audio_col = st.columns([3, 1])
            with main_spec_col:
                self._render_timed_spectrogram(
                    row,
                    dataset_path,
                    window_start=timestamp,
                    window_end=end_time,
                    chart_key_prefix="main_clip"
                )
            with main_audio_col:
                st.caption("Audio")
                if main_audio_path.exists():
                    try:
                        with open(main_audio_path, 'rb') as f:
                            st.audio(f.read(), format='audio/wav')
                    except Exception as e:
                        st.warning(f"Could not load main audio: {e}")
                else:
                    st.caption("Main audio missing")

        with right_col:
            st.markdown("**Relative position plot**")
            self._render_relative_position_plot(row, overlaps)

        if overlaps:
            st.markdown("**Overlap clips**")
            for start_idx in range(0, len(overlaps), 2):
                pair = overlaps[start_idx:start_idx + 2]
                pair_cols = st.columns(2)
                for col_idx, ov in enumerate(pair):
                    ov_filename = Path(str(ov.get('filename', ''))).name
                    ov_spec_path = dataset_path / 'spectrograms' / Path(ov_filename).with_suffix('.png').name
                    ov_audio_path = dataset_path / 'audio' / ov_filename
                    with pair_cols[col_idx]:
                        st.caption(
                            f"{ov.get('label', 'unknown')} | {ov.get('yamnet_class', 'unknown')} | "
                            f"@ {float(ov.get('timestamp', 0.0)):.3f}s | dur {float(ov.get('duration_s', 0.0) or 0.0):.2f}s"
                        )
                        self._render_timed_spectrogram(
                            ov,
                            dataset_path,
                            window_start=timestamp,
                            window_end=end_time,
                            chart_key_prefix=f"overlap_{start_idx}_{col_idx}"
                        )
                        if ov_audio_path.exists():
                            try:
                                with open(ov_audio_path, 'rb') as f:
                                    st.audio(f.read(), format='audio/wav')
                            except Exception as e:
                                st.warning(f"Could not load overlap audio: {e}")
                        else:
                            st.caption("Overlap audio missing")

    def _render_timed_spectrogram(self, clip_row, dataset_path, window_start, window_end, chart_key_prefix):
        """Render a spectrogram on a shared absolute-time axis using the clip's .bin data when available."""
        spectra_file = clip_row.get('spectra_file', '')
        clip_label = str(clip_row.get('label', 'unknown'))
        clip_timestamp = float(clip_row.get('timestamp', 0.0) or 0.0)
        clip_duration = max(float(clip_row.get('duration_s', 0.5) or 0.5), 0.1)

        if spectra_file and isinstance(spectra_file, str) and spectra_file.strip():
            bin_path = dataset_path / spectra_file
            if bin_path.exists():
                try:
                    raw = np.fromfile(bin_path, dtype=np.float32)
                    actual_frames = raw.size // 257
                    if actual_frames > 0:
                        spectra = raw[:actual_frames * 257].reshape(actual_frames, 257)
                        db = 20.0 * np.log10(np.maximum(spectra, 1e-6))
                        frame_times = np.linspace(clip_timestamp, clip_timestamp + clip_duration, actual_frames, endpoint=False)
                        freq_hz = np.linspace(0, 8000, 257)
                        fig = go.Figure(go.Heatmap(
                            z=db.T,
                            x=frame_times,
                            y=freq_hz,
                            colorscale='Turbo',
                            zmin=float(np.percentile(db, 5)),
                            zmax=float(np.percentile(db, 99)),
                            showscale=True,
                            colorbar=dict(title='dB'),
                        ))
                        fig.update_layout(
                            title=f"Spectrogram - {clip_label}",
                            xaxis_title='Time (s)',
                            yaxis_title='Frequency',
                            xaxis=dict(range=[window_start, window_end]),
                            height=320,
                            margin=dict(l=55, r=20, t=40, b=45),
                            paper_bgcolor='#0e1117',
                            plot_bgcolor='#0e1117',
                            font=dict(color='white'),
                        )
                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                            key=f"{chart_key_prefix}_{str(clip_row.get('filename', 'na'))}_{window_start:.3f}_{window_end:.3f}"
                        )
                        return
                except Exception as e:
                    st.warning(f"Could not render aligned spectrogram: {e}")

        spec_path = dataset_path / 'spectrograms' / Path(str(clip_row.get('filename', ''))).with_suffix('.png').name
        if spec_path.exists():
            st.image(str(spec_path), use_container_width=True)
        else:
            st.caption("Spectrogram missing")

    def _render_relative_position_plot(self, main_row, overlaps):
        """Render a quadrant plot of overlap positions relative to the main clip."""
        main_pos = main_row.get('position_dict', {}) if isinstance(main_row.get('position_dict', {}), dict) else {}
        main_x = float(main_pos.get('x', 0.0) or 0.0)
        main_y = float(main_pos.get('y', 0.0) or 0.0)

        points = [{
            'name': f"Main: {main_row.get('label', 'unknown')}",
            'dx': 0.0,
            'dy': 0.0,
            'color': '#1f77b4',
            'symbol': 'star'
        }]

        for ov in overlaps:
            ov_pos = ov.get('position_dict', {}) if isinstance(ov.get('position_dict', {}), dict) else {}
            ov_x = float(ov_pos.get('x', 0.0) or 0.0)
            ov_y = float(ov_pos.get('y', 0.0) or 0.0)
            points.append({
                'name': f"{ov.get('label', 'unknown')} ({ov.get('yamnet_class', 'unknown')})",
                'dx': ov_x - main_x,
                'dy': ov_y - main_y,
                'color': '#ff7f0e',
                'symbol': 'circle'
            })

        max_range = 1.0

        fig = go.Figure()
        for point in points:
            fig.add_trace(go.Scatter(
                x=[point['dx']],
                y=[point['dy']],
                mode='markers+text',
                text=[point['name']],
                textposition='top center',
                marker=dict(size=14 if point['symbol'] == 'star' else 10, color=point['color'], symbol=point['symbol']),
                hovertemplate='%{text}<br>dx=%{x:.2f}<br>dy=%{y:.2f}<extra></extra>',
                showlegend=False,
            ))

        fig.add_shape(type='line', x0=0, y0=-max_range, x1=0, y1=max_range, line=dict(color='gray', width=1, dash='dot'))
        fig.add_shape(type='line', x0=-max_range, y0=0, x1=max_range, y1=0, line=dict(color='gray', width=1, dash='dot'))
        fig.update_layout(
            title='Relative positions around the selected clip (meters)',
            xaxis_title='Relative X (m)',
            yaxis_title='Relative Y (m)',
            xaxis=dict(range=[-max_range, max_range], zeroline=False),
            yaxis=dict(range=[-max_range, max_range], zeroline=False, scaleanchor='x', scaleratio=1),
            height=360,
            margin=dict(l=30, r=20, t=50, b=40),
        )
        chart_key = (
            f"relative_plot_{str(main_row.get('run_id', 'na'))}_"
            f"{str(main_row.get('filename', 'na'))}_"
            f"{float(main_row.get('timestamp', 0.0) or 0.0):.3f}"
        )
        st.plotly_chart(fig, use_container_width=True, key=chart_key)

    def _display_sample(self, row, dataset_path, overlaps=None):
        """Display a single sample with audio playback and overlap inspection."""
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
                    st.text(f"Dataset label: {label}")
                    st.text(f"YAMNet class: {yamnet_class}")
                    st.text(f"Confidence: {confidence:.3f}")
                    st.text(f"Source label: {row.get('source_label', row.get('ground_truth', 'unknown'))}")
                
                with meta_col2:
                    st.text(f"Run: {row.get('run_id', 'N/A')}")
                    st.text(f"Activity: {row.get('activity', 0):.3f}")
                    st.text(f"Duration: {row.get('duration_s', 0.5):.2f}s")
                    if 'curation_reason' in row:
                        st.text(f"Reason: {row.get('curation_reason', 'N/A')}")

                # Source location / distance
                source_distance = row.get('source_distance_m')
                if source_distance is not None:
                    st.text(f"Source distance: {source_distance:.2f} m")
                if 'position' in row and isinstance(row['position'], str):
                    try:
                        pos = json.loads(row['position'].replace("'", '"'))
                        st.text(f"Position: ({pos.get('x', 0):.2f}, {pos.get('y', 0):.2f}, {pos.get('z', 0):.2f})")
                    except Exception:
                        pass

                if overlaps:
                    st.text(f"Overlaps: {len(overlaps)} other call(s)")
            
            with col2:
                # Comparison
                gt = row.get('ground_truth', 'unknown')
                if gt != 'unknown' and gt != yamnet_class:
                    st.markdown("**⚠️ Mismatch:**")
                    st.text(f"Ground Truth: {gt}")
                    st.text(f"YAMNet: {yamnet_class}")
                else:
                    st.markdown("**✅ Match**" if gt != 'unknown' else "**❓ Unknown GT**")
                
                # Bin / spectra info
                spectra_file = row.get('spectra_file', '')
                n_frames = int(row.get('n_frames', 0)) if 'n_frames' in row.index else 0
                if spectra_file and isinstance(spectra_file, str) and spectra_file.strip():
                    bin_path = dataset_path / spectra_file
                    duration_s = n_frames * 128 / 16000  # hop_length=128, sr=16000
                    st.markdown("**🔬 Spectral .bin:**")
                    st.text(f"Frames: {n_frames}  (~{duration_s:.1f}s)")
                    st.text(f"Shape: {n_frames}×257 float32")
                    if not bin_path.exists():
                        st.caption("⚠️ .bin missing — re-run curation")
                elif 'spectra_file' not in row.index:
                    st.caption("⚠️ No spectra_file column — re-run curation")
            
            # Audio playback
            audio_path = dataset_path / 'audio' / row['filename']
            
            if audio_path.exists():
                st.markdown("**🔊 Audio (Griffin-Lim reconstruction):**")
                
                # Load and display audio
                try:
                    with open(audio_path, 'rb') as f:
                        audio_bytes = f.read()
                    st.audio(audio_bytes, format='audio/wav')
                except Exception as e:
                    st.error(f"Failed to load audio: {e}")
            else:
                st.warning(f"Audio file not found: {row['filename']}")
            
            # Spectrogram visualization (PNG saved by curator)
            spec_path = dataset_path / 'spectrograms' / row['filename'].replace('.wav', '.png')
            
            if spec_path.exists():
                st.markdown("**📊 Spectrogram (from .bin):**")
                st.image(str(spec_path), use_column_width=True)

            if overlaps:
                with st.expander("🔄 Side-by-side overlap compare", expanded=False):
                    main_spec = spec_path if spec_path.exists() else None
                    for ov in overlaps:
                        ov_filename = str(ov.get('filename', ''))
                        ov_spec = dataset_path / 'spectrograms' / Path(ov_filename).with_suffix('.png').name
                        if not ov_spec.exists():
                            continue
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.caption(f"Main clip: {Path(row.get('filename', '')).stem}")
                            if main_spec and main_spec.exists():
                                st.image(str(main_spec), use_column_width=True)
                        with col_b:
                            st.caption(f"Overlap: {Path(ov_filename).stem} | {ov['label']} | {ov['yamnet_class']}")
                            st.image(str(ov_spec), use_column_width=True)
            
            # Raw .bin heatmap (interactive)
            spectra_file = row.get('spectra_file', '')
            if spectra_file and isinstance(spectra_file, str) and spectra_file.strip():
                bin_path = dataset_path / spectra_file
                if bin_path.exists():
                    with st.expander("🔬 Raw Spectral Heatmap (.bin)", expanded=False):
                        try:
                            raw = np.fromfile(bin_path, dtype=np.float32)
                            actual_frames = raw.size // 257
                            if actual_frames > 0:
                                spectra = raw[:actual_frames * 257].reshape(actual_frames, 257)
                                # Convert to dB for visibility
                                db = 20.0 * np.log10(np.maximum(spectra, 1e-6))
                                vmin = float(np.percentile(db, 5))
                                vmax = float(np.percentile(db, 99))
                                time_s = (np.arange(actual_frames) * 128 / 16000).tolist()
                                freq_hz = np.linspace(0, 8000, 257).tolist()
                                fig = go.Figure(go.Heatmap(
                                    z=db.T,
                                    x=time_s,
                                    y=freq_hz,
                                    colorscale='Turbo',
                                    zmin=vmin,
                                    zmax=vmax,
                                    showscale=True,
                                    colorbar=dict(title='dB'),
                                ))
                                fig.update_layout(
                                    title=f'Spectrogram (dB) — {actual_frames} frames × 257 bins',
                                    xaxis_title='Time (s)',
                                    yaxis_title='Frequency (Hz)',
                                    height=320,
                                    margin=dict(l=60, r=20, t=40, b=50),
                                    paper_bgcolor='#0e1117',
                                    plot_bgcolor='#0e1117',
                                    font=dict(color='white'),
                                    xaxis=dict(gridcolor='#333'),
                                    yaxis=dict(gridcolor='#333'),
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption(
                                    f"Shape: {actual_frames}×257 | "
                                    f"Duration: ~{actual_frames * 128 / 16000:.2f}s | "
                                    f"Range: {vmin:.0f}–{vmax:.0f} dB | "
                                    "dB magnitude (pre-mel) — same input ODAS feeds to YAMNet"
                                )
                        except Exception as e:
                            st.warning(f"Could not render .bin: {e}")

            if overlaps:
                with st.expander("🔗 Overlapping calls", expanded=False):
                    for ov in overlaps:
                        ov_label = f"{ov['label']} @ {ov['timestamp']:.3f}s"
                        ov_distance = ov.get('source_distance_m')
                        ov_dist_text = f" | dist {ov_distance:.2f}m" if ov_distance is not None else ''
                        st.caption(f"{ov_label} | {ov['yamnet_class']}{ov_dist_text}")
                        ov_path = dataset_path / 'spectrograms' / Path(ov['filename']).with_suffix('.png').name
                        if ov_path.exists():
                            st.image(str(ov_path), use_column_width=True)
    
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
