"""
The streamlit app that has the following components

1. Load sources.csv config file and be able to view it and listen to its audios
2. A scene configurator module (configurator.py) to create scene json files
3. renderer module (renderer.py) to simulate the room and generate multi-channel audio for the mic array
4. A simulator module (simulator.py) to run the simulation based on a selected scene
5. An analyzer module (analyzer.py) to analyze the output of the simulation

We will add more modules later for training and evaluation
"""

import streamlit as st
import pandas as pd
import os
import json
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

# Import custom modules
from configurator import SceneConfigurator, DatasetConfigurator
from renderer import AudioRenderer
from simulator import SimulationRunner
from analyzer import ResultAnalyzer
from custom_simulator import CustomSimulator
from odas_simulator import ODASSimulator

# Configuration
SOURCES_CSV_PATH = "/home/azureuser/config/sources.csv"
SCENES_DIR = "/home/azureuser/config/scenes"
OUTPUT_DIR = "/home/azureuser/simulator/outputs"
# Classifier logs directory - matches classifier_log_dir = "./ClassifierLogs" in config
# Since odaslive runs from z_odas_newbeamform/build, the actual path is:
ODAS_LOGS_DIR = "/home/azureuser/z_odas_newbeamform/build/ClassifierLogs"

# Ensure directories exist
os.makedirs(SCENES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Audio Simulation & Classification Pipeline",
    page_icon="🎵",
    layout="wide"
)

# Initialize session state
if 'sources_df' not in st.session_state:
    st.session_state.sources_df = None
if 'current_scene' not in st.session_state:
    st.session_state.current_scene = None
if 'rendered_audio_path' not in st.session_state:
    st.session_state.rendered_audio_path = None

def load_sources():
    """Load the sources CSV file"""
    try:
        df = pd.read_csv(SOURCES_CSV_PATH)
        # Validate paths
        df['exists'] = df['wav_path'].apply(os.path.exists)
        return df
    except Exception as e:
        st.error(f"Error loading sources: {e}")
        return None

def load_audio(path, sr=16000):
    """Load audio file"""
    try:
        audio, _ = librosa.load(path, sr=sr)
        return audio
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None

def main():
    st.title("🎵 Audio Simulation & Classification Pipeline")
    st.markdown("### Synthetic Data Generation for Directional Audio Classification")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Module",
        ["📁 Sources Library", "🎨 Scene Configurator", "🔊 Audio Renderer", 
         "⚙️ ODAS Simulator", "🔬 Custom DOA Processor", 
         "📊 Results Analyzer", "🎯 YAMNet Datasets"]
         # Removed "🤖 Model Training" - using YAMNet instead
    )
    
    # Load sources if not already loaded
    if st.session_state.sources_df is None:
        with st.spinner("Loading audio sources..."):
            st.session_state.sources_df = load_sources()
    
    # Route to appropriate page
    if page == "📁 Sources Library":
        show_sources_library()
    elif page == "🎨 Scene Configurator":
        show_scene_configurator()
    elif page == "🔊 Audio Renderer":
        show_audio_renderer()
    elif page == "⚙️ ODAS Simulator":
        show_simulator()
    elif page == "🔬 Custom DOA Processor":
        show_custom_simulator()
    # elif page == "🎯 Improved ODAS Processor":
    #     show_odas_processor()
    elif page == "📊 Results Analyzer":
        show_analyzer()
    elif page == "🎯 YAMNet Datasets":
        show_dataset_manager()
    # Removed Model Training - using YAMNet instead
    # elif page == "🤖 Model Training":
    #     show_model_training()

def show_sources_library():
    """Display the sources library"""
    st.header("📁 Audio Sources Library")
    
    if st.session_state.sources_df is None:
        st.error("Failed to load sources CSV")
        return
    
    df = st.session_state.sources_df
    
    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sources", len(df))
    with col2:
        st.metric("Directional", len(df[df['source_type'] == 'directional']))
    with col3:
        st.metric("Ambient", len(df[df['source_type'] == 'ambient']))
    
    # Filter options
    st.subheader("Filters")
    col1, col2 = st.columns(2)
    with col1:
        source_type_filter = st.multiselect(
            "Source Type",
            options=df['source_type'].unique(),
            default=df['source_type'].unique()
        )
    with col2:
        label_filter = st.multiselect(
            "Label",
            options=sorted(df['label'].unique()),
            default=[]
        )
    
    # Apply filters
    filtered_df = df[df['source_type'].isin(source_type_filter)]
    if label_filter:
        filtered_df = filtered_df[filtered_df['label'].isin(label_filter)]
    
    # Display table
    st.subheader("Sources")
    st.dataframe(
        filtered_df[['label', 'source_type', 'wav_path', 'exists']],
        width='stretch'
    )
    
    # Audio preview
    st.subheader("Audio Preview")
    if len(filtered_df) > 0:
        selected_idx = st.selectbox(
            "Select audio to preview",
            options=range(len(filtered_df)),
            format_func=lambda i: f"{filtered_df.iloc[i]['label']} - {filtered_df.iloc[i]['wav_path']}"
        )
        
        if st.button("Load & Play"):
            audio_path = filtered_df.iloc[selected_idx]['wav_path']
            if os.path.exists(audio_path):
                audio = load_audio(audio_path)
                if audio is not None:
                    st.audio(audio, sample_rate=16000)
                    
                    # Show waveform
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.plot(audio)
                    ax.set_title(f"Waveform: {filtered_df.iloc[selected_idx]['label']}")
                    ax.set_xlabel("Sample")
                    ax.set_ylabel("Amplitude")
                    st.pyplot(fig)
            else:
                st.error("Audio file not found!")

def show_scene_configurator():
    """Scene configuration interface"""
    st.header("🎨 Scene Configurator")
    
    configurator = SceneConfigurator(SOURCES_CSV_PATH, SCENES_DIR)
    configurator.render()

def show_audio_renderer():
    """Audio rendering interface"""
    st.header("🔊 Audio Renderer")
    
    renderer = AudioRenderer(SCENES_DIR, OUTPUT_DIR)
    renderer.render()

def show_simulator():
    """Simulation runner interface"""
    st.header("⚙️ ODAS Simulator")
    
    simulator = SimulationRunner(OUTPUT_DIR, ODAS_LOGS_DIR)
    simulator.render()

def show_custom_simulator():
    """Custom DOA processor interface"""
    st.header("🔬 Custom DOA Processor")
    
    renders_dir = Path(OUTPUT_DIR) / 'renders'
    custom_sim = CustomSimulator(OUTPUT_DIR, renders_dir)
    custom_sim.render()

def show_odas_processor():
    """Improved ODAS processor interface"""
    st.header("🎯 Improved ODAS Processor")
    
    odas_sim = ODASSimulator(OUTPUT_DIR)
    odas_sim.render()

def show_analyzer():
    """Results analysis interface"""
    st.header("📊 Results Analyzer")
    
    analyzer = ResultAnalyzer(OUTPUT_DIR, ODAS_LOGS_DIR)
    analyzer.render()

def show_dataset_manager():
    """YAMNet dataset management interface"""
    st.header("🎯 YAMNet Dataset Manager")
    
    dataset_config = DatasetConfigurator(OUTPUT_DIR)
    dataset_config.render()

if __name__ == "__main__":
    main()