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
from gt_dataset_builder import GTDatasetBuilder

PROJECT_ROOT = Path(__file__).resolve().parent
HOME_DIR = Path.home()

# Configuration (can be overridden by environment variables)
SOUNDS_DIR = os.getenv("SOUNDS_DIR", str(HOME_DIR / "sounds"))
SCENES_DIR = os.getenv("SCENES_DIR", str(PROJECT_ROOT / "config" / "scenes"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs"))
ODAS_LOGS_DIR = os.getenv("ODAS_LOGS_DIR", str(PROJECT_ROOT / "ClassifierLogs"))

# Ensure directories exist
os.makedirs(SCENES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ODAS_LOGS_DIR, exist_ok=True)

# Page configuration
st.set_page_config(
    page_title="Audio Simulation & Classification Pipeline",
    page_icon="🎵",
    layout="wide"
)

# Initialize session state
if 'current_scene' not in st.session_state:
    st.session_state.current_scene = None
if 'rendered_audio_path' not in st.session_state:
    st.session_state.rendered_audio_path = None

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
        ["🎨 Scene Configurator", "🔊 Audio Renderer",
         "⚙️ ODAS Simulator", "🔬 Custom DOA Processor",
         "📊 Results Analyzer", "🎯 YAMNet Datasets",
         "🏷️ GT Dataset Builder", "🧠 Fine-Tune YAMNet"]
    )

    # Route to appropriate page
    if page == "🎨 Scene Configurator":
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
    elif page == "🏷️ GT Dataset Builder":
        show_gt_dataset_builder()
    elif page == "🧠 Fine-Tune YAMNet":
        show_yamnet_finetuner()

def show_scene_configurator():
    """Scene configuration interface"""
    st.header("🎨 Scene Configurator")
    
    configurator = SceneConfigurator(SCENES_DIR, SOUNDS_DIR)
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

def show_gt_dataset_builder():
    """Ground-truth dataset builder from rendered audio"""
    st.header("🏷️ GT Dataset Builder")

    renders_dir = Path(OUTPUT_DIR) / 'renders'
    builder = GTDatasetBuilder(renders_dir, OUTPUT_DIR)
    builder.render()


def show_yamnet_finetuner():
    """YAMNet fine-tuning pipeline: dataset prep → train → export → deploy"""
    st.header("🧠 Fine-Tune YAMNet")
    st.markdown(
        "Train a custom YAMNet on your GT datasets, export to TFLite, "
        "and deploy to the ODAS simulator or live firmware."
    )

    from yamnet_finetuner_ui import YAMNetFinetunerUI
    ui = YAMNetFinetunerUI(OUTPUT_DIR)
    ui.render()


if __name__ == "__main__":
    main()