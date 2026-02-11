"""
Audio Classification Demo
Processes an audio file with overlapping windows and visualizes YAMNet predictions.
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple
import argparse

from yamnet_spectrum_classifier import YAMNetSpectrumClassifier, compute_magnitude_spectrum


class AudioWindowClassifier:
    """
    Process audio files with overlapping windows and classify using YAMNet.
    """
    
    def __init__(self, model_path: str, class_map_path: str):
        """
        Initialize classifier.
        
        Args:
            model_path: Path to yamnet_core.tflite
            class_map_path: Path to yamnet_class_map.csv
        """
        self.classifier = YAMNetSpectrumClassifier(model_path, class_map_path)
        self.sample_rate = self.classifier.SAMPLE_RATE
        self.window_size = self.classifier.FFT_SIZE  # Use FFT size for spectrum computation
        self.hop_size = self.classifier.FRAME_STEP
    
    def process_audio_file(self, audio_path: str, 
                          duration: float = None) -> Tuple[List[dict], np.ndarray]:
        """
        Process an audio file and return classifications.
        
        Args:
            audio_path: Path to audio file
            duration: Optional duration limit in seconds
            
        Returns:
            predictions: List of prediction dictionaries
            audio: Loaded audio array
        """
        print(f"\nProcessing: {audio_path}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=duration)
        print(f"  Loaded: {len(audio)} samples ({len(audio)/sr:.2f}s) at {sr} Hz")
        
        # Reset classifier
        self.classifier.reset()
        
        # Process audio in overlapping windows
        predictions = []
        num_frames = (len(audio) - self.window_size) // self.hop_size + 1
        
        print(f"  Processing {num_frames} frames...")
        
        for i in range(num_frames):
            # Extract window
            start_idx = i * self.hop_size
            end_idx = start_idx + self.window_size
            
            if end_idx > len(audio):
                break
            
            audio_window = audio[start_idx:end_idx]
            
            # Pad window if needed and compute magnitude spectrum
            if len(audio_window) < self.window_size:
                audio_window = np.pad(audio_window, (0, self.window_size - len(audio_window)))
            
            magnitude_spectrum = compute_magnitude_spectrum(audio_window, 
                                                           window_size=self.window_size,
                                                           hop_size=self.hop_size)
            
            # Add frame to classifier
            result = self.classifier.add_frame(magnitude_spectrum)
            
            if result is not None:
                class_id, class_name, confidence = result
                timestamp = start_idx / sr
                
                predictions.append({
                    'frame': i,
                    'timestamp': timestamp,
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
                
                print(f"    [{timestamp:6.2f}s] {class_name:30s} (conf: {confidence:.3f})")
        
        print(f"\n  Total predictions: {len(predictions)}")
        
        return predictions, audio
    
    def plot_predictions(self, predictions: List[dict], audio: np.ndarray, 
                        output_path: str = None):
        """
        Plot predictions timeline and audio waveform.
        
        Args:
            predictions: List of prediction dictionaries
            audio: Audio waveform
            output_path: Optional path to save plot
        """
        if not predictions:
            print("No predictions to plot!")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Extract data
        timestamps = [p['timestamp'] for p in predictions]
        confidences = [p['confidence'] for p in predictions]
        class_names = [p['class_name'] for p in predictions]
        
        # Plot 1: Waveform
        ax1 = axes[0]
        time_axis = np.arange(len(audio)) / self.sample_rate
        ax1.plot(time_axis, audio, linewidth=0.5, alpha=0.7)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Audio Waveform')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Confidence over time
        ax2 = axes[1]
        ax2.plot(timestamps, confidences, 'o-', linewidth=2, markersize=4)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Classification Confidence Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.0])
        
        # Plot 3: Class distribution
        ax3 = axes[2]
        
        # Count class occurrences
        class_counts = {}
        for p in predictions:
            name = p['class_name']
            class_counts[name] = class_counts.get(name, 0) + 1
        
        # Sort by frequency
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        top_classes = sorted_classes[:15]  # Show top 15
        
        classes = [c[0] for c in top_classes]
        counts = [c[1] for c in top_classes]
        
        y_pos = np.arange(len(classes))
        ax3.barh(y_pos, counts)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(classes, fontsize=9)
        ax3.set_xlabel('Occurrences')
        ax3.set_title('Top Detected Classes')
        ax3.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
        
        plt.show()
    
    def plot_prediction_timeline(self, predictions: List[dict], 
                                output_path: str = None):
        """
        Plot prediction timeline with color-coded classes.
        
        Args:
            predictions: List of prediction dictionaries
            output_path: Optional path to save plot
        """
        if not predictions:
            print("No predictions to plot!")
            return
        
        # Get unique classes
        unique_classes = sorted(set(p['class_name'] for p in predictions))
        class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
        
        # Create color map
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot timeline
        for i, pred in enumerate(predictions):
            timestamp = pred['timestamp']
            class_name = pred['class_name']
            confidence = pred['confidence']
            class_idx = class_to_idx[class_name]
            
            # Bar height based on confidence
            ax.barh(class_idx, width=0.96, left=timestamp, height=confidence*0.8,
                   color=colors[class_idx], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Formatting
        ax.set_yticks(range(len(unique_classes)))
        ax.set_yticklabels(unique_classes, fontsize=8)
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Detected Class', fontsize=12)
        ax.set_title('YAMNet Classification Timeline\n(Bar opacity indicates confidence)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nTimeline plot saved to: {output_path}")
        
        plt.show()


def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(
        description='Classify audio using YAMNet with overlapping windows'
    )
    parser.add_argument('audio_file', type=str, 
                       help='Path to audio file')
    parser.add_argument('--model', type=str, 
                       default='/home/azureuser/z_odas_newbeamform/models/yamnet_core.tflite',
                       help='Path to YAMNet TFLite model')
    parser.add_argument('--class-map', type=str,
                       default='/home/azureuser/z_odas_newbeamform/models/yamnet_class_map.csv',
                       help='Path to class map CSV')
    parser.add_argument('--duration', type=float, default=None,
                       help='Limit audio duration (seconds)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Directory for output plots')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting')
    
    args = parser.parse_args()
    
    # Check inputs
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    class_map_path = Path(args.class_map)
    if not class_map_path.exists():
        print(f"Error: Class map file not found: {class_map_path}")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize classifier
    print("=" * 70)
    print("YAMNet Audio Classifier Demo")
    print("=" * 70)
    
    classifier = AudioWindowClassifier(str(model_path), str(class_map_path))
    
    # Process audio
    predictions, audio = classifier.process_audio_file(
        str(audio_path), 
        duration=args.duration
    )
    
    if not predictions:
        print("\nNo predictions generated. Audio might be too short.")
        return
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total predictions: {len(predictions)}")
    print(f"Time span: {predictions[0]['timestamp']:.2f}s to {predictions[-1]['timestamp']:.2f}s")
    
    # Most common classes
    class_counts = {}
    for p in predictions:
        name = p['class_name']
        class_counts[name] = class_counts.get(name, 0) + 1
    
    print(f"\nTop 5 detected classes:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
        percentage = (count / len(predictions)) * 100
        print(f"  {class_name:30s}: {count:3d} ({percentage:5.1f}%)")
    
    # Average confidence
    avg_confidence = np.mean([p['confidence'] for p in predictions])
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    
    # Plot results
    if not args.no_plot:
        print("\n" + "=" * 70)
        print("GENERATING PLOTS")
        print("=" * 70)
        
        base_name = audio_path.stem
        
        # Main plot
        plot_path = output_dir / f"{base_name}_analysis.png"
        classifier.plot_predictions(predictions, audio, str(plot_path))
        
        # Timeline plot
        timeline_path = output_dir / f"{base_name}_timeline.png"
        classifier.plot_prediction_timeline(predictions, str(timeline_path))
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
