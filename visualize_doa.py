#!/usr/bin/env python3
"""
Visualize Custom DOA results with matplotlib plots.

Usage:
    python visualize_doa.py <custom_doa.json>
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd


def load_results(filepath):
    """Load custom DOA results"""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_polar_sources(sources, ax):
    """Plot sources on polar coordinate system"""
    if not sources:
        return
    
    # Extract data
    azimuths = np.array([np.radians(s['azimuth']) for s in sources])
    frequencies = np.array([s['frequency'] for s in sources])
    counts = np.array([s['occurrence_count'] for s in sources])
    confidences = np.array([s['avg_confidence'] for s in sources])
    
    # Normalize radius by log(count)
    radii = np.log10(counts + 1)
    
    # Color by frequency
    colors = plt.cm.viridis(frequencies / frequencies.max())
    
    # Size by confidence
    sizes = confidences * 200
    
    # Plot
    scatter = ax.scatter(azimuths, radii, c=frequencies, s=sizes, 
                        alpha=0.7, cmap='viridis', edgecolors='black', linewidth=0.5)
    
    # Configure polar plot
    ax.set_theta_zero_location('N')  # 0° at top (front)
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_title('Source Directions\n(Size=Confidence, Color=Frequency)', pad=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Frequency (Hz)', rotation=270, labelpad=20)
    
    # Add direction labels
    ax.text(0, ax.get_ylim()[1] * 1.1, 'FRONT', ha='center', va='bottom', fontsize=10, weight='bold')
    ax.text(np.pi/2, ax.get_ylim()[1] * 1.1, 'RIGHT', ha='left', va='center', fontsize=10, weight='bold')
    ax.text(np.pi, ax.get_ylim()[1] * 1.1, 'BACK', ha='center', va='top', fontsize=10, weight='bold')
    ax.text(3*np.pi/2, ax.get_ylim()[1] * 1.1, 'LEFT', ha='right', va='center', fontsize=10, weight='bold')


def plot_timeline(frames, ax):
    """Plot detection timeline"""
    times = []
    counts = []
    avg_confidences = []
    
    for frame in frames:
        if frame['detections']:
            times.append(frame['time'])
            counts.append(len(frame['detections']))
            avg_conf = np.mean([d['confidence'] for d in frame['detections']])
            avg_confidences.append(avg_conf)
    
    if not times:
        ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Plot detection counts
    ax.plot(times, counts, 'b-', linewidth=2, alpha=0.7, label='Detections per Frame')
    ax.fill_between(times, counts, alpha=0.3, color='blue')
    
    # Overlay confidence
    ax2 = ax.twinx()
    ax2.plot(times, avg_confidences, 'g-', linewidth=2, alpha=0.7, label='Avg Confidence')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Detections per Frame', color='blue')
    ax2.set_ylabel('Avg Confidence', color='green')
    ax.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.set_ylim([0, 1])
    ax.set_title('Detection Timeline')
    ax.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')


def plot_frequency_distribution(sources, ax):
    """Plot frequency distribution histogram"""
    if not sources:
        return
    
    frequencies = [s['frequency'] for s in sources]
    counts = [s['occurrence_count'] for s in sources]
    
    # Create bins
    bins = [0, 500, 1000, 2000, 4000, 8000]
    bin_labels = ['0-500', '500-1k', '1k-2k', '2k-4k', '4k-8k']
    
    # Count sources in each bin
    bin_counts = []
    for i in range(len(bins) - 1):
        count = sum(1 for f in frequencies if bins[i] <= f < bins[i+1])
        bin_counts.append(count)
    
    # Plot
    bars = ax.bar(range(len(bin_labels)), bin_counts, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right')
    ax.set_xlabel('Frequency Range (Hz)')
    ax.set_ylabel('Number of Sources')
    ax.set_title('Frequency Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, bin_counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(count), ha='center', va='bottom', fontsize=10)


def plot_direction_distribution(sources, ax):
    """Plot direction distribution"""
    if not sources:
        return
    
    azimuths = [s['azimuth'] for s in sources]
    
    # Define directional bins
    bins = [(315, 45, 'Front'), (45, 135, 'Right'), 
            (135, 225, 'Back'), (225, 315, 'Left')]
    
    bin_counts = []
    bin_labels = []
    
    for az_min, az_max, label in bins:
        if az_min > az_max:  # Wrap-around
            count = sum(1 for az in azimuths if az >= az_min or az < az_max)
        else:
            count = sum(1 for az in azimuths if az_min <= az < az_max)
        bin_counts.append(count)
        bin_labels.append(label)
    
    # Plot
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.barh(bin_labels, bin_counts, color=colors, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Number of Sources')
    ax.set_title('Direction Distribution')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add count labels
    for bar, count in zip(bars, bin_counts):
        if count > 0:
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                   str(count), ha='left', va='center', fontsize=10)


def plot_top_sources_table(sources, ax):
    """Plot table of top sources"""
    if not sources:
        return
    
    # Prepare data
    top_sources = sources[:10]
    data = []
    for i, src in enumerate(top_sources, 1):
        data.append([
            i,
            f"{src['frequency']:.1f}",
            f"{src['azimuth']:.1f}",
            src['occurrence_count'],
            f"{src['avg_confidence']:.3f}"
        ])
    
    # Create table
    columns = ['#', 'Freq (Hz)', 'Azimuth (°)', 'Count', 'Confidence']
    
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=data, colLabels=columns, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    ax.set_title('Top 10 Sources by Occurrence', pad=20, fontsize=12, weight='bold')


def plot_spectrogram(frames, ax):
    """Plot frequency over time (spectrogram-like)"""
    if not frames:
        return
    
    # Extract all detections with time and frequency
    times = []
    frequencies = []
    confidences = []
    source_ids = []
    
    for frame in frames:
        for detection in frame['detections']:
            times.append(frame['time'])
            frequencies.append(detection['frequency'])
            confidences.append(detection['confidence'])
            source_ids.append(detection.get('source_id', -1))
    
    if not times:
        ax.text(0.5, 0.5, 'No detections', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Convert to numpy arrays
    times = np.array(times)
    frequencies = np.array(frequencies)
    confidences = np.array(confidences)
    source_ids = np.array(source_ids)
    
    # Use source_id for colors if available
    if source_ids.max() > 0:
        # Create colormap for track IDs
        n_tracks = source_ids.max() + 1
        colors = plt.cm.tab20(source_ids % 20)  # Cycle through 20 colors
        scatter = ax.scatter(times, frequencies, c=colors, s=20, alpha=0.7, edgecolors='black', linewidths=0.3)
        ax.set_title('Detected Peaks Over Time (Colored by Track ID)')
    else:
        # Fall back to confidence coloring
        scatter = ax.scatter(times, frequencies, c=confidences, s=20, 
                            alpha=0.6, cmap='plasma', vmin=0, vmax=1)
        ax.set_title('Detected Peaks Over Time (Colored by Confidence)')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Confidence', rotation=270, labelpad=20)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True, alpha=0.3)


def plot_track_timeline(results, ax):
    """Plot track lifetimes over time"""
    if 'tracks' not in results or not results['tracks']:
        ax.text(0.5, 0.5, 'No track data available', ha='center', va='center', transform=ax.transAxes)
        return
    
    tracks = results['tracks']
    
    # Sort tracks by track_id
    tracks_sorted = sorted(tracks, key=lambda x: x['track_id'])
    
    # Limit to top 20 tracks by duration
    tracks_sorted = sorted(tracks, key=lambda x: x['duration_seconds'], reverse=True)[:20]
    
    # Create color map
    colors = plt.cm.tab20(np.linspace(0, 1, len(tracks_sorted)))
    
    y_positions = []
    y_labels = []
    
    for i, track in enumerate(tracks_sorted):
        start_time = track['start_time']
        end_time = track['end_time']
        track_id = track['track_id']
        avg_freq = track['avg_frequency']
        
        # Plot horizontal bar for track duration
        ax.barh(i, end_time - start_time, left=start_time, height=0.8,
               color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        y_positions.append(i)
        y_labels.append(f"ID {track_id}\n{avg_freq:.0f}Hz")
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Track ID')
    ax.set_title('Track Lifetimes (Top 20 by Duration)')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Top to bottom


def visualize(filepath, output_file=None):
    """Create comprehensive visualization"""
    
    # Load data
    results = load_results(filepath)
    sources = results['summary']['sources']
    frames = results['frames']
    metadata = results['metadata']
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    
    # 1. Polar plot of sources (top-left, double width)
    ax1 = fig.add_subplot(gs[0, :2], projection='polar')
    plot_polar_sources(sources, ax1)
    
    # 2. Top sources table (top-right)
    ax2 = fig.add_subplot(gs[0, 2])
    plot_top_sources_table(sources, ax2)
    
    # 3. Timeline (second row, full width)
    ax3 = fig.add_subplot(gs[1, :])
    plot_timeline(frames, ax3)
    
    # 4. Track timeline (third row, full width)
    ax4 = fig.add_subplot(gs[2, :])
    plot_track_timeline(results, ax4)
    
    # 5. Frequency distribution (bottom-left)
    ax5 = fig.add_subplot(gs[3, 0])
    plot_frequency_distribution(sources, ax5)
    
    # 6. Direction distribution (bottom-middle)
    ax6 = fig.add_subplot(gs[3, 1])
    plot_direction_distribution(sources, ax6)
    
    # 7. Spectrogram with track IDs (bottom-right)
    ax7 = fig.add_subplot(gs[3, 2])
    plot_spectrogram(frames, ax7)
    
    # Overall title
    title = f"Custom DOA Analysis: {Path(metadata['file']).name}\n"
    title += f"Duration: {metadata['duration']:.1f}s | "
    title += f"Frames: {metadata['frames_processed']} | "
    title += f"Sources: {results['summary']['unique_sources']} | "
    title += f"Tracks: {metadata.get('total_tracks', 0)} | "
    title += f"Validation Rate: {metadata['validation_rate']*100:.1f}%"
    
    fig.suptitle(title, fontsize=14, weight='bold', y=0.99)
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_doa.py <custom_doa.json> [output.png]")
        print("\nExample:")
        print("  python visualize_doa.py outputs/renders/test_custom_doa.json")
        print("  python visualize_doa.py outputs/renders/test_custom_doa.json result.png")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Visualizing: {input_file}")
    visualize(input_file, output_file)
    
    if not output_file:
        print("\n💡 Tip: Add an output filename to save the plot:")
        print(f"   python visualize_doa.py {input_file} output.png")


if __name__ == "__main__":
    main()
