#!/usr/bin/env python3
"""
Visualize individual tracks from Custom DOA results.

Shows track trajectories, frequency evolution, and detailed track statistics.

Usage:
    python visualize_tracks.py <custom_doa.json> [output.png]
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd


def load_results(filepath):
    """Load custom DOA results"""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_track_data(frames, track_id):
    """Extract all detections for a specific track"""
    track_detections = []
    
    for frame in frames:
        for detection in frame['detections']:
            if detection.get('source_id') == track_id:
                track_detections.append({
                    'time': frame['time'],
                    'frame': frame['frame'],
                    'frequency': detection['frequency'],
                    'azimuth': detection['azimuth'],
                    'energy': detection['energy'],
                    'confidence': detection['confidence']
                })
    
    return track_detections


def plot_track_trajectories(tracks, frames, ax):
    """Plot spatial trajectories of top tracks"""
    if not tracks:
        return
    
    # Get top 10 tracks by duration
    top_tracks = sorted(tracks, key=lambda x: x['duration_seconds'], reverse=True)[:10]
    
    # Create color map
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_tracks)))
    
    for i, track in enumerate(top_tracks):
        track_id = track['track_id']
        
        # Extract detections for this track
        detections = extract_track_data(frames, track_id)
        
        if not detections:
            continue
        
        # Convert azimuth to radians for plotting
        azimuths = [np.radians(d['azimuth']) for d in detections]
        times = [d['time'] for d in detections]
        
        # Plot as connected line
        ax.plot(azimuths, times, 'o-', color=colors[i], 
               label=f"Track {track_id} ({track['avg_frequency']:.0f}Hz)",
               markersize=4, linewidth=1.5, alpha=0.7)
    
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim([0, max([d['time'] for track in top_tracks 
                         for d in extract_track_data(frames, track['track_id'])])])
    ax.set_xlabel('Time (s)', labelpad=10)
    ax.set_title('Track Trajectories Over Time\\n(Radius = Time)', pad=20)
    ax.legend(bbox_to_anchor=(1.15, 1.0), fontsize=8)


def plot_track_frequency_evolution(tracks, frames, ax):
    """Plot frequency evolution for top tracks"""
    if not tracks:
        return
    
    # Get top 5 tracks by duration
    top_tracks = sorted(tracks, key=lambda x: x['duration_seconds'], reverse=True)[:5]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_tracks)))
    
    for i, track in enumerate(top_tracks):
        track_id = track['track_id']
        detections = extract_track_data(frames, track_id)
        
        if not detections:
            continue
        
        times = [d['time'] for d in detections]
        frequencies = [d['frequency'] for d in detections]
        
        ax.plot(times, frequencies, 'o-', color=colors[i],
               label=f"Track {track_id}",
               markersize=3, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Frequency Evolution (Top 5 Tracks)')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_track_confidence(tracks, frames, ax):
    """Plot confidence over time for top tracks"""
    if not tracks:
        return
    
    top_tracks = sorted(tracks, key=lambda x: x['duration_seconds'], reverse=True)[:5]
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_tracks)))
    
    for i, track in enumerate(top_tracks):
        track_id = track['track_id']
        detections = extract_track_data(frames, track_id)
        
        if not detections:
            continue
        
        times = [d['time'] for d in detections]
        confidences = [d['confidence'] for d in detections]
        
        ax.plot(times, confidences, 'o-', color=colors[i],
               label=f"Track {track_id}",
               markersize=3, linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Over Time (Top 5 Tracks)')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_track_gantt(tracks, ax):
    """Plot Gantt chart of track lifetimes"""
    if not tracks:
        return
    
    # Sort by start time, limit to top 30
    tracks_sorted = sorted(tracks, key=lambda x: x['start_time'])[:30]
    
    colors = plt.cm.tab20(np.arange(len(tracks_sorted)) % 20)
    
    for i, track in enumerate(tracks_sorted):
        start = track['start_time']
        duration = track['duration_seconds']
        
        ax.barh(i, duration, left=start, height=0.8,
               color=colors[i], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Add track ID label
        ax.text(start + duration/2, i, f"{track['track_id']}",
               ha='center', va='center', fontsize=7, weight='bold')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Track Index')
    ax.set_title('Track Lifetimes (First 30 Tracks)')
    ax.set_yticks([])
    ax.grid(True, alpha=0.3, axis='x')


def plot_track_statistics(tracks, ax):
    """Plot track statistics distribution"""
    if not tracks:
        return
    
    durations = [t['duration_seconds'] for t in tracks]
    detection_counts = [t['detection_count'] for t in tracks]
    
    # Create 2x2 subplot within this ax
    ax.axis('off')
    
    # Duration histogram
    ax1 = plt.subplot(4, 4, 13)
    ax1.hist(durations, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Duration (s)', fontsize=8)
    ax1.set_ylabel('Count', fontsize=8)
    ax1.set_title('Track Duration Distribution', fontsize=9)
    ax1.tick_params(labelsize=7)
    ax1.grid(True, alpha=0.3)
    
    # Detection count histogram
    ax2 = plt.subplot(4, 4, 14)
    ax2.hist(detection_counts, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Detections', fontsize=8)
    ax2.set_ylabel('Count', fontsize=8)
    ax2.set_title('Detections per Track', fontsize=9)
    ax2.tick_params(labelsize=7)
    ax2.grid(True, alpha=0.3)


def plot_track_heatmap(tracks, frames, ax):
    """Plot frequency-time heatmap with tracks"""
    if not tracks:
        return
    
    # Create 2D histogram
    all_times = []
    all_freqs = []
    
    for frame in frames:
        for detection in frame['detections']:
            all_times.append(frame['time'])
            all_freqs.append(detection['frequency'])
    
    if not all_times:
        return
    
    # Create heatmap
    h, xedges, yedges = np.histogram2d(all_times, all_freqs, bins=[50, 50])
    
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    im = ax.imshow(h.T, aspect='auto', origin='lower', extent=extent,
                   cmap='YlOrRd', alpha=0.7, interpolation='gaussian')
    
    # Overlay top tracks
    top_tracks = sorted(tracks, key=lambda x: x['duration_seconds'], reverse=True)[:5]
    colors = plt.cm.cool(np.linspace(0, 1, len(top_tracks)))
    
    for i, track in enumerate(top_tracks):
        detections = extract_track_data(frames, track['track_id'])
        if detections:
            times = [d['time'] for d in detections]
            freqs = [d['frequency'] for d in detections]
            ax.plot(times, freqs, 'o-', color=colors[i], 
                   linewidth=2, markersize=2, alpha=0.8,
                   label=f"Track {track['track_id']}")
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Frequency-Time Heatmap with Top 5 Tracks')
    ax.legend(fontsize=8, loc='upper right')
    
    # Colorbar
    plt.colorbar(im, ax=ax, label='Detection Density')


def visualize_tracks(filepath, output_file=None):
    """Create comprehensive track visualization"""
    
    # Load data
    results = load_results(filepath)
    
    if 'tracks' not in results or not results['tracks']:
        print("No track data available in results")
        return
    
    tracks = results['tracks']
    frames = results['frames']
    metadata = results['metadata']
    
    print(f"Loaded {len(tracks)} tracks from {len(frames)} frames")
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.4)
    
    # 1. Track trajectories (polar)
    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    plot_track_trajectories(tracks, frames, ax1)
    
    # 2. Frequency evolution
    ax2 = fig.add_subplot(gs[0, 1])
    plot_track_frequency_evolution(tracks, frames, ax2)
    
    # 3. Confidence over time
    ax3 = fig.add_subplot(gs[0, 2])
    plot_track_confidence(tracks, frames, ax3)
    
    # 4. Gantt chart (full width)
    ax4 = fig.add_subplot(gs[1, :])
    plot_track_gantt(tracks, ax4)
    
    # 5. Frequency-time heatmap (double width)
    ax5 = fig.add_subplot(gs[2, :2])
    plot_track_heatmap(tracks, frames, ax5)
    
    # 6. Statistics
    ax6 = fig.add_subplot(gs[2, 2])
    plot_track_statistics(tracks, ax6)
    
    # Overall title
    title = f"Track Analysis: {Path(metadata['file']).name}\n"
    title += f"Total Tracks: {len(tracks)} | "
    title += f"Duration: {metadata['duration']:.1f}s | "
    title += f"Avg Track Duration: {np.mean([t['duration_seconds'] for t in tracks]):.2f}s"
    
    fig.suptitle(title, fontsize=14, weight='bold')
    
    # Statistics text
    durations = [t['duration_seconds'] for t in tracks]
    det_counts = [t['detection_count'] for t in tracks]
    
    stats_text = f"""Track Statistics:
    • Longest: {max(durations):.2f}s
    • Shortest: {min(durations):.2f}s
    • Median: {np.median(durations):.2f}s
    • Max detections: {max(det_counts)}
    • Avg detections: {np.mean(det_counts):.1f}
    """
    
    fig.text(0.98, 0.02, stats_text, fontsize=9, family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            verticalalignment='bottom', horizontalalignment='right')
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Track visualization saved to: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_tracks.py <custom_doa.json> [output.png]")
        print("\nExample:")
        print("  python visualize_tracks.py outputs/renders/test_custom_doa.json")
        print("  python visualize_tracks.py outputs/renders/test_custom_doa.json tracks.png")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(input_file).exists():
        print(f"Error: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Visualizing tracks from: {input_file}")
    visualize_tracks(input_file, output_file)
    
    if not output_file:
        print("\n💡 Tip: Add an output filename to save the plot:")
        print(f"   python visualize_tracks.py {input_file} tracks.png")


if __name__ == "__main__":
    main()
